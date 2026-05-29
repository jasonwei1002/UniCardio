"""Training loop for the scalar BP regression head.

Mirrors :mod:`src.trainer_module.trainer` (RF trainer) in structure — same
checkpoint / scheduler / AMP / SwanLab plumbing — but, unlike the RF trainer,
**does not randomly sample a task per batch**. The BP head trains on a single
fixed ABP-target task (``trainer.bp_task``, default ``ecgppg2abp``), so every
step supervises the same vital subset:

    ecgppg2abp  -> active_vitals=["ecg","ppg"]  (both vitals, averaged)

The task's condition modalities (via :func:`cond_slots_to_vitals`) become the
BP head's ``active_vitals`` for every step. This matches how the evaluation
cascade feeds the head for the multimodal BP task (see
:mod:`src.trainer_module.bp_metrics`). There is no time embedding, attention
mask, or Euler sampler.

Loss mode (MD-ViSCo §6.2.2, ``L_ref = L_MAE + L_WCL``), resolved from
``(trainer.wcl, stage)`` by :func:`_resolve_loss_mode`:
- ``l1`` — pure L1 in label space (``wcl.enabled=false``; legacy).
- ``wcl_only`` — self-supervised contrastive pretraining (stage-1): only the
  multi-WCL terms, training the encoders/projection + demo MLP (skips the
  fusion encoders + MlpBP heads). Selected by min val WCL loss.
- ``l1+wcl`` — L1 + multi-WCL (stage-2 finetune). Selected by min val L1.

Batch contract (5-tuple): each batch is ``(signal: (B, 3, L), sbp_dbp: (B, 2),
demographics: (B, 6), abp_minmax: (B, 2), age_raw: (B, 1))``. The trainer slices
ECG+PPG (``signal[:, :2, :]``) as input and forwards ``demographics``; the ABP
slot is unused. The regression target is selected by ``bp_label_source``:
``segment_minmax`` (per-segment ABP max/min from ``abp_minmax``, default —
matches MD-ViSCo "SBP/DBP = max/min of ABP") or ``per_cycle_mean`` (CSV labels,
legacy). ``abp_minmax`` also supplies the WCL waveform-term raw SBP/DBP weights;
``age_raw`` the WCL age term. See ``src/data_module/cardiac_dataset.py``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import swanlab
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    _LRScheduler,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..model_module.tasks import (
    Slot,
    TaskSpec,
    cond_slots_to_vitals,
    get_task,
)
from ..utils.checkpoint import load_checkpoint, save_checkpoint, unwrap_model
from ..utils.normalization import BPLabelNorm, split_abp_minmax
from .csv_logger import SimpleCSVLogger
from .wcl import DEFAULT_WCL_TERMS, WCLTerm, multi_wcl

logger = logging.getLogger(__name__)

# Loss modes (set by stage + trainer.wcl config):
#   "l1"       — pure L1 in label space (legacy / WCL disabled).
#   "wcl_only" — self-supervised contrastive pretraining (MD-ViSCo stage-1):
#                only the WCL terms, training the encoders/projection + demo MLP.
#   "l1+wcl"   — L_ref = L_MAE + L_WCL (MD-ViSCo refinement finetune, stage-2).
_LOSS_MODES = ("l1", "wcl_only", "l1+wcl")


def _resolve_loss_mode(cfg: Mapping[str, Any], stage: str) -> str:
    """Map ``(trainer.wcl, stage)`` to a loss mode (see ``_LOSS_MODES``)."""
    wcl_cfg = cfg.get("wcl", {}) or {}
    if not bool(wcl_cfg.get("enabled", False)):
        return "l1"
    if stage == "pretrain" and bool(wcl_cfg.get("pretrain_contrastive_only", True)):
        return "wcl_only"
    return "l1+wcl"


def _wcl_terms(cfg: Mapping[str, Any]) -> tuple[WCLTerm, ...]:
    """WCL term configs — defaults to MD-ViSCo's six-term multi-WCL.

    Optional ``trainer.wcl.terms`` (list of dicts) overrides per-term
    hyperparameters; absent fields fall back to the dataclass defaults.
    """
    raw = (cfg.get("wcl", {}) or {}).get("terms")
    if not raw:
        return DEFAULT_WCL_TERMS
    return tuple(WCLTerm(**dict(t)) for t in raw)


def _segment_minmax_target(abp_minmax: Tensor, bp_norm: BPLabelNorm | None) -> Tensor:
    """``abp_minmax = (dbp_seg, sbp_seg)`` -> ``(SBP, DBP)`` target in label space.

    MD-ViSCo regresses SBP/DBP = the per-segment ABP max/min; the target is
    ``(sbp_seg, dbp_seg)`` normalized by ``bp_norm`` (identity if ``None``).
    """
    sbp_seg, dbp_seg = split_abp_minmax(abp_minmax)
    sbp_dbp_raw = torch.stack([sbp_seg, dbp_seg], dim=-1)
    return bp_norm.normalize(sbp_dbp_raw) if bp_norm is not None else sbp_dbp_raw


def _resolve_target(
    sbp_dbp: Tensor,
    abp_minmax: Tensor | None,
    bp_norm: BPLabelNorm | None,
    bp_label_source: str,
) -> Tensor:
    """Pick the BP regression target for one batch.

    ``segment_minmax`` derives it from ``abp_minmax`` (per-segment max/min);
    ``per_cycle_mean`` uses the dataset's ``sbp_dbp`` (CSV labels) directly.
    """
    if bp_label_source == "segment_minmax":
        if abp_minmax is None:
            raise RuntimeError(
                "bp_label_source='segment_minmax' needs abp_minmax (batch[3])."
            )
        return _segment_minmax_target(abp_minmax, bp_norm)
    return sbp_dbp


def _wcl_weights(
    abp_minmax: Tensor,
    demographics: Tensor | None,
    age_raw: Tensor | None,
) -> dict[str, Tensor]:
    """Build the raw-label weight dict for :func:`multi_wcl`.

    Waveform terms weight by raw mmHg SBP/DBP (= per-segment extrema in
    ``abp_minmax``); PI terms by raw gender (``demographics[:, 1]``) and raw
    age (years). Missing inputs simply omit their keys, so ``multi_wcl`` skips
    the corresponding terms.
    """
    weights: dict[str, Tensor] = {
        "y_sbp_raw": abp_minmax[:, 1],
        "y_dbp_raw": abp_minmax[:, 0],
    }
    if demographics is not None:
        weights["gender_raw"] = demographics[:, 1]
    if age_raw is not None:
        weights["age_raw"] = age_raw.reshape(-1)
    return weights


def _amp_enabled(cfg: Mapping[str, Any], device: torch.device) -> bool:
    amp_cfg = cfg.get("amp", {}) or {}
    return bool(amp_cfg.get("enabled", True)) and device.type == "cuda"


def _build_optimizer(model: nn.Module, cfg: Mapping[str, Any]) -> Optimizer:
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("BPHead has no trainable parameters.")
    fused = torch.cuda.is_available() and any(p.is_cuda for p in trainable)
    return Adam(
        trainable,
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 1.0e-6)),
        fused=fused,
    )


def _build_scheduler(
    optimizer: Optimizer,
    cfg: Mapping[str, Any],
    total_steps: int,
) -> _LRScheduler:
    """Same cosine schedule API as :mod:`src.trainer_module.trainer`."""
    sched_cfg = cfg["lr_scheduler"]
    name = str(sched_cfg["name"]).lower()
    if name != "cosine":
        raise ValueError(f"Unsupported lr_scheduler '{name}'.")
    min_lr = float(sched_cfg.get("min_lr", 0.0))
    first_cycle_pct = float(sched_cfg.get("first_cycle_pct", 1.0))
    if not 0.0 < first_cycle_pct <= 1.0:
        raise ValueError(f"first_cycle_pct must be in (0, 1]; got {first_cycle_pct}")
    if first_cycle_pct >= 1.0:
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=min_lr)
    cycle_mult = int(sched_cfg.get("cycle_mult", 1))
    first_cycle_steps = int(round(first_cycle_pct * total_steps))
    if first_cycle_steps < 2:
        raise ValueError(
            f"first_cycle_pct * total_steps = {first_cycle_steps} < 2"
        )
    return CosineAnnealingWarmRestarts(
        optimizer, T_0=first_cycle_steps, T_mult=cycle_mult, eta_min=min_lr
    )


def _resolve_bp_task(model: nn.Module, task_name: str) -> TaskSpec:
    """Resolve the single ABP-target task the BP head trains on.

    Unlike the RF trainer, the BP head does not sample a task per batch — it
    trains on one fixed task (``trainer.bp_task``, default ``ecgppg2abp``). The
    task must target ABP and condition only on vitals the head has encoders for.
    """
    spec = get_task(task_name)
    if int(spec.target_slot) != int(Slot.ABP):
        raise ValueError(
            f"BP head task '{task_name}' must target ABP (slot {int(Slot.ABP)}); "
            f"got target_slot={int(spec.target_slot)}."
        )
    model_vitals = set(unwrap_model(model).config.vitals)
    vitals = cond_slots_to_vitals(spec)
    if not vitals or not set(vitals) <= model_vitals:
        raise ValueError(
            f"BP head task '{task_name}' needs vitals {vitals} but the head has "
            f"{sorted(model_vitals)}. Check trainer.bp_task and model.vitals."
        )
    return spec


def _empty_acc() -> dict[str, float]:
    return {
        "loss": 0.0,
        "mae_sbp": 0.0,
        "mae_dbp": 0.0,
        "me_sbp": 0.0,
        "me_dbp": 0.0,
        "sq_sbp": 0.0,
        "sq_dbp": 0.0,
        "n": 0.0,
    }


def _finalize_acc(a: Mapping[str, float]) -> dict[str, float]:
    n = max(a["n"], 1.0)
    me_sbp = a["me_sbp"] / n
    me_dbp = a["me_dbp"] / n
    # Sample SD (ddof=1) per AAMI/BHS convention, matching
    # :mod:`src.trainer_module.bp_metrics` (which uses np.std(ddof=1)). The
    # accumulator-based population variance ``sq/n - me^2`` is rescaled by
    # ``n/(n-1)``; needs n > 1 (true for any real val/test split).
    bessel = n / (n - 1.0) if n > 1.0 else 1.0
    var_sbp = max((a["sq_sbp"] / n - me_sbp**2) * bessel, 0.0)
    var_dbp = max((a["sq_dbp"] / n - me_dbp**2) * bessel, 0.0)
    return {
        "loss": a["loss"] / n,
        "mae_sbp": a["mae_sbp"] / n,
        "mae_dbp": a["mae_dbp"] / n,
        "mae_mean": (a["mae_sbp"] + a["mae_dbp"]) / (2 * n),
        "me_sbp": me_sbp,
        "me_dbp": me_dbp,
        "sd_sbp": var_sbp**0.5,
        "sd_dbp": var_dbp**0.5,
    }


def _unpack_batch(
    batch: Any, device: torch.device
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None]:
    """Unpack the dataset tuple to ``(signal, sbp_dbp, demographics, abp_minmax,
    age_raw)`` on ``device``.

    The dataset emits a 5-tuple ``(signal, sbp_dbp, demographics, abp_minmax,
    age_raw)``; ``demographics`` / ``abp_minmax`` / ``age_raw`` are ``None`` for
    shorter legacy tuples (the head then runs waveform-only / per-cycle-mean /
    no-age-WCL respectively). ``abp_minmax`` feeds both the ``segment_minmax``
    target and the WCL waveform-term raw weights; ``age_raw`` the WCL age term.
    """
    if len(batch) < 2:
        raise RuntimeError(
            "BP head training requires at least (signal, sbp_dbp) batches; "
            f"got a {len(batch)}-tuple."
        )
    signal = batch[0].to(device, non_blocking=True)
    sbp_dbp = batch[1].to(device, non_blocking=True)
    demographics = batch[2].to(device, non_blocking=True) if len(batch) >= 3 else None
    abp_minmax = batch[3].to(device, non_blocking=True) if len(batch) >= 4 else None
    age_raw = batch[4].to(device, non_blocking=True) if len(batch) >= 5 else None
    return signal, sbp_dbp, demographics, abp_minmax, age_raw


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    *,
    tasks: Sequence[TaskSpec],
    amp_enabled: bool,
    max_batches: int | None = None,
    bp_norm: BPLabelNorm | None = None,
    bp_label_source: str = "per_cycle_mean",
) -> dict[str, dict[str, float]]:
    """Per-task L1 loss + per-channel MAE / ME / SD on the val set.

    For each task the head is evaluated with ``active_vitals`` restricted to the
    task's condition modalities, matching :mod:`src.trainer_module.bp_metrics`.
    Each vital's ``(SBP, DBP)`` is computed once per batch and tasks are
    assembled by averaging the relevant vitals (exact in eval mode — no dropout
    — see ``test_aggregate_equals_mean_of_single_vitals``), so a forward runs
    once per distinct vital rather than once per task.

    ``loss`` is the L1 training objective in whichever space the dataset emits
    labels (raw mmHg if ``bp_norm is None``; otherwise globally min-max
    normalized [0, 1] following MD-ViSCo Sec III.D). MAE / ME / SD are always
    reported in **mmHg** by
    inverting the normalization on the residual, so AAMI thresholds
    (|ME| ≤ 5, σ ≤ 8) read off the same units regardless of training space.
    """
    model.eval()
    needed_vitals = sorted({v for t in tasks for v in cond_slots_to_vitals(t)})
    acc = {t.name: _empty_acc() for t in tasks}

    for batch_idx, batch in enumerate(val_loader):
        signal, sbp_dbp, demographics, abp_minmax, _ = _unpack_batch(batch, device)
        target = _resolve_target(sbp_dbp, abp_minmax, bp_norm, bp_label_source)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled
        ):
            vital_pred = {
                v: model(signal[:, :2, :], demographics, active_vitals=[v]).float()
                for v in needed_vitals
            }
        bsz = int(target.shape[0])
        target_f = target.float()
        for t in tasks:
            vs = cond_slots_to_vitals(t)
            pred = torch.stack([vital_pred[v] for v in vs], dim=0).mean(dim=0)
            diff = pred - target_f
            diff_mmhg = bp_norm.denormalize_diff(diff) if bp_norm is not None else diff
            a = acc[t.name]
            # Reported ``loss`` mirrors the L1 training objective (label space).
            a["loss"] += float(diff.abs().mean(dim=-1).sum().item())
            a["mae_sbp"] += float(diff_mmhg[:, 0].abs().sum().item())
            a["mae_dbp"] += float(diff_mmhg[:, 1].abs().sum().item())
            a["me_sbp"] += float(diff_mmhg[:, 0].sum().item())
            a["me_dbp"] += float(diff_mmhg[:, 1].sum().item())
            a["sq_sbp"] += float((diff_mmhg[:, 0] ** 2).sum().item())
            a["sq_dbp"] += float((diff_mmhg[:, 1] ** 2).sum().item())
            a["n"] += bsz
        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break

    model.train()
    return {name: _finalize_acc(a) for name, a in acc.items()}


@torch.no_grad()
def _evaluate_wcl(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    *,
    active_vitals: Sequence[str],
    terms: tuple[WCLTerm, ...],
    amp_enabled: bool,
    max_batches: int | None = None,
) -> float:
    """Mean total multi-WCL loss over the val set.

    Selection metric for the contrastive *pretraining* stage (``wcl_only``),
    where per-task BP metrics are meaningless (the MlpBP heads are untrained).
    Runs the encoder-only embedding path (no fusion / heads).
    """
    core = unwrap_model(model)
    core.eval()
    total, n_batches = 0.0, 0
    for batch_idx, batch in enumerate(val_loader):
        signal, _, demographics, abp_minmax, age_raw = _unpack_batch(batch, device)
        if abp_minmax is None:
            raise RuntimeError("WCL eval needs abp_minmax (batch[3]) for raw weights.")
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled
        ):
            emb = core.encode_embeddings(
                signal[:, :2, :], demographics, active_vitals=active_vitals
            )
        wcl_total, _ = multi_wcl(emb, _wcl_weights(abp_minmax, demographics, age_raw), terms)
        total += float(wcl_total.item())
        n_batches += 1
        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break
    core.train()
    return total / max(n_batches, 1)


def _log_eval(prefix: str, per_task: Mapping[str, dict[str, float]]) -> None:
    for name, m in per_task.items():
        logger.info(
            "  %s %-12s | loss %.4f | mae_mean %.2f | SBP ME %+.2f ± %.2f | "
            "DBP ME %+.2f ± %.2f mmHg",
            prefix, name, m["loss"], m["mae_mean"],
            m["me_sbp"], m["sd_sbp"], m["me_dbp"], m["sd_dbp"],
        )


def train(
    model: nn.Module,
    cfg: DictConfig | Mapping[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    *,
    device: torch.device,
    output_dir: str | Path,
    test_loader: DataLoader | None = None,
    bp_norm: BPLabelNorm | None = None,
    bp_label_source: str = "per_cycle_mean",
) -> None:
    """Train the BP head on a single fixed ABP-target task (no task sampling).

    Loss mode is resolved from ``(trainer.wcl, stage)`` (see
    :func:`_resolve_loss_mode`): ``l1`` (legacy / WCL off), ``wcl_only``
    (MD-ViSCo stage-1 self-supervised contrastive pretraining of the encoders),
    or ``l1+wcl`` (MD-ViSCo refinement finetune, ``L_ref = L_MAE + L_WCL``).

    Args:
        model: :class:`BPHead` (optionally wrapped by ``torch.compile``).
        cfg: trainer config (Hydra DictConfig or plain dict); ``bp_task``
            selects the single ABP-target task to train on (default
            ``ecgppg2abp``); ``wcl`` enables/configures the multi-WCL terms.
        train_loader / val_loader / test_loader: DataLoaders returning the
            dataset 5-tuple ``(signal, sbp_dbp, demographics, abp_minmax, age_raw)``.
        device: torch device.
        output_dir: Hydra-created run directory.
        bp_norm: when set, SBP/DBP labels live in globally min-max normalized
            [0, 1] (MD-ViSCo Sec III.D); ``_evaluate`` inverses to mmHg for
            AAMI metric reporting. ``None`` = raw-mmHg labels (legacy).
        bp_label_source: ``"per_cycle_mean"`` (CSV labels, legacy) or
            ``"segment_minmax"`` (per-segment ABP max/min from ``abp_minmax``,
            matching MD-ViSCo's "SBP/DBP = max/min of ABP" definition, making
            the head's ME/SD measure the same quantity as MD-ViSCo Table III).
    """
    if isinstance(cfg, DictConfig):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    else:
        cfg_dict = dict(cfg)

    output_dir = Path(output_dir)
    ckpt_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(cfg_dict["epochs"])
    val_every = int(cfg_dict.get("val_every", 1))
    ckpt_every = int(cfg_dict.get("ckpt_every", 1))
    log_every_n_steps = int(cfg_dict.get("log_every_n_steps", 50))
    grad_clip_norm = float(cfg_dict.get("grad_clip_norm", 1.0))

    model.to(device)

    stage = str(cfg_dict.get("stage", "pretrain"))
    if stage not in ("pretrain", "finetune"):
        raise ValueError(f"trainer.stage must be 'pretrain' or 'finetune'; got {stage!r}")

    init_from = cfg_dict.get("init_from")
    if stage == "finetune":
        if not init_from:
            raise ValueError(
                "trainer.stage='finetune' requires trainer.init_from=<BP head ckpt>"
            )
        load_checkpoint(init_from, model=model, map_location=device)
        logger.info("BP head finetune: loaded weights from %s", init_from)
    elif init_from:
        load_checkpoint(init_from, model=model, map_location=device)
        logger.info("BP head pretrain: warm-started from %s", init_from)

    task = _resolve_bp_task(model, str(cfg_dict.get("bp_task", "ecgppg2abp")))
    tasks = [task]
    active_vitals = cond_slots_to_vitals(task)
    logger.info(
        "BP head fixed task: %s (active_vitals=%s)", task.name, active_vitals
    )

    if bp_label_source not in ("per_cycle_mean", "segment_minmax"):
        raise ValueError(
            f"bp_label_source must be 'per_cycle_mean' or 'segment_minmax'; "
            f"got {bp_label_source!r}"
        )
    loss_mode = _resolve_loss_mode(cfg_dict, stage)
    wcl_terms = _wcl_terms(cfg_dict) if loss_mode != "l1" else ()
    # WCL forward paths return a dict of embeddings; route them through the
    # unwrapped module so torch.compile (which dislikes dict outputs) is bypassed
    # only for those calls. The pure-L1 path keeps the (compiled) ``model``.
    core = unwrap_model(model)
    logger.info(
        "BP head loss_mode=%s | bp_label_source=%s | wcl_terms=%d",
        loss_mode, bp_label_source, len(wcl_terms),
    )

    optimizer = _build_optimizer(model, cfg_dict)
    steps_per_epoch = len(train_loader)
    scheduler = _build_scheduler(optimizer, cfg_dict, epochs * steps_per_epoch)
    amp_enabled = _amp_enabled(cfg_dict, device)

    csv_fields = ["epoch", "lr", "epoch_time_s", "train_loss"]
    csv_fields += [f"train_loss_{t.name}" for t in tasks]
    csv_fields += ["val_loss_mean"]  # selection scalar = mean per-task L1 (MD-ViSCo: min val loss)
    for t in tasks:
        csv_fields += [
            f"val_{t.name}_loss",
            f"val_{t.name}_mae_mean",
            f"val_{t.name}_me_sbp",
            f"val_{t.name}_sd_sbp",
            f"val_{t.name}_me_dbp",
            f"val_{t.name}_sd_dbp",
        ]
    csv_logger = SimpleCSVLogger(
        log_dir / str(cfg_dict.get("log_filename", "bp_head_loss.csv")),
        fieldnames=csv_fields,
    )

    best_val = float("inf")
    global_step = 0
    model.train()

    for epoch in range(epochs):
        epoch_start = time.time()
        train_loss_sum = torch.zeros((), device=device)
        train_n = 0
        window_loss = torch.zeros((), device=device)
        window_n = 0
        task_loss_sum = {t.name: torch.zeros((), device=device) for t in tasks}
        task_loss_count = {t.name: 0 for t in tasks}
        # L1 / WCL component accumulators (on-device; reduced to floats once per
        # epoch at logging time). Unused in "l1" mode.
        l1_sum = torch.zeros((), device=device)
        wcl_sum = torch.zeros((), device=device)
        wcl_term_sum: dict[str, Tensor] = {}

        it = tqdm(
            train_loader,
            mininterval=5.0,
            maxinterval=50.0,
            disable=False,
            desc=f"bp_head ep {epoch}/{epochs - 1}",
        )
        for batch in it:
            signal, sbp_dbp, demographics, abp_minmax, age_raw = _unpack_batch(
                batch, device
            )
            target = _resolve_target(sbp_dbp, abp_minmax, bp_norm, bp_label_source)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                l1 = signal.new_zeros(())
                wcl_total = signal.new_zeros(())
                wcl_vals: dict[str, float] = {}
                if loss_mode == "l1":
                    pred = model(signal[:, :2, :], demographics, active_vitals=active_vitals)
                    l1 = (pred - target).abs().mean()
                elif loss_mode == "wcl_only":
                    # Contrastive pretraining: encoders only, no fusion/heads.
                    emb = core.encode_embeddings(
                        signal[:, :2, :], demographics, active_vitals=active_vitals
                    )
                    wcl_total, wcl_vals = multi_wcl(
                        emb, _wcl_weights(abp_minmax, demographics, age_raw), wcl_terms
                    )
                else:  # "l1+wcl"
                    pred, emb = core(
                        signal[:, :2, :], demographics,
                        active_vitals=active_vitals, return_embeddings=True,
                    )
                    l1 = (pred - target).abs().mean()
                    wcl_total, wcl_vals = multi_wcl(
                        emb, _wcl_weights(abp_minmax, demographics, age_raw), wcl_terms
                    )
                loss = l1 + wcl_total
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            scheduler.step()

            loss_d = loss.detach()
            train_loss_sum += loss_d
            train_n += 1
            window_loss += loss_d
            window_n += 1
            task_loss_sum[task.name] += loss_d
            task_loss_count[task.name] += 1
            if loss_mode != "l1":
                # On-device accumulation only (no per-batch .item() sync); the
                # L1/WCL split + per-term values are reduced to floats once per
                # epoch when logged. In "l1" mode these are unused.
                l1_sum += l1.detach()
                wcl_sum += wcl_total.detach()
                for k, v in wcl_vals.items():
                    wcl_term_sum[k] = wcl_term_sum.get(k, 0.0) + v
            global_step += 1

            if window_n >= log_every_n_steps:
                swanlab.log(
                    {
                        "train/loss": float((window_loss / window_n).item()),
                        "train/lr": float(optimizer.param_groups[0]["lr"]),
                    },
                    step=global_step,
                )
                window_loss.zero_()
                window_n = 0

        if window_n > 0:
            swanlab.log(
                {
                    "train/loss": float((window_loss / window_n).item()),
                    "train/lr": float(optimizer.param_groups[0]["lr"]),
                },
                step=global_step,
            )

        epoch_time = time.time() - epoch_start
        avg_train_loss = float((train_loss_sum / max(train_n, 1)).item())
        lr_val = float(optimizer.param_groups[0]["lr"])
        # nan (not 0.0) when a task was never sampled this epoch, so the log
        # does not read as "loss 0". With real data every task is sampled.
        per_task_train = {
            t.name: (
                float((task_loss_sum[t.name] / task_loss_count[t.name]).item())
                if task_loss_count[t.name] > 0
                else float("nan")
            )
            for t in tasks
        }
        logger.info(
            "bp_head epoch %d/%d | train_loss %.4f | lr %.2e | %.1fs | per-task %s",
            epoch, epochs - 1, avg_train_loss, lr_val, epoch_time,
            {k: round(v, 4) for k, v in per_task_train.items()},
        )

        row: dict[str, Any] = {
            "epoch": epoch,
            "lr": lr_val,
            "epoch_time_s": round(epoch_time, 2),
            "train_loss": avg_train_loss,
        }
        for t in tasks:
            row[f"train_loss_{t.name}"] = per_task_train[t.name]

        epoch_metrics: dict[str, float] = {
            "epoch/train_loss": avg_train_loss,
            "epoch/lr": lr_val,
            "epoch/time_s": epoch_time,
        }
        for t in tasks:
            epoch_metrics[f"epoch/train_loss_{t.name}"] = per_task_train[t.name]
        if loss_mode != "l1":
            denom = max(train_n, 1)
            epoch_metrics["epoch/l1_loss"] = float((l1_sum / denom).item())
            epoch_metrics["epoch/wcl_loss"] = float((wcl_sum / denom).item())
            for k, s in wcl_term_sum.items():
                epoch_metrics[f"epoch/wcl_{k}"] = float((s / denom).item())

        if val_loader is not None and (epoch + 1) % val_every == 0:
            if loss_mode == "wcl_only":
                # Contrastive pretraining: BP metrics are meaningless (heads
                # untrained), so select best.pt by the min val WCL loss.
                sel = _evaluate_wcl(
                    model, val_loader, device,
                    active_vitals=active_vitals, terms=wcl_terms,
                    amp_enabled=amp_enabled,
                )
                row["val_loss_mean"] = sel
                epoch_metrics["val/loss_mean"] = sel
                epoch_metrics["val/wcl_loss"] = sel
                logger.info(
                    "  val | mean WCL loss %.5f (contrastive-pretrain selection)", sel
                )
            else:
                per_task = _evaluate(
                    model, val_loader, device,
                    tasks=tasks,
                    amp_enabled=amp_enabled,
                    bp_norm=bp_norm,
                    bp_label_source=bp_label_source,
                )
                # MD-ViSCo selects best by the minimum validation loss; in
                # l1+wcl the selection scalar is the mean per-task val L1 (label
                # space), matching MD-ViSCo's L_MAE monitoring.
                sel = sum(m["loss"] for m in per_task.values()) / len(per_task)
                row["val_loss_mean"] = sel
                for name, m in per_task.items():
                    row[f"val_{name}_loss"] = m["loss"]
                    row[f"val_{name}_mae_mean"] = m["mae_mean"]
                    row[f"val_{name}_me_sbp"] = m["me_sbp"]
                    row[f"val_{name}_sd_sbp"] = m["sd_sbp"]
                    row[f"val_{name}_me_dbp"] = m["me_dbp"]
                    row[f"val_{name}_sd_dbp"] = m["sd_dbp"]
                    epoch_metrics[f"val/{name}/mae_mean"] = m["mae_mean"]
                    epoch_metrics[f"val/{name}/me_sbp"] = m["me_sbp"]
                    epoch_metrics[f"val/{name}/sd_sbp"] = m["sd_sbp"]
                    epoch_metrics[f"val/{name}/me_dbp"] = m["me_dbp"]
                    epoch_metrics[f"val/{name}/sd_dbp"] = m["sd_dbp"]
                epoch_metrics["val/loss_mean"] = sel
                logger.info("  val | mean L1 loss over tasks %.4f (selection metric)", sel)
                _log_eval("val", per_task)

            if sel < best_val:
                best_val = sel
                save_checkpoint(
                    ckpt_dir / "best.pt",
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=scheduler,
                    config=cfg_dict,
                    extra={"val_loss_mean": best_val},
                )

        csv_logger.log_mapping(row)
        swanlab.log(epoch_metrics, step=epoch)

        if (epoch + 1) % ckpt_every == 0:
            save_checkpoint(
                ckpt_dir / "latest.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                lr_scheduler=scheduler,
                config=cfg_dict,
            )

    if test_loader is not None:
        best_path = ckpt_dir / "best.pt"
        if best_path.exists():
            load_checkpoint(best_path, model=model, map_location=device)
            logger.info("Reloaded best.pt for BP head test evaluation.")
        else:
            logger.warning(
                "best.pt not found at %s; evaluating with current weights.",
                best_path,
            )
        per_task = _evaluate(
            model, test_loader, device,
            tasks=tasks,
            amp_enabled=amp_enabled,
            bp_norm=bp_norm,
            bp_label_source=bp_label_source,
        )
        sel = sum(m["loss"] for m in per_task.values()) / len(per_task)
        logger.info("bp_head test | mean L1 loss over tasks %.4f", sel)
        _log_eval("test", per_task)
        test_swan: dict[str, float] = {"test/loss_mean": sel}
        for name, m in per_task.items():
            for k, v in m.items():
                test_swan[f"test/{name}/{k}"] = v
        swanlab.log(test_swan)
        test_row: dict[str, Any] = {
            "epoch": -1,
            "lr": 0.0,
            "epoch_time_s": 0.0,
            "train_loss": float("nan"),
            "val_loss_mean": sel,
        }
        for t in tasks:
            per_task_train_nan = float("nan")
            test_row[f"train_loss_{t.name}"] = per_task_train_nan
            m = per_task[t.name]
            test_row[f"val_{t.name}_loss"] = m["loss"]
            test_row[f"val_{t.name}_mae_mean"] = m["mae_mean"]
            test_row[f"val_{t.name}_me_sbp"] = m["me_sbp"]
            test_row[f"val_{t.name}_sd_sbp"] = m["sd_sbp"]
            test_row[f"val_{t.name}_me_dbp"] = m["me_dbp"]
            test_row[f"val_{t.name}_sd_dbp"] = m["sd_dbp"]
        csv_logger.log_mapping(test_row)

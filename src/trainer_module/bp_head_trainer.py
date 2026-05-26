"""Training loop for the scalar BP regression head.

Mirrors :mod:`src.trainer_module.trainer` (RF trainer) in structure — same
checkpoint / scheduler / AMP / SwanLab plumbing, and the **same per-batch
weighted task sampling**. Each batch draws one ABP-target task from
``trainer.task_weights``; the task's condition modalities (via
:func:`cond_slots_to_vitals`) become the BP head's ``active_vitals`` for that
step, so each vital subset is supervised directly:

    ecg2abp     -> active_vitals=["ecg"]        (ECG-only stack)
    ppg2abp     -> active_vitals=["ppg"]        (PPG-only stack)
    ecgppg2abp  -> active_vitals=["ecg","ppg"]  (both, averaged)

This matches how the evaluation cascade feeds the head (see
:mod:`src.trainer_module.bp_metrics`) and removes the train/eval gap that
existed when training always averaged over both vitals. The objective is
``L1(pred, target)`` (MD-ViSCo §6.2.2 uses L1 + multi-WCL; the WCL term is
deferred); there is still no time embedding, attention mask, or Euler sampler
— task sampling here only selects the vital subset, not a different target
(every sample has one SBP/DBP label).

Batch contract (3-tuple): each batch is
``(signal: (B, 3, L), sbp_dbp: (B, 2), demographics: (B, 6))``. The trainer
slices ECG+PPG (``signal[:, :2, :]``) as input and forwards ``demographics``;
the ABP slot is unused (the head is strictly causal w.r.t. the source
modalities, like MD-ViSCo's refinement model). SBP/DBP come from PulseDB's CSV
(per-cycle mean labeling), not per-segment min/max — see
``src/data_module/cardiac_dataset.py``.
"""

from __future__ import annotations

import logging
import random
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
    active_task_pairs,
    cond_slots_to_vitals,
)
from ..utils.checkpoint import load_checkpoint, save_checkpoint, unwrap_model
from ..utils.normalization import BPLabelNorm
from .csv_logger import SimpleCSVLogger

logger = logging.getLogger(__name__)


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


def _sample_task(pairs: Sequence[tuple[TaskSpec, float]]) -> TaskSpec:
    """Weighted single-task draw (mirrors the RF trainer's ``_sample_task``)."""
    tasks = [t for t, _ in pairs]
    weights = [w for _, w in pairs]
    return random.choices(tasks, weights=weights, k=1)[0]


def _resolve_bp_tasks(
    model: nn.Module,
    task_weights: Mapping[str, float] | None,
) -> list[tuple[TaskSpec, float]]:
    """ABP-target tasks the BP head can actually serve, with their weights.

    Reuses the global ``task_weights`` convention but keeps only tasks whose
    target is ABP and whose condition vitals (ECG/PPG) are all configured
    encoders on this head. Non-ABP tasks (e.g. ecg2ppg) and tasks needing a
    missing vital are dropped with a WARNING.
    """
    model_vitals = set(unwrap_model(model).config.vitals)
    pairs: list[tuple[TaskSpec, float]] = []
    for spec, w in active_task_pairs(task_weights):
        if int(spec.target_slot) != int(Slot.ABP):
            continue
        vitals = cond_slots_to_vitals(spec)
        if vitals and set(vitals) <= model_vitals:
            pairs.append((spec, w))
        else:
            logger.warning(
                "Skipping task '%s' for BP head: needs vitals %s but head has %s.",
                spec.name, vitals, sorted(model_vitals),
            )
    if not pairs:
        raise ValueError(
            "No trainable BP-head tasks: need ABP-target tasks whose vitals are "
            f"a subset of the head's vitals {sorted(model_vitals)}. Check "
            "trainer.task_weights and model.vitals."
        )
    return pairs


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
    var_sbp = max(a["sq_sbp"] / n - me_sbp**2, 0.0)
    var_dbp = max(a["sq_dbp"] / n - me_dbp**2, 0.0)
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
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Unpack ``(signal, sbp_dbp [, demographics])`` from the dataset.

    Returns a 3-tuple ``(signal, sbp_dbp, demographics)`` where
    ``demographics`` is ``None`` if the loader emits a legacy 2-tuple
    (BP head will then run waveform-only, no demographic branch).
    """
    if len(batch) < 2:
        raise RuntimeError(
            "BP head training requires (signal, sbp_dbp, demographics) "
            f"batches; got a {len(batch)}-tuple. Run the dataset "
            "rewrite before training the BP head."
        )
    signal = batch[0].to(device, non_blocking=True)
    sbp_dbp = batch[1].to(device, non_blocking=True)
    demographics: Tensor | None = None
    if len(batch) >= 3:
        demographics = batch[2].to(device, non_blocking=True)
    return signal, sbp_dbp, demographics


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
        signal, target, demographics = _unpack_batch(batch, device)
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
) -> None:
    """Train the BP head with per-batch weighted task (vital-subset) sampling.

    Args:
        model: :class:`BPHead` (optionally wrapped by ``torch.compile``).
        cfg: trainer config (Hydra DictConfig or plain dict); ``task_weights``
            selects which ABP-target tasks are sampled (defaults to all three).
        train_loader / val_loader / test_loader: DataLoaders returning
            ``(signal, sbp_dbp, demographics)`` tuples (the dataset 3-tuple contract).
        device: torch device.
        output_dir: Hydra-created run directory.
        bp_norm: when set, dataset labels are in globally min-max normalized
            [0, 1] (MD-ViSCo Sec III.D); ``_evaluate`` inverses to mmHg for
            AAMI metric reporting. ``None`` = raw-mmHg labels (legacy).
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

    task_pairs = _resolve_bp_tasks(model, cfg_dict.get("task_weights"))
    tasks = [spec for spec, _ in task_pairs]
    logger.info(
        "BP head tasks (weight): %s",
        {t.name: w for t, w in task_pairs},
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

        it = tqdm(
            train_loader,
            mininterval=5.0,
            maxinterval=50.0,
            disable=False,
            desc=f"bp_head ep {epoch}/{epochs - 1}",
        )
        for batch in it:
            signal, target, demographics = _unpack_batch(batch, device)
            task = _sample_task(task_pairs)
            active_vitals = cond_slots_to_vitals(task)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                pred = model(signal[:, :2, :], demographics, active_vitals=active_vitals)
                # L1 objective (MD-ViSCo §6.2.2 uses L1 + multi-WCL; WCL deferred).
                loss = (pred - target).abs().mean()
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

        if val_loader is not None and (epoch + 1) % val_every == 0:
            per_task = _evaluate(
                model, val_loader, device,
                tasks=tasks,
                amp_enabled=amp_enabled,
                bp_norm=bp_norm,
            )
            # MD-ViSCo selects best by the minimum validation loss (L1 + WCL);
            # with WCL deferred our objective is pure L1, so the selection
            # scalar is the mean per-task val L1 loss (label space).
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

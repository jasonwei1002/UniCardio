"""Training loop for the Path A scalar BP regression head.

Mirrors :mod:`src.trainer_module.trainer` (RF trainer) in structure — same
checkpoint / scheduler / AMP / SwanLab plumbing — but solves a single
regression objective ``MSE(pred_mmHg, target_mmHg)``. There is no task
sampling, no time embedding, no attention mask, and no Euler sampler.

Batch contract (Path A 3-tuple): each batch is
``(signal: (B, 3, L), sbp_dbp: (B, 2), demographics: (B, 6))``. The
trainer slices ECG+PPG (``signal[:, :2, :]``) as input and forwards
``demographics`` to BPHead v2; the ABP slot is unused (BP head is strictly
causal w.r.t. the source modalities, like MD-ViSCo's refinement model).
SBP/DBP come from PulseDB's CSV (per-cycle mean labeling), not from
per-segment min/max — see ``src/data_module/cardiac_dataset.py``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Mapping

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

from ..utils.checkpoint import load_checkpoint, save_checkpoint
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


def _unpack_batch(
    batch: Any, device: torch.device
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Unpack ``(signal, sbp_dbp [, demographics])`` from the Path A dataset.

    Returns a 3-tuple ``(signal, sbp_dbp, demographics)`` where
    ``demographics`` is ``None`` if the loader emits a legacy 2-tuple
    (BP head will then run waveform-only, no demographic branch).
    """
    if len(batch) < 2:
        raise RuntimeError(
            "BP head training requires (signal, sbp_dbp, demographics) "
            f"batches; got a {len(batch)}-tuple. Run the Path A dataset "
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
    amp_enabled: bool,
    max_batches: int | None = None,
    bp_norm: BPLabelNorm | None = None,
) -> dict[str, float]:
    """Return MSE loss + per-channel MAE / ME / SD on the val set.

    ``loss`` is MSE in whichever space the dataset emits labels (raw mmHg if
    ``bp_norm is None``; otherwise globally min-max normalized [0, 1]
    following MD-ViSCo Sec III.D). MAE / ME / SD are always reported in
    **mmHg** by inverting the normalization on the residual — so AAMI
    thresholds (|ME| ≤ 5, σ ≤ 8) read off the same units regardless of
    training space. ME = mean(pred − target); SD = population std of same.
    """
    model.eval()
    loss_sum = 0.0
    mae_sbp_sum = 0.0
    mae_dbp_sum = 0.0
    me_sbp_sum = 0.0
    me_dbp_sum = 0.0
    sq_sbp_sum = 0.0
    sq_dbp_sum = 0.0
    n_samples = 0
    for batch_idx, batch in enumerate(val_loader):
        signal, target, demographics = _unpack_batch(batch, device)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled
        ):
            pred = model(signal[:, :2, :], demographics)
        diff = pred.float() - target.float()
        loss_sum += float((diff**2).mean(dim=-1).sum().item())
        # AAMI-style metrics always in mmHg. ``vmin`` cancels on a residual,
        # so multiply by scale only; skip alloc when norm is off.
        diff_mmhg = bp_norm.denormalize_diff(diff) if bp_norm is not None else diff
        mae_sbp_sum += float(diff_mmhg[:, 0].abs().sum().item())
        mae_dbp_sum += float(diff_mmhg[:, 1].abs().sum().item())
        me_sbp_sum += float(diff_mmhg[:, 0].sum().item())
        me_dbp_sum += float(diff_mmhg[:, 1].sum().item())
        sq_sbp_sum += float((diff_mmhg[:, 0] ** 2).sum().item())
        sq_dbp_sum += float((diff_mmhg[:, 1] ** 2).sum().item())
        n_samples += int(target.shape[0])
        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break
    model.train()
    n = max(n_samples, 1)
    me_sbp = me_sbp_sum / n
    me_dbp = me_dbp_sum / n
    var_sbp = max(sq_sbp_sum / n - me_sbp**2, 0.0)
    var_dbp = max(sq_dbp_sum / n - me_dbp**2, 0.0)
    return {
        "loss": loss_sum / n,
        "mae_sbp": mae_sbp_sum / n,
        "mae_dbp": mae_dbp_sum / n,
        "mae_mean": (mae_sbp_sum + mae_dbp_sum) / (2 * n),
        "me_sbp": me_sbp,
        "me_dbp": me_dbp,
        "sd_sbp": var_sbp**0.5,
        "sd_dbp": var_dbp**0.5,
    }


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
    """Train the BP head end-to-end.

    Args:
        model: :class:`BPHead` (optionally wrapped by ``torch.compile``).
        cfg: trainer config (Hydra DictConfig or plain dict).
        train_loader / val_loader / test_loader: DataLoaders returning
            ``(signal, sbp_dbp)`` 2-tuples (see Path A data contract).
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

    optimizer = _build_optimizer(model, cfg_dict)
    steps_per_epoch = len(train_loader)
    scheduler = _build_scheduler(optimizer, cfg_dict, epochs * steps_per_epoch)
    amp_enabled = _amp_enabled(cfg_dict, device)

    csv_fields = [
        "epoch", "lr", "epoch_time_s",
        "train_loss",
        "val_loss", "val_mae_sbp", "val_mae_dbp", "val_mae_mean",
        "val_me_sbp", "val_sd_sbp", "val_me_dbp", "val_sd_dbp",
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

        it = tqdm(
            train_loader,
            mininterval=5.0,
            maxinterval=50.0,
            disable=False,
            desc=f"bp_head ep {epoch}/{epochs - 1}",
        )
        for batch in it:
            signal, target, demographics = _unpack_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                pred = model(signal[:, :2, :], demographics)
                loss = ((pred - target) ** 2).mean()
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
        logger.info(
            "bp_head epoch %d/%d | train_loss %.4f | lr %.2e | %.1fs",
            epoch, epochs - 1, avg_train_loss, lr_val, epoch_time,
        )

        row: dict[str, Any] = {
            "epoch": epoch,
            "lr": lr_val,
            "epoch_time_s": round(epoch_time, 2),
            "train_loss": avg_train_loss,
        }

        if val_loader is not None and (epoch + 1) % val_every == 0:
            val_metrics = _evaluate(
                model, val_loader, device,
                amp_enabled=amp_enabled,
                bp_norm=bp_norm,
            )
            row.update(
                {
                    "val_loss": val_metrics["loss"],
                    "val_mae_sbp": val_metrics["mae_sbp"],
                    "val_mae_dbp": val_metrics["mae_dbp"],
                    "val_mae_mean": val_metrics["mae_mean"],
                    "val_me_sbp": val_metrics["me_sbp"],
                    "val_sd_sbp": val_metrics["sd_sbp"],
                    "val_me_dbp": val_metrics["me_dbp"],
                    "val_sd_dbp": val_metrics["sd_dbp"],
                }
            )
            logger.info(
                "  val | loss %.4f | mae_sbp %.2f | mae_dbp %.2f | mae_mean %.2f mmHg",
                val_metrics["loss"], val_metrics["mae_sbp"],
                val_metrics["mae_dbp"], val_metrics["mae_mean"],
            )
            logger.info(
                "        SBP: ME %+.2f ± %.2f | DBP: ME %+.2f ± %.2f mmHg",
                val_metrics["me_sbp"], val_metrics["sd_sbp"],
                val_metrics["me_dbp"], val_metrics["sd_dbp"],
            )
            if val_metrics["mae_mean"] < best_val:
                best_val = val_metrics["mae_mean"]
                save_checkpoint(
                    ckpt_dir / "best.pt",
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=scheduler,
                    config=cfg_dict,
                    extra={"val_mae_mean": best_val},
                )

        csv_logger.log_mapping(row)
        epoch_metrics: dict[str, float] = {
            "epoch/train_loss": avg_train_loss,
            "epoch/lr": lr_val,
            "epoch/time_s": epoch_time,
        }
        if "val_loss" in row:
            epoch_metrics.update(
                {
                    "val/loss": row["val_loss"],
                    "val/mae_sbp": row["val_mae_sbp"],
                    "val/mae_dbp": row["val_mae_dbp"],
                    "val/mae_mean": row["val_mae_mean"],
                    "val/me_sbp": row["val_me_sbp"],
                    "val/sd_sbp": row["val_sd_sbp"],
                    "val/me_dbp": row["val_me_dbp"],
                    "val/sd_dbp": row["val_sd_dbp"],
                }
            )
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
        test_metrics = _evaluate(
            model, test_loader, device,
            amp_enabled=amp_enabled,
            bp_norm=bp_norm,
        )
        logger.info(
            "bp_head test | loss %.4f | mae_sbp %.2f | mae_dbp %.2f | mae_mean %.2f mmHg",
            test_metrics["loss"], test_metrics["mae_sbp"],
            test_metrics["mae_dbp"], test_metrics["mae_mean"],
        )
        logger.info(
            "             SBP: ME %+.2f ± %.2f | DBP: ME %+.2f ± %.2f mmHg",
            test_metrics["me_sbp"], test_metrics["sd_sbp"],
            test_metrics["me_dbp"], test_metrics["sd_dbp"],
        )
        swanlab.log({f"test/{k}": v for k, v in test_metrics.items()})
        csv_logger.log_mapping(
            {
                "epoch": -1,
                "lr": 0.0,
                "epoch_time_s": 0.0,
                "train_loss": float("nan"),
                "val_loss": test_metrics["loss"],
                "val_mae_sbp": test_metrics["mae_sbp"],
                "val_mae_dbp": test_metrics["mae_dbp"],
                "val_mae_mean": test_metrics["mae_mean"],
                "val_me_sbp": test_metrics["me_sbp"],
                "val_sd_sbp": test_metrics["sd_sbp"],
                "val_me_dbp": test_metrics["me_dbp"],
                "val_sd_dbp": test_metrics["sd_dbp"],
            }
        )

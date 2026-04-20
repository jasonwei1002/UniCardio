"""Training loop: task-weighted sampling, Adam + MultiStepLR, per-epoch ckpt.

One optimizer step per batch. Each batch independently samples a task from
``TASK_LIST`` (weighted) and calls :func:`rf_train_step` on that task. Per-task
running averages are written to CSV at epoch end, alongside LR and epoch wall
time. Validation (every ``val_every`` epochs) computes mean RF loss across the
5 tasks on the validation split and tracks ``best.pt``.
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import MultiStepLR, _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..model_module.tasks import TASK_LIST, TaskSpec
from ..utils.checkpoint import load_checkpoint, save_checkpoint
from .csv_logger import SimpleCSVLogger
from .rectified_flow import rf_train_step

logger = logging.getLogger(__name__)


def _build_optimizer(model: nn.Module, cfg: Mapping[str, Any]) -> Optimizer:
    return Adam(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 1.0e-6)),
    )


def _build_scheduler(
    optimizer: Optimizer, cfg: Mapping[str, Any], total_epochs: int
) -> _LRScheduler:
    sched_cfg = cfg["lr_scheduler"]
    name = str(sched_cfg["name"]).lower()
    if name != "multistep":
        raise ValueError(f"Unsupported lr_scheduler '{name}'.")
    milestones = [
        max(1, int(round(total_epochs * float(p))))
        for p in sched_cfg["milestones_pct"]
    ]
    return MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=float(sched_cfg["gamma"]),
    )


def _weighted_task_sampler(
    weights_cfg: Mapping[str, float] | None,
) -> list[tuple[TaskSpec, float]]:
    weights_cfg = weights_cfg or {}
    pairs: list[tuple[TaskSpec, float]] = []
    for spec in TASK_LIST:
        w = float(weights_cfg.get(spec.name, 1.0))
        if w < 0:
            raise ValueError(f"Negative task weight for {spec.name}: {w}")
        pairs.append((spec, w))
    total = sum(w for _, w in pairs)
    if total <= 0:
        raise ValueError("All task weights are zero.")
    return pairs


def _sample_task(pairs: Sequence[tuple[TaskSpec, float]]) -> TaskSpec:
    tasks = [t for t, _ in pairs]
    weights = [w for _, w in pairs]
    return random.choices(tasks, weights=weights, k=1)[0]


def _csv_fields() -> list[str]:
    fields = ["epoch", "lr", "epoch_time_s", "avg_loss"]
    fields.extend(f"loss_{t.name}" for t in TASK_LIST)
    fields.extend(f"val_loss_{t.name}" for t in TASK_LIST)
    fields.append("val_loss_mean")
    return fields


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    *,
    t_mean: float,
    t_std: float,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Compute mean RF loss per task on the validation loader."""
    model.eval()
    sums = {t.name: 0.0 for t in TASK_LIST}
    counts = {t.name: 0 for t in TASK_LIST}
    for batch_idx, batch in enumerate(val_loader):
        signal = batch[0].to(device)
        for task in TASK_LIST:
            loss = rf_train_step(
                model, signal, task, t_mean=t_mean, t_std=t_std
            )
            sums[task.name] += float(loss.item())
            counts[task.name] += 1
        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break
    model.train()
    return {
        name: (sums[name] / counts[name]) if counts[name] else float("nan")
        for name in sums
    }


def train(
    model: nn.Module,
    cfg: DictConfig | Mapping[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    *,
    device: torch.device,
    output_dir: str | Path,
) -> None:
    """Run the full Rectified-Flow training loop.

    Args:
        model: :class:`UniCardioRF` (or wrapped with :class:`nn.DataParallel`).
        cfg: Trainer config section (either OmegaConf DictConfig or dict).
        train_loader: DataLoader yielding ``(signal,)`` with shape ``(B, 3, L)``.
        val_loader: Optional validation loader with identical output contract.
        device: Torch device.
        output_dir: Run directory (Hydra creates this).
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
    val_every = int(cfg_dict.get("val_every", 10))
    ckpt_every = int(cfg_dict.get("ckpt_every", 1))
    itr_per_epoch = cfg_dict.get("itr_per_epoch")

    optimizer = _build_optimizer(model, cfg_dict)
    scheduler = _build_scheduler(optimizer, cfg_dict, epochs)
    t_sampler_cfg = cfg_dict.get("t_sampler", {})
    t_mean = float(t_sampler_cfg.get("mean", 0.0))
    t_std = float(t_sampler_cfg.get("std", 1.0))

    task_pairs = _weighted_task_sampler(cfg_dict.get("task_weights"))

    csv_logger = SimpleCSVLogger(
        log_dir / cfg_dict.get("log_filename", "loss.csv"),
        fieldnames=_csv_fields(),
    )

    start_epoch = 0
    best_val = float("inf")
    resume_from = cfg_dict.get("resume_from")
    if resume_from:
        payload = load_checkpoint(
            resume_from,
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            map_location=device,
        )
        start_epoch = int(payload.get("epoch", 0)) + 1
        logger.info("Resuming from epoch %d", start_epoch)

    model.train()
    model.to(device)

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        task_loss_sum = {t.name: 0.0 for t in TASK_LIST}
        task_loss_count = {t.name: 0 for t in TASK_LIST}
        total_loss = 0.0
        total_batches = 0

        it = tqdm(
            train_loader,
            mininterval=5.0,
            maxinterval=50.0,
            disable=not (hasattr(tqdm, "_instances") and True),
            desc=f"epoch {epoch}/{epochs - 1}",
        )
        for batch_idx, batch in enumerate(it, start=1):
            signal = batch[0].to(device, non_blocking=True)
            task = _sample_task(task_pairs)

            optimizer.zero_grad(set_to_none=True)
            loss = rf_train_step(
                model, signal, task, t_mean=t_mean, t_std=t_std
            )
            loss.backward()
            optimizer.step()

            val = float(loss.item())
            task_loss_sum[task.name] += val
            task_loss_count[task.name] += 1
            total_loss += val
            total_batches += 1

            if itr_per_epoch is not None and batch_idx >= int(itr_per_epoch):
                break

        scheduler.step()
        epoch_time = time.time() - epoch_start
        avg = total_loss / max(total_batches, 1)
        per_task = {
            f"loss_{name}": (task_loss_sum[name] / task_loss_count[name])
            if task_loss_count[name] > 0
            else float("nan")
            for name in task_loss_sum
        }
        lr_val = float(scheduler.get_last_lr()[0])
        logger.info(
            "epoch %d/%d | avg_loss %.6f | lr %.2e | %.1fs",
            epoch,
            epochs - 1,
            avg,
            lr_val,
            epoch_time,
        )

        row: dict[str, Any] = {
            "epoch": epoch,
            "lr": lr_val,
            "epoch_time_s": round(epoch_time, 2),
            "avg_loss": avg,
            **per_task,
        }

        # Validation.
        if val_loader is not None and (epoch + 1) % val_every == 0:
            val_losses = _evaluate(
                model, val_loader, device, t_mean=t_mean, t_std=t_std
            )
            for name, v in val_losses.items():
                row[f"val_loss_{name}"] = v
            mean_val = sum(val_losses.values()) / len(val_losses)
            row["val_loss_mean"] = mean_val
            logger.info("  val_loss_mean %.6f", mean_val)
            if mean_val < best_val:
                best_val = mean_val
                save_checkpoint(
                    ckpt_dir / "best.pt",
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=scheduler,
                    config=cfg_dict,
                    task_list=[t.name for t in TASK_LIST],
                    extra={"val_loss_mean": mean_val},
                )

        csv_logger.log_mapping(row)

        if (epoch + 1) % ckpt_every == 0:
            save_checkpoint(
                ckpt_dir / "latest.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                lr_scheduler=scheduler,
                config=cfg_dict,
                task_list=[t.name for t in TASK_LIST],
            )

"""Checkpoint save/load with full training state and RNG preservation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: str | Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    lr_scheduler: Optional[_LRScheduler] = None,
    config: Optional[dict] = None,
    task_list: Optional[list[str]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Save a full-state checkpoint.

    Args:
        path: Destination path (will be overwritten).
        epoch: Current epoch (0-indexed).
        model: Module whose state_dict will be saved (unwraps DataParallel).
        optimizer: Optimizer whose state_dict will be saved.
        lr_scheduler: Optional LR scheduler state.
        config: Optional config dict (serializable).
        task_list: Optional list of task names included in training.
        extra: Optional extra dict merged into the checkpoint.
    """
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state": model_state,
        "optimizer_state": optimizer.state_dict(),
        "lr_scheduler_state": (
            lr_scheduler.state_dict() if lr_scheduler is not None else None
        ),
        "config": config,
        "task_list": task_list,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
            "numpy": np.random.get_state(),
        },
    }
    if extra:
        payload.update(extra)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    logger.info("Saved checkpoint to %s", path)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    lr_scheduler: Optional[_LRScheduler] = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint produced by :func:`save_checkpoint`.

    Returns the full loaded payload so callers can inspect ``epoch`` or ``config``.
    """
    payload = torch.load(Path(path), map_location=map_location)
    target = model.module if isinstance(
        model, (nn.DataParallel, nn.parallel.DistributedDataParallel)
    ) else model
    target.load_state_dict(payload["model_state"], strict=strict)
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if (
        lr_scheduler is not None
        and payload.get("lr_scheduler_state") is not None
    ):
        lr_scheduler.load_state_dict(payload["lr_scheduler_state"])
    logger.info("Loaded checkpoint from %s (epoch %s)", path, payload.get("epoch"))
    return payload

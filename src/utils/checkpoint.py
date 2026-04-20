"""Checkpoint 保存/加载：完整训练状态 + 随机数生成器状态。"""

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
    """保存包含完整训练状态的 checkpoint。

    Args:
        path: 目标路径（若已存在会被覆盖）。
        epoch: 当前 epoch（0 起始）。
        model: 需要保存 state_dict 的模型（会自动解包 DataParallel）。
        optimizer: 需要保存 state_dict 的优化器。
        lr_scheduler: 可选的学习率调度器状态。
        config: 可选的可序列化配置字典。
        task_list: 可选，参与训练的任务名列表。
        extra: 可选，会并入 checkpoint 的额外字段。
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
    """加载由 :func:`save_checkpoint` 产生的 checkpoint。

    返回完整的 payload，方便调用者读取 ``epoch`` 或 ``config`` 等字段。
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

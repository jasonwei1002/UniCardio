"""DataLoader 构造器：全量 RAM 装载 MIMIC-BP 三个 split 的 .npy。

数据来源是 ``script/convert_mimicbp_to_500.py`` 预生成的三份 ``(N, 3, 500)``
float32 文件，channel 顺序已是 (ECG, PPG, ABP)。装载、归一化与样本张量
化全部在 :class:`CardiacDataset` 内完成（``__getitem__`` 是 zero-copy
切片）。
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from torch.utils.data import DataLoader

from ..utils.seed import worker_init_fn
from .cardiac_dataset import CardiacDataset

logger = logging.getLogger(__name__)


def build_loaders(
    cfg: Mapping[str, Any],
    *,
    num_workers_override: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """根据数据配置构造 ``(train, val, test)`` 三个 DataLoader。

    Args:
        cfg: 数据配置，需包含 ``train_path / val_path / test_path``、
            ``batch_size``、``num_workers``、``pin_memory`` 等键。
        num_workers_override: 若提供则覆盖 ``cfg['num_workers']``。CPU
            smoke test 等场景下，worker 启动延迟较大，可设为 0。
    """
    train_path = str(cfg["train_path"])
    val_path = str(cfg["val_path"])
    test_path = str(cfg["test_path"])

    train_ds = CardiacDataset(train_path)
    val_ds = CardiacDataset(val_path)
    test_ds = CardiacDataset(test_path)
    logger.info(
        "Data splits — train %d, val %d, test %d",
        len(train_ds), len(val_ds), len(test_ds),
    )

    batch_size = int(cfg.get("batch_size", 128))
    num_workers = (
        num_workers_override
        if num_workers_override is not None
        else int(cfg.get("num_workers", 8))
    )
    pin_memory = bool(cfg.get("pin_memory", True))

    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": worker_init_fn if num_workers > 0 else None,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=True, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader

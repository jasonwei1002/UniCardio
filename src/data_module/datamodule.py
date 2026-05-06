"""DataLoader 构造器：从单一 train .npy 切出 train/val，外加独立 test .npy。

数据来源是 PulseDB 预生成的两份 ``(N, 3, 1250)`` float32：
  - ``train_path``：内部按 ``val_split`` 用 :func:`sklearn.model_selection.train_test_split`
    随机切训练 / 验证（同一 subject 的多个窗口可能跨切，仅作 best.pt 选择信号）。
  - ``test_path``：独立的 subject-disjoint 测试集。

装载、归一化与样本张量化全部在 :class:`CardiacDataset` 内完成；切分通过
:class:`torch.utils.data.Subset` 包索引，不复制数据 buffer。
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

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
        cfg: 数据配置，需包含 ``train_path / test_path / val_split / split_seed``、
            ``batch_size / num_workers / pin_memory``。
        num_workers_override: 若提供则覆盖 ``cfg['num_workers']``。CPU
            smoke test 等场景下，worker 启动延迟较大，可设为 0。
    """
    train_path = str(cfg["train_path"])
    test_path = str(cfg["test_path"])
    val_split = float(cfg.get("val_split", 0.2))
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"val_split must be in (0, 1); got {val_split}")
    split_seed = int(cfg.get("split_seed", 42))

    full_train = CardiacDataset(train_path)
    test_ds = CardiacDataset(test_path)

    indices = np.arange(len(full_train))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_split, random_state=split_seed, shuffle=True,
    )
    train_ds = Subset(full_train, train_idx)
    val_ds = Subset(full_train, val_idx)
    logger.info(
        "Data splits — train %d, val %d (val_split=%.2f, seed=%d), test %d",
        len(train_ds), len(val_ds), val_split, split_seed, len(test_ds),
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

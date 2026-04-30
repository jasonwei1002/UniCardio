"""DataLoader 构造器：mmap 打开 .npy + 索引划分 + 三个 DataLoader。

数据集通过 ``cfg.name`` 切换：
    - ``combined``：Final_sig_combined.npy 单文件三划分（legacy）
    - ``pulsedb`` ：PulseDB Train_Subset.npy 8:2 划分得 train/val，
                  CalFree_Test_Subset.npy 作为 test。

划分时只切**索引**（``np.arange(N)``），不切数据；样本到张量的转换在
:class:`CardiacDataset.__getitem__` 内按需做，包含通道置换与 BP 归一化。
这样可避免把 13 GB 的 PulseDB 训练文件整个实例化到内存。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from ..utils.seed import worker_init_fn
from .cardiac_dataset import CardiacDataset

logger = logging.getLogger(__name__)


def _peek_length(path: str | Path) -> int:
    """只读 ``.npy`` 头部拿样本数，不实际加载数据。"""
    arr = np.load(Path(path), mmap_mode="r")
    if arr.ndim != 3 or arr.shape[1] != 3:
        raise ValueError(
            f"Expected (N, 3, L) data at {path}, got shape {arr.shape}"
        )
    return int(arr.shape[0])


def _split_three_way_indices(
    n: int,
    val_size: int,
    test_size: int,
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """复现 ``base_model/train_original.py`` 的两步切分逻辑（按索引）。"""
    train_idx, temp_idx = train_test_split(
        np.arange(n), test_size=val_size + test_size, random_state=split_seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=test_size, random_state=split_seed
    )
    return train_idx, val_idx, test_idx


_DatasetSpec = tuple[str, np.ndarray, Sequence[int]]   # (path, indices, channel_permutation)


def _plan_combined(cfg: Mapping[str, Any]) -> tuple[_DatasetSpec, _DatasetSpec, _DatasetSpec]:
    sub = cfg["combined"]
    path = str(sub["data_path"])
    perm = tuple(sub["channel_permutation"])
    n = _peek_length(path)
    train_idx, val_idx, test_idx = _split_three_way_indices(
        n,
        int(sub.get("val_size", 20_000)),
        int(sub.get("test_size", 20_000)),
        int(cfg.get("split_seed", 42)),
    )
    return (path, train_idx, perm), (path, val_idx, perm), (path, test_idx, perm)


def _plan_pulsedb(cfg: Mapping[str, Any]) -> tuple[_DatasetSpec, _DatasetSpec, _DatasetSpec]:
    sub = cfg["pulsedb"]
    train_path = str(sub["train_path"])
    test_path = str(sub["test_path"])
    perm = tuple(sub["channel_permutation"])
    n_train = _peek_length(train_path)
    n_test = _peek_length(test_path)
    val_ratio = float(sub.get("val_ratio", 0.2))
    train_idx, val_idx = train_test_split(
        np.arange(n_train),
        test_size=val_ratio,
        random_state=int(cfg.get("split_seed", 42)),
    )
    test_idx = np.arange(n_test)
    return (train_path, train_idx, perm), (train_path, val_idx, perm), (test_path, test_idx, perm)


_PLANNERS = {
    "combined": _plan_combined,
    "pulsedb": _plan_pulsedb,
}


def build_loaders(
    cfg: Mapping[str, Any],
    *,
    num_workers_override: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """根据配置字典构造 ``(train, val, test)`` 三个 DataLoader（mmap-backed）。

    Args:
        cfg: 数据配置，需包含 ``name``、对应数据集子节、``batch_size``、
            ``num_workers``、``pin_memory`` 等键。
        num_workers_override: 若提供则覆盖 ``cfg['num_workers']``。在 CPU
            smoke test 等场景下，worker 启动延迟较大，可设为 0。
    """
    name = str(cfg.get("name", "combined"))
    if name not in _PLANNERS:
        raise ValueError(
            f"Unknown data.name='{name}'; expected one of {sorted(_PLANNERS)}"
        )
    train_spec, val_spec, test_spec = _PLANNERS[name](cfg)

    logger.info(
        "Data splits (%s) — train %d, val %d, test %d",
        name,
        train_spec[1].size,
        val_spec[1].size,
        test_spec[1].size,
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


    train_loader = DataLoader(
        CardiacDataset(*train_spec), shuffle=True, drop_last=True, **loader_kwargs
    )
    val_loader = DataLoader(
        CardiacDataset(*val_spec), shuffle=False, drop_last=True, **loader_kwargs
    )
    test_loader = DataLoader(
        CardiacDataset(*test_spec), shuffle=False, **loader_kwargs
    )
    return train_loader, val_loader, test_loader

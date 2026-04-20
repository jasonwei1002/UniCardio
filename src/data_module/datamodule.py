"""DataLoader 构造器：加载 -> 通道置换 -> BP 归一化 -> 划分 -> 封装。

代码库中**唯一**处理文件顺序与模型 slot 顺序不一致问题的地方。
调用 :func:`load_and_preprocess` 后，所有张量都处于模型 slot 顺序
``(ECG=0, PPG=1, ABP=2)``，且 ABP 通道（slot 2）已归一化为 ``(x - 100) / 50``。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from ..utils.normalization import bp_normalize
from ..utils.seed import worker_init_fn
from .cardiac_dataset import FILE_TO_MODEL_PERMUTATION, CardiacDataset

logger = logging.getLogger(__name__)


def load_and_preprocess(data_path: str | Path) -> np.ndarray:
    """加载原始 ``.npy`` 文件，执行通道置换并归一化 ABP。

    Args:
        data_path: ``Final_sig_combined.npy`` 的路径（形状 ``(N, 3, L)``）。

    Returns:
        形状为 ``(N, 3, L)`` 的 ``float32`` 数组，已按模型 slot 顺序排列，
        且 BP 通道已归一化。
    """
    arr = np.load(Path(data_path))
    if arr.ndim != 3 or arr.shape[1] != 3:
        raise ValueError(
            f"Expected (N, 3, L) data, got shape {arr.shape}"
        )
    arr = arr.astype(np.float32, copy=False)

    # 文件顺序 (PPG, BP, ECG) -> 模型顺序 (ECG, PPG, ABP)。
    arr = arr[:, list(FILE_TO_MODEL_PERMUTATION), :]

    # 此时 BP 位于 slot 2（ABP），归一化到模型量纲 (x - 100) / 50。
    arr[:, 2, :] = bp_normalize(arr[:, 2, :])
    return arr


def _split_three_way(
    signals: np.ndarray,
    val_size: int,
    test_size: int,
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """复现 ``base_model/train_original.py`` 中两步切分的逻辑。"""
    train, temp = train_test_split(
        signals, test_size=val_size + test_size, random_state=split_seed
    )
    val, test = train_test_split(
        temp, test_size=test_size, random_state=split_seed
    )
    return train, val, test


def build_loaders(
    cfg: Mapping[str, Any],
    *,
    num_workers_override: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """根据配置字典构造 ``(train, val, test)`` 三个 DataLoader。

    Args:
        cfg: 数据配置，需包含 ``data_path``、``val_size``、``test_size``、
            ``split_seed``、``batch_size``、``num_workers``、``pin_memory`` 等键。
        num_workers_override: 若提供则覆盖 ``cfg['num_workers']``。在 CPU
            smoke test 等场景下，worker 启动延迟较大，可设为 0。

    Returns:
        三个 :class:`DataLoader` 实例组成的元组。
    """
    signals = load_and_preprocess(cfg["data_path"])
    val_size = int(cfg.get("val_size", 20_000))
    test_size = int(cfg.get("test_size", 20_000))
    split_seed = int(cfg.get("split_seed", 42))
    train, val, test = _split_three_way(signals, val_size, test_size, split_seed)

    logger.info(
        "Data splits — train %d, val %d, test %d (slot_length=%d)",
        train.shape[0],
        val.shape[0],
        test.shape[0],
        train.shape[-1],
    )

    batch_size = int(cfg.get("batch_size", 128))
    num_workers = (
        num_workers_override
        if num_workers_override is not None
        else int(cfg.get("num_workers", 8))
    )
    pin_memory = bool(cfg.get("pin_memory", True))

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": worker_init_fn if num_workers > 0 else None,
    }

    train_loader = DataLoader(
        CardiacDataset(train), shuffle=True, **loader_kwargs
    )
    val_loader = DataLoader(
        CardiacDataset(val), shuffle=False, **loader_kwargs
    )
    test_loader = DataLoader(
        CardiacDataset(test), shuffle=False, **loader_kwargs
    )
    return train_loader, val_loader, test_loader

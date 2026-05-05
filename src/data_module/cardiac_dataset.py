"""一次性全量加载到 RAM 的模型空间信号数据集。

通道顺序约定（非常重要）：所有 ``.npy`` 数据文件在磁盘上必须已经按
**模型** slot 顺序存好 ``ECG=0, PPG=1, ABP=2``——预处理脚本（见
``script/convert_*_to_500.py``）负责在生成 ``.npy`` 时把通道排好，
``CardiacDataset`` 不再做任何重排。

构造时把磁盘字节全部 ``np.load`` 进 RAM，做以下 per-slot 归一化（一次
性、in-place）：

  - slot 0 / 1（ECG / PPG）：per-sample min-max 到 ``[0, 1]``，消除跨
    设备 / 跨受试者的幅值偏移；
  - slot 2（ABP）：``(x - 100) / 50``，保留 mmHg 物理量。

之后 ``__getitem__`` 仅做 zero-copy 切片，避免每个 batch 重复 numpy
操作。模块级 ``_CACHE`` 按 ``path`` 去重，多次构造同一文件的 Dataset
共用同一份 buffer。如需子集，包一层 ``torch.utils.data.Subset``。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..utils.normalization import bp_normalize, minmax_normalize_per_sample_inplace

logger = logging.getLogger(__name__)

_CACHE: dict[str, np.ndarray] = {}


def _load_full(path: str) -> np.ndarray:
    """读 ``.npy`` 到 RAM 并对三个 slot 各自归一化，返回 ``(N, 3, L)`` float32。

    同一 ``path`` 只读盘一次。返回的 ndarray 由多个 split 共享 buffer，调用
    方约定**不得**原地修改；``__getitem__`` 通过 ``torch.from_numpy`` 生成
    的张量同样不应在 in-place 算子里被改动（不能用 ``writeable=False`` 锁
    定，否则 ``torch.from_numpy`` 会持续抛 UserWarning）。
    """
    cached = _CACHE.get(path)
    if cached is not None:
        return cached
    raw = np.load(path)
    if raw.ndim != 3 or raw.shape[1] != 3:
        raise ValueError(
            f"Expected (N, 3, L) data at {path}, got shape {raw.shape}; "
            "channels must be ordered (ECG, PPG, ABP) — reorder in your "
            "preprocessing script."
        )
    arr = raw if raw.dtype == np.float32 else raw.astype(np.float32)
    minmax_normalize_per_sample_inplace(arr[:, 0:2])  # ECG + PPG per-sample [0, 1]
    arr[:, 2] = bp_normalize(arr[:, 2])
    logger.info(
        "Loaded %s into RAM: shape=%s dtype=%s size=%.2f GB",
        path, arr.shape, arr.dtype, arr.nbytes / 2**30,
    )
    _CACHE[path] = arr
    return arr


def clear_cache() -> None:
    """释放 ``_load_full`` 维护的全部数据 buffer；测试或长会话里手动回收 RAM。"""
    _CACHE.clear()


class CardiacDataset(Dataset):
    """``(N, 3, L)`` 信号数据集；channel 顺序固定为 (ECG, PPG, ABP)。

    Args:
        data_path: ``.npy`` 文件路径，形状必须为 ``(N, 3, L)``。channel 顺序
            必须已经是 ``(ECG, PPG, ABP)``——若手头数据不一致，应在生成
            ``.npy`` 时一次性重排，而不是在 dataset 里做。
    """

    def __init__(self, data_path: str | Path) -> None:
        self._data = _load_full(str(Path(data_path)))

    def __len__(self) -> int:
        return int(self._data.shape[0])

    def __getitem__(self, idx: int) -> tuple[Tensor]:
        return (torch.from_numpy(self._data[idx]),)

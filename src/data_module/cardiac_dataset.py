"""按 ``np.memmap`` 流式访问 ``.npy`` 的模型空间信号数据集。

通道顺序约定（非常重要）：下游所有代码（模型、mask、采样器、指标）
都按**模型** slot 顺序 ``ECG=0, PPG=1, ABP=2`` 工作。``CardiacDataset``
内部按 ``channel_permutation`` 把磁盘顺序重排到模型 slot 顺序，并在 slot 2
上应用 ``(x - 100) / 50`` 的 BP 归一化——两步都在 ``__getitem__`` 时
按需做，避免把整个文件实例化进内存（PulseDB Train_Subset.npy ~13 GB）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..utils.normalization import bp_normalize


class CardiacDataset(Dataset):
    """``np.memmap``-backed ``(N, 3, slot_length)`` 数据集，按需置换 + 归一化。

    用 ``np.load(path, mmap_mode='r')`` 打开 ``.npy`` 文件，仅按 ``indices``
    索引行；这样 train/val 划分用的 ``train_test_split(np.arange(N), ...)``
    只产出索引数组，不会触发任何数据 copy。

    Args:
        data_path: ``.npy`` 文件路径，形状必须为 ``(N, 3, L)``。
        indices: 该 split 选用的样本下标（``int`` 一维数组）。
        channel_permutation: 把磁盘通道顺序重排到模型 slot 顺序的 3 个索引。
            例：Final_sig_combined.npy 存 (PPG, BP, ECG) → ``(2, 0, 1)``；
            PulseDB 存 (ECG, PPG, ABP) → ``(0, 1, 2)``。
    """

    def __init__(
        self,
        data_path: str | Path,
        indices: np.ndarray | Sequence[int],
        channel_permutation: Sequence[int] = (0, 1, 2),
    ) -> None:
        path = str(Path(data_path))
        self._mm = np.load(path, mmap_mode="r")
        if self._mm.ndim != 3 or self._mm.shape[1] != 3:
            raise ValueError(
                f"Expected (N, 3, L) data at {path}, got shape {self._mm.shape}"
            )
        self._indices = np.asarray(indices, dtype=np.int64)
        self._perm = np.asarray(channel_permutation, dtype=np.int64)
        if self._perm.shape != (3,):
            raise ValueError(
                f"channel_permutation must have 3 entries, got {self._perm.tolist()}"
            )

    def __len__(self) -> int:
        return int(self._indices.shape[0])

    def __getitem__(self, idx: int) -> tuple[Tensor]:
        # 一次 advanced indexing 即完成「行选取 + 通道置换」，并产出一份 owned、
        # writable、C-contiguous 的 ndarray（numpy 高级索引语义保证）；之后
        # `from_numpy` 可零拷贝接管，DataLoader 的 collate + pin_memory 也安全。
        x = self._mm[self._indices[idx], self._perm]
        x[2] = bp_normalize(x[2])
        return (torch.from_numpy(x),)

"""Path A: ``np.memmap``-backed dataset returning shape-only ABP + scalar BP.

Channel order (重要): 下游所有代码（模型、mask、采样器、指标）按
**模型** slot 顺序 ``ECG=0, PPG=1, ABP=2`` 工作。``CardiacDataset``
内部按 ``channel_permutation`` 把磁盘顺序重排到模型 slot 顺序，然后对
slot 2（ABP）做 **per-sample min-max** 归一化到 ``[0, 1]``（Path A 决定，
取代旧的全局 ``(x - 100) / 50``）。ECG / PPG 保持 raw scale。

每次 ``__getitem__`` 返回 2-tuple：

* ``signal``: ``(3, L)`` float32，ABP slot ∈ [0, 1]，其它 slot 为原始尺度。
* ``sbp_dbp``: ``(2,)`` float32 = ``(sbp_mmHg, dbp_mmHg)``，即原始 ABP
  片段的 max / min。RF 训练 (Stream 1) 忽略它；BP head 训练 (Stream 2)
  把它当作回归目标；推理时用 :func:`reconstruct_mmHg` 把 shape × scalar
  组合回 mmHg 量纲的完整波形。
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..utils.normalization import MINMAX_EPS


class CardiacDataset(Dataset):
    """``np.memmap``-backed ``(N, 3, slot_length)`` 数据集。

    用 ``np.load(path, mmap_mode='r')`` 打开 ``.npy``；仅按 ``indices``
    索引行；这样 train/val 划分用的 ``train_test_split(np.arange(N), ...)``
    只产出索引数组，不会触发任何数据 copy。

    Args:
        data_path: ``.npy`` 文件路径，形状必须为 ``(N, 3, L)``。
        indices: 该 split 选用的样本下标。
        channel_permutation: 把磁盘通道顺序重排到模型 slot 顺序的 3 个索引。
            例：PulseDB 已按 (ECG, PPG, ABP) 存盘 → ``(0, 1, 2)``；
            旧的 Final_sig_combined.npy 存 (PPG, BP, ECG) → ``(2, 0, 1)``。
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

    @property
    def indices(self) -> np.ndarray:
        """该 split 在原始 .npy / 同名 .csv 里的行号（一维 int64 数组）。"""
        return self._indices

    def __len__(self) -> int:
        return int(self._indices.shape[0])

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        # Advanced indexing returns an owned, writable, C-contiguous ndarray
        # (safe for from_numpy + pin_memory). astype(copy=False) is a no-op on
        # float32 sources, else casts to avoid conv1d dtype mismatch.
        x = self._mm[self._indices[idx], self._perm].astype(np.float32, copy=False)
        abp = x[2]
        dbp = float(abp.min())
        sbp = float(abp.max())
        denom = max(sbp - dbp, MINMAX_EPS)
        x[2] = (abp - dbp) / denom  # per-sample min-max → [0, 1]
        signal = torch.from_numpy(x)
        sbp_dbp = torch.tensor([sbp, dbp], dtype=torch.float32)
        return signal, sbp_dbp

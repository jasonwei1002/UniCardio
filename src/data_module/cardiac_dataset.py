"""完全加载到内存的模型空间信号数据集。

通道顺序约定（非常重要）：下游所有代码（模型、mask、采样器、指标）
都按**模型** slot 顺序 ``ECG=0, PPG=1, ABP=2`` 工作。
``Final_sig_combined.npy`` 在磁盘上的顺序为 ``PPG=0, BP=1, ECG=2``；
:data:`FILE_TO_MODEL_PERMUTATION` 元组是这次重排唯一发生的位置。
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

FILE_TO_MODEL_PERMUTATION: tuple[int, int, int] = (2, 0, 1)
"""将文件顺序（PPG, BP, ECG）重排为模型顺序（ECG, PPG, ABP）的索引元组。"""


class CardiacDataset(Dataset):
    """形状为 ``(N, 3, slot_length)`` 的模型 slot 顺序数据集。

    张量完全驻留在内存中（``Final_sig_combined.npy`` 以 float32 加载约 3.4 GB，
    在训练节点上可从容装载）。如果需要 on-disk 流式读取，可替换为 mmap 版本，
    接口契约保持不变。
    """

    def __init__(self, signals: np.ndarray | Tensor) -> None:
        if isinstance(signals, np.ndarray):
            tensor = torch.from_numpy(signals).float()
        else:
            tensor = signals.float()
        if tensor.ndim != 3 or tensor.size(1) != 3:
            raise ValueError(
                f"Expected shape (N, 3, L), got {tuple(tensor.shape)}"
            )
        self.signals = tensor

    def __len__(self) -> int:
        return self.signals.size(0)

    def __getitem__(self, idx: int) -> tuple[Tensor]:
        return (self.signals[idx],)  # (3, L)

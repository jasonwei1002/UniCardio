"""In-memory dataset wrapping the model-space signal tensor.

Channel ordering contract (critical): everything downstream (model, masks,
sampler, metrics) operates in *model* slot order ``ECG=0, PPG=1, ABP=2``.
The on-disk array in ``Final_sig_combined.npy`` uses file order
``PPG=0, BP=1, ECG=2``; the :data:`FILE_TO_MODEL_PERMUTATION` tuple documents
the single place where that re-ordering happens.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

FILE_TO_MODEL_PERMUTATION: tuple[int, int, int] = (2, 0, 1)
"""Indexes to permute file order (PPG, BP, ECG) into model order (ECG, PPG, ABP)."""


class CardiacDataset(Dataset):
    """Dataset of shape ``(N, 3, slot_length)`` in model slot order.

    The tensor is held entirely in RAM (the full ``Final_sig_combined.npy``
    is ~3.4 GB as float32 and fits comfortably on a training node). For
    on-disk streaming, swap this class for a memory-mapped variant; the
    interface contract is identical.
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

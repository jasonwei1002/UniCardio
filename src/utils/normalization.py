"""BP 信号归一化。

Final_sig_combined.npy 中 ABP 通道的原始数值单位为 mmHg（范围约 50~200），
若直接缩放到 [-1, 1] 会扭曲其物理均值。这里以标称 100 mmHg 作为偏移、
50 mmHg 作为尺度，把信号中心化到 0 附近。
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch

BP_OFFSET: float = 100.0
BP_SCALE: float = 50.0

Array = Union[np.ndarray, torch.Tensor]


def bp_normalize(x: Array) -> Array:
    """将原始 BP（mmHg）归一化到模型空间：``(x - 100) / 50``。"""
    return (x - BP_OFFSET) / BP_SCALE


def bp_denormalize(x: Array) -> Array:
    """:func:`bp_normalize` 的逆运算，返回 mmHg 量纲的数值。"""
    return x * BP_SCALE + BP_OFFSET

"""信号归一化。

ABP 通道原始数值单位为 mmHg（范围约 50~200），若直接缩放到 [-1, 1]
会扭曲其物理均值。这里以标称 100 mmHg 作为偏移、50 mmHg 作为尺度，
把信号中心化到 0 附近。

ECG / PPG 通道的绝对幅值由采集设备决定，没有可解释的单位；不同
受试者间幅值差可达数十倍。这里对每个样本沿时间维 min-max 归一化
到 [0, 1]，保留形态信息、消除幅值偏移。
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


def minmax_normalize_per_sample_inplace(x: np.ndarray) -> None:
    """对前缀任意维 + 最后一维 ``L`` 的浮点数组做 in-place per-sample min-max → ``[0, 1]``。

    用于 ECG / PPG：每个样本独立按自身极值缩放，保留波形形态、消除采集
    设备造成的幅值偏移。``x`` 必须是可写的浮点 view，函数直接改写其缓冲
    区，不分配新数组。对 flat trace（max==min）退化为零向量（除以 eps）。
    支持任意 leading dims，例如 ``(N, L)`` 或 ``(N, C, L)``——后者表示对
    每个样本的每个通道独立归一化。
    """
    if not np.issubdtype(x.dtype, np.floating):
        raise TypeError(
            f"minmax_normalize_per_sample_inplace expects floating dtype, got {x.dtype}"
        )
    mn = x.min(axis=-1, keepdims=True)
    mx = x.max(axis=-1, keepdims=True)
    rng = np.maximum(mx - mn, np.finfo(x.dtype).eps)
    x -= mn
    x /= rng

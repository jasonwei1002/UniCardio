"""面向多模态信号重建的回归指标。"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def _to_numpy(x: Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def rmse(pred: Tensor | np.ndarray, target: Tensor | np.ndarray) -> float:
    """对所有元素求均方根误差。"""
    p, t = _to_numpy(pred), _to_numpy(target)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def mae(pred: Tensor | np.ndarray, target: Tensor | np.ndarray) -> float:
    """对所有元素求平均绝对误差。"""
    p, t = _to_numpy(pred), _to_numpy(target)
    return float(np.mean(np.abs(p - t)))


def ks_statistic(
    pred: Tensor | np.ndarray, target: Tensor | np.ndarray
) -> float:
    """在展平后的取值分布上计算两样本 Kolmogorov-Smirnov 统计量。"""
    try:
        from scipy.stats import ks_2samp
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "scipy is required for ks_statistic; install scipy."
        ) from e
    p = _to_numpy(pred).ravel()
    t = _to_numpy(target).ravel()
    return float(ks_2samp(p, t).statistic)


def pearson_corr(
    pred: Tensor | np.ndarray, target: Tensor | np.ndarray
) -> float:
    """逐样本 Pearson 相关系数，再对所有样本取均值。

    形状假设 ``(N, ..., L)``——首维是样本，其余被展平为时间维。波形重建里
    全局展平后求 r 会被各样本的 DC 偏移偏倚，逐样本中心化再平均更贴合
    "波形相似度" 这一直觉。常数样本（方差为 0）会从分母上跳过；若整批
    都常数，返回 ``nan``。
    """
    p = _to_numpy(pred)
    t = _to_numpy(target)
    p = p.reshape(p.shape[0], -1).astype(np.float64)
    t = t.reshape(t.shape[0], -1).astype(np.float64)
    pm = p - p.mean(axis=1, keepdims=True)
    tm = t - t.mean(axis=1, keepdims=True)
    num = (pm * tm).sum(axis=1)
    den = np.sqrt((pm * pm).sum(axis=1) * (tm * tm).sum(axis=1))
    valid = den > 0
    if not valid.any():
        return float("nan")
    return float(np.mean(num[valid] / den[valid]))

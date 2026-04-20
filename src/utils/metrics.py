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

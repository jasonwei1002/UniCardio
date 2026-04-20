"""Regression metrics for multimodal signal reconstruction."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def _to_numpy(x: Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def rmse(pred: Tensor | np.ndarray, target: Tensor | np.ndarray) -> float:
    """Root-mean-squared error over all elements."""
    p, t = _to_numpy(pred), _to_numpy(target)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def mae(pred: Tensor | np.ndarray, target: Tensor | np.ndarray) -> float:
    """Mean absolute error over all elements."""
    p, t = _to_numpy(pred), _to_numpy(target)
    return float(np.mean(np.abs(p - t)))


def ks_statistic(
    pred: Tensor | np.ndarray, target: Tensor | np.ndarray
) -> float:
    """Two-sample Kolmogorov-Smirnov statistic on the flattened value distributions."""
    try:
        from scipy.stats import ks_2samp
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "scipy is required for ks_statistic; install scipy."
        ) from e
    p = _to_numpy(pred).ravel()
    t = _to_numpy(target).ravel()
    return float(ks_2samp(p, t).statistic)

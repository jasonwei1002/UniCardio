"""ABP signal normalization.

Path A (approximation + refinement decomposition) uses **per-sample min-max
normalization**: each ABP segment is mapped to ``[0, 1]`` so the RF model
learns shape only. Absolute SBP/DBP values are predicted by a separate
``BPHead`` and recombined at inference via :func:`reconstruct_mmHg`.

The legacy global ``(x - 100) / 50`` path is retained for backwards
compatibility with pre-Path-A checkpoints but should not be used by new
training runs.
"""

from __future__ import annotations

import warnings
from typing import Union

import numpy as np
import torch

# Legacy global-affine normalization (deprecated, kept for ckpt back-compat).
BP_OFFSET: float = 100.0
BP_SCALE: float = 50.0

# Guard against division by ~0 on flat ABP segments (e.g. clipped / saturated).
MINMAX_EPS: float = 1e-3

Array = Union[np.ndarray, torch.Tensor]


def bp_normalize(x: Array) -> Array:
    """Legacy: ``(x - 100) / 50``. Deprecated since Path A."""
    warnings.warn(
        "bp_normalize is deprecated; use per-sample minmax_normalize (Path A).",
        DeprecationWarning,
        stacklevel=2,
    )
    return (x - BP_OFFSET) / BP_SCALE


def bp_denormalize(x: Array) -> Array:
    """Inverse of :func:`bp_normalize`. Deprecated since Path A."""
    warnings.warn(
        "bp_denormalize is deprecated; use reconstruct_mmHg (Path A).",
        DeprecationWarning,
        stacklevel=2,
    )
    return x * BP_SCALE + BP_OFFSET


def minmax_normalize(
    x: Array, eps: float = MINMAX_EPS
) -> tuple[Array, float, float]:
    """Per-sample min-max normalize a 1-D ABP segment to ``[0, 1]``.

    Args:
        x: 1-D array/tensor of raw ABP samples (mmHg). Higher dims are not
            supported here — caller should iterate over batch / channel.
        eps: minimum span ``(max - min)`` to avoid /0 on flat segments.

    Returns:
        ``(x_norm, x_min, x_max)``. ``x_min`` and ``x_max`` are the raw
        mmHg DBP/SBP values for this segment (post-eps clamping for max).
    """
    if isinstance(x, torch.Tensor):
        x_min = float(x.min().item())
        x_max = float(x.max().item())
    else:
        x_min = float(np.min(x))
        x_max = float(np.max(x))
    denom = max(x_max - x_min, eps)
    if isinstance(x, torch.Tensor):
        x_norm = (x - x_min) / denom
    else:
        x_norm = (x - x_min) / denom
    # If the segment was effectively flat, the eps-clamped denominator
    # produced a near-zero range output; report the true max anyway so
    # downstream reconstruct_mmHg yields the original flat value.
    return x_norm, x_min, x_max


def reconstruct_mmHg(shape: Array, sbp: Array, dbp: Array) -> Array:
    """Recover mmHg waveform from shape-only prediction + scalar BP.

    ``wave_mmHg = shape * (sbp - dbp).unsqueeze(-1) + dbp.unsqueeze(-1)``.

    Args:
        shape: shape-only prediction in ``[0, 1]``. Last dim is time.
            Accepts any leading batch shape, e.g. ``(B, L)`` or
            ``(B, 1, L)``.
        sbp, dbp: per-sample SBP/DBP in mmHg. Shape must broadcast with
            ``shape[..., 0]`` (i.e. ``shape.shape[:-1]``).

    Returns:
        Tensor / array same shape as ``shape``, units mmHg.
    """
    sbp_b = sbp[..., None] if isinstance(sbp, (np.ndarray, torch.Tensor)) else sbp
    dbp_b = dbp[..., None] if isinstance(dbp, (np.ndarray, torch.Tensor)) else dbp
    return shape * (sbp_b - dbp_b) + dbp_b

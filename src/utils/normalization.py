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
from dataclasses import dataclass
from typing import Any, Mapping, Union

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


@dataclass(frozen=True)
class BPLabelNorm:
    """Global min-max normalization for SBP/DBP scalar labels.

    Matches MD-ViSCo Sec III.D (IEEE JBHI 2026): refinement model predicts
    SBP/DBP in a globally min-max normalized [0, 1] domain (using fixed
    constants computed once on the training set) rather than directly in
    mmHg. Their ablation (Appendix XI) shows this is substantially more
    stable than direct mmHg regression.

    Constants live in ``data.bp_label_norm`` in the Hydra config. ``None``
    everywhere disables this normalization (legacy raw-mmHg path).
    """

    vmin: float
    vmax: float

    def __post_init__(self) -> None:
        if self.vmax <= self.vmin:
            raise ValueError(
                f"BPLabelNorm.vmax ({self.vmax}) must exceed vmin ({self.vmin})"
            )

    @property
    def scale(self) -> float:
        return self.vmax - self.vmin

    def normalize(self, x: Array) -> Array:
        """mmHg → [0, 1]. Over-range samples land at <0 or >1 (no clip)."""
        return (x - self.vmin) / self.scale

    def denormalize(self, x: Array) -> Array:
        """[0, 1] → mmHg. Inverse of :meth:`normalize`."""
        return x * self.scale + self.vmin

    def denormalize_diff(self, x: Array) -> Array:
        """Inverse for residuals ``(pred - target)``: ``vmin`` cancels."""
        return x * self.scale

    @classmethod
    def from_cfg(cls, data_cfg: Mapping[str, Any] | None) -> "BPLabelNorm | None":
        """Build from a Hydra ``data`` config node. Returns ``None`` when
        ``bp_label_norm`` is absent or empty; raises on partial config."""
        if data_cfg is None:
            return None
        node = data_cfg.get("bp_label_norm")
        if not node:
            return None
        vmin = node.get("vmin")
        vmax = node.get("vmax")
        if vmin is None or vmax is None:
            raise ValueError(
                "data.bp_label_norm must specify both vmin and vmax; "
                f"got vmin={vmin}, vmax={vmax}"
            )
        return cls(vmin=float(vmin), vmax=float(vmax))


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

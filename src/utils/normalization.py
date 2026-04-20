"""BP signal normalization.

The raw ABP channel in Final_sig_combined.npy is in mmHg (~50-200 range) and
cannot be rescaled to [-1, 1] without distorting the physical mean. We shift
to zero-centered values around a nominal 100 mmHg with a 50 mmHg scale.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch

BP_OFFSET: float = 100.0
BP_SCALE: float = 50.0

Array = Union[np.ndarray, torch.Tensor]


def bp_normalize(x: Array) -> Array:
    """Normalize raw BP (mmHg) to model-space: ``(x - 100) / 50``."""
    return (x - BP_OFFSET) / BP_SCALE


def bp_denormalize(x: Array) -> Array:
    """Inverse of :func:`bp_normalize`; returns values in mmHg."""
    return x * BP_SCALE + BP_OFFSET

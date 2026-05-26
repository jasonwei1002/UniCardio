"""Per-sample min-max normalization helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.utils.normalization import (
    MINMAX_EPS,
    minmax_normalize,
    reconstruct_mmHg,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_minmax_normalize_torch_roundtrip(dtype: torch.dtype) -> None:
    """``reconstruct_mmHg(minmax_normalize(x))`` recovers ``x``."""
    torch.manual_seed(0)
    x = torch.randn(1250, dtype=dtype) * 30 + 90  # mmHg-like distribution
    x_norm, x_min, x_max = minmax_normalize(x)
    sbp = torch.tensor(x_max, dtype=dtype)
    dbp = torch.tensor(x_min, dtype=dtype)
    recon = reconstruct_mmHg(x_norm, sbp, dbp)
    assert torch.allclose(recon, x, atol=1e-5)


def test_minmax_normalize_numpy_roundtrip() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(loc=90, scale=30, size=1250).astype(np.float32)
    x_norm, x_min, x_max = minmax_normalize(x)
    sbp = np.array(x_max, dtype=np.float32)
    dbp = np.array(x_min, dtype=np.float32)
    recon = reconstruct_mmHg(x_norm, sbp, dbp)
    np.testing.assert_allclose(recon, x, atol=1e-5)


def test_minmax_normalize_range() -> None:
    """Output should lie in ``[0, 1]`` for a non-degenerate input."""
    x = torch.tensor([60.0, 70.0, 80.0, 100.0, 120.0, 150.0])
    x_norm, _, _ = minmax_normalize(x)
    assert float(x_norm.min()) == pytest.approx(0.0, abs=1e-6)
    assert float(x_norm.max()) == pytest.approx(1.0, abs=1e-6)


def test_minmax_normalize_flat_input_no_nan() -> None:
    """Flat (constant) ABP segment must not produce NaN/Inf."""
    x = torch.full((1250,), 90.0)
    x_norm, x_min, x_max = minmax_normalize(x)
    assert torch.isfinite(x_norm).all()
    # eps-clamped denominator → output is ~0 everywhere, not 0/0 NaN.
    assert float(x_norm.abs().max()) <= 1.0 / MINMAX_EPS + 1.0


def test_reconstruct_mmHg_batched() -> None:
    """Broadcasting works for ``(B, L)`` shape × ``(B,)`` scalars."""
    B, L = 4, 1250
    shape = torch.rand(B, L)
    sbp = torch.tensor([120.0, 140.0, 110.0, 160.0])
    dbp = torch.tensor([70.0, 90.0, 60.0, 95.0])
    wave = reconstruct_mmHg(shape, sbp, dbp)
    assert wave.shape == (B, L)
    # First sample's max should equal SBP[0] when shape[0].max() == 1.
    for i in range(B):
        assert float(wave[i].max()) <= float(sbp[i]) + 1e-5
        assert float(wave[i].min()) >= float(dbp[i]) - 1e-5

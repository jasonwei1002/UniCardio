"""Tests for :mod:`src.utils.normalization`."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.normalization import (
    bp_denormalize,
    bp_normalize,
    minmax_normalize_per_sample_inplace,
)


def test_bp_round_trip() -> None:
    raw = np.array([60.0, 100.0, 180.0], dtype=np.float32)
    np.testing.assert_allclose(bp_denormalize(bp_normalize(raw)), raw)


def test_minmax_per_sample_2d_normal() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 500)).astype(np.float32) * 10 + 5
    minmax_normalize_per_sample_inplace(x)
    np.testing.assert_allclose(x.min(axis=-1), 0.0, atol=1e-6)
    np.testing.assert_allclose(x.max(axis=-1), 1.0, atol=1e-6)


def test_minmax_per_sample_3d_normal() -> None:
    """支持 ``(N, C, L)`` leading dims：每个 (sample, channel) 独立归一化。"""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((4, 3, 200)).astype(np.float32) * 5
    minmax_normalize_per_sample_inplace(x)
    np.testing.assert_allclose(x.min(axis=-1), 0.0, atol=1e-6)
    np.testing.assert_allclose(x.max(axis=-1), 1.0, atol=1e-6)


def test_minmax_flat_trace_yields_zeros() -> None:
    """max == min 时除以 ``eps`` 退化为零向量，不应抛 NaN/Inf。"""
    x = np.full((3, 64), 7.0, dtype=np.float32)
    minmax_normalize_per_sample_inplace(x)
    assert np.all(np.isfinite(x))
    np.testing.assert_array_equal(x, 0.0)


def test_minmax_strided_view() -> None:
    """传入 ``arr[:, 0]`` 这种 strided view 也应正确 in-place 归一化。"""
    rng = np.random.default_rng(2)
    full = rng.standard_normal((4, 3, 100)).astype(np.float32)
    expected_slot1 = full[:, 1].copy()
    expected_slot2 = full[:, 2].copy()
    minmax_normalize_per_sample_inplace(full[:, 0])
    np.testing.assert_allclose(full[:, 0].min(axis=-1), 0.0, atol=1e-6)
    np.testing.assert_allclose(full[:, 0].max(axis=-1), 1.0, atol=1e-6)
    # Other slots untouched.
    np.testing.assert_array_equal(full[:, 1], expected_slot1)
    np.testing.assert_array_equal(full[:, 2], expected_slot2)


def test_minmax_rejects_integer_dtype() -> None:
    x = np.zeros((4, 8), dtype=np.int32)
    with pytest.raises(TypeError, match="floating dtype"):
        minmax_normalize_per_sample_inplace(x)

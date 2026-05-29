"""Unit tests for :mod:`src.trainer_module.wcl` (MD-ViSCo multi-WCL port).

Validates the single-term math against an independent reference, plus the
multi-term composition's skip-on-missing behavior, threshold semantics, and
device/dtype robustness.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.trainer_module.wcl import (
    DEFAULT_WCL_TERMS,
    multi_wcl,
    weighted_contrastive_loss,
)


def _reference_wcl(
    emb: torch.Tensor,
    w: torch.Tensor,
    *,
    temp_emb: float,
    temp_w: float,
    threshold: float,
    scale: float,
) -> float:
    """Independent loop-based reference (no broadcasting tricks)."""
    emb = emb.double()
    w = w.double().reshape(-1)
    b = emb.shape[0]
    # weight similarity + threshold
    W = torch.empty(b, b, dtype=torch.double)
    for i in range(b):
        for j in range(b):
            W[i, j] = math.exp(-abs(w[i].item() - w[j].item()) / temp_w)
    W = torch.where(W >= threshold, W, torch.zeros_like(W))
    S = (emb @ emb.T) / temp_emb
    logp = torch.log_softmax(S, dim=-1)
    per = torch.empty(b, dtype=torch.double)
    for i in range(b):
        num = sum(W[i, j].item() * logp[i, j].item() for j in range(b))
        per[i] = -num / (W[i].sum().item() + 1e-8)
    return float((per * scale).mean())


def test_single_term_matches_reference():
    torch.manual_seed(0)
    emb = torch.randn(8, 16)
    w = torch.randn(8) * 20 + 120  # mmHg-ish
    got = weighted_contrastive_loss(
        emb, w,
        temperature_embeddings=1.0, temperature_weight=1.0,
        threshold=0.0, scale_factor=1e-3,
    )
    ref = _reference_wcl(emb, w, temp_emb=1.0, temp_w=1.0, threshold=0.0, scale=1e-3)
    assert got.item() == pytest.approx(ref, rel=1e-6, abs=1e-9)


def test_single_term_matches_reference_with_threshold_and_temps():
    torch.manual_seed(1)
    emb = torch.randn(10, 12)
    w = torch.randint(0, 2, (10,)).float()  # gender-like
    got = weighted_contrastive_loss(
        emb, w,
        temperature_embeddings=4.0, temperature_weight=1.0,
        threshold=1.0, scale_factor=1e-2,
    )
    ref = _reference_wcl(emb, w, temp_emb=4.0, temp_w=1.0, threshold=1.0, scale=1e-2)
    assert got.item() == pytest.approx(ref, rel=1e-6, abs=1e-9)


def test_accepts_1d_and_2d_weights():
    torch.manual_seed(2)
    emb = torch.randn(6, 8)
    w1 = torch.randn(6)
    a = weighted_contrastive_loss(emb, w1)
    b = weighted_contrastive_loss(emb, w1.unsqueeze(-1))
    assert a.item() == pytest.approx(b.item(), rel=1e-7)


def test_loss_is_nonnegative_and_scalar():
    torch.manual_seed(3)
    emb = torch.randn(16, 32)
    w = torch.randn(16) * 15 + 80
    loss = weighted_contrastive_loss(emb, w, scale_factor=1e-3)
    assert loss.ndim == 0
    assert loss.item() >= 0.0  # -log_softmax >= 0, weights >= 0


def test_gradient_flows_to_embeddings():
    torch.manual_seed(4)
    emb = torch.randn(8, 16, requires_grad=True)
    w = torch.randn(8) * 20 + 120
    loss = weighted_contrastive_loss(emb, w, scale_factor=1.0)
    loss.backward()
    assert emb.grad is not None
    assert torch.isfinite(emb.grad).all()


def test_fp32_compute_under_bf16_inputs():
    torch.manual_seed(5)
    emb = torch.randn(8, 16, dtype=torch.bfloat16)
    w = (torch.randn(8) * 20 + 120).to(torch.bfloat16)
    loss = weighted_contrastive_loss(emb, w, scale_factor=1e-3)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)


def test_multi_wcl_sums_active_terms():
    torch.manual_seed(6)
    b, d = 12, 16
    emb = {
        "ecg_embeddings": torch.randn(b, d),
        "ppg_embeddings": torch.randn(b, d),
        "text_embeddings": torch.randn(b, d),
    }
    w = {
        "y_sbp_raw": torch.randn(b) * 20 + 120,
        "y_dbp_raw": torch.randn(b) * 10 + 70,
        "gender_raw": torch.randint(0, 2, (b,)).float(),
        "age_raw": torch.randn(b) * 15 + 50,
    }
    total, per_term = multi_wcl(emb, w)
    # All 6 default terms active.
    assert set(per_term) == {t.name for t in DEFAULT_WCL_TERMS}
    # per_term holds detached scalar tensors (no per-batch .item() sync).
    assert all(isinstance(v, torch.Tensor) for v in per_term.values())
    # total == sum of the individual term values (within fp tolerance).
    assert total.item() == pytest.approx(
        sum(v.item() for v in per_term.values()), rel=1e-5, abs=1e-8
    )


def test_multi_wcl_skips_missing_embedding_or_weight():
    torch.manual_seed(7)
    b, d = 10, 16
    # ppg2abp-like: only PPG embedding present, no demographics.
    emb = {"ppg_embeddings": torch.randn(b, d)}
    w = {"y_sbp_raw": torch.randn(b) * 20 + 120, "y_dbp_raw": torch.randn(b) * 10 + 70}
    total, per_term = multi_wcl(emb, w)
    assert set(per_term) == {"ppg_sbp", "ppg_dbp"}  # ecg_* and text_* skipped
    assert torch.isfinite(total)


def test_multi_wcl_no_active_terms_returns_zero():
    total, per_term = multi_wcl({"unused": torch.randn(4, 8)}, {})
    assert per_term == {}
    assert total.item() == 0.0


def test_default_terms_match_mdvisco_spec():
    by_name = {t.name: t for t in DEFAULT_WCL_TERMS}
    assert len(DEFAULT_WCL_TERMS) == 6
    for n in ("ecg_sbp", "ppg_sbp", "ecg_dbp", "ppg_dbp"):
        assert by_name[n].scale_factor == pytest.approx(1e-3)
        assert by_name[n].embedding_key.endswith("_embeddings")
    g = by_name["text_gender_wcl"]
    assert (g.threshold, g.scale_factor, g.temperature_embeddings) == (1.0, 1e-2, 4.0)
    a = by_name["text_age_wcl"]
    assert (a.threshold, a.temperature_weight, a.scale_factor) == (0.0235, 4.0, 1e-2)

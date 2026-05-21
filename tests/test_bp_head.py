"""Unit tests for :class:`src.model_module.bp_head.BPHead`."""

from __future__ import annotations

import pytest
import torch

from src.model_module.bp_head import BPHead, BPHeadConfig, build_bp_head


def test_bp_head_forward_shape() -> None:
    model = build_bp_head({})
    x = torch.randn(4, 2, 1250)
    out = model(x)
    assert out.shape == (4, 2)
    assert out.dtype == torch.float32


def test_bp_head_param_budget() -> None:
    """Sanity ceiling: default config should stay under 200 k params.

    Stays >100x smaller than the ~30 M UniCardio backbone so train cost is
    a rounding error. Tighten this if the architecture is intentionally
    shrunk.
    """
    model = build_bp_head({})
    assert model.num_parameters() < 200_000


def test_bp_head_backward_grads() -> None:
    """All trainable params receive non-zero grad on a single MSE step."""
    model = build_bp_head({})
    x = torch.randn(4, 2, 1250, requires_grad=False)
    target = torch.tensor([[120.0, 70.0]] * 4)
    pred = model(x)
    loss = ((pred - target) ** 2).mean()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"missing grad: {name}"
        assert torch.any(p.grad != 0), f"all-zero grad: {name}"


def test_bp_head_overfit_single_batch() -> None:
    """Should overfit a tiny synthetic batch to MAE < 5 mmHg in 400 steps."""
    torch.manual_seed(0)
    model = build_bp_head({"hidden": 32, "depth": 2})
    optim = torch.optim.Adam(model.parameters(), lr=3e-3)

    B = 8
    x = torch.randn(B, 2, 256)
    # Construct deterministic targets from the input — relating x to (sbp, dbp)
    # via fixed projections so the task is learnable.
    proj_sbp = torch.randn(2 * 256)
    proj_dbp = torch.randn(2 * 256)
    flat = x.reshape(B, -1)
    sbp_true = 100.0 + 20.0 * (flat @ proj_sbp / proj_sbp.numel() ** 0.5)
    dbp_true = 60.0 + 10.0 * (flat @ proj_dbp / proj_dbp.numel() ** 0.5)
    target = torch.stack([sbp_true, dbp_true], dim=-1)

    losses = []
    for _ in range(400):
        optim.zero_grad(set_to_none=True)
        pred = model(x)
        loss = ((pred - target) ** 2).mean()
        loss.backward()
        optim.step()
        losses.append(float(loss.item()))
    final_mae = float((model(x) - target).abs().mean().item())
    assert final_mae < 5.0, (
        f"BP head failed to overfit: final MAE={final_mae:.2f} mmHg "
        f"(loss trace: {losses[0]:.1f} -> {losses[-1]:.4f})"
    )


def test_bp_head_config_from_mapping() -> None:
    cfg = BPHeadConfig.from_mapping({"in_channels": 3, "hidden": 32})
    assert cfg.in_channels == 3
    assert cfg.hidden == 32
    assert cfg.depth == 3  # default
    # Idempotent on BPHeadConfig instance
    assert BPHeadConfig.from_mapping(cfg) is cfg

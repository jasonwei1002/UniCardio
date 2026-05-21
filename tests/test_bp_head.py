"""Unit tests for :class:`src.model_module.bp_head.BPHead` (Path A v2).

BPHead v2 = PatchTSMixer waveform encoder + demographic MLP + fusion head,
mirroring MD-ViSCo's refinement model. Input contract:
``forward(ecg_ppg: (B, 2, slot_length), demographics: (B, 6) | None)``.
"""

from __future__ import annotations

import pytest
import torch

from src.model_module.bp_head import BPHead, BPHeadConfig, build_bp_head


SLOT_LEN = 1250
DEMO_DIM = 6


def test_bp_head_forward_shape_with_demographics() -> None:
    model = build_bp_head({})
    x = torch.randn(4, 2, SLOT_LEN)
    d = torch.randn(4, DEMO_DIM)
    out = model(x, d)
    assert out.shape == (4, 2)
    assert out.dtype == torch.float32


def test_bp_head_forward_shape_without_demographics() -> None:
    """Demographics arg is optional; missing → zero-fill on the demo branch."""
    model = build_bp_head({})
    x = torch.randn(4, 2, SLOT_LEN)
    out = model(x)
    assert out.shape == (4, 2)


def test_bp_head_param_budget() -> None:
    """Default config sits at ~0.5M params — well under the 30M backbone."""
    model = build_bp_head({})
    assert 100_000 < model.num_parameters() < 1_000_000


def test_bp_head_backward_grads() -> None:
    model = build_bp_head({})
    x = torch.randn(4, 2, SLOT_LEN)
    d = torch.randn(4, DEMO_DIM)
    target = torch.tensor([[120.0, 70.0]] * 4)
    pred = model(x, d)
    loss = ((pred - target) ** 2).mean()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"missing grad: {name}"
        assert torch.any(p.grad != 0), f"all-zero grad: {name}"


def test_bp_head_overfit_single_batch() -> None:
    """Tiny BPHead overfits a synthetic batch to MAE < 5 mmHg in 400 steps."""
    torch.manual_seed(0)
    model = build_bp_head({
        "slot_length": 256, "patch_len": 32,
        "dim": 32, "depth": 2, "mlp_ratio": 1,
        "demo_hidden": 16, "fusion_hidden": 32,
    })
    optim = torch.optim.Adam(model.parameters(), lr=3e-3)

    B = 8
    x = torch.randn(B, 2, 256)
    d = torch.randn(B, DEMO_DIM)
    # Deterministic targets from input via fixed projections so the task is
    # learnable. SBP ~ 100 mmHg ± 20, DBP ~ 60 ± 10.
    proj_sbp = torch.randn(2 * 256 + DEMO_DIM)
    proj_dbp = torch.randn(2 * 256 + DEMO_DIM)
    flat = torch.cat([x.reshape(B, -1), d], dim=-1)
    sbp_true = 100.0 + 20.0 * (flat @ proj_sbp / proj_sbp.numel() ** 0.5)
    dbp_true = 60.0 + 10.0 * (flat @ proj_dbp / proj_dbp.numel() ** 0.5)
    target = torch.stack([sbp_true, dbp_true], dim=-1)

    for _ in range(400):
        optim.zero_grad(set_to_none=True)
        pred = model(x, d)
        loss = ((pred - target) ** 2).mean()
        loss.backward()
        optim.step()
    final_mae = float((model(x, d) - target).abs().mean().item())
    assert final_mae < 5.0, f"final MAE {final_mae:.2f} mmHg (failed to overfit)"


def test_bp_head_config_from_mapping() -> None:
    cfg = BPHeadConfig.from_mapping({"dim": 32, "depth": 2})
    assert cfg.dim == 32
    assert cfg.depth == 2
    # Defaults preserved
    assert cfg.in_channels == 2
    assert cfg.slot_length == 1250
    # Idempotent on BPHeadConfig instance
    assert BPHeadConfig.from_mapping(cfg) is cfg


def test_bp_head_rejects_mismatched_input_shape() -> None:
    model = build_bp_head({})
    bad = torch.randn(2, 2, 1234)  # wrong slot_length
    with pytest.raises(ValueError, match="BPHead expects"):
        model(bad)


def test_bp_head_patchify_divisibility() -> None:
    with pytest.raises(ValueError, match="divisible"):
        BPHeadConfig(slot_length=1250, patch_len=51).n_patches

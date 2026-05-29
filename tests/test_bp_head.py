"""Unit tests for :class:`src.model_module.bp_head.BPHead` (v4).

v4 = faithful MD-ViSCo ``VitalEncoder`` (pi=true): per-vital
``waveform_encoder -> flatten -> ProjectionHead -> 2-channel fusion encoder
(wave emb + numeric demo emb) -> flatten -> MlpBP(sbp/dbp)``, averaged over
``active_vitals``. Input contract:
``forward(ecg_ppg: (B, 2, slot_length), demographics: (B, 6) | None,
         active_vitals: Iterable[str] | None)``.
"""

from __future__ import annotations

import pytest
import torch

from src.model_module.bp_head import BPHead, BPHeadConfig, build_bp_head

DEMO_DIM = 6

# Small config keeps the two PatchTSMixer stages + forward fast on CPU.
TINY = {
    "slot_length": 256,
    "patch_len": 32,
    "patch_stride": 32,
    "d_model": 16,
    "num_layers": 2,
    "expansion_factor": 2,
    "projection_dim": 64,
    "proj_dropout": 0.0,
    "mlp_dropout": 0.0,
    "demo_hidden": 16,
}


def _tiny() -> BPHead:
    return build_bp_head(TINY)


def test_forward_shape_with_demographics() -> None:
    model = _tiny()
    x = torch.randn(4, 2, TINY["slot_length"])
    d = torch.randn(4, DEMO_DIM)
    out = model(x, d)
    assert out.shape == (4, 2)
    assert out.dtype == torch.float32


def test_forward_shape_without_demographics() -> None:
    model = _tiny()
    x = torch.randn(4, 2, TINY["slot_length"])
    out = model(x)
    assert out.shape == (4, 2)


def test_aggregate_equals_mean_of_single_vitals() -> None:
    """In eval mode (no dropout), the both-vital average equals the mean of
    the two single-vital predictions."""
    model = _tiny().eval()
    x = torch.randn(4, 2, TINY["slot_length"])
    d = torch.randn(4, DEMO_DIM)
    with torch.no_grad():
        both = model(x, d, active_vitals=None)
        ecg = model(x, d, active_vitals=["ecg"])
        ppg = model(x, d, active_vitals=["ppg"])
    assert torch.allclose(both, (ecg + ppg) / 2, atol=1e-5)
    assert not torch.allclose(ecg, ppg)


def test_active_vitals_single_uses_only_that_encoder() -> None:
    model = _tiny().eval()
    x = torch.randn(2, 2, TINY["slot_length"])
    # Zeroing the PPG channel must not change an ECG-only prediction.
    x_zero_ppg = x.clone()
    x_zero_ppg[:, 1, :] = 0.0
    with torch.no_grad():
        a = model(x, active_vitals=["ecg"])
        b = model(x_zero_ppg, active_vitals=["ecg"])
    assert torch.allclose(a, b, atol=1e-6)


def test_demographics_changes_prediction() -> None:
    """The demo channel actually feeds the fusion encoder (pi=true)."""
    model = _tiny().eval()
    x = torch.randn(3, 2, TINY["slot_length"])
    d = torch.randn(3, DEMO_DIM)
    with torch.no_grad():
        with_demo = model(x, d)
        no_demo = model(x, None)
    assert not torch.allclose(with_demo, no_demo, atol=1e-4)


def test_backward_grads() -> None:
    model = _tiny()
    x = torch.randn(4, 2, TINY["slot_length"])
    d = torch.randn(4, DEMO_DIM)
    target = torch.tensor([[120.0, 70.0]] * 4)
    loss = ((model(x, d) - target) ** 2).mean()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"missing grad: {name}"


def test_overfit_single_batch() -> None:
    """Tiny BPHead overfits a synthetic batch to MAE < 5 mmHg."""
    torch.manual_seed(0)
    model = _tiny()
    optim = torch.optim.Adam(model.parameters(), lr=3e-3)
    B, L = 8, TINY["slot_length"]
    x = torch.randn(B, 2, L)
    d = torch.randn(B, DEMO_DIM)
    proj = torch.randn(2 * L + DEMO_DIM, 2)
    flat = torch.cat([x.reshape(B, -1), d], dim=-1)
    base = torch.tensor([100.0, 60.0])
    scale = torch.tensor([20.0, 10.0])
    target = base + scale * (flat @ proj / proj.shape[0] ** 0.5)
    for _ in range(400):
        optim.zero_grad(set_to_none=True)
        loss = ((model(x, d) - target) ** 2).mean()
        loss.backward()
        optim.step()
    final_mae = float((model(x, d) - target).abs().mean().item())
    assert final_mae < 5.0, f"final MAE {final_mae:.2f} mmHg (failed to overfit)"


def test_rejects_mismatched_slot_length() -> None:
    model = _tiny()
    bad = torch.randn(2, 2, TINY["slot_length"] + 7)
    with pytest.raises(ValueError, match="BPHead expects"):
        model(bad)


def test_config_from_mapping_defaults_and_override() -> None:
    cfg = BPHeadConfig.from_mapping({"d_model": 32, "num_layers": 3, "projection_dim": 256})
    assert cfg.d_model == 32
    assert cfg.num_layers == 3
    assert cfg.projection_dim == 256
    assert cfg.slot_length == 1250
    assert cfg.vitals == ("ecg", "ppg")
    assert BPHeadConfig.from_mapping(cfg) is cfg


def test_config_rejects_unknown_vital() -> None:
    with pytest.raises(ValueError, match="unknown vital"):
        build_bp_head({**TINY, "vitals": ["ecg", "abp"]})


def test_default_config_param_budget() -> None:
    """Faithful MD-ViSCo pi=true config (depth 15, d_model 64, proj 512, 2 vitals)
    is ~385M params. Instantiation only (no forward)."""
    model = build_bp_head({})
    n = model.num_parameters()
    assert 350_000_000 < n < 420_000_000, f"param count {n:,} outside 350M-420M"


def test_return_embeddings_keys_and_shapes() -> None:
    """return_embeddings exposes per-vital waveform embeddings + text embedding
    for WCL, keyed to match src.trainer_module.wcl term embedding_keys."""
    model = _tiny().eval()
    x = torch.randn(4, 2, TINY["slot_length"])
    d = torch.randn(4, DEMO_DIM)
    with torch.no_grad():
        out, emb = model(x, d, return_embeddings=True)
    assert out.shape == (4, 2)
    assert set(emb) == {"ecg_embeddings", "ppg_embeddings", "text_embeddings"}
    for v in emb.values():
        assert v.shape == (4, TINY["projection_dim"])


def test_return_embeddings_matches_plain_forward() -> None:
    """Predictions are identical whether or not embeddings are returned."""
    model = _tiny().eval()
    x = torch.randn(3, 2, TINY["slot_length"])
    d = torch.randn(3, DEMO_DIM)
    with torch.no_grad():
        plain = model(x, d)
        out, _ = model(x, d, return_embeddings=True)
    assert torch.allclose(plain, out, atol=1e-6)


def test_return_embeddings_omits_text_without_demographics() -> None:
    model = _tiny().eval()
    x = torch.randn(2, 2, TINY["slot_length"])
    with torch.no_grad():
        _, emb = model(x, None, return_embeddings=True)
    assert "text_embeddings" not in emb  # no demographics → no PI embedding
    assert set(emb) == {"ecg_embeddings", "ppg_embeddings"}


def test_return_embeddings_single_active_vital() -> None:
    model = _tiny().eval()
    x = torch.randn(2, 2, TINY["slot_length"])
    d = torch.randn(2, DEMO_DIM)
    with torch.no_grad():
        _, emb = model(x, d, active_vitals=["ppg"], return_embeddings=True)
    assert "ppg_embeddings" in emb and "ecg_embeddings" not in emb


def test_encode_embeddings_matches_forward_embeddings() -> None:
    """encode_embeddings (no fusion/heads) yields the same waveform/text
    embeddings as forward(return_embeddings=True)."""
    model = _tiny().eval()
    x = torch.randn(3, 2, TINY["slot_length"])
    d = torch.randn(3, DEMO_DIM)
    with torch.no_grad():
        _, emb_full = model(x, d, return_embeddings=True)
        emb_only = model.encode_embeddings(x, d)
    assert set(emb_only) == set(emb_full)
    for k in emb_full:
        assert torch.allclose(emb_only[k], emb_full[k], atol=1e-6)


def test_encode_embeddings_omits_text_without_demographics() -> None:
    model = _tiny().eval()
    x = torch.randn(2, 2, TINY["slot_length"])
    with torch.no_grad():
        emb = model.encode_embeddings(x, None)
    assert set(emb) == {"ecg_embeddings", "ppg_embeddings"}


def test_encode_embeddings_grad_reaches_encoder_not_heads() -> None:
    """Contrastive pretraining trains the encoders/projection (+ demo MLP) but
    NOT the fusion encoders or MlpBP heads."""
    model = _tiny()
    x = torch.randn(4, 2, TINY["slot_length"])
    d = torch.randn(4, DEMO_DIM)
    emb = model.encode_embeddings(x, d)
    loss = sum(e.pow(2).mean() for e in emb.values())
    loss.backward()
    # Encoder + projection + demo MLP receive gradient.
    assert model.projections["ecg"].projection.weight.grad is not None
    assert model.demo_encoder[0].weight.grad is not None
    # MlpBP heads + fusion encoders are untouched (not in the embedding path).
    assert model.sbp_heads["ecg"].fc1.weight.grad is None

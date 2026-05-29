"""CPU integration smoke for the BP-head trainer's WCL + label-source paths.

Builds a tiny synthetic ``.npy`` + sibling ``.csv``, real :class:`CardiacDataset`
loaders, and runs :func:`src.trainer_module.bp_head_trainer.train` end-to-end for
a couple of epochs across all three loss modes (``l1`` / ``wcl_only`` /
``l1+wcl``) and both label sources. Verifies the run completes, produces finite
losses, and writes ``best.pt``. This exercises the dataset 5-tuple, WCL weight
plumbing, ``segment_minmax`` target derivation, eval, and best-selection without
the 13 GB PulseDB data or a GPU.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.data_module.cardiac_dataset import CardiacDataset
from src.model_module.bp_head import build_bp_head
from src.trainer_module import bp_head_trainer
from src.utils.normalization import BPLabelNorm

L = 256
TINY_MODEL = {
    "slot_length": L,
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


@pytest.fixture
def synthetic_data(tmp_path):
    """Tiny (N, 3, L) npy + sibling csv; returns (npy_path, n)."""
    rng = np.random.default_rng(0)
    n = 24
    x = rng.standard_normal((n, 3, L)).astype(np.float32)
    # Make slot-2 (ABP) positive-ish mmHg so per-segment min/max are sane.
    x[:, 2, :] = 80.0 + 40.0 * rng.random((n, L)).astype(np.float32)
    npy = tmp_path / "tiny.npy"
    np.save(npy, x)
    df = pd.DataFrame(
        {
            "sbp": rng.uniform(100, 160, n),
            "dbp": rng.uniform(60, 90, n),
            "age": rng.uniform(20, 80, n),
            "gender": rng.integers(0, 2, n),
            "height": rng.uniform(150, 190, n),
            "weight": rng.uniform(50, 100, n),
            "bmi": rng.uniform(18, 35, n),
        }
    )
    df.to_csv(tmp_path / "tiny.csv", index=False)
    return npy, n


def _loaders(npy, n, bp_norm):
    ds_train = CardiacDataset(npy, np.arange(0, 16), bp_label_norm=bp_norm)
    ds_val = CardiacDataset(
        npy, np.arange(16, n),
        bp_labels_table=ds_train.bp_labels_table,
        demographics_table=ds_train.demographics_table,
        age_raw_table=ds_train.age_raw_table,
        bp_label_norm=bp_norm,
    )
    return (
        DataLoader(ds_train, batch_size=4, shuffle=True, drop_last=True),
        DataLoader(ds_val, batch_size=4, shuffle=False),
    )


def _cfg(*, wcl_enabled, contrastive_only):
    return {
        "epochs": 2,
        "lr": 1e-3,
        "weight_decay": 1e-6,
        "grad_clip_norm": 1.0,
        "lr_scheduler": {"name": "cosine", "min_lr": 1e-5, "first_cycle_pct": 1.0, "cycle_mult": 1},
        "amp": {"enabled": False},
        "val_every": 1,
        "ckpt_every": 1,
        "log_filename": "bp_head_loss.csv",
        "log_every_n_steps": 2,
        "stage": "pretrain",
        "init_from": None,
        "bp_task": "ecgppg2abp",
        "wcl": {"enabled": wcl_enabled, "pretrain_contrastive_only": contrastive_only, "terms": None},
    }


def test_dataset_returns_five_tuple_with_age(synthetic_data):
    npy, n = synthetic_data
    ds = CardiacDataset(npy, np.arange(0, n))
    item = ds[0]
    assert len(item) == 5
    signal, sbp_dbp, demo, abp_minmax, age_raw = item
    assert signal.shape == (3, L)
    assert sbp_dbp.shape == (2,)
    assert demo.shape == (6,)
    assert abp_minmax.shape == (2,)
    assert age_raw.shape == (1,)
    # age_raw is in raw years (20..80), not z-scored.
    assert 15.0 <= float(age_raw[0]) <= 90.0
    # abp_minmax = (dbp_seg, sbp_seg) = (min, max) of the raw ABP slot.
    assert abp_minmax[0] <= abp_minmax[1]


@pytest.mark.parametrize(
    "wcl_enabled,contrastive_only,expected_mode",
    [(False, True, "l1"), (True, True, "wcl_only"), (True, False, "l1+wcl")],
)
@pytest.mark.parametrize("bp_label_source", ["segment_minmax", "per_cycle_mean"])
def test_train_smoke_all_modes(
    synthetic_data, tmp_path, monkeypatch, wcl_enabled, contrastive_only,
    expected_mode, bp_label_source,
):
    npy, n = synthetic_data
    # No-op SwanLab so the trainer's logging calls don't require swanlab.init.
    monkeypatch.setattr(bp_head_trainer.swanlab, "log", lambda *a, **k: None)

    assert bp_head_trainer._resolve_loss_mode(
        {"wcl": {"enabled": wcl_enabled, "pretrain_contrastive_only": contrastive_only}},
        "pretrain",
    ) == expected_mode

    bp_norm = BPLabelNorm(vmin=40.0, vmax=180.0)
    train_loader, val_loader = _loaders(npy, n, bp_norm)
    model = build_bp_head(TINY_MODEL)
    out_dir = tmp_path / f"run_{expected_mode}_{bp_label_source}"

    bp_head_trainer.train(
        model,
        _cfg(wcl_enabled=wcl_enabled, contrastive_only=contrastive_only),
        train_loader,
        val_loader,
        device=torch.device("cpu"),
        output_dir=out_dir,
        bp_norm=bp_norm,
        bp_label_source=bp_label_source,
    )

    # best.pt written (selection ran) and loads back.
    best = out_dir / "checkpoints" / "best.pt"
    assert best.exists()
    ckpt = torch.load(best, map_location="cpu", weights_only=False)
    assert "model_state" in ckpt
    # save_checkpoint merges `extra` into the top-level payload.
    assert np.isfinite(ckpt["val_loss_mean"])

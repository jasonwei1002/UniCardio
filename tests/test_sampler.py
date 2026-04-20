"""Unit tests for the Euler ODE sampler."""

from __future__ import annotations

import pytest
import torch

from src.model_module.backbone import BackboneConfig
from src.model_module.tasks import TASK_LIST
from src.model_module.unicardio_rf import UniCardioRF
from src.trainer_module.sampler import euler_sample

SLOT_LEN = 32
B = 2


@pytest.fixture(scope="module")
def tiny_model() -> UniCardioRF:
    torch.manual_seed(0)
    cfg = BackboneConfig(
        slot_length=SLOT_LEN,
        channels=288,
        n_layers=2,
        nheads=4,
        time_embedding_dim=64,
        ffn_dim=32,
    )
    return UniCardioRF(cfg)


@pytest.mark.parametrize("task", TASK_LIST, ids=lambda t: t.name)
def test_euler_sample_shape(tiny_model, task):
    conditions = torch.randn(B, 3, SLOT_LEN)
    out = euler_sample(tiny_model, conditions, task, n_steps=4)
    assert out.shape == (B, 1, SLOT_LEN)
    assert torch.isfinite(out).all()


def test_euler_sample_trajectory_length(tiny_model):
    conditions = torch.randn(B, 3, SLOT_LEN)
    x0, traj = euler_sample(
        tiny_model, conditions, TASK_LIST[0], n_steps=4, return_trajectory=True
    )
    assert x0.shape == (B, 1, SLOT_LEN)
    assert traj.shape == (5, B, 1, SLOT_LEN)  # n_steps + 1


def test_euler_sample_restores_train_mode(tiny_model):
    tiny_model.train()
    conditions = torch.randn(B, 3, SLOT_LEN)
    _ = euler_sample(tiny_model, conditions, TASK_LIST[0], n_steps=2)
    assert tiny_model.training is True


def test_euler_sample_rejects_bad_n_steps(tiny_model):
    conditions = torch.randn(B, 3, SLOT_LEN)
    with pytest.raises(ValueError):
        euler_sample(tiny_model, conditions, TASK_LIST[0], n_steps=0)

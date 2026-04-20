"""Rectified Flow 训练步的单元测试。"""

from __future__ import annotations

import pytest
import torch

from src.model_module.backbone import BackboneConfig
from src.model_module.tasks import TASK_LIST
from src.model_module.unicardio_rf import UniCardioRF
from src.trainer_module.rectified_flow import (
    assemble_x_full,
    build_rf_inputs,
    rf_train_step,
    sample_t_logit_normal,
)

SLOT_LEN = 32
B = 2


@pytest.fixture(scope="module")
def tiny_model() -> UniCardioRF:
    torch.manual_seed(0)
    cfg = BackboneConfig(
        slot_length=SLOT_LEN,
        channels=288,  # 由编码器 bank 固定：6 个卷积核 * 每核 48 通道
        n_layers=2,
        nheads=4,
        time_embedding_dim=64,
        ffn_dim=32,
    )
    return UniCardioRF(cfg)


def _random_batch() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(B, 3, SLOT_LEN)


def test_sample_t_logit_normal_shape_and_range():
    t = sample_t_logit_normal(128, device="cpu")
    assert t.shape == (128,)
    assert torch.all(t > 0.0) and torch.all(t < 1.0)


def test_assemble_x_full_replaces_target_slot():
    signal = torch.arange(B * 3 * SLOT_LEN, dtype=torch.float32).reshape(B, 3, SLOT_LEN)
    xt = torch.full((B, 1, SLOT_LEN), -99.0)
    out = assemble_x_full(signal, xt, target_slot=1, L=SLOT_LEN)
    assert out.shape == (B, 1, 3 * SLOT_LEN)
    # slot 0 保持不变。
    assert torch.equal(out[:, 0, :SLOT_LEN], signal[:, 0, :])
    # slot 1 被 xt 替换。
    assert torch.all(out[:, 0, SLOT_LEN:2 * SLOT_LEN] == -99.0)
    # slot 2 保持不变。
    assert torch.equal(out[:, 0, 2 * SLOT_LEN:], signal[:, 2, :])


def test_build_rf_inputs_shapes_and_values():
    signal = _random_batch()
    task = TASK_LIST[0]
    x_full, t, x0, v = build_rf_inputs(signal, task)
    assert x_full.shape == (B, 1, 3 * SLOT_LEN)
    assert t.shape == (B,)
    assert x0.shape == (B, 1, SLOT_LEN)
    assert v.shape == (B, 1, SLOT_LEN)
    # x0 必须正好等于原信号中 target slot 那一列。
    assert torch.equal(x0[:, 0, :], signal[:, task.target_slot, :])


@pytest.mark.parametrize("task", TASK_LIST, ids=lambda t: t.name)
def test_rf_train_step_produces_scalar_grad(tiny_model, task):
    signal = _random_batch()
    loss = rf_train_step(tiny_model, signal, task)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.requires_grad
    loss.backward()
    # target slot 对应的输出头必须累积到非零梯度。
    head = tiny_model.backbone.output_heads[int(task.target_slot)]
    assert head.proj2.weight.grad is not None
    assert torch.any(head.proj2.weight.grad != 0)


@pytest.mark.parametrize("task", TASK_LIST, ids=lambda t: t.name)
def test_non_target_heads_receive_no_grad(tiny_model, task):
    tiny_model.zero_grad(set_to_none=True)
    signal = _random_batch()
    loss = rf_train_step(tiny_model, signal, task)
    loss.backward()
    for i, head in enumerate(tiny_model.backbone.output_heads):
        if i == int(task.target_slot):
            continue
        # 非 target slot 的输出头从未被调用，其梯度应保持为 None。
        assert head.proj1.weight.grad is None
        assert head.proj2.weight.grad is None

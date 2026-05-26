"""Rectified Flow training step under Lipman convention (``t=0`` noise,
``t=1`` data). Time sampling is SD3-style logit-normal.

Convention: ``x1 = batch_signal[:, target_slot, :]`` for ABP-target
tasks is **per-sample min-max normalized to [0, 1]** (shape only), not the
legacy global ``(x - 100) / 50``. The velocity field therefore learns
shape geometry; absolute SBP/DBP is recovered by the separate ``BPHead``
at inference time via :func:`src.utils.normalization.reconstruct_mmHg`.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn

from ..model_module.attention_masks import select_task_mask
from ..model_module.tasks import TaskSpec


def sample_t_logit_normal(
    batch_size: int,
    device: str | torch.device,
    *,
    mean: float = 0.0,
    std: float = 1.0,
) -> Tensor:
    """SD3 风格的 logit-normal 时间采样。

    返回形状为 ``(batch_size,)`` 的 float 张量，取值于 ``(0, 1)``。
    """
    u = torch.randn(batch_size, device=device) * std + mean
    return torch.sigmoid(u)


def assemble_x_full(
    signal_3slot: Tensor,
    x_t_target: Tensor,
    target_slot: int,
    L: int,
) -> Tensor:
    """把按 slot 组织的信号展平为 ``(B, 1, 3 * L)``，同时替换 target slot。

    Args:
        signal_3slot: ``(B, 3, L)`` —— 模型空间下、按 slot 顺序的干净信号。
        x_t_target: ``(B, 1, L)`` —— 经加噪后的 target ``x_t`` 张量。
        target_slot: 当前正在重建的 slot 编号。
        L: ``slot_length``。

    Returns:
        形状为 ``(B, 1, 3 * L)`` 的张量，可直接传入 :meth:`UniCardioRF.forward`。
    """
    if signal_3slot.size(1) != 3:
        raise ValueError(
            f"Expected (B, 3, L); got {tuple(signal_3slot.shape)}"
        )
    if x_t_target.size(-1) != L:
        raise ValueError(
            f"Target x_t length {x_t_target.size(-1)} != L={L}"
        )
    B = signal_3slot.size(0)
    flat = signal_3slot.clone().reshape(B, 1, 3 * L)
    start = target_slot * L
    end = start + L
    flat[:, :, start:end] = x_t_target
    return flat


def build_rf_inputs(
    batch_signal: Tensor,
    task: TaskSpec,
    *,
    t: Tensor | None = None,
    eps: Tensor | None = None,
    t_mean: float = 0.0,
    t_std: float = 1.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """返回单步训练所需的 ``(x_full, t, x1_target, v_target)`` 张量。

    ``x1_target`` 是 target slot 的干净数据（Lipman 记号下 ``t = 1`` 端点）。
    从 :func:`rf_train_step` 中拆出来，方便测试在不触发反向传播的情况下
    直接检查各个张量。
    """
    if batch_signal.dim() != 3 or batch_signal.size(1) != 3:
        raise ValueError(
            f"batch_signal must be (B, 3, L); got {tuple(batch_signal.shape)}"
        )
    B, _, L = batch_signal.shape
    device = batch_signal.device
    target = int(task.target_slot)

    x1 = batch_signal[:, target : target + 1, :]  # (B, 1, L)，数据端点

    if eps is None:
        eps = torch.randn_like(x1)
    if t is None:
        t = sample_t_logit_normal(B, device=device, mean=t_mean, std=t_std)
    t_b = t.view(B, 1, 1)

    x_t = (1.0 - t_b) * eps + t_b * x1
    v_target = x1 - eps
    x_full = assemble_x_full(batch_signal, x_t, target_slot=target, L=L)
    return x_full, t, x1, v_target


def rf_train_step(
    model: nn.Module,
    batch_signal: Tensor,
    task: TaskSpec,
    *,
    t_mean: float = 0.0,
    t_std: float = 1.0,
) -> Tensor:
    """Scalar Rectified Flow loss for one batch.

    ``task`` is unpacked into a bool mask + ``int target_slot`` outside the
    compile region so ``torch.compile`` specializes on 3 ``target_slot`` values,
    not 15 (slot × task name) Dynamo frames.
    """
    x_full, t, _, v_target = build_rf_inputs(
        batch_signal, task, t_mean=t_mean, t_std=t_std
    )
    _, _, L_total = x_full.shape
    L_slot = L_total // 3
    mask = select_task_mask(task.name, L_slot, x_full.device)
    v_pred = model(x_full, t, mask, int(task.target_slot))
    if v_pred.shape != v_target.shape:
        raise RuntimeError(
            f"Velocity prediction shape {tuple(v_pred.shape)} "
            f"!= target {tuple(v_target.shape)}"
        )
    return (v_pred - v_target).pow(2).mean()

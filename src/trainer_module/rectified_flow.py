"""Rectified-Flow training step: time sampling, x_t construction, loss.

Training objective (from Liu et al., 2022 / SD3):

    x_t     = (1 - t) * x_0 + t * eps,   eps ~ N(0, I),   t ∈ (0, 1)
    v_true  = eps - x_0
    loss    = E_{x_0, eps, t} [ || v_theta(x_t, t) - v_true ||^2 ]

The network only predicts velocity for the *target* slot; non-target slots
hold the clean conditioning signals and never contribute to the loss.

Time sampling follows SD3's logit-normal: ``u ~ N(0, 1); t = sigmoid(u)``.
Compared with uniform ``t``, this concentrates training signal around
``t ≈ 0.5`` where the trajectory is most ambiguous, which empirically speeds
convergence.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn

from ..model_module.tasks import TaskSpec


def sample_t_logit_normal(
    batch_size: int,
    device: str | torch.device,
    *,
    mean: float = 0.0,
    std: float = 1.0,
) -> Tensor:
    """SD3-style logit-normal time sampling.

    Returns ``(batch_size,)`` float tensor with values in ``(0, 1)``.
    """
    u = torch.randn(batch_size, device=device) * std + mean
    return torch.sigmoid(u)


def assemble_x_full(
    signal_3slot: Tensor,
    x_t_target: Tensor,
    target_slot: int,
    L: int,
) -> Tensor:
    """Flatten per-slot signals to ``(B, 1, 3 * L)`` with target slot replaced.

    Args:
        signal_3slot: ``(B, 3, L)`` — clean slot-ordered signals in model space.
        x_t_target: ``(B, 1, L)`` — the noised target ``x_t`` tensor.
        target_slot: Which slot is being reconstructed.
        L: ``slot_length``.

    Returns:
        ``(B, 1, 3 * L)`` tensor suitable for :meth:`UniCardioRF.forward`.
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
    """Return ``(x_full, t, x0_target, v_target)`` tensors for one step.

    Factored out of :func:`rf_train_step` so tests can inspect the tensors
    without having to run a backward pass.
    """
    if batch_signal.dim() != 3 or batch_signal.size(1) != 3:
        raise ValueError(
            f"batch_signal must be (B, 3, L); got {tuple(batch_signal.shape)}"
        )
    B, _, L = batch_signal.shape
    device = batch_signal.device
    target = int(task.target_slot)

    x0 = batch_signal[:, target : target + 1, :]  # (B, 1, L)

    if eps is None:
        eps = torch.randn_like(x0)
    if t is None:
        t = sample_t_logit_normal(B, device=device, mean=t_mean, std=t_std)
    t_b = t.view(B, 1, 1)

    x_t = (1.0 - t_b) * x0 + t_b * eps
    v_target = eps - x0
    x_full = assemble_x_full(batch_signal, x_t, target_slot=target, L=L)
    return x_full, t, x0, v_target


def rf_train_step(
    model: nn.Module,
    batch_signal: Tensor,
    task: TaskSpec,
    *,
    t_mean: float = 0.0,
    t_std: float = 1.0,
) -> Tensor:
    """Compute the scalar Rectified-Flow loss for one batch.

    The model is expected to accept ``(x_full, t, task)`` and return a
    velocity prediction of shape ``(B, 1, L)``.
    """
    x_full, t, _, v_target = build_rf_inputs(
        batch_signal, task, t_mean=t_mean, t_std=t_std
    )
    v_pred = model(x_full, t, task)
    if v_pred.shape != v_target.shape:
        raise RuntimeError(
            f"Velocity prediction shape {tuple(v_pred.shape)} "
            f"!= target {tuple(v_target.shape)}"
        )
    return (v_pred - v_target).pow(2).mean()

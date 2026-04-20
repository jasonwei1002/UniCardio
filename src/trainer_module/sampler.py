"""Euler ODE sampler for Rectified Flow inference.

Given clean condition slots and a task, integrate the learned velocity field
from ``t = 1`` (pure noise) to ``t = 0`` (data) with a fixed number of Euler
steps. Default ``n_steps = 8`` follows the Rectified Flow paper's observation
that 4-16 steps already match quality of many-step diffusion samplers on
comparable tasks; tune via the sampler config.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn

from ..model_module.tasks import TaskSpec
from .rectified_flow import assemble_x_full


@torch.no_grad()
def euler_sample(
    model: nn.Module,
    conditions: Tensor,
    task: TaskSpec,
    *,
    n_steps: int = 8,
    device: str | torch.device | None = None,
    return_trajectory: bool = False,
) -> Tensor | Tuple[Tensor, Tensor]:
    """Sample the target slot via an Euler ODE integration of ``v_theta``.

    Args:
        model: Trained :class:`UniCardioRF` (or equivalent interface).
        conditions: ``(B, 3, L)`` — condition slots contain clean signals;
            the target slot's contents are ignored (replaced by ``x_t``).
        task: :class:`TaskSpec` describing which slot to sample.
        n_steps: Number of Euler steps over ``t ∈ [1, 0]``.
        device: Target device; defaults to ``conditions.device``.
        return_trajectory: If True, also return the full trajectory of shape
            ``(n_steps + 1, B, 1, L)`` for diagnostics.

    Returns:
        Either ``x_0 ≈ (B, 1, L)`` or ``(x_0, trajectory)``.
    """
    if conditions.dim() != 3 or conditions.size(1) != 3:
        raise ValueError(
            f"conditions must be (B, 3, L); got {tuple(conditions.shape)}"
        )
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1; got {n_steps}")

    device = torch.device(device) if device is not None else conditions.device
    model_was_training = model.training
    model.eval()

    B, _, L = conditions.shape
    target = int(task.target_slot)
    conditions = conditions.to(device)

    x = torch.randn(B, 1, L, device=device)  # x_1 ~ N(0, I)
    ts = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
    traj: list[Tensor] = [x.clone()] if return_trajectory else []

    for i in range(n_steps):
        t_cur, t_next = ts[i], ts[i + 1]
        dt = t_next - t_cur  # negative
        t_b = torch.full((B,), float(t_cur), device=device)
        x_full = assemble_x_full(conditions, x, target_slot=target, L=L)
        v = model(x_full, t_b, task)
        x = x + v * dt
        if return_trajectory:
            traj.append(x.clone())

    if model_was_training:
        model.train()

    if return_trajectory:
        return x, torch.stack(traj, dim=0)
    return x

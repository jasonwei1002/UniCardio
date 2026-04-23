"""Rectified Flow 推断阶段的 Euler ODE 采样器。

给定干净的条件 slot 和任务描述后，用学到的速度场做 Euler 积分，从
``t = 1``（纯噪声）积到 ``t = 0``（数据），步数固定。
默认 ``n_steps = 8`` 参照 Rectified Flow 论文的观察：4~16 步已经能在
同类任务上逼近多步扩散采样器的质量，可通过 sampler 配置继续调整。
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn

from ..model_module.attention_masks import build_task_mask
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
    """通过对 ``v_theta`` 做 Euler 积分采样 target slot。

    Args:
        model: 训练好的 :class:`UniCardioRF`（或具备相同接口的模型）。
        conditions: ``(B, 3, L)`` —— condition slot 填入干净信号；target slot
            的内容会被忽略（由 ``x_t`` 覆盖）。
        task: :class:`TaskSpec`，描述要采样的 slot。
        n_steps: 在 ``t ∈ [1, 0]`` 上积分的 Euler 步数。
        device: 目标设备；默认为 ``conditions.device``。
        return_trajectory: 为 True 时，额外返回形状为
            ``(n_steps + 1, B, 1, L)`` 的完整轨迹，便于诊断。

    Returns:
        ``x_0 ≈ (B, 1, L)``，或者 ``(x_0, trajectory)``。
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
    # mask 只与 (task.name, L, device, dtype) 有关，n_steps 个步骤可共用一份。
    mask = build_task_mask(
        task.name, L, device=str(device), dtype=torch.bool
    )

    x = torch.randn(B, 1, L, device=device)  # x_1 ~ N(0, I)
    ts = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
    traj: list[Tensor] = [x.clone()] if return_trajectory else []

    for i in range(n_steps):
        t_cur, t_next = ts[i], ts[i + 1]
        dt = t_next - t_cur  # 为负值
        t_b = torch.full((B,), float(t_cur), device=device)
        x_full = assemble_x_full(conditions, x, target_slot=target, L=L)
        v = model(x_full, t_b, mask, target)
        x = x + v * dt
        if return_trajectory:
            traj.append(x.clone())

    if model_was_training:
        model.train()

    if return_trajectory:
        return x, torch.stack(traj, dim=0)
    return x

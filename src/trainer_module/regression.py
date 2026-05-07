"""Deterministic regression baseline：与 RF 共享 backbone，但只做 1-shot MSE 回归。

在条件信号→目标信号近乎 deterministic 的场景下（PPG → ABP），RF/Diffusion 框架
的随机采样会引入 MC 方差并把指标天花板托高。本模块提供与 RF 等价签名的训练 step
和"采样器"，作为 ``trainer.objective`` 的另一条路径。

约定（保持 backbone 接口不变，零模型改动）：

* ``t`` 固定为 ``0.0``（一个常量），让 time-MLP 输出一个稳定的"regression mode"
  token；不再对每条样本采新的 ``t``。
* target slot 在 ``x_full`` 中填 0，与 RF 训练里 ``t=0`` 端的 ``x_t = ε``
  统计性质不同（这里是确定的零向量），但训练 / 评估端口完全一致——网络
  在这一种 (target_zeros, t=0) 配置下学回归，评估也跑同一种配置。
* loss = ``MSE(pred_x1, x1)``。``pred_x1`` 直接取 backbone 输出。

调用方约定与 RF 完全一致：``regression_train_step(model, batch, task)`` 返回
标量 loss；``regression_sample(model, conditions, task)`` 返回 ``(B, 1, L)``。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..model_module.attention_masks import build_task_mask
from ..model_module.tasks import TaskSpec
from ..utils import unwrap_model
from .rectified_flow import assemble_x_full


def _build_inputs(
    batch_signal: Tensor, task: TaskSpec
) -> tuple[Tensor, Tensor, Tensor]:
    """构造 ``(x_full, t_zero, x1)``。

    ``x_full`` 把 target slot 替换成全零张量，``t_zero`` 是一个全 0 的 ``(B,)``。
    """
    if batch_signal.dim() != 3 or batch_signal.size(1) != 3:
        raise ValueError(
            f"batch_signal must be (B, 3, L); got {tuple(batch_signal.shape)}"
        )
    B, _, L = batch_signal.shape
    target = int(task.target_slot)
    x1 = batch_signal[:, target : target + 1, :]
    x_target_zero = torch.zeros_like(x1)
    x_full = assemble_x_full(batch_signal, x_target_zero, target_slot=target, L=L)
    t_zero = torch.zeros(B, device=batch_signal.device, dtype=batch_signal.dtype)
    return x_full, t_zero, x1


def regression_train_step(
    model: nn.Module,
    batch_signal: Tensor,
    task: TaskSpec,
) -> Tensor:
    """单个 batch 的回归 MSE loss。

    与 :func:`rf_train_step` 同签名，便于 trainer 按 ``objective`` 切换。
    """
    x_full, t_zero, x1 = _build_inputs(batch_signal, task)
    n_patches = unwrap_model(model).n_patches_per_slot
    mask = build_task_mask(
        task.name, n_patches, device=str(x_full.device), dtype=torch.bool
    )
    pred_x1 = model(x_full, t_zero, mask, int(task.target_slot))
    if pred_x1.shape != x1.shape:
        raise RuntimeError(
            f"Regression prediction shape {tuple(pred_x1.shape)} "
            f"!= target {tuple(x1.shape)}"
        )
    return (pred_x1 - x1).pow(2).mean()


@torch.no_grad()
def regression_sample(
    model: nn.Module,
    conditions: Tensor,
    task: TaskSpec,
    *,
    device: str | torch.device | None = None,
) -> Tensor:
    """单 forward 拿 ``pred_x1``，与 :func:`euler_sample` 同签名。

    Args:
        model: 训练好的 :class:`UniCardioRF`。
        conditions: ``(B, 3, L)`` —— condition slot 是干净信号；target slot 内容
            会被忽略（由全零张量覆盖）。
        task: :class:`TaskSpec`。
        device: 默认 ``conditions.device``。

    Returns:
        ``(B, 1, L)`` —— target slot 的回归预测。
    """
    if conditions.dim() != 3 or conditions.size(1) != 3:
        raise ValueError(
            f"conditions must be (B, 3, L); got {tuple(conditions.shape)}"
        )

    device = torch.device(device) if device is not None else conditions.device
    model_was_training = model.training
    model.eval()

    conditions = conditions.to(device)
    x_full, t_zero, _ = _build_inputs(conditions, task)
    n_patches = unwrap_model(model).n_patches_per_slot
    mask = build_task_mask(
        task.name, n_patches, device=str(device), dtype=torch.bool
    )
    pred_x1 = model(x_full, t_zero, mask, int(task.target_slot))

    if model_was_training:
        model.train()
    return pred_x1

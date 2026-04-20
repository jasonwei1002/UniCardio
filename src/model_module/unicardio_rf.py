"""包裹 :class:`UniCardioBackbone` 的顶层 Rectified Flow 模型。

设计决定：模型内部**不**计算 loss。``UniCardioRF.forward(x_full, t, task)``
只返回 target slot 的原始速度预测 ``(B, 1, L)``；loss、x_t 的构造、采样
都放在 ``src.trainer_module`` 中，使本模块更易测试、checkpoint 以及在
将来替换不同 backbone。
"""

from __future__ import annotations

from typing import Any, Mapping

import torch.nn as nn
from torch import Tensor

from .attention_masks import build_task_mask
from .backbone import BackboneConfig, UniCardioBackbone
from .tasks import TaskSpec


class UniCardioRF(nn.Module):
    """包裹 :class:`UniCardioBackbone` 的 Rectified Flow 模型。

    该 wrapper 会按需构造任务专属的注意力 mask（由 :func:`build_task_mask`
    缓存），并以正确的 target slot 调用 backbone。
    """

    def __init__(
        self, config: Mapping[str, Any] | BackboneConfig
    ) -> None:
        super().__init__()
        self.backbone = UniCardioBackbone(config)
        self.L = self.backbone.L

    def forward(
        self,
        x_full: Tensor,
        t: Tensor,
        task: TaskSpec,
    ) -> Tensor:
        """预测 ``task.target_slot`` 的速度。

        Args:
            x_full: ``(B, 1, 3 * L_slot)`` —— condition slot + 填充了 ``x_t``
                的 target slot 拼接后的张量。
            t: ``(B,)`` —— 取值于 ``[0, 1]`` 的连续 flow 时间。
            task: :class:`TaskSpec`，指定哪一个 slot 是 target。

        Returns:
            target slot 的速度预测 ``(B, 1, L_slot)``。
        """
        mask = build_task_mask(
            task.name,
            self.L,
            device=str(x_full.device),
        )
        return self.backbone(
            x_full, t, mask, target_slot=int(task.target_slot)
        )

"""Top-level Rectified Flow model wrapping :class:`UniCardioBackbone`.

``forward(x_full, t, mask, target_slot)`` returns the raw velocity prediction
for ``target_slot`` only; loss, ``x_t`` construction, sampling, and the
``task → mask/target_slot`` mapping all live in ``src.trainer_module``.

``forward`` takes a Python ``int`` rather than a ``TaskSpec`` so
``torch.compile`` specializes on the 3 ``target_slot`` values, not 15
(slot × task name) Dynamo frames — which would otherwise hit
``recompile_limit=8`` and fall back to eager.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch.nn as nn
from torch import Tensor

from .attention_masks import AttentionMask
from .backbone import BackboneConfig, UniCardioBackbone


class UniCardioRF(nn.Module):
    """包裹 :class:`UniCardioBackbone` 的 Rectified Flow 模型。

    forward 签名是纯张量 + mask + ``int``，调用方负责从 :class:`TaskSpec`
    里取 ``target_slot`` 并按 device 选择 mask（CUDA 走
    :func:`build_task_block_mask`，CPU 走 :func:`build_task_mask` bool）。
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
        mask: AttentionMask,
        target_slot: int,
    ) -> Tensor:
        """预测 ``target_slot`` 对应的速度。

        Args:
            x_full: ``(B, 1, 3 * L_slot)`` —— condition slot + 填充了 ``x_t``
                的 target slot 拼接后的张量。
            t: ``(B,)`` —— 取值于 ``[0, 1]`` 的连续 flow 时间。
            mask: ``BlockMask``（CUDA / flex_attention 路径）或 dense bool
                ``Tensor``（CPU / SDPA fallback）。
            target_slot: 参与重建的 slot 编号（0/1/2）。

        Returns:
            target slot 的速度预测 ``(B, 1, L_slot)``。
        """
        return self.backbone(x_full, t, mask, target_slot=target_slot)

    def freeze_for_finetune(self, n_unfrozen_blocks: int) -> None:
        """阶段二微调：冻结 backbone，仅解冻最后 N 个 ResidualBlock + 全部 output_heads。

        - ``n_unfrozen_blocks=0``：仅解冻 ``output_heads``。
        - ``n_unfrozen_blocks >= len(residual_layers)``：解冻所有 ResidualBlock
          + ``output_heads``，等价于只冻结 stem (encoders / norms + time embedding)。
        """
        for p in self.parameters():
            p.requires_grad = False
        bb = self.backbone
        n = max(0, min(int(n_unfrozen_blocks), len(bb.residual_layers)))
        if n > 0:
            for blk in bb.residual_layers[-n:]:
                for p in blk.parameters():
                    p.requires_grad = True
        for head in bb.output_heads:
            for p in head.parameters():
                p.requires_grad = True

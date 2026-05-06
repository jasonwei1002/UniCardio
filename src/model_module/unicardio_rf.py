"""包裹 :class:`UniCardioBackbone` 的顶层 Rectified Flow 模型。

设计决定：模型内部**不**计算 loss。``UniCardioRF.forward(x_full, t, mask,
target_slot)`` 只返回 target slot 的原始速度预测 ``(B, 1, L)``；loss、
x_t 的构造、采样、**以及任务 → mask/target_slot 的映射**都放在
``src.trainer_module`` 中，使本模块更易测试、checkpoint 以及在将来替换
不同 backbone。

forward 只接收张量和一个 Python ``int``，不接受 ``TaskSpec``。这样
``torch.compile(model)`` 只会对 ``target_slot`` 的 3 个值做特化（Dynamo
guard），而不会对 ``task.name`` 的 5 种字符串分别编一份图。历史上用
``task`` 作为参数时，Dynamo 会把 5 × 3 = 15 种组合当成不同 frame，撞
``recompile_limit=8`` 之后直接回退 eager。

注意：``torch.compile`` 由训练入口 (``run/pipeline/train.py``) 在构造
完成后按 ``cfg.trainer.compile`` 包一层。这样 ``UniCardioRF.state_dict()``
的 key 前缀不会被 ``OptimizedModule`` 污染成 ``_orig_mod.*``（checkpoint
代码里还有兜底的解包逻辑）。
"""

from __future__ import annotations

from typing import Any, Mapping

import torch.nn as nn
from torch import Tensor

from .backbone import BackboneConfig, UniCardioBackbone


class UniCardioRF(nn.Module):
    """包裹 :class:`UniCardioBackbone` 的 Rectified Flow 模型。

    forward 签名是纯张量 + ``int``，调用方负责从 :class:`TaskSpec` 里
    取 ``target_slot`` 并通过 :func:`build_task_mask` 构造 ``mask``。
    """

    def __init__(
        self, config: Mapping[str, Any] | BackboneConfig
    ) -> None:
        super().__init__()
        self.backbone = UniCardioBackbone(config)
        self.L = self.backbone.L
        # patch-tokenization 后 mask 长度 = patch 数；trainer / sampler 通过
        # ``model.n_patches_per_slot`` 直读，避免穿到 backbone.cfg。
        self.n_patches_per_slot = self.backbone.n_patches_per_slot

    def forward(
        self,
        x_full: Tensor,
        t: Tensor,
        mask: Tensor,
        target_slot: int,
    ) -> Tensor:
        """预测 ``target_slot`` 对应的速度。

        Args:
            x_full: ``(B, 1, 3 * L_slot)`` —— condition slot + 填充了 ``x_t``
                的 target slot 拼接后的张量。
            t: ``(B,)`` —— 取值于 ``[0, 1]`` 的连续 flow 时间。
            mask: ``(3 * L_slot, 3 * L_slot)`` 的 bool 注意力 mask；由
                :func:`src.model_module.attention_masks.build_task_mask`
                以 ``dtype=torch.bool`` 生成。
            target_slot: 参与重建的 slot 编号（0/1/2）。

        Returns:
            target slot 的速度预测 ``(B, 1, L_slot)``。
        """
        return self.backbone(x_full, t, mask, target_slot=target_slot)

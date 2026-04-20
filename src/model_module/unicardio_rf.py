"""Top-level Rectified Flow model wrapping :class:`UniCardioBackbone`.

Design choice: the loss is intentionally *not* computed inside the model.
``UniCardioRF.forward(x_full, t, task)`` returns the raw velocity prediction
``(B, 1, L)`` for the target slot; loss, x_t construction, and sampling live in
``src.trainer_module`` so this module stays easy to test, checkpoint, and
swap for a different backbone later.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch.nn as nn
from torch import Tensor

from .attention_masks import build_task_mask
from .backbone import BackboneConfig, UniCardioBackbone
from .tasks import TaskSpec


class UniCardioRF(nn.Module):
    """Rectified-Flow wrapper around :class:`UniCardioBackbone`.

    The wrapper builds the task-specific attention mask lazily (cached by
    :func:`build_task_mask`) and forwards to the backbone with the correct
    target slot.
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
        """Predict velocity for ``task.target_slot``.

        Args:
            x_full: ``(B, 1, 3 * L_slot)`` — condition slots plus the target
                slot populated with ``x_t``.
            t: ``(B,)`` — continuous flow time in ``[0, 1]``.
            task: :class:`TaskSpec` identifying which slot is the target.

        Returns:
            Velocity prediction ``(B, 1, L_slot)`` for the target slot only.
        """
        mask = build_task_mask(
            task.name,
            self.L,
            device=str(x_full.device),
        )
        return self.backbone(
            x_full, t, mask, target_slot=int(task.target_slot)
        )

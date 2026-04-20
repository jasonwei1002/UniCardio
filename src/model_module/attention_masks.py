"""Per-task additive attention masks.

Shape: ``(3 * L_slot, 3 * L_slot)``. Slot token ranges with model slot order:

    ECG = [0, L)
    PPG = [L, 2L)
    ABP = [2L, 3L)

Semantics (additive, PyTorch nn.MultiheadAttention-compatible):

    0.0   → attention allowed
    -inf  → attention blocked

Rule per task:

* Every participating slot attends to itself (self-attention block on diagonal).
* The target slot additionally attends to all condition slots.
* Condition slots cross-attend to each other (relevant only for ecgppg2abp).
* Non-participating slot rows are fully blocked.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import product

import torch
from torch import Tensor

from .tasks import TASK_SPECS


def _block(mask: Tensor, q_slot: int, k_slot: int, L: int) -> None:
    """Set the ``(q_slot, k_slot)`` L x L sub-block of ``mask`` to 0 (allow)."""
    qs, qe = q_slot * L, (q_slot + 1) * L
    ks, ke = k_slot * L, (k_slot + 1) * L
    mask[qs:qe, ks:ke] = 0.0


@lru_cache(maxsize=32)
def build_task_mask(
    task_name: str, L_slot: int, device: str = "cpu"
) -> Tensor:
    """Build the additive attention mask for the given task.

    The mask is cached per ``(task_name, L_slot, device)`` tuple; in DDP each
    process holds its own cache.

    Returns a tensor of shape ``(3 * L_slot, 3 * L_slot)`` containing ``0.0``
    on allowed cells and ``-inf`` on blocked cells.
    """
    spec = TASK_SPECS[task_name]
    total = 3 * L_slot
    mask = torch.full((total, total), float("-inf"))
    participants = set(spec.cond_slots) | {spec.target_slot}

    # Self-attention within each participating slot.
    for s in participants:
        _block(mask, int(s), int(s), L_slot)

    # Condition-to-condition cross attention (matters only for multi-condition).
    for s1, s2 in product(spec.cond_slots, repeat=2):
        if s1 != s2:
            _block(mask, int(s1), int(s2), L_slot)

    # Target reads from all conditions.
    for c in spec.cond_slots:
        _block(mask, int(spec.target_slot), int(c), L_slot)

    return mask.to(device)


def clear_mask_cache() -> None:
    """Flush the lru_cache; useful in tests after device switches."""
    build_task_mask.cache_clear()

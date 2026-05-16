"""Per-task attention masks of shape ``(3 * L_slot, 3 * L_slot)``.

Production path uses :func:`build_task_block_mask` → ``BlockMask`` for
``flex_attention`` (sparse 128-block kernel). :func:`build_task_mask` (dense
bool / float32) is kept as a reference for unit tests. Rules: each
participating slot self-attends; target reads all conds; conds cross-attend
each other; non-participating slot rows are fully blocked.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import product
from typing import Any

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from .tasks import TASK_SPECS, TaskSpec

AttentionMask = BlockMask | Tensor


def _block(mask: Tensor, q_slot: int, k_slot: int, L: int, fill: Any) -> None:
    """将 ``mask`` 中 ``(q_slot, k_slot)`` 对应的 L x L 子块置为 ``fill``（允许注意力）。"""
    qs, qe = q_slot * L, (q_slot + 1) * L
    ks, ke = k_slot * L, (k_slot + 1) * L
    mask[qs:qe, ks:ke] = fill


def _allowed_slot_pairs(spec: TaskSpec) -> set[tuple[int, int]]:
    """返回 (q_slot, k_slot) 允许 attention 的 slot 对集合。"""
    pairs: set[tuple[int, int]] = set()
    participants = set(spec.cond_slots) | {spec.target_slot}
    for s in participants:
        pairs.add((int(s), int(s)))
    for c1, c2 in product(spec.cond_slots, repeat=2):
        pairs.add((int(c1), int(c2)))
    for c in spec.cond_slots:
        pairs.add((int(spec.target_slot), int(c)))
    return pairs


@lru_cache(maxsize=32)
def build_task_mask(
    task_name: str,
    L_slot: int,
    device: str = "cpu",
    *,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Build a dense ``(3 * L_slot, 3 * L_slot)`` attention mask for ``task_name``.

    Kept for tests and tooling. Production attention uses
    :func:`build_task_block_mask`.

    Cached by ``(task_name, L_slot, device, dtype)``; DDP-safe per process.
    """
    spec = TASK_SPECS[task_name]
    total = 3 * L_slot
    if dtype == torch.bool:
        mask = torch.zeros((total, total), dtype=torch.bool)
        allowed_fill: Any = True
    elif dtype == torch.float32:
        mask = torch.full((total, total), float("-inf"))
        allowed_fill = 0.0
    else:
        raise ValueError(
            f"Unsupported mask dtype {dtype}; use torch.float32 or torch.bool."
        )
    for (qs, ks) in _allowed_slot_pairs(spec):
        _block(mask, qs, ks, L_slot, allowed_fill)
    return mask.to(device)


def _make_mask_mod(spec: TaskSpec, L_slot: int, device: str = "cpu"):
    """Build a ``mask_mod`` closure for ``create_block_mask``.

    Captures a small ``(3, 3)`` bool lookup table for the task's allowed
    slot pairs; returns a function suitable for ``flex_attention``'s
    ``(b, h, q_idx, kv_idx) -> bool`` contract.
    """
    allowed_table = torch.zeros(3, 3, dtype=torch.bool, device=device)
    for qs, ks in _allowed_slot_pairs(spec):
        allowed_table[qs, ks] = True

    def mask_mod(b, h, q_idx, kv_idx):
        return allowed_table[q_idx // L_slot, kv_idx // L_slot]

    return mask_mod


@lru_cache(maxsize=32)
def build_task_block_mask(
    task_name: str,
    L_slot: int,
    device: str = "cpu",
    *,
    BLOCK_SIZE: int = 128,
) -> BlockMask:
    """Build a sparse ``BlockMask`` for ``flex_attention``.

    Cached by ``(task_name, L_slot, device, BLOCK_SIZE)``. Use this — not
    :func:`build_task_mask` — at training and inference sites.
    """
    spec = TASK_SPECS[task_name]
    mask_mod = _make_mask_mod(spec, L_slot, device=device)
    total = 3 * L_slot
    return create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=total,
        KV_LEN=total,
        device=device,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def select_task_mask(
    task_name: str, L_slot: int, device: torch.device
) -> AttentionMask:
    """Pick the right mask type for ``device``.

    FlexAttention has no CPU backward as of torch 2.11; fall back to dense
    bool SDPA on non-CUDA devices.
    """
    if device.type == "cuda":
        return build_task_block_mask(task_name, L_slot, device=str(device))
    return build_task_mask(
        task_name, L_slot, device=str(device), dtype=torch.bool
    )


def clear_mask_cache() -> None:
    """清空 mask lru_cache；在测试切换设备后会用到。"""
    build_task_mask.cache_clear()
    build_task_block_mask.cache_clear()

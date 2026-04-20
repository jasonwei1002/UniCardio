"""按任务生成的加性注意力 mask。

形状：``(3 * L_slot, 3 * L_slot)``。按模型 slot 顺序划分 token 段：

    ECG = [0, L)
    PPG = [L, 2L)
    ABP = [2L, 3L)

语义（与 PyTorch nn.MultiheadAttention 的加性 mask 兼容）：

    0.0   → 允许注意力
    -inf  → 屏蔽注意力

每个任务的规则：

* 每个参与的 slot 都对自身做 self-attention（对角块）。
* target slot 额外可以读取所有 condition slot。
* condition slot 之间互相 cross-attention（仅在 ecgppg2abp 中实际生效）。
* 未参与的 slot 整行全部屏蔽。
"""

from __future__ import annotations

from functools import lru_cache
from itertools import product

import torch
from torch import Tensor

from .tasks import TASK_SPECS


def _block(mask: Tensor, q_slot: int, k_slot: int, L: int) -> None:
    """将 ``mask`` 中 ``(q_slot, k_slot)`` 对应的 L x L 子块置 0（允许注意力）。"""
    qs, qe = q_slot * L, (q_slot + 1) * L
    ks, ke = k_slot * L, (k_slot + 1) * L
    mask[qs:qe, ks:ke] = 0.0


@lru_cache(maxsize=32)
def build_task_mask(
    task_name: str, L_slot: int, device: str = "cpu"
) -> Tensor:
    """为指定任务构建加性注意力 mask。

    mask 按 ``(task_name, L_slot, device)`` 三元组缓存；在 DDP 场景下每个
    进程持有独立的缓存副本。

    返回形状为 ``(3 * L_slot, 3 * L_slot)`` 的张量，允许位置为 ``0.0``，
    屏蔽位置为 ``-inf``。
    """
    spec = TASK_SPECS[task_name]
    total = 3 * L_slot
    mask = torch.full((total, total), float("-inf"))
    participants = set(spec.cond_slots) | {spec.target_slot}

    # 每个参与 slot 内部做 self-attention。
    for s in participants:
        _block(mask, int(s), int(s), L_slot)

    # condition 之间的 cross-attention（仅在多条件任务中实际生效）。
    for s1, s2 in product(spec.cond_slots, repeat=2):
        if s1 != s2:
            _block(mask, int(s1), int(s2), L_slot)

    # target 读取所有 condition。
    for c in spec.cond_slots:
        _block(mask, int(spec.target_slot), int(c), L_slot)

    return mask.to(device)


def clear_mask_cache() -> None:
    """清空 lru_cache；在测试切换设备后会用到。"""
    build_task_mask.cache_clear()

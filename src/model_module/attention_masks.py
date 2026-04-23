"""按任务生成的注意力 mask。

形状：``(3 * L_slot, 3 * L_slot)``。按模型 slot 顺序划分 token 段：

    ECG = [0, L)
    PPG = [L, 2L)
    ABP = [2L, 3L)

支持两种 dtype（由 ``dtype`` kwarg 选择）：

* ``torch.float32``（默认，兼容 ``nn.MultiheadAttention`` 的加性 mask）：
  ``0.0`` 允许注意力，``-inf`` 屏蔽注意力。
* ``torch.bool``（供 ``F.scaled_dot_product_attention`` 使用，
  在 Hopper 上可路由到 Flash Attention）：``True`` 允许注意力，
  ``False`` 屏蔽注意力。

每个任务的规则：

* 每个参与的 slot 都对自身做 self-attention（对角块）。
* target slot 额外可以读取所有 condition slot。
* condition slot 之间互相 cross-attention（仅在 ecgppg2abp 中实际生效）。
* 未参与的 slot 整行全部屏蔽。
"""

from __future__ import annotations

from functools import lru_cache
from itertools import product
from typing import Any

import torch
from torch import Tensor

from .tasks import TASK_SPECS


def _block(mask: Tensor, q_slot: int, k_slot: int, L: int, fill: Any) -> None:
    """将 ``mask`` 中 ``(q_slot, k_slot)`` 对应的 L x L 子块置为 ``fill``（允许注意力）。"""
    qs, qe = q_slot * L, (q_slot + 1) * L
    ks, ke = k_slot * L, (k_slot + 1) * L
    mask[qs:qe, ks:ke] = fill


@lru_cache(maxsize=32)
def build_task_mask(
    task_name: str,
    L_slot: int,
    device: str = "cpu",
    *,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """为指定任务构建注意力 mask。

    mask 按 ``(task_name, L_slot, device, dtype)`` 缓存；在 DDP 场景下每个
    进程持有独立的缓存副本。``dtype`` 作为 kwarg 也会参与 ``lru_cache``
    键计算，float 与 bool 版本互不污染。

    Args:
        task_name: 任务名，必须在 :data:`TASK_SPECS` 里。
        L_slot: 单 slot 长度（token 数）。
        device: 目标设备字符串。
        dtype: ``torch.float32``（加性，0/-inf）或 ``torch.bool``（True/False）。

    Returns:
        形状为 ``(3 * L_slot, 3 * L_slot)`` 的张量。
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
    participants = set(spec.cond_slots) | {spec.target_slot}

    # 每个参与 slot 内部做 self-attention。
    for s in participants:
        _block(mask, int(s), int(s), L_slot, allowed_fill)

    # condition 之间的 cross-attention（仅在多条件任务中实际生效）。
    for s1, s2 in product(spec.cond_slots, repeat=2):
        if s1 != s2:
            _block(mask, int(s1), int(s2), L_slot, allowed_fill)

    # target 读取所有 condition。
    for c in spec.cond_slots:
        _block(mask, int(spec.target_slot), int(c), L_slot, allowed_fill)

    return mask.to(device)


def clear_mask_cache() -> None:
    """清空 lru_cache；在测试切换设备后会用到。"""
    build_task_mask.cache_clear()

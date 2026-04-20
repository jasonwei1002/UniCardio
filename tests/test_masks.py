""":func:`build_task_mask` 的单元测试。

5 个任务中每个任务都有一组手算得到的允许 ``(query_slot, key_slot)`` 块。
测试验证：

1. 允许位置的值恰好为 0（加性 mask 语义）。
2. 非允许位置全部为 ``-inf``。
3. 取值为 0 的 cell 数量等于 ``(允许块数) * L * L``。
"""

from __future__ import annotations

import math

import pytest
import torch

from src.model_module.attention_masks import build_task_mask, clear_mask_cache
from src.model_module.tasks import TASK_LIST

L_SLOT = 16  # 取小一点，可以逐 cell 检查且速度快

# 每个任务手工推导出的允许 (query, key) 块集合。
# 规则回顾：target↔target、cond→cond 自注意力、cond↔cond 成对、target←cond。
EXPECTED_ALLOWED: dict[str, set[tuple[int, int]]] = {
    # ecg2ppg  —— cond={0}, target=1
    "ecg2ppg": {(0, 0), (1, 1), (1, 0)},
    # ppg2ecg  —— cond={1}, target=0
    "ppg2ecg": {(0, 0), (1, 1), (0, 1)},
    # ecg2abp  —— cond={0}, target=2
    "ecg2abp": {(0, 0), (2, 2), (2, 0)},
    # ppg2abp  —— cond={1}, target=2
    "ppg2abp": {(1, 1), (2, 2), (2, 1)},
    # ecgppg2abp —— cond={0,1}, target=2 → cond↔cond + target←两个 cond + 自注意力
    "ecgppg2abp": {
        (0, 0), (1, 1), (2, 2),
        (0, 1), (1, 0),
        (2, 0), (2, 1),
    },
}


@pytest.fixture(autouse=True)
def _reset_cache():
    clear_mask_cache()
    yield
    clear_mask_cache()


@pytest.mark.parametrize("task", TASK_LIST, ids=lambda t: t.name)
def test_mask_allowed_blocks_are_zero(task):
    mask = build_task_mask(task.name, L_SLOT, device="cpu")
    allowed = EXPECTED_ALLOWED[task.name]
    for q in range(3):
        for k in range(3):
            block = mask[q * L_SLOT:(q + 1) * L_SLOT, k * L_SLOT:(k + 1) * L_SLOT]
            if (q, k) in allowed:
                assert torch.all(block == 0.0), (
                    f"Task {task.name} block ({q},{k}) should be all-zero; "
                    f"got min={block.min().item()}, max={block.max().item()}"
                )
            else:
                assert torch.all(torch.isneginf(block)), (
                    f"Task {task.name} block ({q},{k}) should be all -inf"
                )


@pytest.mark.parametrize("task", TASK_LIST, ids=lambda t: t.name)
def test_mask_zero_cell_count(task):
    mask = build_task_mask(task.name, L_SLOT, device="cpu")
    expected_zeros = len(EXPECTED_ALLOWED[task.name]) * L_SLOT * L_SLOT
    actual_zeros = int((mask == 0.0).sum().item())
    assert actual_zeros == expected_zeros, (
        f"Task {task.name}: expected {expected_zeros} zero cells, got {actual_zeros}"
    )


def test_mask_shape_and_dtype():
    mask = build_task_mask("ecg2ppg", L_SLOT)
    assert mask.shape == (3 * L_SLOT, 3 * L_SLOT)
    assert mask.dtype == torch.float32
    assert torch.isneginf(mask).any() and (mask == 0.0).any()


def test_mask_cache_hits():
    clear_mask_cache()
    m1 = build_task_mask("ppg2abp", L_SLOT)
    m2 = build_task_mask("ppg2abp", L_SLOT)
    # lru_cache 命中时会返回同一个对象。
    assert m1 is m2

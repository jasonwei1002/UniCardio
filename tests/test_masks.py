"""Unit tests for :func:`build_task_mask`.

Each of the 5 tasks has a hand-computed expected set of allowed ``(query_slot,
key_slot)`` blocks. The tests verify that:

1. Allowed blocks are exactly zero (additive mask semantics).
2. Non-allowed blocks are entirely ``-inf``.
3. The zero-cell count matches ``(# allowed blocks) * L * L``.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.model_module.attention_masks import build_task_mask, clear_mask_cache
from src.model_module.tasks import TASK_LIST

L_SLOT = 16  # small, fast to check cell-by-cell

# Hand-derived allowed (query, key) slot pairs per task.
# Rule recap: target↔target, cond→cond (self), cond↔cond (pairs), target←cond.
EXPECTED_ALLOWED: dict[str, set[tuple[int, int]]] = {
    # ecg2ppg  — cond={0}, target=1
    "ecg2ppg": {(0, 0), (1, 1), (1, 0)},
    # ppg2ecg  — cond={1}, target=0
    "ppg2ecg": {(0, 0), (1, 1), (0, 1)},
    # ecg2abp  — cond={0}, target=2
    "ecg2abp": {(0, 0), (2, 2), (2, 0)},
    # ppg2abp  — cond={1}, target=2
    "ppg2abp": {(1, 1), (2, 2), (2, 1)},
    # ecgppg2abp — cond={0,1}, target=2 → cond↔cond + target←both + self
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
    # lru_cache returns same object on repeat hit.
    assert m1 is m2

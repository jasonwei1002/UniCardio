"""Task specification for 5-task multimodal reconstruction.

Slot convention (model order, distinct from the on-disk file order PPG/BP/ECG):

    ECG = 0
    PPG = 1
    ABP = 2

The data loader applies a one-time channel permutation ``[2, 0, 1]`` so all
downstream code (model, masks, loss, sampler, metrics) speaks exclusively in
model slot indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Slot(IntEnum):
    """Model-space slot indices for the three cardiovascular modalities."""

    ECG = 0
    PPG = 1
    ABP = 2


@dataclass(frozen=True)
class TaskSpec:
    """Immutable description of one reconstruction direction.

    Attributes:
        name: Short string ID (used for logging, mask cache key).
        cond_slots: Slots that provide clean conditioning signals.
        target_slot: Slot the model generates.
        task_id: Integer index in ``TASK_LIST`` (for optional task embedding).
    """

    name: str
    cond_slots: tuple[Slot, ...]
    target_slot: Slot
    task_id: int


TASK_SPECS: dict[str, TaskSpec] = {
    "ecg2ppg": TaskSpec(
        name="ecg2ppg",
        cond_slots=(Slot.ECG,),
        target_slot=Slot.PPG,
        task_id=0,
    ),
    "ppg2ecg": TaskSpec(
        name="ppg2ecg",
        cond_slots=(Slot.PPG,),
        target_slot=Slot.ECG,
        task_id=1,
    ),
    "ecg2abp": TaskSpec(
        name="ecg2abp",
        cond_slots=(Slot.ECG,),
        target_slot=Slot.ABP,
        task_id=2,
    ),
    "ppg2abp": TaskSpec(
        name="ppg2abp",
        cond_slots=(Slot.PPG,),
        target_slot=Slot.ABP,
        task_id=3,
    ),
    "ecgppg2abp": TaskSpec(
        name="ecgppg2abp",
        cond_slots=(Slot.ECG, Slot.PPG),
        target_slot=Slot.ABP,
        task_id=4,
    ),
}

TASK_LIST: list[TaskSpec] = list(TASK_SPECS.values())


def get_task(name: str) -> TaskSpec:
    """Lookup helper that raises a clear error for unknown task IDs."""
    if name not in TASK_SPECS:
        raise KeyError(
            f"Unknown task '{name}'. Known tasks: {sorted(TASK_SPECS.keys())}"
        )
    return TASK_SPECS[name]

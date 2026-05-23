"""5 个多模态重建任务的任务规范。

Slot 约定（模型顺序，区别于磁盘文件中的 PPG/BP/ECG 顺序）：

    ECG = 0
    PPG = 1
    ABP = 2

数据加载器会执行一次 ``[2, 0, 1]`` 的通道置换，因此下游所有代码（模型、
mask、损失、采样器、指标）都只使用模型 slot 索引。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Mapping


class Slot(IntEnum):
    """三种心血管模态在模型空间中的 slot 索引。"""

    ECG = 0
    PPG = 1
    ABP = 2


@dataclass(frozen=True)
class TaskSpec:
    """单个重建方向的不可变描述。

    Attributes:
        name: 字符串 ID（用于日志和 mask 缓存键）。
        cond_slots: 提供干净条件信号的 slot。
        target_slot: 模型需要生成的 slot。
        task_id: 在 ``TASK_LIST`` 中的整数索引（可用于可选的 task embedding）。
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
    """按名称查询任务，未知任务 ID 时抛出可读错误。"""
    if name not in TASK_SPECS:
        raise KeyError(
            f"Unknown task '{name}'. Known tasks: {sorted(TASK_SPECS.keys())}"
        )
    return TASK_SPECS[name]


def active_task_pairs(
    weights_cfg: Mapping[str, float] | None,
) -> list[tuple[TaskSpec, float]]:
    """Return ``(task, weight)`` pairs with weight > 0.

    Weight 0 disables the task; NaN/Inf/negative raises.
    """
    weights_cfg = weights_cfg or {}
    pairs: list[tuple[TaskSpec, float]] = []
    for spec in TASK_LIST:
        w = float(weights_cfg.get(spec.name, 1.0))
        if math.isnan(w) or math.isinf(w) or w < 0:
            raise ValueError(f"Invalid task weight for {spec.name}: {w}")
        if w > 0:
            pairs.append((spec, w))
    if not pairs:
        raise ValueError("All task weights are zero.")
    return pairs


# Slot → BP-head vital name. Only ECG/PPG are valid BP-head inputs; ABP is the
# target waveform and is never fed to the scalar head.
_SLOT_TO_VITAL: dict[Slot, str] = {Slot.ECG: "ecg", Slot.PPG: "ppg"}


def cond_slots_to_vitals(task: TaskSpec) -> list[str]:
    """Map a task's condition slots to BP-head vital names (ECG/PPG only).

    Used by the evaluation cascade so the BP head averages only over the
    modalities the task actually conditions on (e.g. ``ppg2abp`` -> ``["ppg"]``,
    no ECG leakage). Order follows ``task.cond_slots``.
    """
    return [_SLOT_TO_VITAL[s] for s in task.cond_slots if s in _SLOT_TO_VITAL]

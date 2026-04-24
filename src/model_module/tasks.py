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
    """返回权重 > 0 的 ``(task, weight)`` 对，用于训练 / 验证 / 评估。

    权重为 0 视作"显式禁用该任务"；NaN / Inf / 负数直接 raise。训练采样、
    val、evaluate 都以这里的返回值作为 active tasks，不再各自过滤。
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

"""Unit tests for task → vital-name mapping used by the BP-head cascade."""

from __future__ import annotations

from src.model_module.tasks import TASK_SPECS, cond_slots_to_vitals


def test_cond_slots_to_vitals_single_ppg() -> None:
    assert cond_slots_to_vitals(TASK_SPECS["ppg2abp"]) == ["ppg"]


def test_cond_slots_to_vitals_single_ecg() -> None:
    assert cond_slots_to_vitals(TASK_SPECS["ecg2abp"]) == ["ecg"]


def test_cond_slots_to_vitals_multimodal() -> None:
    assert cond_slots_to_vitals(TASK_SPECS["ecgppg2abp"]) == ["ecg", "ppg"]


def test_cond_slots_to_vitals_skips_non_ecg_ppg() -> None:
    # ecg2ppg conditions on ECG only; PPG/ABP targets never appear as vitals.
    assert cond_slots_to_vitals(TASK_SPECS["ecg2ppg"]) == ["ecg"]


def test_cond_slots_to_vitals_filters_abp() -> None:
    """A synthetic task with ABP in cond_slots must drop ABP and keep ECG/PPG order."""
    from src.model_module.tasks import Slot, TaskSpec

    spec = TaskSpec(
        name="abp_probe",
        cond_slots=(Slot.ABP, Slot.PPG),
        target_slot=Slot.ECG,
        task_id=99,
    )
    assert cond_slots_to_vitals(spec) == ["ppg"]

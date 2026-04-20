"""Convert MIMIC-BP down_stream data to (ECG, PPG, ABP)-ordered .npy + .csv.

Source (per subject):
    down_stream/mimic-bp/{abp,ecg,ppg}/{pid}_{mod}.npy  shape=(30, 3750)
    down_stream/mimic-bp/labels/{pid}_labels.npy        shape=(30, 2)  -> [SBP, DBP]

Each 3750-sample (30 s @ 125 Hz) parent segment is split into three non-overlapping
1250-sample (10 s) sub-segments. Labels are inherited from the parent segment.

Output (per split, train/val/test):
    down_stream/mimic-bp/mimic_bp_{split}.npy  shape=(N*90, 3, 1250) float32
        axis 1: [0]=ECG, [1]=PPG, [2]=ABP (mmHg raw)
    down_stream/mimic-bp/mimic_bp_{split}.csv  columns:
        subject_id, parent_seg_idx (0..29), sub_seg_idx (0..2), sbp, dbp

Split membership is read from the provided train/val/test_subjects.txt files
(stored as a single Python-list literal).
"""
from __future__ import annotations

import ast
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path("/Volumes/Disk/projects/UniCardio/down_stream/mimic-bp")
PARENT_LEN = 3750
SUB_LEN = 1250
N_SUB_PER_PARENT = PARENT_LEN // SUB_LEN  # 3
N_PARENT_PER_SUBJECT = 30                 # dataset invariant
SEGS_PER_SUBJECT = N_PARENT_PER_SUBJECT * N_SUB_PER_PARENT  # 90

SPLITS = {
    "train": "train_subjects.txt",
    "val": "val_subjects.txt",
    "test": "test_subjects.txt",
}


def load_subjects(split_file: Path) -> list[str]:
    return list(ast.literal_eval(split_file.read_text().strip()))


def load_subject(pid: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ecg = np.load(ROOT / "ecg" / f"{pid}_ecg.npy")
    ppg = np.load(ROOT / "ppg" / f"{pid}_ppg.npy")
    abp = np.load(ROOT / "abp" / f"{pid}_abp.npy")
    labels = np.load(ROOT / "labels" / f"{pid}_labels.npy")
    expected = (N_PARENT_PER_SUBJECT, PARENT_LEN)
    assert ecg.shape == expected, f"{pid} ecg shape={ecg.shape}"
    assert ppg.shape == expected, f"{pid} ppg shape={ppg.shape}"
    assert abp.shape == expected, f"{pid} abp shape={abp.shape}"
    assert labels.shape == (N_PARENT_PER_SUBJECT, 2), f"{pid} labels shape={labels.shape}"
    return ecg, ppg, abp, labels


def process_split(split: str, subjects_file: str) -> None:
    subjects = load_subjects(ROOT / subjects_file)
    n_total = len(subjects) * SEGS_PER_SUBJECT
    logger.info("=" * 72)
    logger.info("%s split: %d subjects -> %d sub-segments", split, len(subjects), n_total)

    out_npy = ROOT / f"mimic_bp_{split}.npy"
    waves = np.lib.format.open_memmap(
        out_npy, mode="w+", dtype=np.float32, shape=(n_total, 3, SUB_LEN)
    )

    rows: list[dict] = []
    cursor = 0
    for i, pid in enumerate(subjects, start=1):
        ecg, ppg, abp, labels = load_subject(pid)
        # Reshape (30, 3750) -> (30, 3, 1250), then flatten parent/sub axes.
        ecg_sub = ecg.reshape(N_PARENT_PER_SUBJECT, N_SUB_PER_PARENT, SUB_LEN)
        ppg_sub = ppg.reshape(N_PARENT_PER_SUBJECT, N_SUB_PER_PARENT, SUB_LEN)
        abp_sub = abp.reshape(N_PARENT_PER_SUBJECT, N_SUB_PER_PARENT, SUB_LEN)
        # Stack modalities along a new axis -> (30, 3, 3, 1250); order [ECG, PPG, ABP].
        stacked = np.stack([ecg_sub, ppg_sub, abp_sub], axis=2)
        # (30 parents, 3 subs, 3 mods, 1250) -> (90, 3, 1250)
        flat = stacked.reshape(SEGS_PER_SUBJECT, 3, SUB_LEN).astype(np.float32)
        waves[cursor : cursor + SEGS_PER_SUBJECT] = flat
        cursor += SEGS_PER_SUBJECT

        for parent_idx in range(N_PARENT_PER_SUBJECT):
            sbp, dbp = labels[parent_idx]
            for sub_idx in range(N_SUB_PER_PARENT):
                rows.append(
                    {
                        "subject_id": pid,
                        "parent_seg_idx": parent_idx,
                        "sub_seg_idx": sub_idx,
                        "sbp": float(sbp),
                        "dbp": float(dbp),
                    }
                )
        if i % 100 == 0 or i == len(subjects):
            logger.info("  %s  subjects=%d/%d  segs=%d", split, i, len(subjects), cursor)

    waves.flush()
    del waves

    out_csv = ROOT / f"mimic_bp_{split}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logger.info(
        "wrote %s  (%.2f GB)  segments=%d",
        out_npy.name,
        out_npy.stat().st_size / 2**30,
        cursor,
    )
    logger.info("wrote %s  rows=%d", out_csv.name, len(rows))


def main() -> None:
    for split, fname in SPLITS.items():
        process_split(split, fname)


if __name__ == "__main__":
    main()

"""Convert PulseDB HDF5 files to (ECG, PPG, ABP)-ordered .npy + metadata .csv.

Waveform output:
    {stem}.npy   shape=(N, 3, 1250)  dtype=float32
                 axis 1: [0]=ECG, [1]=PPG, [2]=ABP (raw mmHg)

Metadata output:
    {stem}.csv   columns: subject_id, sbp, dbp, map, age, gender, height, weight, bmi

Streamed block-by-block to keep peak RAM under ~1.5 GB.
"""
from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PULSEDB_DIR = Path("/Volumes/Disk/projects/UniCardio/PulseDB")
FILES = ["CalFree_Test_Subset.h5", "Train_Subset.h5"]
BLOCK = 20_000  # ~ 20k * 3 * 1250 * 4 B ≈ 0.28 GB per waveform block


def convert(h5_path: Path) -> None:
    stem = h5_path.stem
    npy_path = h5_path.with_suffix(".npy")
    csv_path = h5_path.with_suffix(".csv")
    logger.info("=" * 72)
    logger.info("Converting %s", h5_path.name)

    with h5py.File(h5_path, "r") as h:
        n = h["ppg_raw"].shape[0]
        length = h["ppg_raw"].shape[-1]
        logger.info("samples=%d  length=%d", n, length)

        # Preallocate .npy on disk via memmap; written block-wise below.
        out = np.lib.format.open_memmap(
            npy_path, mode="w+", dtype=np.float32, shape=(n, 3, length)
        )
        for start in range(0, n, BLOCK):
            end = min(start + BLOCK, n)
            out[start:end, 0] = h["ecg_raw"][start:end, 0].astype(np.float32)
            out[start:end, 1] = h["ppg_raw"][start:end, 0].astype(np.float32)
            out[start:end, 2] = h["abp_raw"][start:end, 0].astype(np.float32)
            logger.info("  waveform %d / %d", end, n)
        out.flush()
        del out

        bp = h["bp_raw"][:]  # (N, 3) -> SBP, DBP, MAP
        meta = pd.DataFrame(
            {
                "subject_id": [s.decode() for s in h["subject_ids"][:]],
                "sbp": bp[:, 0].astype(np.float32),
                "dbp": bp[:, 1].astype(np.float32),
                "map": bp[:, 2].astype(np.float32),
                "age": h["age"][:],
                "gender": h["gender"][:],
                "height": h["height"][:],
                "weight": h["weight"][:],
                "bmi": h["bmi"][:],
            }
        )
    meta.to_csv(csv_path, index=False)
    logger.info("wrote %s  (%.2f GB)", npy_path.name, npy_path.stat().st_size / 2**30)
    logger.info("wrote %s  (%d rows)", csv_path.name, len(meta))


def main() -> None:
    for fname in FILES:
        convert(PULSEDB_DIR / fname)


if __name__ == "__main__":
    main()

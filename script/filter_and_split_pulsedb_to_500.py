"""按 CSV 元数据筛选 PulseDB 行后切成两段非重叠的 500 长度样本。

仅保留 ``age / gender / height / weight / bmi`` 五列同时非空的行（约占 51.6%），
然后对剩余的 (3, 1250) 样本沿用 ``split_pulsedb_to_500.py`` 的切分策略：

  out[2i]   = src_kept[i, :,   0: 500]
  out[2i+1] = src_kept[i, :, 500:1000]

末尾 250 个时间点直接丢弃。

输入: data/pulsedb/Train_Subset.npy + Train_Subset.csv
       data/pulsedb/CalFree_Test_Subset.npy + CalFree_Test_Subset.csv
输出: data/pulsedb/Train_Subset_vitaldb_500.npy        shape=(2 * n_kept, 3, 500)
       data/pulsedb/Train_Subset_vitaldb_500.csv       rows=2 * n_kept（每行复制 2 份）
       data/pulsedb/CalFree_Test_Subset_vitaldb_500.npy shape=(2 * n_kept, 3, 500)
       data/pulsedb/CalFree_Test_Subset_vitaldb_500.csv rows=2 * n_kept

CSV 与 npy 严格按下标对齐：``csv.iloc[2k]`` 和 ``csv.iloc[2k+1]`` 同源，
都对应原始 npy 第 ``keep_idx[k]`` 行；它们分别 mapping 到 ``npy[2k]``（前
500 时间步切片）和 ``npy[2k+1]``（后 500 时间步切片）。

读输入 npy 用 mmap；输出也用 mmap，分块流式处理。
"""
from __future__ import annotations

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

PULSEDB_DIR = Path("/Volumes/Disk/projects/UniCardio/data/pulsedb")
FILES = ["Train_Subset", "CalFree_Test_Subset"]
DEMOGRAPHIC_COLS = ["age", "gender", "height", "weight", "bmi"]
SLOT = 500
N_SLICES = 2
BLOCK = 20_000  # 输入 20k * 3 * 1250 * 4B ≈ 0.28 GB 每块


def filter_and_split(stem: str) -> Path:
    npy_path = PULSEDB_DIR / f"{stem}.npy"
    csv_path = PULSEDB_DIR / f"{stem}.csv"
    dst_path = PULSEDB_DIR / f"{stem}_vitaldb_500.npy"
    dst_csv_path = PULSEDB_DIR / f"{stem}_vitaldb_500.csv"

    df = pd.read_csv(csv_path)
    mask = df[DEMOGRAPHIC_COLS].notna().all(axis=1).to_numpy()
    keep_idx = np.flatnonzero(mask)
    n_kept = int(keep_idx.size)

    src = np.load(npy_path, mmap_mode="r")
    if src.shape[0] != len(df):
        raise ValueError(
            f"Row count mismatch for {stem}: npy={src.shape[0]} csv={len(df)}"
        )
    if src.ndim != 3 or src.shape[1] != 3 or src.shape[2] < SLOT * N_SLICES:
        raise ValueError(
            f"Unexpected shape {src.shape} for {npy_path.name}; "
            f"expected (N, 3, >= {SLOT * N_SLICES})."
        )

    n_out = n_kept * N_SLICES
    logger.info("=" * 72)
    logger.info(
        "filter+split %s  src=%s  kept=%d / %d (%.2f%%)  ->  %s  (%d, 3, %d)",
        stem, src.shape, n_kept, src.shape[0], 100.0 * n_kept / src.shape[0],
        dst_path.name, n_out, SLOT,
    )

    dst = np.lib.format.open_memmap(
        dst_path, mode="w+", dtype=src.dtype, shape=(n_out, 3, SLOT),
    )
    try:
        # 按输入块迭代；每块内取交集 keep_idx 后再切片。
        write_cursor = 0
        for start in range(0, src.shape[0], BLOCK):
            end = min(start + BLOCK, src.shape[0])
            block_keep = keep_idx[(keep_idx >= start) & (keep_idx < end)]
            if block_keep.size == 0:
                continue
            chunk = np.asarray(src[block_keep])  # (b, 3, 1250)
            b = chunk.shape[0]
            two = chunk[:, :, : SLOT * N_SLICES].reshape(b, 3, N_SLICES, SLOT)
            two = np.ascontiguousarray(two.transpose(0, 2, 1, 3))
            two = two.reshape(b * N_SLICES, 3, SLOT)
            dst[write_cursor : write_cursor + two.shape[0]] = two
            write_cursor += two.shape[0]
            logger.info("  rows %d / %d  (kept so far %d)", end, src.shape[0], write_cursor // N_SLICES)
        if write_cursor != n_out:
            raise RuntimeError(
                f"write_cursor={write_cursor} != n_out={n_out} for {stem}"
            )
        dst.flush()
    finally:
        del dst

    size_gb = dst_path.stat().st_size / 2**30
    logger.info("wrote %s  (%.2f GB)", dst_path.name, size_gb)

    # 与 npy 对齐的 CSV：先按 keep_idx 取子集，再把每行复制 N_SLICES 份。
    # `iloc[index.repeat(N)]` 保持原顺序，使 csv.iloc[2k] / [2k+1] 都对应
    # 第 k 个保留样本。
    filtered = df.iloc[keep_idx].reset_index(drop=True)
    duplicated = filtered.loc[filtered.index.repeat(N_SLICES)].reset_index(drop=True)
    duplicated.to_csv(dst_csv_path, index=False)
    logger.info("wrote %s  (%d rows)", dst_csv_path.name, len(duplicated))
    return dst_path


def main() -> None:
    for stem in FILES:
        filter_and_split(stem)


if __name__ == "__main__":
    main()

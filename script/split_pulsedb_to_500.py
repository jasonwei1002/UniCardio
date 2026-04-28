"""把 PulseDB 1250 长度的样本按 0:500 / 500:1000 切成两段非重叠的 500 长度样本。

输入: data/pulsedb/Train_Subset.npy            shape=(N, 3, 1250)
       data/pulsedb/CalFree_Test_Subset.npy    shape=(N, 3, 1250)

输出: data/pulsedb/Train_Subset_500.npy        shape=(2N, 3, 500)
       data/pulsedb/CalFree_Test_Subset_500.npy shape=(2N, 3, 500)

每条 (3, 1250) 样本切成两条 (3, 500)：[:, 0:500] 和 [:, 500:1000]；
最后 250 个时间点直接丢弃。两条切片在输出里相邻排列（即 out[2i], out[2i+1]）。

读输入用 mmap；写输出也用 mmap，分块处理，峰值内存只受 BLOCK 控制（默认约 0.45 GB）。
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PULSEDB_DIR = Path("/Volumes/Disk/projects/UniCardio/data/pulsedb")
FILES = ["Train_Subset.npy", "CalFree_Test_Subset.npy"]
SLOT = 500            # 目标 slot 长度
N_SLICES = 2          # 一条 1250 → 两条 500
BLOCK = 20_000        # 每块输入样本数：20k * 3 * 1250 * 4B ≈ 0.28 GB，输出 0.18 GB


def split_one(src_path: Path) -> Path:
    dst_path = src_path.with_name(f"{src_path.stem}_500.npy")
    src = np.load(src_path, mmap_mode="r")
    if src.ndim != 3 or src.shape[1] != 3 or src.shape[2] < SLOT * N_SLICES:
        raise ValueError(
            f"Unexpected shape {src.shape} for {src_path.name}; "
            f"expected (N, 3, >= {SLOT * N_SLICES})."
        )
    n_in = src.shape[0]
    n_out = n_in * N_SLICES
    logger.info("=" * 72)
    logger.info("splitting %s  %s -> %s  (%d, 3, %d)",
                src_path.name, src.shape, dst_path.name, n_out, SLOT)

    dst = np.lib.format.open_memmap(
        dst_path, mode="w+", dtype=src.dtype, shape=(n_out, 3, SLOT),
    )
    try:
        for start in range(0, n_in, BLOCK):
            end = min(start + BLOCK, n_in)
            chunk = np.asarray(src[start:end])  # 物化当前块进内存
            # chunk: (b, 3, 1250) -> (b, 3, 2, 500) -> (b, 2, 3, 500) -> (b*2, 3, 500)
            b = end - start
            two = chunk[:, :, : SLOT * N_SLICES].reshape(b, 3, N_SLICES, SLOT)
            two = np.ascontiguousarray(two.transpose(0, 2, 1, 3))
            dst[start * N_SLICES : end * N_SLICES] = two.reshape(b * N_SLICES, 3, SLOT)
            logger.info("  rows %d / %d", end, n_in)
        dst.flush()
    finally:
        del dst
    size_gb = dst_path.stat().st_size / 2**30
    logger.info("wrote %s  (%.2f GB)", dst_path.name, size_gb)
    return dst_path


def main() -> None:
    for fname in FILES:
        split_one(PULSEDB_DIR / fname)


if __name__ == "__main__":
    main()

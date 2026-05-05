"""按 train/val/test_subjects.txt 把 MIMIC-BP 30 s 段切成不重叠的 4 s 样本。

每个受试者在 ``data/mimicbp/{ecg,ppg,abp}/<sid>_<wav>.npy`` 下各有一份
``(30, 3750)`` 数组（125 Hz × 30 s）。本脚本：

  - 读取 ``train_subjects.txt / val_subjects.txt / test_subjects.txt``
    （文件内是单行 Python list 字面量）；
  - 对每个 30 s 段取前 28 s，切成 7 个 ``500`` 长度的不重叠 4 s 片段，
    丢弃最后 250 个样本点（2 s）；
  - 沿 channel 轴按 ``(ECG, PPG, ABP)`` 拼接，输出形状 ``(N, 3, 500)``；
  - 不做任何归一化（``CardiacDataset`` 内部对 slot=2 应用 BP 归一化）；
  - dtype 统一为 ``float32``。

输出：
  data/mimicbp/Train_mimicbp_500.npy        (n_train_subj * 30 * 7, 3, 500)
  data/mimicbp/Val_mimicbp_500.npy
  data/mimicbp/Test_mimicbp_500.npy

写入用 ``np.lib.format.open_memmap``，避免把整组数组在 RAM 里拼起来。
"""
from __future__ import annotations

import ast
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
MIMIC_DIR = ROOT / "data" / "mimicbp"
WAVES = ("ecg", "ppg", "abp")  # 输出 channel 顺序：slot 0/1/2
FS = 125
SEG_LEN = 30 * FS  # 3750
SLOT = 4 * FS      # 500
N_SLICES = 7       # 28 s / 4 s，丢弃最后 2 s
SEGS_PER_SUBJ = 30

SPLITS = {
    "Train": "train_subjects.txt",
    "Val": "val_subjects.txt",
    "Test": "test_subjects.txt",
}


def load_subject_ids(txt_path: Path) -> list[str]:
    text = txt_path.read_text().strip()
    ids = ast.literal_eval(text)
    if not isinstance(ids, list) or not all(isinstance(s, str) for s in ids):
        raise ValueError(f"{txt_path} must contain a Python list of strings")
    return ids


def load_subject_signals(sid: str) -> np.ndarray:
    """返回 ``(SEGS_PER_SUBJ, 3, SEG_LEN)``，channel 顺序 = (ECG, PPG, ABP)。"""
    arrs = []
    for wav in WAVES:
        path = MIMIC_DIR / wav / f"{sid}_{wav}.npy"
        a = np.load(path, mmap_mode="r")
        if a.shape != (SEGS_PER_SUBJ, SEG_LEN):
            raise ValueError(f"{path} shape {a.shape} != ({SEGS_PER_SUBJ}, {SEG_LEN})")
        arrs.append(a)
    return np.stack(arrs, axis=1).astype(np.float32, copy=False)  # (30, 3, 3750)


def slice_subject(stack: np.ndarray) -> np.ndarray:
    """``(30, 3, 3750)`` -> ``(30 * 7, 3, 500)``，按时间维 reshape，时间序保留。"""
    truncated = stack[:, :, : SLOT * N_SLICES]              # (30, 3, 3500)
    sliced = truncated.reshape(SEGS_PER_SUBJ, 3, N_SLICES, SLOT)  # (30, 3, 7, 500)
    sliced = np.ascontiguousarray(sliced.transpose(0, 2, 1, 3))   # (30, 7, 3, 500)
    return sliced.reshape(SEGS_PER_SUBJ * N_SLICES, 3, SLOT)      # (210, 3, 500)


def convert_split(split_name: str, txt_name: str) -> Path:
    ids = load_subject_ids(MIMIC_DIR / txt_name)
    n_out = len(ids) * SEGS_PER_SUBJ * N_SLICES
    dst_path = MIMIC_DIR / f"{split_name}_mimicbp_500.npy"

    logger.info("=" * 72)
    logger.info(
        "split=%s  subjects=%d  -> %s  shape=(%d, 3, %d)",
        split_name, len(ids), dst_path.name, n_out, SLOT,
    )

    dst = np.lib.format.open_memmap(
        dst_path, mode="w+", dtype=np.float32, shape=(n_out, 3, SLOT),
    )
    try:
        cursor = 0
        log_every = max(1, len(ids) // 20)
        for i, sid in enumerate(ids):
            stack = load_subject_signals(sid)
            sliced = slice_subject(stack)
            dst[cursor : cursor + sliced.shape[0]] = sliced
            cursor += sliced.shape[0]
            if (i + 1) % log_every == 0 or (i + 1) == len(ids):
                logger.info(
                    "  %s: %d / %d subjects, %d slices written",
                    split_name, i + 1, len(ids), cursor,
                )
        if cursor != n_out:
            raise RuntimeError(f"cursor={cursor} != n_out={n_out}")
        dst.flush()
    finally:
        del dst

    size_gb = dst_path.stat().st_size / 2**30
    logger.info("wrote %s  (%.2f GB)", dst_path.name, size_gb)
    return dst_path


def main() -> None:
    for split_name, txt_name in SPLITS.items():
        convert_split(split_name, txt_name)


if __name__ == "__main__":
    main()

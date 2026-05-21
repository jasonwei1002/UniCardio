"""Path A: ``np.memmap``-backed dataset returning shape-only ABP + scalars + demographics.

Channel order: 下游所有代码（模型、mask、采样器、指标）按
**模型** slot 顺序 ``ECG=0, PPG=1, ABP=2`` 工作。``CardiacDataset``
内部按 ``channel_permutation`` 把磁盘顺序重排到模型 slot 顺序，并对
slot 2（ABP）做 **per-sample min-max** 归一化到 ``[0, 1]``。
ECG / PPG 保持 raw scale。

每次 ``__getitem__`` 返回 3-tuple：

* ``signal``: ``(3, L)`` float32，ABP slot ∈ [0, 1]，其它 slot 为原始尺度。
* ``sbp_dbp``: ``(2,)`` float32 = ``(sbp_mmHg, dbp_mmHg)`` 从 CSV 读取
  （PulseDB 的 ground-truth per-cycle mean，不是 segment max/min）。
* ``demographics``: ``(6,)`` float32 = ``[age_z, gender, height_z, weight_z, bmi_z, mask]``
  - ``age_z`` / ``height_z`` / ``weight_z`` / ``bmi_z``: z-score 归一化（数据集
    整体统计），缺失值 -> 0。
  - ``gender`` ∈ {0, 1}：原始 categorical，不归一化。
  - ``mask`` ∈ {0, 1}：anthropometric (height/weight/bmi) 存在与否的指示位。
    PulseDB 约 48% 样本（MIMIC-III 子集）缺三者，mask=0；剩下的 VitalDB
    子集 mask=1。BP head 用这一位区分 "已知中位值" 和 "真实缺失"。

CSV path 默认从 ``data_path`` 推断（同目录、同 stem，扩展名换 ``.csv``）；
若 CSV 不存在或缺关键列，``sbp_dbp`` / ``demographics`` 会用零张量代替并
emit 一条 WARNING，方便对老 (legacy) 数据集做调试。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..utils.normalization import MINMAX_EPS, BPLabelNorm

logger = logging.getLogger(__name__)

_BP_LABEL_COLUMNS = ("sbp", "dbp")
_NUMERIC_DEMOGRAPHIC_COLUMNS = ("age", "height", "weight", "bmi")  # z-scored
_GENDER_COLUMN = "gender"  # binary {0, 1}, no z-score
_ANTHROPOMETRIC_COLUMNS = ("height", "weight", "bmi")  # drive the missingness mask


def _zscore_stats(values: np.ndarray) -> tuple[float, float]:
    """Per-column ``(mean, std)`` over non-NaN entries. Returns ``(0, 1)``
    if no usable rows or std collapses to ~0."""
    mask = ~np.isnan(values)
    if not mask.any():
        return 0.0, 1.0
    mean = float(values[mask].mean())
    std = float(values[mask].std(ddof=1))
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0
    return mean, std


def _build_csv_tables(
    data_path: Path,
    csv_path: Path | None,
    n_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Single-pass CSV → ``(bp_labels (N, 2), demographics (N, 6))``.

    Demographics layout (6 columns):
        ``[age_z, gender, height_z, weight_z, bmi_z, mask]``
    where ``mask=1`` iff all anthropometric values (height/weight/bmi)
    were present in this row, ``0`` otherwise. Z-scores use the full
    CSV's non-NaN stats; NaN → 0 after z-scoring.

    Returns zero-filled tables (with a WARNING) when the CSV is missing.
    """
    bp_labels = np.zeros((n_rows, 2), dtype=np.float32)
    demographics = np.zeros((n_rows, 6), dtype=np.float32)
    if csv_path is None:
        csv_path = data_path.with_suffix(".csv")
    if not csv_path.exists():
        logger.warning(
            "No CSV sibling at %s; sbp_dbp + demographics fall back to zero "
            "tensors. Path A BP head training requires the CSV.",
            csv_path,
        )
        return bp_labels, demographics

    df = pd.read_csv(csv_path)
    if len(df) != n_rows:
        logger.warning(
            "CSV row count %d != npy row count %d; alignment may be off.",
            len(df), n_rows,
        )
    n = min(len(df), n_rows)

    for col_idx, col in enumerate(_BP_LABEL_COLUMNS):
        if col in df.columns:
            bp_labels[:n, col_idx] = df[col].to_numpy(dtype=np.float32)[:n]
        else:
            logger.warning("CSV missing column '%s'; %s left at 0.", col, col)

    # Age (column 0) — z-score, NaN→0.
    if "age" in df.columns:
        age = df["age"].to_numpy(dtype=np.float64)
        mu, sd = _zscore_stats(age)
        demographics[:n, 0] = np.nan_to_num((age - mu) / sd, nan=0.0).astype(np.float32)[:n]
    # Gender (column 1) — raw {0, 1}, NaN→0.
    if _GENDER_COLUMN in df.columns:
        gender = df[_GENDER_COLUMN].to_numpy(dtype=np.float32)
        demographics[:n, 1] = np.nan_to_num(gender, nan=0.0)[:n]
    # Anthropometrics (columns 2..4) — z-score; the per-row 'all-present' AND
    # drives the mask in column 5.
    has_anthro = np.ones(n, dtype=np.bool_)
    for offset, col in enumerate(_ANTHROPOMETRIC_COLUMNS):
        col_idx = 2 + offset
        if col not in df.columns:
            has_anthro[:] = False
            continue
        values = df[col].to_numpy(dtype=np.float64)[:n]
        nan_mask = np.isnan(values)
        has_anthro &= ~nan_mask
        mu, sd = _zscore_stats(df[col].to_numpy(dtype=np.float64))
        demographics[:n, col_idx] = np.nan_to_num(
            (values - mu) / sd, nan=0.0
        ).astype(np.float32)
    demographics[:n, 5] = has_anthro.astype(np.float32)
    return bp_labels, demographics


class CardiacDataset(Dataset):
    """``np.memmap``-backed ``(N, 3, slot_length)`` 数据集 + CSV-derived 标签。

    Args:
        data_path: ``.npy`` 文件路径，形状必须为 ``(N, 3, L)``。
        indices: 该 split 选用的样本下标。
        channel_permutation: 把磁盘通道顺序重排到模型 slot 顺序的 3 个索引。
        bp_labels_table: 可选预构建的 ``(N_total, 2)`` SBP/DBP 表。若 None，
            构造时尝试从 ``data_path.with_suffix('.csv')`` 加载。
        demographics_table: 可选预构建的 ``(N_total, 6)`` demographics 表。
        csv_path: 显式覆盖 CSV 路径（默认从 data_path 推断）。
    """

    def __init__(
        self,
        data_path: str | Path,
        indices: np.ndarray | Sequence[int],
        channel_permutation: Sequence[int] = (0, 1, 2),
        *,
        bp_labels_table: np.ndarray | None = None,
        demographics_table: np.ndarray | None = None,
        csv_path: str | Path | None = None,
        bp_label_norm: "BPLabelNorm | None" = None,
    ) -> None:
        path = Path(data_path)
        self._mm = np.load(str(path), mmap_mode="r")
        if self._mm.ndim != 3 or self._mm.shape[1] != 3:
            raise ValueError(
                f"Expected (N, 3, L) data at {path}, got shape {self._mm.shape}"
            )
        self._indices = np.asarray(indices, dtype=np.int64)
        self._perm = np.asarray(channel_permutation, dtype=np.int64)
        if self._perm.shape != (3,):
            raise ValueError(
                f"channel_permutation must have 3 entries, got {self._perm.tolist()}"
            )

        n_total = int(self._mm.shape[0])
        built_from_csv = bp_labels_table is None
        if bp_labels_table is None or demographics_table is None:
            csv = Path(csv_path) if csv_path is not None else None
            bp_built, demo_built = _build_csv_tables(path, csv, n_total)
            if bp_labels_table is None:
                bp_labels_table = bp_built
            if demographics_table is None:
                demographics_table = demo_built
        if bp_labels_table.shape != (n_total, 2):
            raise ValueError(
                f"bp_labels_table shape must be ({n_total}, 2), got {bp_labels_table.shape}"
            )
        if demographics_table.shape != (n_total, 6):
            raise ValueError(
                f"demographics_table shape must be ({n_total}, 6), got "
                f"{demographics_table.shape}"
            )

        # When ``bp_label_norm`` is set, pre-normalize once at construction
        # (MD-ViSCo Sec III.D). __getitem__ stays branchless. Cached tables
        # (passed from a sibling dataset via ``bp_labels_table=...``) are
        # assumed to already be normalized by their original owner; sharing
        # is safe because data.bp_label_norm is a single top-level config.
        if bp_label_norm is not None and built_from_csv:
            bp_labels_table = bp_label_norm.normalize(bp_labels_table)

        self._bp_labels = np.ascontiguousarray(bp_labels_table, dtype=np.float32)
        self._demographics = np.ascontiguousarray(demographics_table, dtype=np.float32)
        self._bp_norm = bp_label_norm

    @property
    def indices(self) -> np.ndarray:
        """该 split 在原始 .npy / 同名 .csv 里的行号（一维 int64 数组）。"""
        return self._indices

    @property
    def bp_labels_table(self) -> np.ndarray:
        """Full ``(N_total, 2)`` SBP/DBP table — used by build_loaders to share."""
        return self._bp_labels

    @property
    def demographics_table(self) -> np.ndarray:
        """Full ``(N_total, 6)`` demographics table — used by build_loaders to share."""
        return self._demographics

    def __len__(self) -> int:
        return int(self._indices.shape[0])

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        # Advanced indexing returns an owned, writable, C-contiguous ndarray
        # (safe for from_numpy + pin_memory). astype(copy=False) is a no-op on
        # float32 sources, else casts to avoid conv1d dtype mismatch.
        row = int(self._indices[idx])
        x = self._mm[row, self._perm].astype(np.float32, copy=False)
        abp = x[2]
        dbp_seg = float(abp.min())
        sbp_seg = float(abp.max())
        denom = max(sbp_seg - dbp_seg, MINMAX_EPS)
        x[2] = (abp - dbp_seg) / denom  # per-sample min-max → [0, 1]
        signal = torch.from_numpy(x)
        # SBP/DBP come from CSV (PulseDB per-cycle-mean labeling), not from
        # the per-segment min/max we just computed for normalization.
        # `_bp_labels` / `_demographics` are C-contiguous owned arrays; advanced
        # indexing with a single int returns an owned row, so torch.from_numpy
        # is safe without .copy() (table is never mutated after dataset init).
        # ``_bp_labels`` is already pre-normalized in __init__ when bp_label_norm
        # is active, so __getitem__ is branchless on the hot path (called
        # ~902k × n_epochs × n_workers times).
        sbp_dbp = torch.from_numpy(self._bp_labels[row])
        demographics = torch.from_numpy(self._demographics[row])
        return signal, sbp_dbp, demographics

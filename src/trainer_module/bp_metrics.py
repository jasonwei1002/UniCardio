"""Signal-domain BP metrics: SBP/DBP ME and SD on the test split.

The absolute SBP/DBP are predicted by a trained :class:`BPHead` directly
from the raw ECG/PPG of each task's condition modalities (matching
MD-ViSCo's scalar refinement metric — no waveform reconstruction, no RF
sampler). A BP-head ckpt is **required**; the legacy sampler max/min
fallback on the inverse-``(x-100)/50`` waveform has been removed.

CSV alignment invariant: with ``shuffle=False`` the loader consumes the
dataset in index order, so ``dataset.indices[k]`` is the row of sample
``k`` in both the ``.npy`` and the sibling ``.csv``. SD uses ``ddof=1``
per BHS / AAMI convention.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..model_module.bp_head import BPHead, build_bp_head
from ..model_module.tasks import TaskSpec, cond_slots_to_vitals
from ..utils.checkpoint import load_checkpoint
from ..utils.normalization import BPLabelNorm, split_abp_minmax

logger = logging.getLogger(__name__)


def _read_bp_labels(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """只读 ``sbp`` / ``dbp`` 两列，避免把整个元数据表都进内存。"""
    df = pd.read_csv(csv_path, usecols=["sbp", "dbp"])
    return df["sbp"].to_numpy(dtype=np.float64), df["dbp"].to_numpy(dtype=np.float64)


def load_bp_head_ckpt(
    ckpt_path: str | Path,
    device: torch.device,
) -> BPHead:
    """Instantiate a :class:`BPHead` and load weights from ``ckpt_path``.

    The architecture is read from the ckpt's ``config`` field (saved by
    :func:`save_checkpoint`). Falls back to default ``BPHeadConfig`` if the
    config blob is missing.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("config", {})
    bp_head_cfg = cfg_dict.get("model") if isinstance(cfg_dict, dict) else None
    model = build_bp_head(bp_head_cfg or {})
    model.to(device)
    load_checkpoint(ckpt_path, model=model, map_location=device)
    model.eval()
    return model


@torch.no_grad()
def _predict_one_task(
    bp_head: BPHead,
    test_loader: DataLoader,
    task: TaskSpec,
    *,
    device: torch.device,
    amp_enabled: bool,
    bp_norm: BPLabelNorm | None = None,
    collect_seg: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(sbp_pred, dbp_pred, sbp_seg, dbp_seg)`` per sample, mmHg.

    SBP/DBP predictions come straight from the BP head's regression on the
    task's condition modalities (``active_vitals=cond_slots_to_vitals(task)``,
    so e.g. ppg2abp does not leak the ECG channel). When ``bp_norm`` is set the
    head output is in the globally min-max normalized ``[0, 1]`` label space it
    was trained in and is denormalized back to mmHg. When ``collect_seg`` is set,
    ``sbp_seg`` / ``dbp_seg`` are the per-segment ABP max/min (from ``abp_minmax``,
    batch[3]) collected in loader order — the ground truth for
    ``bp_label_source='segment_minmax'``; otherwise they are empty arrays.
    """
    sbp_chunks: list[np.ndarray] = []
    dbp_chunks: list[np.ndarray] = []
    sbp_seg_chunks: list[np.ndarray] = []
    dbp_seg_chunks: list[np.ndarray] = []
    for batch in test_loader:
        signal = batch[0].to(device, non_blocking=True)
        demographics = (
            batch[2].to(device, non_blocking=True) if len(batch) >= 3 else None
        )
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled
        ):
            bp_pred = bp_head(
                signal[:, :2, :],
                demographics,
                active_vitals=cond_slots_to_vitals(task),
            ).float()
        if bp_norm is not None:
            # Head was trained on min-max normalized [0, 1] labels; recover mmHg.
            bp_pred = bp_norm.denormalize(bp_pred)
        sbp_chunks.append(bp_pred[:, 0].detach().cpu().numpy())
        dbp_chunks.append(bp_pred[:, 1].detach().cpu().numpy())
        if collect_seg and len(batch) >= 4:
            sbp_seg_b, dbp_seg_b = split_abp_minmax(batch[3])  # (dbp,sbp)->(sbp,dbp)
            sbp_seg_chunks.append(sbp_seg_b.detach().cpu().numpy())
            dbp_seg_chunks.append(dbp_seg_b.detach().cpu().numpy())
    sbp_seg = np.concatenate(sbp_seg_chunks) if sbp_seg_chunks else np.empty(0)
    dbp_seg = np.concatenate(dbp_seg_chunks) if dbp_seg_chunks else np.empty(0)
    return np.concatenate(sbp_chunks), np.concatenate(dbp_chunks), sbp_seg, dbp_seg


def evaluate_bp_test(
    test_loader: DataLoader,
    *,
    tasks: Sequence[TaskSpec],
    csv_path: str | Path,
    device: torch.device,
    bp_head_ckpt: str | Path,
    amp_enabled: bool = False,
    bp_norm: BPLabelNorm | None = None,
    bp_label_source: str = "per_cycle_mean",
) -> dict[str, dict[str, float]]:
    """Per-task SBP/DBP ME ± SD on the test split.

    Args:
        test_loader: must be ``shuffle=False``; ``dataset`` must expose
            ``.indices`` (CSV row indices).
        tasks: only ``target_slot == 2`` (ABP) tasks contribute; others
            are skipped with INFO.
        csv_path: ``.csv`` co-located with the test ``.npy``; must contain
            ``sbp`` and ``dbp`` columns aligned row-by-row with the ``.npy``.
        device: torch device.
        bp_head_ckpt: path to a trained :class:`BPHead` ckpt (required).
        amp_enabled: enable bf16 autocast.
        bp_norm: SBP/DBP label normalization the BP head was trained under.
            When set, the head's ``[0, 1]`` output is denormalized back to
            mmHg before comparison. ``None`` assumes the head emits raw mmHg.
        bp_label_source: ground-truth target definition. ``"per_cycle_mean"``
            (CSV ``sbp``/``dbp`` labels, legacy) or ``"segment_minmax"``
            (per-segment ABP max/min from ``abp_minmax``, matching the head's
            training target and MD-ViSCo Table III). Must match the value the
            head was trained with.

    Returns:
        ``{task_name: {sbp_me, sbp_sd, dbp_me, dbp_sd, n}}``. ``error =
        pred - true``; ``SD`` is sample stddev (``ddof=1``).
    """
    if not bp_head_ckpt:
        raise ValueError("evaluate_bp_test requires a trained bp_head_ckpt.")
    if bp_label_source not in ("per_cycle_mean", "segment_minmax"):
        raise ValueError(
            f"bp_label_source must be 'per_cycle_mean' or 'segment_minmax'; "
            f"got {bp_label_source!r}"
        )
    dataset = test_loader.dataset
    if not hasattr(dataset, "indices"):
        raise AttributeError(
            "test_loader.dataset must expose `.indices` (CSV row indices)"
        )
    n = int(np.asarray(dataset.indices, dtype=np.int64).size)

    # per_cycle_mean targets come from the CSV (aligned by dataset.indices);
    # segment_minmax targets are the per-segment extrema collected per task
    # from the loader's abp_minmax, so no CSV is needed.
    sbp_true = dbp_true = None
    if bp_label_source == "per_cycle_mean":
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"BP labels CSV not found: {csv_path}")
        sbp_label_all, dbp_label_all = _read_bp_labels(csv_path)
        indices = np.asarray(dataset.indices, dtype=np.int64)
        if indices.max(initial=-1) >= sbp_label_all.size:
            raise ValueError(
                f"dataset.indices max={indices.max()} but CSV has only "
                f"{sbp_label_all.size} rows ({csv_path})"
            )
        sbp_true = sbp_label_all[indices]
        dbp_true = dbp_label_all[indices]

    bp_head = load_bp_head_ckpt(bp_head_ckpt, device)
    logger.info(
        "BP head loaded from %s for BP inference (bp_label_source=%s).",
        bp_head_ckpt, bp_label_source,
    )

    results: dict[str, dict[str, float]] = {}
    for task in tasks:
        if int(task.target_slot) != 2:
            logger.info("Skipping BP metrics for non-ABP task '%s'", task.name)
            continue

        sbp_pred, dbp_pred, sbp_seg, dbp_seg = _predict_one_task(
            bp_head, test_loader, task,
            device=device, amp_enabled=amp_enabled,
            bp_norm=bp_norm,
            collect_seg=(bp_label_source == "segment_minmax"),
        )
        if sbp_pred.size != n:
            raise RuntimeError(
                f"Predicted {sbp_pred.size} samples but loader covers {n} samples "
                f"for task '{task.name}'; check shuffle / drop_last."
            )
        if bp_label_source == "segment_minmax":
            if sbp_seg.size != n:
                raise RuntimeError(
                    "bp_label_source='segment_minmax' needs abp_minmax (batch[3]) "
                    f"from the loader; got {sbp_seg.size} extrema vs {n} samples."
                )
            sbp_err = sbp_pred - sbp_seg
            dbp_err = dbp_pred - dbp_seg
        else:
            sbp_err = sbp_pred - sbp_true
            dbp_err = dbp_pred - dbp_true
        results[task.name] = {
            "sbp_me": float(sbp_err.mean()),
            "sbp_sd": float(sbp_err.std(ddof=1)),
            "dbp_me": float(dbp_err.mean()),
            "dbp_sd": float(dbp_err.std(ddof=1)),
            "n": float(n),
        }
        logger.info(
            "BP metrics %-12s | SBP ME=%+.2f SD=%.2f mmHg | DBP ME=%+.2f SD=%.2f mmHg | N=%d",
            task.name,
            results[task.name]["sbp_me"], results[task.name]["sbp_sd"],
            results[task.name]["dbp_me"], results[task.name]["dbp_sd"],
            n,
        )
    return results

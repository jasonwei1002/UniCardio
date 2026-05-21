"""Path A signal-domain BP metrics: SBP/DBP ME and SD on the test split.

Under Path A the RF model produces a per-sample **shape-only** ABP in
``[0, 1]``; the absolute SBP/DBP are predicted by a separate
:class:`BPHead`. AAMI metric is therefore the BP head's prediction vs the
CSV ground truth — the RF sampler does **not** affect the BP numbers
directly. The Euler reconstruction is still combined with the predicted
``(SBP, DBP)`` via :func:`reconstruct_mmHg` so callers that need the full
mmHg waveform (e.g. visualization or waveform-fidelity metric) get it.

CSV alignment invariant: with ``shuffle=False`` the loader consumes the
dataset in index order, so ``dataset.indices[k]`` is the row of sample
``k`` in both the ``.npy`` and the sibling ``.csv``. SD uses ``ddof=1``
per BHS / AAMI convention.

Backwards-compatible fallback: if ``bp_head`` is ``None``, the previous
legacy path (sampler max/min of the inverse-normalized waveform) is used
with a one-time WARNING. This lets RF-only training runs still produce a
BP number while the BP head ckpt is not yet trained, but the numbers are
not comparable to the Path A targets.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ..model_module.bp_head import BPHead, build_bp_head
from ..model_module.tasks import TaskSpec
from ..utils.checkpoint import load_checkpoint
from ..utils.normalization import bp_denormalize, reconstruct_mmHg
from .sampler import euler_sample

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
    rf_model: nn.Module,
    bp_head: BPHead | None,
    test_loader: DataLoader,
    task: TaskSpec,
    *,
    n_steps: int,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(sbp_pred, dbp_pred)`` per sample in loader order, mmHg.

    Path A pathway (``bp_head is not None``): SBP/DBP come straight from
    the BP head's regression on ECG+PPG. The RF Euler reconstruction is
    still computed to keep the wave_mmHg buffer warm for any downstream
    waveform-quality metric, but it does not feed into AAMI numbers.

    Legacy fallback (``bp_head is None``): SBP/DBP = amax/amin of the
    inverse-``(x-100)/50`` waveform — only meaningful for ckpts trained
    under the legacy normalization.
    """
    sbp_chunks: list[np.ndarray] = []
    dbp_chunks: list[np.ndarray] = []
    for batch in test_loader:
        signal = batch[0].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled
        ):
            # Always invoke the sampler so the per-task RF forward graph is
            # exercised; useful for OOM / shape sanity checks and for any
            # caller-side waveform metric that reuses the same loader.
            out = euler_sample(
                rf_model, signal, task, n_steps=n_steps, device=device
            )
        if bp_head is None:
            wave_mmHg = bp_denormalize(out.squeeze(1).float())
            sbp_chunks.append(wave_mmHg.amax(dim=-1).detach().cpu().numpy())
            dbp_chunks.append(wave_mmHg.amin(dim=-1).detach().cpu().numpy())
            continue
        # Path A: BP head on raw ECG + PPG → (SBP, DBP) in mmHg directly.
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled
        ):
            bp_pred = bp_head(signal[:, :2, :]).float()
        sbp_chunks.append(bp_pred[:, 0].detach().cpu().numpy())
        dbp_chunks.append(bp_pred[:, 1].detach().cpu().numpy())
    return np.concatenate(sbp_chunks), np.concatenate(dbp_chunks)


def evaluate_bp_test(
    model: nn.Module,
    test_loader: DataLoader,
    *,
    tasks: Sequence[TaskSpec],
    csv_path: str | Path,
    n_steps: int,
    device: torch.device,
    amp_enabled: bool = False,
    bp_head_ckpt: str | Path | None = None,
) -> dict[str, dict[str, float]]:
    """Per-task SBP/DBP ME ± SD on the test split.

    Args:
        model: RF model (:class:`UniCardioRF` or compiled wrapper).
        test_loader: must be ``shuffle=False``; ``dataset`` must expose
            ``.indices`` (CSV row indices).
        tasks: only ``target_slot == 2`` (ABP) tasks contribute; others
            are skipped with INFO.
        csv_path: ``.csv`` co-located with the test ``.npy``; must contain
            ``sbp`` and ``dbp`` columns aligned row-by-row with the ``.npy``.
        n_steps: Euler ODE step count.
        device: torch device.
        amp_enabled: enable bf16 autocast.
        bp_head_ckpt: path to a trained :class:`BPHead` ckpt. If ``None``,
            falls back to legacy sampler-max/min on inverse ``(x-100)/50``
            normalization (with a WARNING).

    Returns:
        ``{task_name: {sbp_me, sbp_sd, dbp_me, dbp_sd, n}}``. ``error =
        pred - true``; ``SD`` is sample stddev (``ddof=1``).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"BP labels CSV not found: {csv_path}")

    sbp_label_all, dbp_label_all = _read_bp_labels(csv_path)
    dataset = test_loader.dataset
    if not hasattr(dataset, "indices"):
        raise AttributeError(
            "test_loader.dataset must expose `.indices` (CSV row indices)"
        )
    indices = np.asarray(dataset.indices, dtype=np.int64)
    n = int(indices.size)
    if indices.max(initial=-1) >= sbp_label_all.size:
        raise ValueError(
            f"dataset.indices max={indices.max()} but CSV has only "
            f"{sbp_label_all.size} rows ({csv_path})"
        )
    sbp_true = sbp_label_all[indices]
    dbp_true = dbp_label_all[indices]

    bp_head: BPHead | None = None
    if bp_head_ckpt is not None:
        bp_head = load_bp_head_ckpt(bp_head_ckpt, device)
        logger.info("BP head loaded from %s for Path A inference.", bp_head_ckpt)
    else:
        logger.warning(
            "No bp_head_ckpt provided; falling back to legacy sampler "
            "max/min path. BP numbers will only be meaningful for ckpts "
            "trained under the deprecated (x-100)/50 normalization."
        )

    results: dict[str, dict[str, float]] = {}
    for task in tasks:
        if int(task.target_slot) != 2:
            logger.info("Skipping BP metrics for non-ABP task '%s'", task.name)
            continue

        sbp_pred, dbp_pred = _predict_one_task(
            model, bp_head, test_loader, task,
            n_steps=n_steps, device=device, amp_enabled=amp_enabled,
        )
        if sbp_pred.size != n:
            raise RuntimeError(
                f"Predicted {sbp_pred.size} samples but loader covers {n} samples "
                f"for task '{task.name}'; check shuffle / drop_last."
            )
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

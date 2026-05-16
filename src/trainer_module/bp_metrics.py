"""Signal-domain BP metrics: SBP/DBP ME and SD from Euler-reconstructed ABP.

CSV alignment invariant: with ``shuffle=False`` the loader consumes the
dataset in index order, so ``dataset.indices[k]`` is the row of sample ``k``
in both the ``.npy`` and the sibling ``.csv``. SD uses ``ddof=1`` per
BHS / AAMI convention.
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

from ..model_module.tasks import TaskSpec
from ..utils.normalization import bp_denormalize
from .sampler import euler_sample

logger = logging.getLogger(__name__)


def _read_bp_labels(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """只读 ``sbp`` / ``dbp`` 两列，避免把整个元数据表都进内存。"""
    df = pd.read_csv(csv_path, usecols=["sbp", "dbp"])
    return df["sbp"].to_numpy(dtype=np.float64), df["dbp"].to_numpy(dtype=np.float64)


@torch.no_grad()
def _predict_one_task(
    model: nn.Module,
    test_loader: DataLoader,
    task: TaskSpec,
    *,
    n_steps: int,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """跑一遍 Euler 重建，返回 (sbp_pred_mmHg, dbp_pred_mmHg)，按 loader 顺序拼接。"""
    sbp_chunks: list[np.ndarray] = []
    dbp_chunks: list[np.ndarray] = []
    for batch in test_loader:
        signal = batch[0].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled
        ):
            out = euler_sample(model, signal, task, n_steps=n_steps, device=device)
        wave_mmHg = bp_denormalize(out.squeeze(1).float())
        sbp_chunks.append(wave_mmHg.amax(dim=-1).detach().cpu().numpy())
        dbp_chunks.append(wave_mmHg.amin(dim=-1).detach().cpu().numpy())
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
) -> dict[str, dict[str, float]]:
    """对每个 ABP-target task 计算 SBP/DBP 的 ME 与 SD。

    Args:
        model: 训练后的 :class:`UniCardioRF`（或同接口模型）。
        test_loader: 必须 ``shuffle=False``，underlying ``CardiacDataset``
            通过 ``.indices`` 暴露原始 .csv 行号。
        tasks: 仅对 ``target_slot==2`` (ABP) 的 task 计算 BP 指标，
            其它 task 会被跳过并记一条 INFO。
        csv_path: 与 test_loader 数据 ``.npy`` 同源的 CSV，必须含 ``sbp``、
            ``dbp`` 列；行号与 .npy 对齐。
        n_steps: Euler ODE 步数。
        device: PyTorch device。
        amp_enabled: 是否启用 bf16 autocast。

    Returns:
        ``{task_name: {sbp_me, sbp_sd, dbp_me, dbp_sd, n}}``。
        ``error = pred - true``；``SD`` 为样本标准差（ddof=1）。
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

    results: dict[str, dict[str, float]] = {}
    for task in tasks:
        if int(task.target_slot) != 2:
            logger.info("Skipping BP metrics for non-ABP task '%s'", task.name)
            continue

        sbp_pred, dbp_pred = _predict_one_task(
            model, test_loader, task,
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

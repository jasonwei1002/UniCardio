"""Path A 评估入口：加载 RF ckpt（+ 可选 BPHead ckpt）并对所有任务打分。

例：
    # ABP-target 任务在 mmHg 量纲下报指标（推荐）：
    python run/pipeline/evaluate.py \
        +checkpoint=run/outputs/<rf>/checkpoints/best.pt \
        +bp_head_checkpoint=run/outputs/<bp_head>/checkpoints/best.pt \
        device=cuda data.num_workers=8

    # 仅 RF，ABP 输出留在 [0, 1] shape 空间（Pearson 仍然可比，
    # RMSE / MAE 是无量纲量）：
    python run/pipeline/evaluate.py \
        +checkpoint=run/outputs/<rf>/checkpoints/best.pt
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data_module.datamodule import build_loaders
from src.model_module.tasks import Slot, active_task_pairs
from src.model_module.unicardio_rf import UniCardioRF
from src.trainer_module.bp_metrics import load_bp_head_ckpt
from src.trainer_module.sampler import euler_sample
from src.utils.checkpoint import load_checkpoint
from src.utils.metrics import ks_statistic, mae, pearson_corr, rmse
from src.utils.normalization import BPLabelNorm, reconstruct_mmHg
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def _resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


@torch.no_grad()
def _eval_task(
    rf_model: torch.nn.Module,
    bp_head: torch.nn.Module | None,
    loader,
    task,
    *,
    device: torch.device,
    n_steps: int,
    limit_batches: int | None,
    bp_norm: BPLabelNorm | None = None,
) -> dict[str, float]:
    """Per-task waveform metrics.

    For ABP-target tasks under Path A: if ``bp_head`` is provided, both
    pred and target waveforms are reconstructed to mmHg via the predicted
    ``(SBP, DBP)`` (pred) and the ground-truth ``(SBP, DBP)`` (target).
    Otherwise both stay in shape-only ``[0, 1]`` and only Pearson is
    physically meaningful.

    When ``bp_norm`` is provided (MD-ViSCo refinement-style globally min-max
    normalized BP labels), both the BP head output and the ground-truth
    ``sbp_dbp`` from the loader are in [0, 1] space; this function inverts
    them to mmHg before feeding into ``reconstruct_mmHg``.
    """
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    total = limit_batches if limit_batches is not None else len(loader)
    pbar = tqdm(
        loader,
        total=total,
        desc=task.name,
        mininterval=5.0,
        maxinterval=50.0,
    )
    is_abp = int(task.target_slot) == int(Slot.ABP)
    for batch_idx, batch in enumerate(pbar):
        signal = batch[0].to(device)
        sbp_dbp_true = batch[1].to(device) if len(batch) > 1 else None
        demographics = batch[2].to(device) if len(batch) > 2 else None
        target = signal[:, int(task.target_slot):int(task.target_slot) + 1, :]
        pred = euler_sample(rf_model, signal, task, n_steps=n_steps, device=device)

        if is_abp and bp_head is not None and sbp_dbp_true is not None:
            bp_pred = bp_head(signal[:, :2, :], demographics).float()
            # reconstruct_mmHg expects SBP/DBP in mmHg; inverse if labels are
            # in normalized space.
            if bp_norm is not None:
                bp_pred = bp_norm.denormalize(bp_pred)
                sbp_dbp_true = bp_norm.denormalize(sbp_dbp_true)
            sbp_p, dbp_p = bp_pred[:, 0], bp_pred[:, 1]
            sbp_t, dbp_t = sbp_dbp_true[:, 0], sbp_dbp_true[:, 1]
            pred = reconstruct_mmHg(pred.squeeze(1), sbp_p, dbp_p).unsqueeze(1)
            target = reconstruct_mmHg(target.squeeze(1), sbp_t, dbp_t).unsqueeze(1)
        preds.append(pred.cpu().numpy())
        targets.append(target.cpu().numpy())
        if limit_batches is not None and (batch_idx + 1) >= limit_batches:
            break
    preds_np = np.concatenate(preds, axis=0)
    targets_np = np.concatenate(targets, axis=0)
    return {
        "rmse": rmse(preds_np, targets_np),
        "mae": mae(preds_np, targets_np),
        "pearson": pearson_corr(preds_np, targets_np),
        "ks": ks_statistic(preds_np, targets_np),
        "n": int(preds_np.shape[0]),
    }


@hydra.main(
    config_path="../conf",
    config_name="config",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if "checkpoint" not in cfg:
        raise ValueError("Pass +checkpoint=path/to/best.pt to evaluate.py.")

    set_seed(int(cfg.seed), deterministic=bool(cfg.deterministic))
    device = _resolve_device(str(cfg.device))
    logger.info("Using device: %s", device)

    _, _, test_loader = build_loaders(cfg.data)

    rf_model = UniCardioRF(cfg.model)
    load_checkpoint(cfg.checkpoint, model=rf_model, map_location=device)
    rf_model.to(device).eval()

    bp_head_ckpt = cfg.get("bp_head_checkpoint", None)
    bp_head = None
    if bp_head_ckpt is not None:
        bp_head = load_bp_head_ckpt(str(bp_head_ckpt), device)
        logger.info("BPHead loaded from %s; ABP-target metrics in mmHg.", bp_head_ckpt)
    else:
        logger.warning(
            "No +bp_head_checkpoint provided; ABP-target metrics stay in "
            "shape-only [0, 1] space (Pearson still valid; RMSE/MAE unit-less)."
        )

    n_steps = int(cfg.sampler.n_steps)
    limit_batches = cfg.get("eval", {}).get("limit_batches", None)
    if limit_batches is not None:
        limit_batches = int(limit_batches)

    active_tasks = [spec for spec, _ in active_task_pairs(cfg.trainer.task_weights)]
    logger.info("Active tasks: %s", [t.name for t in active_tasks])

    bp_norm = BPLabelNorm.from_cfg(cfg.data)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "per_task_metrics.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "rmse", "mae", "pearson", "ks", "n"])
        for task in active_tasks:
            logger.info("Evaluating task %s", task.name)
            metrics = _eval_task(
                rf_model, bp_head, test_loader, task,
                device=device, n_steps=n_steps, limit_batches=limit_batches,
                bp_norm=bp_norm,
            )
            writer.writerow([
                task.name,
                metrics["rmse"], metrics["mae"],
                metrics["pearson"], metrics["ks"], metrics["n"],
            ])
            logger.info(
                "%s | rmse=%.4f | mae=%.4f | pearson=%.4f | ks=%.4f | n=%d",
                task.name,
                metrics["rmse"], metrics["mae"],
                metrics["pearson"], metrics["ks"], metrics["n"],
            )
    logger.info("Wrote %s", out_csv)


if __name__ == "__main__":
    main()

"""评估入口：加载 checkpoint 并对全部 5 个任务打分。

示例：
    python run/pipeline/evaluate.py \
        +checkpoint=run/outputs/2026-04-20_21-00-00/checkpoints/best.pt \
        device=cpu data.num_workers=0 eval.limit_batches=4
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data_module.datamodule import build_loaders
from src.model_module.tasks import Slot, active_task_pairs
from src.model_module.unicardio_rf import UniCardioRF
from src.trainer_module.sampler import euler_sample
from src.utils.bp_metrics import bp_errors
from src.utils.checkpoint import load_checkpoint
from src.utils.metrics import ks_statistic, mae, pearson_corr, rmse
from src.utils.normalization import bp_denormalize
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def _resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _maybe_denormalize(
    tensor: torch.Tensor, target_slot: Slot
) -> torch.Tensor:
    """对 ABP 做 BP 反归一化，使指标以物理单位（mmHg）给出。"""
    if target_slot == Slot.ABP:
        return bp_denormalize(tensor)
    return tensor


@torch.no_grad()
def _eval_task(
    model: torch.nn.Module,
    loader,
    task,
    *,
    device: torch.device,
    n_steps: int,
    limit_batches: int | None,
    srate: int,
) -> dict[str, float]:
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    total = limit_batches if limit_batches is not None else len(loader)
    # mininterval=5s 避免刷屏；leave=True 让最后一行进度条留在日志里；
    # 同时每 ~10% 主动 logger.info 一行，便于 `tee` 到文件后仍能看到推进。
    log_every = max(1, total // 10)
    test_iter = tqdm(
        loader,
        desc=f"test {task.name}",
        mininterval=5.0,
        maxinterval=50.0,
        total=total,
        leave=True,
    )
    for batch_idx, batch in enumerate(test_iter):
        signal = batch[0].to(device)
        target = signal[:, task.target_slot:task.target_slot + 1, :]
        pred = euler_sample(model, signal, task, n_steps=n_steps, device=device)
        pred = _maybe_denormalize(pred, task.target_slot)
        target = _maybe_denormalize(target, task.target_slot)
        preds.append(pred.cpu().numpy())
        targets.append(target.cpu().numpy())
        done = batch_idx + 1
        if done % log_every == 0 or done == total:
            logger.info(
                "task=%s batch=%d/%d (%.1f%%)",
                task.name, done, total, 100.0 * done / total,
            )
        if limit_batches is not None and done >= limit_batches:
            break
    preds_np = np.concatenate(preds, axis=0)
    targets_np = np.concatenate(targets, axis=0)
    metrics: dict[str, float] = {
        "rmse": rmse(preds_np, targets_np),
        "mae": mae(preds_np, targets_np),
        "pearson": pearson_corr(preds_np, targets_np),
        "ks": ks_statistic(preds_np, targets_np),
        "n": int(preds_np.shape[0]),
    }
    # 仅 ABP 目标任务额外计算 SBP/DBP 误差（pyvital 经典 DSP 峰检测）。
    if task.target_slot == Slot.ABP:
        metrics.update(bp_errors(preds_np, targets_np, srate=srate))
    return metrics


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

    # 必须通过 +checkpoint=... 显式传入 checkpoint 路径。
    if "checkpoint" not in cfg:
        raise ValueError(
            "Pass +checkpoint=path/to/best.pt to evaluate.py."
        )

    set_seed(int(cfg.seed), deterministic=bool(cfg.deterministic))
    device = _resolve_device(str(cfg.device))
    logger.info("Using device: %s", device)

    _, _, test_loader = build_loaders(cfg.data)

    model = UniCardioRF(cfg.model)
    load_checkpoint(cfg.checkpoint, model=model, map_location=device)
    model.to(device).eval()

    n_steps = int(cfg.sampler.n_steps)
    limit_batches = cfg.get("eval", {}).get("limit_batches", None)
    if limit_batches is not None:
        limit_batches = int(limit_batches)
    srate = int(cfg.data.srate)

    # 与训练保持一致：权重为 0 的任务不参与评估。
    active_tasks = [spec for spec, _ in active_task_pairs(
        cfg.trainer.task_weights
    )]
    logger.info("Active tasks: %s", [t.name for t in active_tasks])

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "per_task_metrics.csv"
    # (csv_column, metrics_key) — n_valid/n_total 加 bp_ 前缀以避免与已存在的 'n' 列混淆。
    bp_fields: list[tuple[str, str]] = [
        ("sbp_mae", "sbp_mae"),
        ("dbp_mae", "dbp_mae"),
        ("sbp_me", "sbp_me"),
        ("dbp_me", "dbp_me"),
        ("sbp_std", "sbp_std"),
        ("dbp_std", "dbp_std"),
        ("n_bp_valid", "n_valid"),
        ("n_bp_total", "n_total"),
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task", "rmse", "mae", "pearson", "ks", "n",
            *(c for c, _ in bp_fields),
        ])
        task_iter = tqdm(active_tasks, desc="tasks", total=len(active_tasks), leave=True)
        for ti, task in enumerate(task_iter, 1):
            logger.info("[%d/%d] Evaluating task %s", ti, len(active_tasks), task.name)
            metrics = _eval_task(
                model,
                test_loader,
                task,
                device=device,
                n_steps=n_steps,
                limit_batches=limit_batches,
                srate=srate,
            )
            row = [
                task.name,
                metrics["rmse"],
                metrics["mae"],
                metrics["pearson"],
                metrics["ks"],
                metrics["n"],
                *(metrics.get(k, "") for _, k in bp_fields),
            ]
            writer.writerow(row)
            logger.info(
                "%s | rmse=%.4f | mae=%.4f | pearson=%.4f | ks=%.4f | n=%d",
                task.name,
                metrics["rmse"],
                metrics["mae"],
                metrics["pearson"],
                metrics["ks"],
                metrics["n"],
            )
            if "sbp_mae" in metrics:
                logger.info(
                    "%s | SBP MAE=%.2f ± %.2f mmHg (ME=%+.2f) | "
                    "DBP MAE=%.2f ± %.2f mmHg (ME=%+.2f) | bp_n=%d/%d",
                    task.name,
                    metrics["sbp_mae"], metrics["sbp_std"], metrics["sbp_me"],
                    metrics["dbp_mae"], metrics["dbp_std"], metrics["dbp_me"],
                    metrics["n_valid"], metrics["n_total"],
                )
    logger.info("Wrote %s", out_csv)


if __name__ == "__main__":
    main()

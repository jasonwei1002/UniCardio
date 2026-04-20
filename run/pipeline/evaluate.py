"""Evaluation entrypoint: load a checkpoint and score all 5 tasks.

Example:
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

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data_module.datamodule import build_loaders
from src.model_module.tasks import TASK_LIST, Slot
from src.model_module.unicardio_rf import UniCardioRF
from src.trainer_module.sampler import euler_sample
from src.utils.checkpoint import load_checkpoint
from src.utils.metrics import ks_statistic, mae, rmse
from src.utils.normalization import bp_denormalize
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def _resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _maybe_denormalize(
    tensor: torch.Tensor, target_slot: int
) -> torch.Tensor:
    """Undo BP normalization so ABP metrics are in physical units (mmHg)."""
    if target_slot == int(Slot.ABP):
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
) -> dict[str, float]:
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for batch_idx, batch in enumerate(loader):
        signal = batch[0].to(device)
        target = signal[:, int(task.target_slot):int(task.target_slot) + 1, :]
        pred = euler_sample(model, signal, task, n_steps=n_steps, device=device)
        pred = _maybe_denormalize(pred, int(task.target_slot))
        target = _maybe_denormalize(target, int(task.target_slot))
        preds.append(pred.cpu().numpy())
        targets.append(target.cpu().numpy())
        if limit_batches is not None and (batch_idx + 1) >= limit_batches:
            break
    preds_np = np.concatenate(preds, axis=0)
    targets_np = np.concatenate(targets, axis=0)
    return {
        "rmse": rmse(preds_np, targets_np),
        "mae": mae(preds_np, targets_np),
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

    out_dir = Path(cfg.output_dir) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "per_task_metrics.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "rmse", "mae", "ks", "n"])
        for task in TASK_LIST:
            logger.info("Evaluating task %s", task.name)
            metrics = _eval_task(
                model,
                test_loader,
                task,
                device=device,
                n_steps=n_steps,
                limit_batches=limit_batches,
            )
            writer.writerow(
                [task.name, metrics["rmse"], metrics["mae"], metrics["ks"], metrics["n"]]
            )
            logger.info(
                "%s | rmse=%.4f | mae=%.4f | ks=%.4f | n=%d",
                task.name,
                metrics["rmse"],
                metrics["mae"],
                metrics["ks"],
                metrics["n"],
            )
    logger.info("Wrote %s", out_csv)


if __name__ == "__main__":
    main()

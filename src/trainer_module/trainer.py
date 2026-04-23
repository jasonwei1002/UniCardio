"""训练循环：按权重采样任务、Adam + MultiStepLR、按 epoch 保存 checkpoint。

每个 batch 进行一次优化器更新：先从 ``TASK_LIST`` 中按权重随机采样一个任务，
再对该任务调用 :func:`rf_train_step`。每个 epoch 结束时把按任务的滑动均值、
学习率、epoch 耗时写入 CSV。验证（每 ``val_every`` 个 epoch 触发一次）会在
验证集上计算 5 个任务的平均 RF 损失，并以此维护 ``best.pt``。
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import swanlab
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import (
    LinearLR,
    MultiStepLR,
    SequentialLR,
    _LRScheduler,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..model_module.tasks import TASK_LIST, TaskSpec
from ..utils.checkpoint import load_checkpoint, save_checkpoint
from .csv_logger import SimpleCSVLogger
from .rectified_flow import rf_train_step

logger = logging.getLogger(__name__)

def _amp_enabled(cfg: Mapping[str, Any], device: torch.device) -> bool:
    """AMP 仅在 CUDA 上启用，始终使用 bfloat16（无需 GradScaler）。"""
    amp_cfg = cfg.get("amp", {}) or {}
    return bool(amp_cfg.get("enabled", True)) and device.type == "cuda"


def _build_optimizer(model: nn.Module, cfg: Mapping[str, Any]) -> Optimizer:
    return Adam(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 1.0e-6)),
    )


def _build_scheduler(
    optimizer: Optimizer,
    cfg: Mapping[str, Any],
    total_epochs: int,
    steps_per_epoch: int,
) -> _LRScheduler:
    """构造 step 级调度器：可选线性 warmup + MultiStepLR。

    与旧实现的差异：整个调度器按 **step** 计数（而非 epoch），trainer 每个
    training step 后调一次 ``scheduler.step()``。这让我们可以用 PyTorch 官方
    的 ``SequentialLR`` 把 ``LinearLR`` warmup 和 ``MultiStepLR`` 主干衔接起来，
    不需要在 trainer 里按阶段切换调度粒度。

    ``milestones_pct`` 的语义保持"占总训练进度的比例"（因为
    ``total_steps = epochs * steps_per_epoch``，占比解释不变），只是换算时
    用的是总 step 数。
    """
    sched_cfg = cfg["lr_scheduler"]
    name = str(sched_cfg["name"]).lower()
    if name != "multistep":
        raise ValueError(f"Unsupported lr_scheduler '{name}'.")

    total_steps = max(1, int(total_epochs) * int(steps_per_epoch))
    milestones_steps = [
        max(1, int(round(total_steps * float(p))))
        for p in sched_cfg["milestones_pct"]
    ]
    gamma = float(sched_cfg["gamma"])

    warmup_steps = int(cfg.get("warmup_steps", 0))
    main = MultiStepLR(optimizer, milestones=milestones_steps, gamma=gamma)
    if warmup_steps <= 0:
        return main

    # LinearLR 从 start_factor 线性升到 1.0，持续 total_iters 步。
    # start_factor 取 1/warmup_steps 以避免 LR=0 的第一步（Adam 的数值稳定性）。
    warmup = LinearLR(
        optimizer,
        start_factor=1.0 / float(warmup_steps),
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    return SequentialLR(
        optimizer, schedulers=[warmup, main], milestones=[warmup_steps]
    )


def _weighted_task_sampler(
    weights_cfg: Mapping[str, float] | None,
) -> list[tuple[TaskSpec, float]]:
    weights_cfg = weights_cfg or {}
    pairs: list[tuple[TaskSpec, float]] = []
    for spec in TASK_LIST:
        w = float(weights_cfg.get(spec.name, 1.0))
        if w < 0:
            raise ValueError(f"Negative task weight for {spec.name}: {w}")
        pairs.append((spec, w))
    total = sum(w for _, w in pairs)
    if total <= 0:
        raise ValueError("All task weights are zero.")
    return pairs


def _sample_task(pairs: Sequence[tuple[TaskSpec, float]]) -> TaskSpec:
    tasks = [t for t, _ in pairs]
    weights = [w for _, w in pairs]
    return random.choices(tasks, weights=weights, k=1)[0]


def _csv_fields() -> list[str]:
    fields = ["epoch", "lr", "epoch_time_s", "avg_loss"]
    fields.extend(f"loss_{t.name}" for t in TASK_LIST)
    fields.extend(f"val_loss_{t.name}" for t in TASK_LIST)
    fields.append("val_loss_mean")
    return fields


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    *,
    t_mean: float,
    t_std: float,
    amp_enabled: bool = False,
    max_batches: int | None = None,
) -> dict[str, float]:
    """在验证集上按任务计算平均 RF loss。"""
    model.eval()
    sums = {t.name: 0.0 for t in TASK_LIST}
    counts = {t.name: 0 for t in TASK_LIST}
    for batch_idx, batch in enumerate(val_loader):
        signal = batch[0].to(device)
        for task in TASK_LIST:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                loss = rf_train_step(
                    model, signal, task, t_mean=t_mean, t_std=t_std
                )
            sums[task.name] += float(loss.item())
            counts[task.name] += 1
        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break
    model.train()
    return {
        name: (sums[name] / counts[name]) if counts[name] else float("nan")
        for name in sums
    }


def train(
    model: nn.Module,
    cfg: DictConfig | Mapping[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    *,
    device: torch.device,
    output_dir: str | Path,
) -> None:
    """执行完整的 Rectified Flow 训练循环。

    Args:
        model: :class:`UniCardioRF`（或由 :class:`nn.DataParallel` 等包裹过的版本）。
        cfg: 训练器配置（OmegaConf DictConfig 或普通 dict 均可）。
        train_loader: 产出 ``(signal,)`` 的 DataLoader，signal 形状 ``(B, 3, L)``。
        val_loader: 可选的验证 loader，需满足相同的输出契约。
        device: PyTorch device。
        output_dir: 本次运行的输出目录（由 Hydra 创建）。
    """
    if isinstance(cfg, DictConfig):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    else:
        cfg_dict = dict(cfg)

    output_dir = Path(output_dir)
    ckpt_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(cfg_dict["epochs"])
    val_every = int(cfg_dict.get("val_every", 10))
    ckpt_every = int(cfg_dict.get("ckpt_every", 1))
    itr_per_epoch = cfg_dict.get("itr_per_epoch")

    optimizer = _build_optimizer(model, cfg_dict)
    steps_per_epoch = len(train_loader) if itr_per_epoch is None else int(itr_per_epoch)
    scheduler = _build_scheduler(
        optimizer, cfg_dict, epochs, steps_per_epoch=steps_per_epoch
    )
    t_sampler_cfg = cfg_dict.get("t_sampler", {})
    t_mean = float(t_sampler_cfg.get("mean", 0.0))
    t_std = float(t_sampler_cfg.get("std", 1.0))

    amp_enabled = _amp_enabled(cfg_dict, device)
    logger.info("AMP: enabled=%s dtype=bfloat16", amp_enabled)

    task_pairs = _weighted_task_sampler(cfg_dict.get("task_weights"))

    csv_logger = SimpleCSVLogger(
        log_dir / cfg_dict.get("log_filename", "loss.csv"),
        fieldnames=_csv_fields(),
    )

    start_epoch = 0
    best_val = float("inf")
    resume_from = cfg_dict.get("resume_from")
    if resume_from:
        payload = load_checkpoint(
            resume_from,
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            map_location=device,
        )
        start_epoch = int(payload.get("epoch", 0)) + 1
        logger.info("Resuming from epoch %d", start_epoch)

    model.train()
    model.to(device)

    global_step = 0
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        task_loss_sum = {t.name: 0.0 for t in TASK_LIST}
        task_loss_count = {t.name: 0 for t in TASK_LIST}
        total_loss = 0.0
        total_batches = 0

        it = tqdm(
            train_loader,
            mininterval=5.0,
            maxinterval=50.0,
            disable=not (hasattr(tqdm, "_instances") and True),
            desc=f"epoch {epoch}/{epochs - 1}",
        )
        for batch_idx, batch in enumerate(it, start=1):
            signal = batch[0].to(device, non_blocking=True)
            task = _sample_task(task_pairs)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                loss = rf_train_step(
                    model, signal, task, t_mean=t_mean, t_std=t_std
                )
            loss.backward()
            optimizer.step()
            scheduler.step()

            val = float(loss.item())
            task_loss_sum[task.name] += val
            task_loss_count[task.name] += 1
            total_loss += val
            total_batches += 1

            swanlab.log(
                {
                    "train/loss": val,
                    f"train/loss_{task.name}": val,
                    "train/lr": float(optimizer.param_groups[0]["lr"]),
                },
                step=global_step,
            )
            global_step += 1

            if itr_per_epoch is not None and batch_idx >= int(itr_per_epoch):
                break

        epoch_time = time.time() - epoch_start
        avg = total_loss / max(total_batches, 1)
        per_task = {
            f"loss_{name}": (task_loss_sum[name] / task_loss_count[name])
            if task_loss_count[name] > 0
            else float("nan")
            for name in task_loss_sum
        }
        lr_val = float(scheduler.get_last_lr()[0])
        logger.info(
            "epoch %d/%d | avg_loss %.6f | lr %.2e | %.1fs",
            epoch,
            epochs - 1,
            avg,
            lr_val,
            epoch_time,
        )

        row: dict[str, Any] = {
            "epoch": epoch,
            "lr": lr_val,
            "epoch_time_s": round(epoch_time, 2),
            "avg_loss": avg,
            **per_task,
        }

        if val_loader is not None and (epoch + 1) % val_every == 0:
            val_losses = _evaluate(
                model, val_loader, device,
                t_mean=t_mean, t_std=t_std,
                amp_enabled=amp_enabled,
            )
            for name, v in val_losses.items():
                row[f"val_loss_{name}"] = v
            mean_val = sum(val_losses.values()) / len(val_losses)
            row["val_loss_mean"] = mean_val
            logger.info("  val_loss_mean %.6f", mean_val)
            if mean_val < best_val:
                best_val = mean_val
                save_checkpoint(
                    ckpt_dir / "best.pt",
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=scheduler,
                    config=cfg_dict,
                    task_list=[t.name for t in TASK_LIST],
                    extra={"val_loss_mean": mean_val},
                )

        csv_logger.log_mapping(row)
        # 用 epoch/、val/ 前缀，避免与 batch 级 train/loss 共享 step 轴时相互覆盖。
        epoch_metrics: dict[str, float] = {
            "epoch/avg_loss": avg,
            "epoch/lr": lr_val,
            "epoch/time_s": epoch_time,
            **{f"epoch/loss_{name}": per_task[f"loss_{name}"] for name in task_loss_sum},
        }
        if "val_loss_mean" in row:
            epoch_metrics["val/loss_mean"] = row["val_loss_mean"]
            for t in TASK_LIST:
                epoch_metrics[f"val/loss_{t.name}"] = row[f"val_loss_{t.name}"]
        swanlab.log(epoch_metrics, step=global_step)

        if (epoch + 1) % ckpt_every == 0:
            save_checkpoint(
                ckpt_dir / "latest.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                lr_scheduler=scheduler,
                config=cfg_dict,
                task_list=[t.name for t in TASK_LIST],
            )

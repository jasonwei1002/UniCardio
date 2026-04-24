"""训练循环：按权重采样任务、Adam + cosine annealing (含 warmup)、按 epoch 保存 checkpoint。

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
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from ..model_module.tasks import TASK_LIST, TaskSpec, active_task_pairs
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
    total_steps: int,
) -> _LRScheduler:
    """构造 step 级调度器：线性 warmup + cosine 退火（无 restart）。

    使用 ``cosine_annealing_warmup.CosineAnnealingWarmupRestarts``（自带
    warmup）。把 ``first_cycle_steps = total_steps`` 让整个训练留在第一个
    cycle 内，就实现了"no restart"：warmup 阶段 lr 从 ``min_lr`` 线性升到
    ``max_lr = cfg.lr``，之后剩余步数从 ``max_lr`` cosine 退火到 ``min_lr``。
    整套调度按 step 推进，trainer 每个 step 后 ``scheduler.step()`` 一次。

    Warmup 长度用 ``cfg.warmup_pct`` 配置，填 [0, 1) 的小数表示占总训练的
    比例（例如 ``0.05`` 即 5%）。真正的 step 数在这里实时换算：
    ``warmup_steps = round(warmup_pct * total_steps)``，这样 epochs / batch /
    数据量变了，warmup 占比仍然保持不变。
    """
    sched_cfg = cfg["lr_scheduler"]
    name = str(sched_cfg["name"]).lower()
    if name != "cosine":
        raise ValueError(f"Unsupported lr_scheduler '{name}'.")

    warmup_pct = float(cfg.get("warmup_pct", 0.0))
    if not 0.0 <= warmup_pct < 1.0:
        raise ValueError(
            f"warmup_pct must be in [0, 1); got {warmup_pct}"
        )
    # CosineAnnealingWarmupRestarts 内部 assert warmup_steps < first_cycle_steps；
    # round 在 warmup_pct → 1.0 时会把结果推到 total_steps 触发断言，在这里截断。
    warmup_steps = min(int(round(warmup_pct * total_steps)), total_steps - 1)
    return CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=total_steps,
        max_lr=float(cfg["lr"]),
        min_lr=float(sched_cfg.get("min_lr", 0.0)),
        warmup_steps=warmup_steps,
    )


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
    tasks: Sequence[TaskSpec] = TASK_LIST,
) -> dict[str, float]:
    """在验证集上按任务计算平均 RF loss；``tasks`` 指定评估集合。"""
    model.eval()
    sums = {t.name: 0.0 for t in tasks}
    counts = {t.name: 0 for t in tasks}
    for batch_idx, batch in enumerate(val_loader):
        signal = batch[0].to(device)
        for task in tasks:
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
    scheduler = _build_scheduler(optimizer, cfg_dict, epochs * steps_per_epoch)
    t_sampler_cfg = cfg_dict.get("t_sampler", {})
    t_mean = float(t_sampler_cfg.get("mean", 0.0))
    t_std = float(t_sampler_cfg.get("std", 1.0))

    amp_enabled = _amp_enabled(cfg_dict, device)
    logger.info("AMP: enabled=%s dtype=bfloat16", amp_enabled)

    # 梯度范数裁剪：防 Adam 第二动量累积后被单个 outlier batch 爆出 1e3 级 step。
    # ≤0 或 None 都关闭裁剪，交给调用方兜底。
    grad_clip_norm = cfg_dict.get("grad_clip_norm")
    grad_clip_norm = float(grad_clip_norm) if grad_clip_norm else 0.0
    if grad_clip_norm > 0:
        logger.info("Grad clip: max_norm=%.2f", grad_clip_norm)

    task_pairs = active_task_pairs(cfg_dict.get("task_weights"))
    active_tasks: list[TaskSpec] = [spec for spec, _ in task_pairs]
    logger.info("Training tasks: %s", [t.name for t in active_tasks])

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
        task_loss_sum = {t.name: 0.0 for t in active_tasks}
        task_loss_count = {t.name: 0 for t in active_tasks}
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
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip_norm
                )
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
        # 直接读 optimizer 而不是 scheduler.get_last_lr()：CosineAnnealingWarmupRestarts
        # 在 torch 2.7+ 下 __init__ 不一定给 _last_lr 赋值，访问会 AttributeError。
        lr_val = float(optimizer.param_groups[0]["lr"])
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
                tasks=active_tasks,
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
                    task_list=[t.name for t in active_tasks],
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
            for t in active_tasks:
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
                task_list=[t.name for t in active_tasks],
            )

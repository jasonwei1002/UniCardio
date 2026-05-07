"""训练循环：按权重采样任务、Adam + cosine annealing (含 warmup)、按 epoch 保存 checkpoint。

每个 batch 进行一次优化器更新：先从 ``TASK_LIST`` 中按权重随机采样一个任务，
再按 ``cfg.trainer.objective`` 调用 :func:`rf_train_step`（``rf``）或
:func:`regression_train_step`（``regression``）。每个 epoch 结束时把按任务的
滑动均值、学习率、epoch 耗时写入 CSV。验证（每 ``val_every`` 个 epoch 触发一次）
会在验证集上计算各任务的平均 loss，并以此维护 ``best.pt``。
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
from torch import nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from ..model_module.tasks import TASK_LIST, TaskSpec, active_task_pairs
from ..utils.checkpoint import save_checkpoint
from .csv_logger import SimpleCSVLogger
from .rectified_flow import rf_loss_at_fixed_t, rf_train_step
from .regression import regression_sample, regression_train_step
from .sampler import euler_sample

logger = logging.getLogger(__name__)


def _resolve_step_fn(objective: str):
    """根据 ``objective`` 选 RF / regression 的训练 step。"""
    name = str(objective).lower()
    if name == "rf":
        return rf_train_step, True   # 第二个返回值：是否需要 t_mean/t_std
    if name == "regression":
        return regression_train_step, False
    raise ValueError(
        f"Unknown trainer.objective '{objective}'; expected 'rf' or 'regression'."
    )


def _call_step(step_fn, needs_t, model, signal, task, *, t_mean, t_std):
    """统一调用入口；regression 路径忽略 t_mean / t_std。"""
    if needs_t:
        return step_fn(model, signal, task, t_mean=t_mean, t_std=t_std)
    return step_fn(model, signal, task)

def _amp_enabled(cfg: Mapping[str, Any], device: torch.device) -> bool:
    """AMP 仅在 CUDA 上启用，始终使用 bfloat16（无需 GradScaler）。"""
    amp_cfg = cfg.get("amp", {}) or {}
    return bool(amp_cfg.get("enabled", True)) and device.type == "cuda"


def _build_optimizer(model: nn.Module, cfg: Mapping[str, Any]) -> Optimizer:
    # fused=True 走 CUDA 的多张量融合路径，单 step 比逐参数版本省可观的 launch 开销；
    # CPU 上不支持，退回默认实现。
    fused = torch.cuda.is_available() and any(
        p.is_cuda for p in model.parameters()
    )
    return Adam(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 1.0e-6)),
        fused=fused,
    )


def _build_scheduler(
    optimizer: Optimizer,
    cfg: Mapping[str, Any],
    total_steps: int,
) -> _LRScheduler:

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
    fields.extend(f"val_recon_mse_{t.name}" for t in TASK_LIST)
    fields.append("val_recon_mse_mean")
    return fields


@torch.inference_mode()
def _evaluate_t_bins(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    *,
    t_values: Sequence[float],
    amp_enabled: bool,
    max_batches: int,
    tasks: Sequence[TaskSpec],
) -> dict[str, float]:
    """RF 专用：把 ``t`` 固定在若干 bin 中心，看 plateau 由哪段 ``t`` 贡献。

    返回 ``{"t={t_value}/{task}": loss}`` 字典，便于直接转 swanlab metrics。
    """
    model.eval()
    sums: dict[str, torch.Tensor] = {
        f"t={t_v:.1f}/{task.name}": torch.zeros((), device=device)
        for t_v in t_values for task in tasks
    }
    counts: dict[str, int] = {k: 0 for k in sums}
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= max_batches:
            break
        signal = batch[0].to(device, non_blocking=True)
        for t_v in t_values:
            for task in tasks:
                key = f"t={t_v:.1f}/{task.name}"
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=amp_enabled,
                ):
                    loss = rf_loss_at_fixed_t(model, signal, task, float(t_v))
                sums[key] += loss
                counts[key] += 1
    model.train()
    return {
        k: (sums[k].item() / counts[k]) if counts[k] else float("nan")
        for k in sums
    }


@torch.inference_mode()
def _evaluate_recon(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    *,
    objective: str,
    n_steps: int,
    max_batches: int,
    amp_enabled: bool,
    tasks: Sequence[TaskSpec],
) -> dict[str, float]:
    """跨 objective 统一的端到端重建 MSE：调各自 sampler 拿 pred，再和 target 求 MSE。

    这是真正可比的指标——不依赖 ``t``、不依赖 v / x1 反推：
      - RF objective:        ``euler_sample`` 跑 ``n_steps`` 步 Euler ODE。
      - regression objective: ``regression_sample`` 单 forward。
    返回每任务的 mean MSE，单位是模型空间（BP 已除 50）。
    """
    model.eval()
    sums: dict[str, torch.Tensor] = {
        t.name: torch.zeros((), device=device) for t in tasks
    }
    counts: dict[str, int] = {t.name: 0 for t in tasks}
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= max_batches:
            break
        signal = batch[0].to(device, non_blocking=True)
        for task in tasks:
            target = signal[:, int(task.target_slot):int(task.target_slot) + 1, :]
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                if objective == "regression":
                    pred = regression_sample(model, signal, task, device=device)
                else:
                    pred = euler_sample(
                        model, signal, task, n_steps=n_steps, device=device
                    )
            sums[task.name] += (pred.float() - target.float()).pow(2).mean()
            counts[task.name] += 1
    model.train()
    return {
        name: (sums[name].item() / counts[name]) if counts[name] else float("nan")
        for name in sums
    }


@torch.inference_mode()
def _evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    *,
    step_fn,
    needs_t: bool,
    t_mean: float,
    t_std: float,
    amp_enabled: bool = False,
    max_batches: int | None = None,
    tasks: Sequence[TaskSpec] = TASK_LIST,
) -> dict[str, float]:
    """在验证集上按任务计算平均 loss；``step_fn`` 与训练侧保持一致。"""
    model.eval()
    sums = {t.name: torch.zeros((), device=device) for t in tasks}
    counts = {t.name: 0 for t in tasks}
    total = max_batches if max_batches is not None else len(val_loader)
    val_iter = tqdm(
        val_loader,
        desc="val",
        mininterval=5.0,
        maxinterval=50.0,
        total=total,
        leave=False,
    )
    for batch_idx, batch in enumerate(val_iter):
        signal = batch[0].to(device, non_blocking=True)
        for task in tasks:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                loss = _call_step(
                    step_fn, needs_t, model, signal, task,
                    t_mean=t_mean, t_std=t_std,
                )
            sums[task.name] += loss
            counts[task.name] += 1
        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break
    model.train()
    return {
        name: (sums[name].item() / counts[name]) if counts[name] else float("nan")
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
    objective = str(cfg_dict.get("objective", "rf"))
    step_fn, needs_t = _resolve_step_fn(objective)
    logger.info("Training objective: %s", objective)

    # RF-only：t-bin 验证诊断（plateau 是否由 t≈0 / t≈1 段贡献）。
    t_bins_cfg = cfg_dict.get("t_bin_diagnostic", {}) or {}
    t_bins_enabled = bool(t_bins_cfg.get("enabled", False)) and objective == "rf"
    t_bin_values = list(t_bins_cfg.get("t_values", [0.1, 0.3, 0.5, 0.7, 0.9]))
    t_bin_max_batches = int(t_bins_cfg.get("max_batches", 64))
    t_bin_every = int(t_bins_cfg.get("every", 5))  # 每 N epoch 跑一次诊断
    if t_bins_enabled:
        logger.info(
            "RF t-bin diagnostic enabled: t_values=%s, max_batches=%d, every %d epochs",
            t_bin_values, t_bin_max_batches, t_bin_every,
        )

    # Reconstruction MSE (跨 objective 可比)：每 val 时跑一小批 sampler，对齐 evaluate 阶段输出。
    recon_cfg = cfg_dict.get("recon_eval", {}) or {}
    recon_enabled = bool(recon_cfg.get("enabled", True))
    recon_max_batches = int(recon_cfg.get("max_batches", 8))
    recon_n_steps = int(recon_cfg.get("n_steps", 8))
    if recon_enabled:
        logger.info(
            "recon eval enabled: max_batches=%d, n_steps=%d (only used for RF objective)",
            recon_max_batches, recon_n_steps,
        )

    model.to(device)
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

    model.train()

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
                loss = _call_step(
                    step_fn, needs_t, model, signal, task,
                    t_mean=t_mean, t_std=t_std,
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

        t_bin_metrics: dict[str, float] = {}
        recon_metrics: dict[str, float] = {}
        if val_loader is not None and (epoch + 1) % val_every == 0:
            val_losses = _evaluate(
                model, val_loader, device,
                step_fn=step_fn, needs_t=needs_t,
                t_mean=t_mean, t_std=t_std,
                amp_enabled=amp_enabled,
                tasks=active_tasks,
            )
            if t_bins_enabled and (epoch + 1) % t_bin_every == 0:
                t_bin_metrics = _evaluate_t_bins(
                    model, val_loader, device,
                    t_values=t_bin_values,
                    amp_enabled=amp_enabled,
                    max_batches=t_bin_max_batches,
                    tasks=active_tasks,
                )
                logger.info(
                    "  t-bin loss summary: %s",
                    {k: round(v, 5) for k, v in t_bin_metrics.items()},
                )
            if recon_enabled:
                recon_metrics = _evaluate_recon(
                    model, val_loader, device,
                    objective=objective,
                    n_steps=recon_n_steps,
                    max_batches=recon_max_batches,
                    amp_enabled=amp_enabled,
                    tasks=active_tasks,
                )
                recon_mean = (
                    sum(recon_metrics.values()) / len(recon_metrics)
                    if recon_metrics else float("nan")
                )
                logger.info(
                    "  recon_mse_mean=%.6f | per-task=%s",
                    recon_mean,
                    {k: round(v, 5) for k, v in recon_metrics.items()},
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
        for k, v in t_bin_metrics.items():
            epoch_metrics[f"val_tbin/{k}"] = v
        if recon_metrics:
            recon_mean_val = sum(recon_metrics.values()) / len(recon_metrics)
            epoch_metrics["val/recon_mse_mean"] = recon_mean_val
            for name, v in recon_metrics.items():
                epoch_metrics[f"val/recon_mse_{name}"] = v
            row["val_recon_mse_mean"] = recon_mean_val
            for name, v in recon_metrics.items():
                row[f"val_recon_mse_{name}"] = v
        swanlab.log(epoch_metrics, step=epoch)

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

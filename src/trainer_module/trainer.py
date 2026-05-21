"""Training loop: weighted task sampling, Adam + cosine annealing (optional
warm restart), per-epoch checkpointing and CSV logging."""

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
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    _LRScheduler,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..model_module.tasks import TASK_LIST, TaskSpec, active_task_pairs
from ..utils.checkpoint import load_checkpoint, save_checkpoint, unwrap_model
from .bp_metrics import evaluate_bp_test
from .csv_logger import SimpleCSVLogger
from .rectified_flow import rf_train_step

logger = logging.getLogger(__name__)

def _amp_enabled(cfg: Mapping[str, Any], device: torch.device) -> bool:
    """AMP 仅在 CUDA 上启用，始终使用 bfloat16（无需 GradScaler）。"""
    amp_cfg = cfg.get("amp", {}) or {}
    return bool(amp_cfg.get("enabled", True)) and device.type == "cuda"


def _build_optimizer(model: nn.Module, cfg: Mapping[str, Any]) -> Optimizer:
    # Filter frozen params (finetune stage); fused-Adam needs all-CUDA tensors.
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters; check freeze configuration.")
    fused = torch.cuda.is_available() and any(p.is_cuda for p in trainable)
    return Adam(
        trainable,
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 1.0e-6)),
        fused=fused,
    )


def _build_scheduler(
    optimizer: Optimizer,
    cfg: Mapping[str, Any],
    total_steps: int,
) -> _LRScheduler:
    """Step 级 cosine 调度；``first_cycle_pct < 1.0`` 时启用 warm restart。"""
    sched_cfg = cfg["lr_scheduler"]
    name = str(sched_cfg["name"]).lower()
    if name != "cosine":
        raise ValueError(f"Unsupported lr_scheduler '{name}'.")

    min_lr = float(sched_cfg.get("min_lr", 0.0))
    first_cycle_pct = float(sched_cfg.get("first_cycle_pct", 1.0))
    if not 0.0 < first_cycle_pct <= 1.0:
        raise ValueError(
            f"first_cycle_pct must be in (0, 1]; got {first_cycle_pct}"
        )

    if first_cycle_pct >= 1.0:
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=min_lr)

    cycle_mult = int(sched_cfg.get("cycle_mult", 1))
    if cycle_mult < 1:
        raise ValueError(f"cycle_mult must be a positive int; got {cycle_mult}")
    first_cycle_steps = int(round(first_cycle_pct * total_steps))
    if first_cycle_steps < 2:
        raise ValueError(
            f"first_cycle_pct * total_steps = {first_cycle_steps} < 2; "
            f"increase first_cycle_pct (got {first_cycle_pct}) or total_steps."
        )
    return CosineAnnealingWarmRestarts(
        optimizer,
        T_0=first_cycle_steps,
        T_mult=cycle_mult,
        eta_min=min_lr,
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


def _flush_train_window(
    *,
    task_sums: Mapping[str, Tensor],
    task_counts: Mapping[str, int],
    lr: float,
    step: int,
) -> None:
    """把窗口内的均值用单次 stack→tolist sync 到 CPU，再写 swanlab。"""
    total_count = sum(task_counts.values())
    if total_count <= 0:
        return
    keys: list[str] = ["train/loss"]
    vals: list[Tensor] = [sum(task_sums.values()) / total_count]
    for name, s in task_sums.items():
        c = task_counts[name]
        if c > 0:
            keys.append(f"train/loss_{name}")
            vals.append(s / c)
    cpu_vals = torch.stack(vals).tolist()
    metrics = dict(zip(keys, cpu_vals))
    metrics["train/lr"] = lr
    swanlab.log(metrics, step=step)


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
    test_loader: DataLoader | None = None,
    bp_test_csv: str | Path | None = None,
    sampler_n_steps: int = 8,
) -> None:
    """执行完整的 Rectified Flow 训练循环。

    Args:
        model: :class:`UniCardioRF`（或由 :class:`nn.DataParallel` 等包裹过的版本）。
        cfg: 训练器配置（OmegaConf DictConfig 或普通 dict 均可）。
        train_loader: 产出 ``(signal, sbp_dbp)`` 的 DataLoader（Path A 契约）。
            signal 形状 ``(B, 3, L)``，ABP slot 已 per-sample minmax 到 [0,1]。
            RF 训练只用 signal，忽略 sbp_dbp（后者属于 Stream 2 BP head）。
        val_loader: 可选的验证 loader，需满足相同的输出契约。
        device: PyTorch device。
        output_dir: 本次运行的输出目录（由 Hydra 创建）。
        test_loader: 仅 ``stage='finetune'`` 使用。训练结束后用 best.pt 在该
            split 上跑一次 RF loss 评估并写入 SwanLab + CSV。
        bp_test_csv: 仅 ``stage='finetune'`` 使用。指向与 test_loader 同源的
            ``.csv``（含 ``sbp``、``dbp`` 列）。提供时会在 RF loss 之外，
            额外用 :func:`evaluate_bp_test` 计算每个 ABP-target task 的
            SBP/DBP ME 与 SD（mmHg）。
        sampler_n_steps: BP 评估用的 Euler ODE 步数。
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

    # Move model to device BEFORE building optimizer / loading state. Otherwise
    # optimizer is built over CPU params, and load_checkpoint(..., map_location=device)
    # would land Adam moments on CPU while later model.to(device) moves only params,
    # crashing on the next optimizer.step().
    model.to(device)

    # Stage routing: 阶段二（finetune）必须在 _build_optimizer 之前完成
    # ckpt 加载 + 参数冻结，否则 Adam 会把 frozen 参数也收进 param_groups。
    stage = str(cfg_dict.get("stage", "pretrain"))
    if stage not in ("pretrain", "finetune"):
        raise ValueError(f"trainer.stage must be 'pretrain' or 'finetune'; got {stage!r}")

    init_from = cfg_dict.get("init_from")

    if stage == "finetune":
        if not init_from:
            raise ValueError(
                "trainer.stage='finetune' requires trainer.init_from=<阶段一 ckpt 路径>"
            )
        load_checkpoint(init_from, model=model, map_location=device)
        n_unfrozen = int(cfg_dict.get("finetune", {}).get("n_unfrozen_blocks", 2))
        unwrap_model(model).freeze_for_finetune(n_unfrozen)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        logger.info(
            "Finetune: loaded backbone from %s; %.2fM / %.2fM params trainable "
            "(n_unfrozen_blocks=%d)",
            init_from, n_train / 1e6, n_total / 1e6, n_unfrozen,
        )

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

    log_every_n_steps = max(1, int(cfg_dict.get("log_every_n_steps", 50)))
    logger.info("SwanLab train-curve granularity: every %d steps", log_every_n_steps)

    task_pairs = active_task_pairs(cfg_dict.get("task_weights"))
    active_tasks: list[TaskSpec] = [spec for spec, _ in task_pairs]
    logger.info("Training tasks: %s", [t.name for t in active_tasks])

    csv_logger = SimpleCSVLogger(
        log_dir / cfg_dict.get("log_filename", "loss.csv"),
        fieldnames=_csv_fields(),
    )

    best_val = float("inf")
    if init_from and stage == "pretrain":
        load_checkpoint(init_from, model=model, map_location=device)
        logger.info("Initialized model from %s (fine-tune mode, fresh optim/sched)", init_from)

    model.train()

    global_step = 0
    for epoch in range(epochs):
        epoch_start = time.time()
        task_loss_sum = {t.name: torch.zeros((), device=device) for t in active_tasks}
        task_loss_count = {t.name: 0 for t in active_tasks}
        total_loss = torch.zeros((), device=device)
        total_batches = 0
        window_task_sum = {t.name: torch.zeros((), device=device) for t in active_tasks}
        window_task_count = {t.name: 0 for t in active_tasks}

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

            loss_d = loss.detach()
            total_loss += loss_d
            task_loss_sum[task.name] += loss_d
            task_loss_count[task.name] += 1
            total_batches += 1
            window_task_sum[task.name] += loss_d
            window_task_count[task.name] += 1

            global_step += 1
            if sum(window_task_count.values()) >= log_every_n_steps:
                _flush_train_window(
                    task_sums=window_task_sum,
                    task_counts=window_task_count,
                    lr=float(optimizer.param_groups[0]["lr"]),
                    step=global_step,
                )
                for t in window_task_sum.values():
                    t.zero_()
                window_task_count = dict.fromkeys(window_task_count, 0)

            if itr_per_epoch is not None and batch_idx >= int(itr_per_epoch):
                break

        if sum(window_task_count.values()) > 0:
            _flush_train_window(
                task_sums=window_task_sum,
                task_counts=window_task_count,
                lr=float(optimizer.param_groups[0]["lr"]),
                step=global_step,
            )

        epoch_time = time.time() - epoch_start
        denom = max(total_batches, 1)
        nan_gpu = torch.full((), float("nan"), device=device)
        epoch_tensors: list[Tensor] = [total_loss / denom]
        for t in active_tasks:
            c = task_loss_count[t.name]
            epoch_tensors.append(task_loss_sum[t.name] / c if c > 0 else nan_gpu)
        epoch_vals = torch.stack(epoch_tensors).tolist()
        avg = epoch_vals[0]
        per_task = {
            f"loss_{t.name}": epoch_vals[i + 1] for i, t in enumerate(active_tasks)
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

    if stage == "finetune" and test_loader is not None:
        best_path = ckpt_dir / "best.pt"
        if best_path.exists():
            load_checkpoint(best_path, model=model, map_location=device)
            logger.info("Reloaded best.pt for finetune test evaluation.")
        else:
            # epochs < val_every 时 best.pt 永不会被保存。退化为用最后 step 的
            # 权重评估，并显式 warn 让用户感知（默认 val_every=1 时不会触发）。
            logger.warning(
                "best.pt not found at %s; evaluating with current (last-step) weights. "
                "Increase trainer.val_every coverage if this is unexpected.",
                best_path,
            )
        test_losses = _evaluate(
            model, test_loader, device,
            t_mean=t_mean, t_std=t_std,
            amp_enabled=amp_enabled,
            tasks=active_tasks,
        )
        test_mean = sum(test_losses.values()) / len(test_losses)
        logger.info("Finetune test (RF loss) mean=%.6f per_task=%s", test_mean, test_losses)
        swanlab.log(
            {
                "finetune/test_loss_mean": test_mean,
                **{f"finetune/test_loss_{name}": v for name, v in test_losses.items()},
            }
        )
        csv_logger.log_mapping(
            {
                "epoch": -1,
                "lr": 0.0,
                "epoch_time_s": 0.0,
                "avg_loss": test_mean,
                **{f"loss_{t.name}": float("nan") for t in active_tasks},
                **{f"val_loss_{name}": v for name, v in test_losses.items()},
                "val_loss_mean": test_mean,
            }
        )

        if bp_test_csv is not None:
            bp_head_ckpt = cfg_dict.get("bp_head_ckpt")
            bp_results = evaluate_bp_test(
                model, test_loader,
                tasks=active_tasks,
                csv_path=bp_test_csv,
                n_steps=sampler_n_steps,
                device=device,
                amp_enabled=amp_enabled,
                bp_head_ckpt=bp_head_ckpt,
            )
            bp_swan: dict[str, float] = {}
            for task_name, m in bp_results.items():
                for k, v in m.items():
                    bp_swan[f"finetune/bp_{task_name}_{k}"] = v
            if bp_swan:
                swanlab.log(bp_swan)
            bp_csv_path = log_dir / "finetune_bp_metrics.csv"
            bp_fields = ["task", "n", "sbp_me", "sbp_sd", "dbp_me", "dbp_sd"]
            bp_logger = SimpleCSVLogger(bp_csv_path, fieldnames=bp_fields)
            for task_name, m in bp_results.items():
                bp_logger.log_mapping({
                    "task": task_name,
                    "n": int(m["n"]),
                    "sbp_me": m["sbp_me"], "sbp_sd": m["sbp_sd"],
                    "dbp_me": m["dbp_me"], "dbp_sd": m["dbp_sd"],
                })

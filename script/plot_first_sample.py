"""加载 checkpoint，对 test 集第 0 条样本绘制各 active task 的预测 vs 真值波形，
并额外画出 Euler 采样过程从 t=0 (噪声) 到 t=1 (数据) 的逐步演化。

运行：
    python script/plot_first_sample.py \
        +checkpoint=run/outputs/<run>/checkpoints/best.pt

输出 PNG 落到 ``${output_dir}/``：
    - first_test_sample.png                —— 终态预测 vs 真值（所有 task 叠一张）
    - sampling_trajectory_<task>.png       —— 每个 task 独立的 Euler 轨迹图
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")  # 服务器无 GUI；必须在 pyplot 之前设置后端。

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data_module.datamodule import build_loaders
from src.model_module.tasks import Slot, active_task_pairs
from src.model_module.unicardio_rf import UniCardioRF
from src.trainer_module.sampler import euler_sample
from src.utils.checkpoint import load_checkpoint
from src.utils.normalization import bp_denormalize
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)

_SLOT_NAMES = {Slot.ECG: "ECG", Slot.PPG: "PPG", Slot.ABP: "ABP"}


def _resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _to_physical(tensor: torch.Tensor, target_slot: int) -> np.ndarray:
    """ABP 反归一化到 mmHg；其余 slot 留在归一化空间。"""
    if target_slot == int(Slot.ABP):
        tensor = bp_denormalize(tensor)
    return tensor.detach().cpu().numpy().squeeze()


def _plot_one_trajectory(
    out_path: Path,
    traj_phys: np.ndarray,
    target_np: np.ndarray,
    task,
    *,
    n_steps: int,
    ckpt_name: str,
) -> None:
    """单个 task 的 Euler 轨迹图：3×3 网格，每格画一帧 + target 黑线参考。

    默认 ``n_steps=8`` → 9 帧（含 ``t=0`` 噪声），刚好填满 3×3。其它步数下
    用 ``np.linspace`` 在轨迹里挑 9 个最接近均匀的索引，可视化保持一致。
    所有子图共享 X/Y 轴，方便横向对比每一步与真值的距离。
    """
    target_slot = int(task.target_slot)
    target_name = _SLOT_NAMES[task.target_slot]
    cond_names = "+".join(_SLOT_NAMES[s] for s in task.cond_slots)
    unit = "mmHg" if target_slot == int(Slot.ABP) else "normalized"
    L = target_np.shape[-1]
    x_axis = np.arange(L)
    cmap = plt.get_cmap("viridis")

    n_frames = traj_phys.shape[0]
    grid_n = 9
    if n_frames == grid_n:
        idxs = np.arange(grid_n)
    else:
        idxs = np.linspace(0, n_frames - 1, grid_n).round().astype(int)

    fig, axes = plt.subplots(
        3, 3, figsize=(15, 9), sharex=True, sharey=True, squeeze=False,
    )
    for ax_pos, idx in enumerate(idxs):
        r, c = divmod(ax_pos, 3)
        ax = axes[r, c]
        t = idx / (n_frames - 1)
        ax.plot(x_axis, target_np, color="black", linewidth=1.2, alpha=0.6, label="target")
        ax.plot(x_axis, traj_phys[idx], color=cmap(t), linewidth=1.1, label=f"x_t")
        if idx == 0:
            tag = "  (noise)"
        elif idx == n_frames - 1:
            tag = "  (pred)"
        else:
            tag = ""
        ax.set_title(f"t = {t:.3f}{tag}", fontsize=10)
        ax.grid(alpha=0.3)
        if c == 0:
            ax.set_ylabel(f"{target_name} ({unit})")
        if r == 2:
            ax.set_xlabel("sample index (slot)")

    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        f"{task.name}: {cond_names} → {target_name}    Euler {n_steps}-step    "
        f"checkpoint: {ckpt_name}",
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_path)


@hydra.main(
    config_path="../run/conf",
    config_name="config",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if "checkpoint" not in cfg:
        raise ValueError("Pass +checkpoint=path/to/best.pt to plot_first_sample.py.")

    set_seed(int(cfg.seed), deterministic=bool(cfg.deterministic))
    device = _resolve_device(str(cfg.device))
    logger.info("Using device: %s", device)

    _, _, test_loader = build_loaders(cfg.data, num_workers_override=0)

    # 取 test 集第 0 条样本（DataLoader shuffle=False，确定性）。
    signal = next(iter(test_loader))[0][:1].to(device)  # (1, 3, L)
    L = signal.size(-1)

    model = UniCardioRF(cfg.model)
    load_checkpoint(cfg.checkpoint, model=model, map_location=device)
    model.to(device).eval()

    n_steps = int(cfg.sampler.n_steps)
    active_tasks = [spec for spec, _ in active_task_pairs(cfg.trainer.task_weights)]
    logger.info("Plotting %d active tasks: %s", len(active_tasks), [t.name for t in active_tasks])

    fig, axes = plt.subplots(
        len(active_tasks), 1,
        figsize=(12, 3 * len(active_tasks)),
        sharex=True,
        squeeze=False,
    )
    x_axis = np.arange(L)

    # 缓存每个 task 的 trajectory 用于第二张图，避免再跑一遍 n_steps 次前向。
    trajectories: list[np.ndarray] = []
    targets_np: list[np.ndarray] = []

    for ax, task in zip(axes[:, 0], active_tasks):
        target_slot = int(task.target_slot)
        target = signal[:, target_slot:target_slot + 1, :]
        pred, traj = euler_sample(
            model, signal, task,
            n_steps=n_steps, device=device, return_trajectory=True,
        )   # pred: (1,1,L); traj: (n_steps+1, 1, 1, L)

        target_np = _to_physical(target, target_slot)
        pred_np = _to_physical(pred, target_slot)
        # traj 也走 ABP 反归一化，保证与 target 同单位
        traj_phys = _to_physical(traj.view(-1, 1, L), target_slot).reshape(traj.size(0), -1)

        rmse = float(np.sqrt(np.mean((pred_np - target_np) ** 2)))
        mae = float(np.mean(np.abs(pred_np - target_np)))

        ax.plot(x_axis, target_np, color="black", linewidth=1.4, label="target")
        ax.plot(x_axis, pred_np, color="tab:red", linewidth=1.2, linestyle="--", label="pred")

        cond_names = "+".join(_SLOT_NAMES[s] for s in task.cond_slots)
        target_name = _SLOT_NAMES[task.target_slot]
        unit = "mmHg" if target_slot == int(Slot.ABP) else "normalized"
        ax.set_title(
            f"{task.name}: {cond_names} → {target_name}    "
            f"RMSE={rmse:.4f} {unit}    MAE={mae:.4f} {unit}"
        )
        ax.set_ylabel(f"{target_name} ({unit})")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")

        trajectories.append(traj_phys)
        targets_np.append(target_np)

    axes[-1, 0].set_xlabel("sample index (slot)")
    fig.suptitle(
        f"First test sample — checkpoint: {Path(cfg.checkpoint).name}",
        y=1.0,
    )
    fig.tight_layout()

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "first_test_sample.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("Wrote %s", out_path)
    plt.close(fig)

    for task, traj_phys, target_np in zip(active_tasks, trajectories, targets_np):
        _plot_one_trajectory(
            out_dir / f"sampling_trajectory_{task.name}.png",
            traj_phys,
            target_np,
            task,
            n_steps=n_steps,
            ckpt_name=Path(cfg.checkpoint).name,
        )


if __name__ == "__main__":
    main()

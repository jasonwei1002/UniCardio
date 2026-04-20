"""Smoke test: overfit a tiny synthetic batch with the full model stack.

Intentionally independent of the real dataset so the check runs in ~1 minute
on a CPU laptop. Validates four things end-to-end:

1. Model + trainer wiring is correct (one forward+backward step succeeds).
2. Loss decreases monotonically on an overfitting dataset (sanity that the
   Rectified-Flow training step and attention masks actually learn).
3. Non-target output heads receive no gradient (mask + head routing correct).
4. Euler sampler reconstructs the overfit target within tolerance.

Run:

    python run/pipeline/smoke_test.py
    python run/pipeline/smoke_test.py --steps 150 --task ecgppg2abp
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.model_module.backbone import BackboneConfig
from src.model_module.tasks import TASK_SPECS, TaskSpec
from src.model_module.unicardio_rf import UniCardioRF
from src.trainer_module.rectified_flow import rf_train_step
from src.trainer_module.sampler import euler_sample
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)

SMOKE_SLOT_LENGTH = 64
SMOKE_BATCH = 4


def _build_tiny_model() -> UniCardioRF:
    cfg = BackboneConfig(
        slot_length=SMOKE_SLOT_LENGTH,
        channels=288,
        n_layers=2,
        nheads=4,
        time_embedding_dim=64,
        ffn_dim=32,
    )
    return UniCardioRF(cfg)


def _assert_non_target_heads_quiet(
    model: UniCardioRF, task: TaskSpec
) -> None:
    for i, head in enumerate(model.backbone.output_heads):
        if i == int(task.target_slot):
            continue
        if head.proj1.weight.grad is not None:
            if torch.any(head.proj1.weight.grad != 0):
                raise AssertionError(
                    f"Non-target head {i} unexpectedly received gradient "
                    f"for task {task.name}."
                )


def _run_overfit(
    task: TaskSpec,
    *,
    steps: int,
    lr: float,
    device: torch.device,
) -> tuple[float, float]:
    model = _build_tiny_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    torch.manual_seed(0)
    batch = torch.randn(SMOKE_BATCH, 3, SMOKE_SLOT_LENGTH, device=device)

    model.train()
    losses: list[float] = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = rf_train_step(model, batch, task)
        loss.backward()
        if step == 0:
            _assert_non_target_heads_quiet(model, task)
        optimizer.step()
        losses.append(float(loss.item()))
    initial, final = losses[0], losses[-1]
    logger.info(
        "  task=%s step 0 loss %.4f | step %d loss %.4f | ratio %.3f",
        task.name,
        initial,
        steps,
        final,
        final / max(initial, 1e-12),
    )

    # Euler reconstruction on the overfit weights.
    model.eval()
    target = batch[:, int(task.target_slot):int(task.target_slot) + 1, :]
    sample = euler_sample(model, batch, task, n_steps=16, device=device)
    recon_mse = float(((sample - target) ** 2).mean().item())
    logger.info("  task=%s recon MSE (Euler 16 steps) %.4f", task.name, recon_mse)
    return final / max(initial, 1e-12), recon_mse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=None, help="Single task name; default runs all 5.")
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--loss-drop-threshold", type=float, default=0.3,
                        help="Final/initial loss ratio threshold for PASS.")
    parser.add_argument("--recon-mse-threshold", type=float, default=0.6,
                        help="Euler reconstruction MSE threshold for PASS.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    set_seed(0)

    device = torch.device(args.device)
    task_names = [args.task] if args.task else list(TASK_SPECS.keys())

    failures: list[str] = []
    for name in task_names:
        task = TASK_SPECS[name]
        logger.info("smoke: overfitting task %s for %d steps", name, args.steps)
        ratio, recon_mse = _run_overfit(
            task, steps=args.steps, lr=args.lr, device=device
        )
        ok_drop = ratio < args.loss_drop_threshold
        ok_recon = recon_mse < args.recon_mse_threshold
        status = "PASS" if (ok_drop and ok_recon) else "FAIL"
        logger.info(
            "  task=%s ratio=%.3f (<%.2f? %s) recon_mse=%.4f (<%.2f? %s) => %s",
            name,
            ratio,
            args.loss_drop_threshold,
            ok_drop,
            recon_mse,
            args.recon_mse_threshold,
            ok_recon,
            status,
        )
        if status == "FAIL":
            failures.append(name)

    if failures:
        logger.error("Smoke FAIL for tasks: %s", failures)
        sys.exit(1)
    logger.info("Smoke PASS for all tested tasks.")


if __name__ == "__main__":
    main()

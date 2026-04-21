#!/usr/bin/env python3
"""SessionStart hook: 打印最近一次训练的尾部进度。

支持两种 CSV 来源：

* **Hydra 训练 run**（服务器上直接训练产物）::

      run/outputs/<YYYY-MM-DD>/<HH-MM-SS>/logs/loss.csv

  schema 由 ``src/trainer_module/trainer.py::_csv_fields`` 决定，核心列：
  ``epoch``, ``lr``, ``avg_loss``, ``val_loss_mean``。

* **SwanLab 拉取**（本机用 ``/swanlog`` 从云端下载）::

      run/outputs/swanlog_<exp_id>/metrics.csv

  列来自 SwanLab key 命名：``step`` (index), ``epoch/avg_loss``,
  ``epoch/lr``, ``val/loss_mean`` 等；非 epoch 边界的行这些列会是空串。
"""

from __future__ import annotations

import csv
import sys
from collections import deque
from pathlib import Path
from typing import NamedTuple

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent
OUTPUTS_DIR = PROJECT_DIR / "run" / "outputs"

STALL_WINDOW = 10
STALL_TOL = 1e-5


class Schema(NamedTuple):
    source: str
    progress: str
    avg_loss: str
    val_loss: str
    lr: str


HYDRA = Schema("hydra", "epoch", "avg_loss", "val_loss_mean", "lr")
SWANLOG = Schema("swanlog", "step", "epoch/avg_loss", "val/loss_mean", "epoch/lr")


def _find_csv() -> tuple[Path, Schema] | None:
    # Hydra run dirs are YYYY-MM-DD/HH-MM-SS — lexicographic == chronological.
    for path in sorted(OUTPUTS_DIR.glob("*/*/logs/loss.csv"), reverse=True):
        return path, HYDRA
    swan = list(OUTPUTS_DIR.glob("swanlog_*/metrics.csv"))
    if swan:
        return max(swan, key=lambda p: p.stat().st_mtime), SWANLOG
    return None


def _run_dir(csv_path: Path, schema: Schema) -> Path:
    # Hydra: .../TIME/logs/loss.csv → .../TIME ; SwanLog: .../swanlog_id/metrics.csv → .../swanlog_id
    return csv_path.parent.parent if schema is HYDRA else csv_path.parent


def handle_session_start() -> None:
    found = _find_csv()
    if found is None:
        print("[UniCardio] No hydra runs or swanlog pulls under run/outputs/.", flush=True)
        return
    csv_path, schema = found

    epoch_rows: deque[dict[str, str]] = deque(maxlen=STALL_WINDOW)
    try:
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get(schema.avg_loss):
                    epoch_rows.append(row)
    except OSError as exc:
        print(f"[UniCardio] Could not read {csv_path}: {exc}", flush=True)
        return

    rel = _run_dir(csv_path, schema).relative_to(PROJECT_DIR)
    if not epoch_rows:
        print(f"[UniCardio] {rel} has no epoch-summary rows yet.", flush=True)
        return

    last = epoch_rows[-1]
    print(
        f"[UniCardio] Latest {schema.source} {rel} — "
        f"{schema.progress} {last.get(schema.progress, '?')} | "
        f"avg_loss {last[schema.avg_loss]} | "
        f"val_loss {last.get(schema.val_loss) or 'n/a'} | "
        f"lr {last.get(schema.lr, '?')}",
        flush=True,
    )

    if len(epoch_rows) < STALL_WINDOW:
        return
    try:
        losses = [float(r[schema.avg_loss]) for r in epoch_rows]
    except ValueError:
        return
    if abs(losses[-1] - losses[0]) < STALL_TOL:
        print(
            f"[UniCardio] WARNING: avg_loss flat across last {STALL_WINDOW} epoch rows — possible stall.",
            flush=True,
        )


def main() -> None:
    try:
        sys.stdin.read()
    except OSError:
        pass
    handle_session_start()
    sys.exit(0)


if __name__ == "__main__":
    main()

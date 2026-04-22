---
name: experiment-compare
description: Compare UniCardio SwanLab runs pulled locally via /swanlog. Summarize a single run's final metrics, or compare multiple runs side-by-side on val loss, per-task loss, and config deltas.
user-invocable: true
disable-model-invocation: true
allowed-tools:
  - Bash
  - Read
  - Glob
---

# Experiment Compare Skill

All analysis is laptop-local. Training happens on the GPU server; results come back via `/swanlog` which dumps into `run/outputs/swanlog_<exp_id>/`. This skill reads those dumps — **never** tries to tail a live log.

## Usage

```
/experiment-compare                             # summary of every swanlog_* pull on disk
/experiment-compare <exp_id>                    # detailed view of one run
/experiment-compare <exp_id_a> <exp_id_b> ...   # side-by-side compare on val loss + config
```

## Data Contract

Each `run/outputs/swanlog_<exp_id>/` contains:
- `metrics.csv` — one row per logged step; epoch-level columns (`epoch/avg_loss`, `epoch/loss_<task>`, `val/loss_mean`, `val/loss_<task>`) are only filled on epoch boundaries (other rows are empty strings — always filter before aggregating).
- `config.yaml` — resolved Hydra config.
- `run_info.json` — `{id, name, state, created_at, finished_at, url, user}`.

Five tasks: `ecg2ppg`, `ppg2ecg`, `ecg2abp`, `ppg2abp`, `ecgppg2abp`.

## Task 1: List all local pulls

```bash
for info in run/outputs/swanlog_*/run_info.json; do
    python3 -c "
import json, sys
d = json.load(open('$info'))
print(f\"{d['id']:>20s}  {d['state']:>9s}  {d.get('finished_at') or d.get('created_at','?')}  {d['name']}\")
"
done | sort
```

## Task 2: Summarize one run

```bash
exp_id="$1"
dir="run/outputs/swanlog_${exp_id}"
[ -d "$dir" ] || { echo "no such pull: $dir"; exit 1; }

python3 - <<PY
import csv, json
from pathlib import Path

dir = Path("$dir")
info = json.load(open(dir / "run_info.json"))
print(f"run   : {info['name']} ({info['id']})  state={info['state']}")
print(f"url   : {info['url']}")

rows = [r for r in csv.DictReader(open(dir / "metrics.csv")) if r.get("epoch/avg_loss")]
if not rows:
    print("no epoch-summary rows yet")
    raise SystemExit

last = rows[-1]
print(f"epochs logged : {len(rows)}")
print(f"last avg_loss : {last['epoch/avg_loss']}")
print(f"last val_mean : {last.get('val/loss_mean') or 'n/a'}")

print("\nPer-task last-epoch losses:")
tasks = ["ecg2ppg", "ppg2ecg", "ecg2abp", "ppg2abp", "ecgppg2abp"]
for t in tasks:
    tr = last.get(f"epoch/loss_{t}", "")
    va = last.get(f"val/loss_{t}", "")
    print(f"  {t:<12s}  train={tr or '-':>10}  val={va or '-':>10}")

# Best val epoch
val_rows = [(i, float(r["val/loss_mean"])) for i, r in enumerate(rows) if r.get("val/loss_mean")]
if val_rows:
    best_i, best_v = min(val_rows, key=lambda x: x[1])
    print(f"\nbest val/loss_mean: {best_v:.6f} at epoch row {best_i}")
PY
```

## Task 3: Compare multiple runs

```bash
python3 - "$@" <<'PY'
import csv, json, sys
from pathlib import Path

ids = sys.argv[1:]
tasks = ["ecg2ppg", "ppg2ecg", "ecg2abp", "ppg2abp", "ecgppg2abp"]
rows = []
for exp in ids:
    d = Path(f"run/outputs/swanlog_{exp}")
    if not d.exists():
        print(f"skip {exp}: no such pull", file=sys.stderr)
        continue
    info = json.load(open(d / "run_info.json"))
    epochs = [r for r in csv.DictReader(open(d / "metrics.csv")) if r.get("epoch/avg_loss")]
    if not epochs:
        continue
    last = epochs[-1]
    best_val = min(
        (float(r["val/loss_mean"]) for r in epochs if r.get("val/loss_mean")),
        default=None,
    )
    rows.append(dict(
        id=exp, name=info["name"], state=info["state"],
        epochs=len(epochs), last_avg=float(last["epoch/avg_loss"]),
        best_val=best_val,
        per_task={t: float(last.get(f"val/loss_{t}", "nan") or "nan") for t in tasks},
    ))

if not rows:
    print("no runs to compare")
    raise SystemExit

w = max(len(r["id"]) for r in rows)
header = f"{'id':<{w}s}  {'state':>9s}  {'epochs':>6s}  {'avg_loss':>10s}  {'best_val':>10s}  " + "  ".join(f"{t:>10s}" for t in tasks)
print(header)
print("-" * len(header))
for r in rows:
    bv = f"{r['best_val']:.4f}" if r["best_val"] is not None else "-"
    tvals = "  ".join(f"{r['per_task'][t]:>10.4f}" for t in tasks)
    print(f"{r['id']:<{w}s}  {r['state']:>9s}  {r['epochs']:>6d}  {r['last_avg']:>10.4f}  {bv:>10s}  {tvals}")
PY
```

## Task 4: Config diff

```bash
python3 - "$@" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf

configs = {}
for exp in sys.argv[1:]:
    p = Path(f"run/outputs/swanlog_{exp}/config.yaml")
    if p.exists():
        configs[exp] = OmegaConf.to_container(OmegaConf.load(p), resolve=True)

if len(configs) < 2:
    print("need at least 2 configs")
    raise SystemExit

def flatten(d, prefix=""):
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            yield from flatten(v, key)
        else:
            yield key, v

all_keys = set()
flat = {}
for exp, cfg in configs.items():
    flat[exp] = dict(flatten(cfg))
    all_keys.update(flat[exp])

for k in sorted(all_keys):
    vals = [flat[e].get(k, "<missing>") for e in configs]
    if len(set(map(repr, vals))) > 1:
        print(f"{k}:")
        for e, v in zip(configs, vals):
            print(f"  {e}: {v}")
PY
```

## Notes

- When `/swanlog` is re-run on the same `exp_id`, the dir is overwritten — no stale-data risk.
- If fewer epoch rows exist than `cfg.trainer.epochs`, the run was still going when pulled (or crashed). Compare against `run_info.json::state`.
- Never reference `base_model/check/`, 800-epoch stage schedules, or `down_stream_code/` — all removed in commit `6ecc770`.

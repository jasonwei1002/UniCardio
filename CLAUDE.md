# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UniCardio is a Rectified-Flow transformer for cardiovascular signal reconstruction. The model defines **5 tasks** across 3 modalities (ECG / PPG / ABP); the **active training scope is the 3 ABP-target tasks** (cuffless BP is the primary clinical product). The two ECG↔PPG directions are kept in `TASK_SPECS` but their `task_weights` default to `0.0` and the sampler filters them out.

| Task ID        | Condition  | Target | Default weight | Use case |
|----------------|------------|--------|----------------|----------|
| `ecg2ppg`      | ECG        | PPG    | `0.0` (off)    | Infer photoplethysmogram from an ECG lead |
| `ppg2ecg`      | PPG        | ECG    | `0.0` (off)    | Infer ECG from a wearable PPG |
| `ecg2abp`      | ECG        | ABP    | `1.0`          | Non-invasive blood-pressure proxy |
| `ppg2abp`      | PPG        | ABP    | `1.0`          | Cuffless blood-pressure |
| `ecgppg2abp`   | ECG + PPG  | ABP    | `1.0`          | Multimodal blood-pressure |

All three active tasks share `target_slot=2`, which is also what makes `torch.compile` happy (Dynamo only needs to specialize one target-slot variant). To run all 5 tasks again, set non-zero weights in `run/conf/trainer/default.yaml::task_weights`.

Training objective is **Rectified Flow** with the **Lipman Flow-Matching time convention** (`t = 0` noise, `t = 1` data): `x_t = (1-t) ε + t x_1`, velocity target `v = x_1 - ε`, logit-normal `t` sampling, 8-step Euler ODE sampler integrating from `t=0` to `t=1` at inference.

See `reports/diffusion-rectified-flow-expressive-cascade.md` for the refactor plan and architectural decisions.

## Development Environment

Hard rules — do not deviate without asking:

- **Local = macOS Apple Silicon, CPU only.** For error debugging and smoke tests. Never launch real training here; the dataset is 3.4 GB and epochs would take hours. No MPS — some ops are unimplemented and it's not worth maintaining.
- **Remote = Linux GPU server, CUDA only.** Only place full training runs. Access is **one-way via GitHub** — the laptop cannot SSH into the server. Push code from laptop → `git pull` on server → `bash train.sh`. Do not try to rsync / ssh the server.
- **AMP is bf16-only on CUDA**, fp16 path removed entirely (no GradScaler). `run/conf/trainer/default.yaml::amp.enabled` gates it; CPU falls back to fp32 automatically.
- **Data / checkpoints / logs are never in git** — `.gitignore` excludes `data/`, `run/outputs/`, `logs/`. First setup on a new server: scp `data/Final_sig_combined.npy` once.

Device is read from the Hydra config (`device: cuda | cpu`); on macOS `cuda` auto-falls back to `cpu` so local debug just works with the default config.

## Environment Setup

```bash
pip install -r requirements.txt
```

Python 3.10+, PyTorch 2.x, Hydra 1.3.

## Key Commands

Run from the **repo root** — Hydra finds configs in `run/conf`:

```bash
# Unit tests (mask semantics, RF step, sampler — 13 tests, ~1 s on CPU).
python -m pytest tests/
python -m pytest tests/test_masks.py::test_self_only_for_active_slots   # single test

# CPU smoke test: overfits a tiny synthetic batch for each task,
# verifies loss drop + Euler reconstruction. ~70 s total.
python run/pipeline/smoke_test.py

# Full training run (overrides via Hydra CLI). On the GPU server prefer `bash train.sh`,
# which fixes the canonical Tier-1 overrides (batch 512, bf16 AMP, torch.compile, warmup).
python run/pipeline/train.py
python run/pipeline/train.py device=cpu trainer.epochs=2 data.num_workers=0
python run/pipeline/train.py trainer.compile.enabled=false   # skip Dynamo when iterating

# Evaluation with a checkpoint.
python run/pipeline/evaluate.py +checkpoint=run/outputs/<run>/checkpoints/best.pt
```

`data_path` resolves through `${oc.env:UNICARDIO_DATA, data/Final_sig_combined.npy}` — set `UNICARDIO_DATA=/abs/path/to.npy` to point at a different dataset without editing the config.

Hydra writes per-run artifacts under `run/outputs/<timestamp>/`:
- `checkpoints/latest.pt`, `checkpoints/best.pt` — full state (model, optimizer, scheduler, config, RNG).
- `logs/loss.csv` — per-epoch per-task training loss + validation loss.
- `swanlog/` — local SwanLab event store (only when `swanlab.enabled=true`).
- Hydra's own `.hydra/` snapshot of the resolved config.

### Experiment tracking

SwanLab is wired in `run/pipeline/train.py::_init_swanlab` and `src/trainer_module/trainer.py`. `run/conf/config.yaml::swanlab.mode` switches between `cloud` (requires `swanlab login`), `local`, `offline`, and `disabled`. Per-step `train/lr` + per-task losses, and per-epoch validation losses, are logged with the global step. The laptop cannot reach the GPU server directly, so to inspect a remote run use `script/fetch_swanlog.py` (also exposed as the `/swanlog` slash command) which pulls the latest experiment's full log into `run/outputs/swanlog_<exp_id>/`.

## Data Preparation

Training data: `data/Final_sig_combined.npy`, shape `(N, 3, 500)`.

**Important**: the on-disk channel order is PPG=0, BP=1, ECG=2, but the model operates in slot order **ECG=0, PPG=1, ABP=2**. The data loader applies a single permutation `[2, 0, 1]` in `src/data_module/datamodule.py::load_and_preprocess` and also applies the BP normalization `(x - 100) / 50` on model slot 2. Everything downstream speaks exclusively in model slot indices.

Data split: 20 000 validation + 20 000 test + remainder train, `random_state=42` (matches the original codebase).

## Architecture

### Code Layout

```
src/
├── data_module/          # CardiacDataset + loader factory (channel permute + BP normalize)
├── model_module/
│   ├── tasks.py          # Slot enum + TaskSpec dataclass + TASK_SPECS
│   ├── attention_masks.py# build_task_mask (lru_cache) — 1 mask per task
│   ├── embeddings.py     # SignalEncoder (preserved) + FlowTimeEmbedding (new)
│   ├── residual_block.py # Gated transformer block (preserved)
│   ├── backbone.py       # UniCardioBackbone — 3-slot version of old diff_CSDI
│   └── unicardio_rf.py   # UniCardioRF wrapper — forward(x, t, task)
├── trainer_module/
│   ├── rectified_flow.py # sample_t_logit_normal, rf_train_step, assemble_x_full
│   ├── sampler.py        # euler_sample (N-step ODE)
│   ├── trainer.py        # train loop with weighted task sampling + ckpt
│   └── csv_logger.py     # Append-only CSV logger
└── utils/                # seed, checkpoint, normalization, metrics

run/
├── conf/                 # Hydra configs: data / model / trainer / sampler / root
├── pipeline/
│   ├── train.py          # Hydra entrypoint for training
│   ├── evaluate.py       # 5-task RMSE / MAE / KS on the test split
│   └── smoke_test.py     # Per-task overfit-one-batch sanity test
└── outputs/              # Hydra run directories (gitignored)

tests/                    # pytest unit tests (masks, rf_step, sampler)
script/                   # Reusable utility scripts (data conversion, swanlog pull)
data/Final_sig_combined.npy   # Training data (gitignored)
train.sh                  # 一键启动训练（GPU 服务器使用，固定 Tier-1 overrides）
```

Convention: any reusable helper script lives in `script/` at the repo root — do not scatter them under `data/` or feature subdirs (see `convert_h5_to_npy_csv.py`, `convert_mimicbp_to_npy_csv.py`, `fetch_swanlog.py`).

### Slot Layout & Attention Masks

Total token sequence length = `3 * slot_length = 1500`. Slot token ranges in the flattened `(B, 1, 1500)` input:

| Slot | Modality | Tokens       |
|------|----------|--------------|
| 0    | ECG      | `[0, 500)`   |
| 1    | PPG      | `[500, 1000)`|
| 2    | ABP      | `[1000, 1500)`|

`build_task_mask(task_name, L_slot)` returns an additive `(L_total, L_total)` mask with `0` on allowed cells and `-inf` elsewhere. Rule: every participating slot attends to itself; the target slot additionally attends to all condition slots; condition slots cross-attend to each other (only matters for `ecgppg2abp`); non-participating slot rows are fully blocked.

### Model (`UniCardioRF` / `UniCardioBackbone`)

```
x (B, 1, 3*L)
  → split into 3 slots along last dim
  → 3× SignalEncoder (multi-scale Conv1d, kernels 1,3,5,7,9,11 → 6×48=288 channels)
  → 3× LayerNorm
  → concat back → (B, 288, 1500)
  → FlowTimeEmbedding(t ∈ [0,1])
  → 5× ResidualBlock (Transformer self-attn with task mask + time emb + sinusoidal pos emb)
  → skip-sum aggregation / sqrt(5)
  → per-slot output head (channels → 1) indexed by target_slot
  → velocity v (B, 1, L) for target slot only
```

The old placeholder slot 3 and `borrow_mode` are removed. Loss is computed externally in `rf_train_step` so the model stays pure.

### Training Loop (`src/trainer_module/trainer.py`)

Per batch: sample one task from the active set (zero-weight tasks are pre-filtered, then weighted via `trainer.task_weights`), call `rf_train_step(model, signal, task)` which:
1. Extracts `x_1 = signal[:, target_slot, :]` (clean data, at `t=1` under Lipman convention).
2. Samples `t ~ sigmoid(Normal(mean, std))` (logit-normal, SD3-style; symmetric about 0.5).
3. Computes `x_t = (1 - t) ε + t x_1`, `v_target = x_1 - ε`.
4. Assembles `(B, 1, 3*L)` input with clean conditions + `x_t` in the target slot.
5. Calls the model and returns MSE between predicted and target velocity.

Optimizer: Adam (default lr 2e-3, wd 1e-6 — sqrt-scaled for batch 512). Schedule: **`cosine_annealing_warmup.CosineAnnealingWarmupRestarts`** (no warm restart — `first_cycle_steps = total_steps` so training ends before the cycle wraps). Linear warmup for `trainer.warmup_pct` (fraction of total steps in `[0, 1)`, default `0.05`; step count derived at runtime via `round(warmup_pct * total_steps)`) from `min_lr` up to `lr` (= the library's `max_lr`), then cosine anneal from `lr` down to `lr_scheduler.min_lr` (default `1e-6`) across the remaining steps; all step-granularity internally. AMP: bf16-only on CUDA, no GradScaler; CPU silently falls back to fp32. `torch.compile(mode="default")` wraps the top-level `UniCardioRF` on CUDA — `mode="reduce-overhead"` was tried and rejected because the CUDA-Graphs private pool burns ~14 GB at this batch size. Checkpoints every epoch (`latest.pt`); best tracked by mean validation RF loss (`best.pt`); `trainer.resume_from` resumes from any saved checkpoint.

### Inference API

```python
from src.trainer_module.sampler import euler_sample
from src.model_module.tasks import TASK_SPECS

task = TASK_SPECS['ecg2ppg']                    # 5 tasks total
out = euler_sample(
    model, conditions, task,                     # conditions: (B, 3, L)
    n_steps=8, device=device,
)                                                # returns (B, 1, L) target-slot reconstruction
```

The sampler integrates `v_θ` from `t = 1` → `t = 0` with Euler steps. `n_steps=8` is the default (trades speed for quality; 4-16 typical).

## Development Notes

- **Adding a new task**: extend `TASK_SPECS` in `src/model_module/tasks.py` *and* give it a non-zero entry in `trainer.task_weights` (the sampler drops zero-weight tasks). The mask builder handles any slot subset; no model surgery needed. Adding a task with a new `target_slot` will trigger a `torch.compile` re-specialization on the first batch hitting it.
- **Single-GPU → multi-GPU**: the checkpoint save/load path already unwraps `nn.DataParallel`. For DDP, wrap before calling `train()`; the logger and ckpt code are DDP-safe per process.
- **macOS**: 本地只用 CPU 做错误调试与 smoke test；不走 MPS（部分算子未实现，不值得维护）。
- **Reproducibility**: `set_seed(cfg.seed)` is called at both entrypoints. Set `deterministic: true` to force cuDNN determinism (slower).

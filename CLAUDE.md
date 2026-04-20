# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UniCardio is a Rectified-Flow transformer for cardiovascular signal reconstruction. One model supports exactly **5 tasks** across 3 modalities (ECG / PPG / ABP):

| Task ID        | Condition  | Target | Use case |
|----------------|------------|--------|----------|
| `ecg2ppg`      | ECG        | PPG    | Infer photoplethysmogram from an ECG lead |
| `ppg2ecg`      | PPG        | ECG    | Infer ECG from a wearable PPG |
| `ecg2abp`      | ECG        | ABP    | Non-invasive blood-pressure proxy |
| `ppg2abp`      | PPG        | ABP    | Cuffless blood-pressure |
| `ecgppg2abp`   | ECG + PPG  | ABP    | Multimodal blood-pressure |

Training objective is **Rectified Flow** (Liu et al., 2022 / SD3): `x_t = (1-t) x_0 + t ε`, velocity target `v = ε - x_0`, logit-normal `t` sampling, 8-step Euler ODE sampler at inference.

See `reports/diffusion-rectified-flow-expressive-cascade.md` for the refactor plan and architectural decisions.

## Development Environment

- **Local (macOS Apple Silicon, arm64)** — code edits, unit tests, CPU smoke tests.
- **Linux training server** — full training runs with CUDA.

The training server is `hdu-baiyang`; code is rsync'd there via `script/sync.sh`. No script hardcodes `CUDA` any more — the device is read from the Hydra config (`device: cuda | mps | cpu`).

## Environment Setup

```bash
pip install -r requirements.txt
```

Python 3.10+, PyTorch 2.x, Hydra 1.3.

## Key Commands

Run from the **repo root** — Hydra finds configs in `run/conf`:

```bash
# Unit tests (mask semantics, RF step, sampler — 33 tests, ~1 s on CPU).
python -m pytest tests/

# CPU smoke test: overfits a tiny synthetic batch for each of the 5 tasks,
# verifies loss drop + Euler reconstruction. ~73 s total.
python run/pipeline/smoke_test.py

# Full training run (overrides via Hydra CLI).
python run/pipeline/train.py
python run/pipeline/train.py device=cpu trainer.epochs=2 data.num_workers=0

# Evaluation with a checkpoint.
python run/pipeline/evaluate.py +checkpoint=run/outputs/<run>/checkpoints/best.pt
```

Hydra writes per-run artifacts under `run/outputs/<timestamp>/`:
- `checkpoints/latest.pt`, `checkpoints/best.pt` — full state (model, optimizer, scheduler, config, RNG).
- `logs/loss.csv` — per-epoch per-task training loss + validation loss.
- Hydra's own `.hydra/` snapshot of the resolved config.

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
data/Final_sig_combined.npy   # Training data (gitignored)
script/sync.sh            # rsync helper for macOS → Linux server
```

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

Per batch: sample one task uniformly from the 5 (weighted via `trainer.task_weights`), call `rf_train_step(model, signal, task)` which:
1. Extracts `x_0 = signal[:, target_slot, :]`.
2. Samples `t ~ sigmoid(Normal(mean, std))` (logit-normal, SD3-style).
3. Computes `x_t = (1 - t) x_0 + t ε`, `v_target = ε - x_0`.
4. Assembles `(B, 1, 3*L)` input with clean conditions + `x_t` in the target slot.
5. Calls the model and returns MSE between predicted and target velocity.

Optimizer: Adam (lr 1e-3, wd 1e-6). Scheduler: MultiStepLR at 18% and 75% of total epochs, γ=0.1. Checkpoints every epoch (`latest.pt`); best model tracked by mean validation RF loss (`best.pt`).

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

- **Adding a new task**: extend `TASK_SPECS` in `src/model_module/tasks.py`. The mask builder handles any slot subset; no model surgery needed.
- **Single-GPU → multi-GPU**: the checkpoint save/load path already unwraps `nn.DataParallel`. For DDP, wrap before calling `train()`; the logger and ckpt code are DDP-safe per process.
- **macOS**: `device: mps` works for most ops; fall back to `device: cpu` for anything `MPS` doesn't implement.
- **Reproducibility**: `set_seed(cfg.seed)` is called at both entrypoints. Set `deterministic: true` to force cuDNN determinism (slower).

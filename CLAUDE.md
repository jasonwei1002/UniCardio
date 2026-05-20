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
- **Data / checkpoints / logs are never in git** — `.gitignore` excludes `data/`, `run/outputs/`, `logs/`. First setup on a new server: scp `data/pulsedb/Train_Subset.npy` + `data/pulsedb/CalFree_Test_Subset.npy` once (legacy `data/Final_sig_combined.npy` is still supported via `data.name=combined`).

Device is read from the Hydra config (`device: cuda | cpu`); on macOS `cuda` auto-falls back to `cpu` so local debug just works with the default config.

## Environment Setup

```bash
pip install -r requirements.txt
```

Python 3.10+, PyTorch ≥2.5 (FlexAttention 用 BlockMask + flex_attention，CPU 走 SDPA fallback), Hydra 1.3. 当前实测 torch 2.11。

## Key Commands

Run from the **repo root** — Hydra finds configs in `run/conf`:

```bash
# Unit tests (masks, RF step, sampler — 43 cases, ~1-2 s on CPU).
python -m pytest tests/
python -m pytest tests/test_masks.py::test_self_only_for_active_slots   # single test

# CPU smoke test: overfits a tiny synthetic batch for each task,
# verifies loss drop + Euler reconstruction. ~70 s total.
python run/pipeline/smoke_test.py

# Stage-1 (pretrain) full run. GPU server prefers `bash train.sh`, which fixes
# canonical Tier-1 overrides (batch 256, 12 workers, bf16 AMP, torch.compile max-autotune).
python run/pipeline/train.py
python run/pipeline/train.py device=cpu trainer.epochs=2 data.num_workers=0
python run/pipeline/train.py trainer.compile.enabled=false   # skip Dynamo when iterating

# Stage-2 (CalFree finetune): load stage-1 weights, freeze backbone,
# only last N residual blocks + output heads trainable. Auto-runs RF + BP
# metric eval on CalFree test split at the end.
bash finetune.sh run/outputs/<stage1>/checkpoints/best.pt
bash finetune.sh <ckpt> trainer.finetune.n_unfrozen_blocks=1 trainer.lr=5e-5

# Evaluation with a checkpoint. On the GPU server prefer `bash test.sh [ckpt]`,
# which auto-picks the latest `run/outputs/*/checkpoints/best.pt` if no path is given.
python run/pipeline/evaluate.py +checkpoint=run/outputs/<run>/checkpoints/best.pt
```

Default `data.name: pulsedb` reads from `data/pulsedb/{Train_Subset,CalFree_Test_Subset}.npy`. Legacy single-file mode (`data.name=combined`) honors `${oc.env:UNICARDIO_DATA, data/Final_sig_combined.npy}` — set `UNICARDIO_DATA=/abs/path/to.npy` to point at a different dataset without editing the config.

Hydra writes per-run artifacts under `run/outputs/<timestamp>_<stage>/` (stage suffix from `UNICARDIO_STAGE`, set by `train.sh`/`finetune.sh` — defaults to `pretrain` for plain `python` invocations):
- `checkpoints/latest.pt`, `checkpoints/best.pt` — full state (model, optimizer, scheduler, config, RNG).
- `logs/loss.csv` — per-epoch per-task training loss + validation loss.
- `logs/finetune_bp_metrics.csv` — only in stage-2; SBP/DBP ME+SD per task on CalFree test split.
- `swanlog/` — local SwanLab event store (only when `swanlab.enabled=true`).
- Hydra's own `.hydra/` snapshot of the resolved config.

### Experiment tracking

SwanLab is wired in `run/pipeline/train.py::_init_swanlab` and `src/trainer_module/trainer.py`. `run/conf/config.yaml::swanlab.mode` switches between `cloud` (requires `swanlab login`), `local`, `offline`, and `disabled`. Per-step `train/{loss,loss_<task>,lr}` are logged with the global step; per-epoch `epoch/{avg_loss,lr,time_s,loss_<task>}` and validation `val/{loss_mean,loss_<task>}` are logged with `step=epoch` so the chart x-axis reads as epoch numbers, not flat per-step counters. The laptop cannot reach the GPU server directly, so to inspect a remote run use `script/fetch_swanlog.py` (also exposed as the `/swanlog` slash command) which pulls the latest experiment's full log into `run/outputs/swanlog_<exp_id>/`.

## Data Preparation

Two datasets are wired through `data.name`:

- **`pulsedb` (default, active)**: `data/pulsedb/Train_Subset.npy` (~13 GB, mmap-backed) for pretrain + `data/pulsedb/CalFree_Test_Subset.npy` for finetune/test. Slot length `L = 1250`, so `L_total = 3 * 1250 = 3750`. Files are already in model slot order `(ECG, PPG, ABP)` (see `script/convert_h5_to_npy_csv.py`); `channel_permutation` is identity. Split logic depends on `mode` (set by `train.sh`/`finetune.sh`):
  - `mode='pretrain'`: `Train_Subset.npy` 80/20 → train/val, `CalFree_Test_Subset.npy` → test.
  - `mode='finetune'`: `CalFree_Test_Subset.npy` 80/10/10 → train/val/test, `Train_Subset.npy` ignored.
- **`combined` (legacy)**: `data/Final_sig_combined.npy`, shape `(N, 3, 500)`, on-disk channel order `(PPG, BP, ECG)` → permuted to `(ECG, PPG, ABP)` with `[2, 0, 1]`. 20 000 val + 20 000 test + remainder train.

Both paths apply the BP normalization `(x - 100) / 50` on model slot 2 inside `CardiacDataset.__getitem__`. Everything downstream speaks exclusively in model slot indices. `split_seed: 42` for reproducibility.

## Architecture

### Code Layout

```
src/
├── data_module/
│   ├── cardiac_dataset.py# mmap-backed __getitem__ (channel permute + BP normalize)
│   └── datamodule.py     # build_loaders: dataset planner per (name, mode)
├── model_module/
│   ├── tasks.py          # Slot enum + TaskSpec dataclass + TASK_SPECS + active_task_pairs
│   ├── attention_masks.py# build_task_mask (dense bool, CPU) + build_task_block_mask (BlockMask, CUDA)
│   ├── embeddings.py     # SignalEncoder (multi-scale Conv1d) + FlowTimeEmbedding
│   ├── residual_block.py # _FlexAttentionFFN (BlockMask→flex, Tensor→SDPA) + gated FFN block
│   ├── backbone.py       # UniCardioBackbone: 3-slot encode → 5× residual → per-slot head
│   └── unicardio_rf.py   # UniCardioRF wrapper + freeze_for_finetune
├── trainer_module/
│   ├── rectified_flow.py # sample_t_logit_normal, build_rf_inputs, rf_train_step
│   ├── sampler.py        # euler_sample (N-step ODE under Lipman convention)
│   ├── trainer.py        # train loop: task sampling, AMP, compile, scheduler, ckpt
│   ├── bp_metrics.py     # evaluate_bp_test: SBP/DBP ME+SD per ABP-target task
│   └── csv_logger.py     # Append-only CSV logger
└── utils/                # seed, checkpoint (unwraps DP/DDP/OptimizedModule), normalization

run/
├── conf/                 # Hydra configs: data / model / trainer / sampler / root
├── pipeline/
│   ├── train.py          # Hydra entrypoint; dispatches stage='pretrain'|'finetune'
│   ├── evaluate.py       # 5-task RMSE / MAE / KS on the test split
│   └── smoke_test.py     # Per-task overfit-one-batch sanity test
└── outputs/              # Hydra run directories (gitignored)

tests/                    # pytest unit tests (masks, rf_step, sampler) — 43 cases
script/                   # Reusable utilities (data conversion, swanlog pull, sample plot)
data/pulsedb/             # Active dataset (gitignored): Train_Subset.npy + CalFree_Test_Subset.npy
train.sh                  # Stage-1 pretrain one-shot (GPU server)
finetune.sh <ckpt>        # Stage-2 CalFree finetune (loads ckpt, freezes backbone)
test.sh [ckpt]            # 评估入口（默认挑最新 best.pt，可传入 ckpt 路径）
```

Convention: any reusable helper script lives in `script/` at the repo root — do not scatter them under `data/` or feature subdirs (see `convert_h5_to_npy_csv.py`, `convert_mimicbp_to_npy_csv.py`, `fetch_swanlog.py`).

### Slot Layout & Attention Masks

Total token sequence length = `3 * L_slot`. `L_slot` is fixed by the active dataset (pulsedb: `1250` → `L_total = 3750`; combined: `500` → `L_total = 1500`). Slot token ranges in the flattened `(B, 1, L_total)` input:

| Slot | Modality | Tokens                  |
|------|----------|-------------------------|
| 0    | ECG      | `[0, L_slot)`           |
| 1    | PPG      | `[L_slot, 2*L_slot)`    |
| 2    | ABP      | `[2*L_slot, 3*L_slot)`  |

Attention is dispatched on mask type inside `_FlexAttentionFFN`:

- **CUDA path**: `build_task_block_mask(task_name, L_slot, device)` returns a `BlockMask`; `flex_attention(q, k, v, block_mask=...)` skips all-False 128-block tiles at kernel level. Inside the top-level `torch.compile` region this fuses into a Triton kernel. Sparsity savings: single-cond tasks ~33% density → ~3× FLOP cut; `ecgppg2abp` ~79% density → modest cut.
- **CPU fallback**: `build_task_mask(task_name, L_slot, dtype=torch.bool)` returns a dense `(L_total, L_total)` bool tensor; `_FlexAttentionFFN` routes it to `F.scaled_dot_product_attention`. **Required because FlexAttention has no CPU backward as of torch 2.11** — without this, `rf_train_step` on CPU crashes. Selection happens in `rf_train_step` / `euler_sample` via `device.type == "cuda"` branch.

Mask rule (same for both paths): every participating slot attends to itself; the target slot additionally attends to all condition slots; condition slots cross-attend each other (only matters for `ecgppg2abp`); non-participating slot rows are fully blocked.

### Model (`UniCardioRF` / `UniCardioBackbone`)

```
x (B, 1, 3*L_slot)
  → split into 3 slots along last dim
  → 3× SignalEncoder (multi-scale Conv1d, kernels 1,3,5,7,9,11 → 6×48=288 channels)
  → 3× LayerNorm
  → concat back → (B, 288, 3*L_slot)
  → FlowTimeEmbedding(t ∈ [0,1])
  → 5× ResidualBlock (nheads=9, gated FFN, FlexAttention/SDPA + sinusoidal pos bias)
  → skip-sum aggregation / sqrt(5)
  → per-slot output head (channels → 1) indexed by target_slot
  → velocity v (B, 1, L_slot) for target slot only
```

The old placeholder slot 3 and `borrow_mode` are removed. Loss is computed externally in `rf_train_step` so the model stays pure. `BackboneConfig.downsample_factor` (default `1` = identity) exposes per-slot Conv1d/ConvTranspose1d down/up-sampling around the transformer stack — currently disabled but available to halve attention length on pulsedb.

### Training Loop (`src/trainer_module/trainer.py`)

Per batch: sample one task from the active set (zero-weight tasks are pre-filtered, then weighted via `trainer.task_weights`), call `rf_train_step(model, signal, task)` which:
1. Extracts `x_1 = signal[:, target_slot, :]` (clean data, at `t=1` under Lipman convention).
2. Samples `t ~ sigmoid(Normal(mean, std))` (logit-normal, SD3-style; symmetric about 0.5).
3. Computes `x_t = (1 - t) ε + t x_1`, `v_target = x_1 - ε`.
4. Assembles `(B, 1, 3*L)` input with clean conditions + `x_t` in the target slot.
5. Calls the model and returns MSE between predicted and target velocity.

Optimizer: Adam (default lr 1e-3, wd 1e-6, fused on CUDA). Gradient clipping: L2 norm capped at `trainer.grad_clip_norm` (default `1.0`; set to 0/null to disable) between `backward()` and `optimizer.step()` — non-optional under bf16 + Adam on RF velocity MSE, an earlier 2e-3 run diverged around epoch 10 without clipping. Schedule: torch built-in `CosineAnnealingLR` (`lr_scheduler.first_cycle_pct=1.0`, single cosine) or `CosineAnnealingWarmRestarts` (`<1.0`, `cycle_mult` controls cycle growth). **No warmup.** Default `first_cycle_pct=1.0` runs a single cosine over the full schedule, decaying from `lr=1e-3` to `min_lr=1e-6`; set `<1.0` to enable warm restarts (each restart returns to peak `lr`; torch built-in does not support peak decay).

AMP: bf16-only on CUDA, no GradScaler; CPU silently falls back to fp32. `torch.compile(mode=trainer.compile.mode)` wraps the top-level `UniCardioRF` on CUDA — default `mode="max-autotune"` autotunes Triton kernels + CUDA Graphs on epoch 1 (a few minutes of compile overhead), then runs 5-15% faster than `reduce-overhead`. Private pool ≈10-15 GB; if VRAM is tight, fall back to `mode=default` (no CUDA Graphs).

DataLoader uses `persistent_workers=True` + `prefetch_factor=4`; `train`/`val` set `drop_last=True` to keep batch shape constant for compile/CUDA-Graphs (val drops ≤ `batch_size-1` samples out of ~tens of thousands, < 0.16% statistical impact). Checkpoints every epoch (`latest.pt`); best tracked by mean validation RF loss (`best.pt`). `trainer.init_from` loads **model weights only** as a fresh-optim fine-tune entry — no full-state resume path; killed training restarts from scratch (use `init_from=<run>/checkpoints/latest.pt` to skip stem retraining).

### Two-stage training (`trainer.stage`)

- **`pretrain`** (default, `bash train.sh`): standard full training on the pulsedb `Train_Subset.npy` 80/20 split.
- **`finetune`** (`bash finetune.sh <stage1 ckpt>`): loads `trainer.init_from` model weights → `UniCardioRF.freeze_for_finetune(n)` freezes the backbone stem and unfreezes only the last `trainer.finetune.n_unfrozen_blocks` `ResidualBlock`s plus all `output_heads` → fused Adam is built **after** the freeze, so `param_groups` only see trainable tensors. Data switches to `mode='finetune'` (CalFree 80/10/10). After the last epoch, `best.pt` is reloaded and `bp_metrics.evaluate_bp_test` runs an 8-step Euler sampler on the test split, logging per-task SBP/DBP ME+SD (mmHg) to SwanLab + `logs/finetune_bp_metrics.csv`.

### Inference API

```python
from src.trainer_module.sampler import euler_sample
from src.model_module.tasks import TASK_SPECS

task = TASK_SPECS['ppg2abp']                    # any active task; 5 specs total
out = euler_sample(
    model, conditions, task,                     # conditions: (B, 3, L_slot)
    n_steps=8, device=device,
)                                                # returns (B, 1, L_slot) target-slot reconstruction
```

The sampler integrates `v_θ` from `t = 0` (noise) → `t = 1` (data) with Euler steps, matching the Lipman convention. `n_steps=8` is the default (trades speed for quality; 4-16 typical).

## Development Notes

- **Adding a new task**: extend `TASK_SPECS` in `src/model_module/tasks.py` *and* give it a non-zero entry in `trainer.task_weights` (the sampler drops zero-weight tasks). The mask builder handles any slot subset; no model surgery needed. Adding a task with a new `target_slot` will trigger a `torch.compile` re-specialization on the first batch hitting it.
- **Single-GPU → multi-GPU**: the checkpoint save/load path unwraps `nn.DataParallel`, `DistributedDataParallel`, and `torch.compile`'s `OptimizedModule` (`_orig_mod`) — so saved `state_dict` keys stay aligned with the architecture regardless of wrapping order. For DDP, wrap before calling `train()`; the logger and ckpt code are DDP-safe per process.
- **macOS**: 本地只用 CPU 做错误调试与 smoke test；不走 MPS（部分算子未实现，不值得维护）。
- **Reproducibility**: `set_seed(cfg.seed)` is called at both entrypoints. Set `deterministic: true` to force cuDNN determinism (slower).

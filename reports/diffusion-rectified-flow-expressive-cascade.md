# UniCardio Refactor: Diffusion → Rectified Flow, 5 Tasks, ML Project Layout

## Context

Current `base_model/` hard-codes a 4-slot diffusion model (PPG/BP/ECG + zero-padded "slot 3") with 8+ task code paths, an integer-step DDPM/DDIM sampler stack, and `task_dice / dirty_dice / condition_dice / thresholder` stochastic task routing. The user only needs **5 reconstruction tasks** and wants to move to **Rectified Flow** for faster sampling (~8 Euler steps) and a simpler objective. This refactor:

1. Drops the placeholder slot 3 + `borrow_mode` machinery — slots become 3 native modalities in model order ECG=0, PPG=1, ABP=2 (a fixed `[2, 0, 1]` permute of the on-disk channel order).
2. Replaces diffusion schedule (α/β buffers, DDPM/DDIM samplers) with RF: `x_t = (1-t)x₀ + tε`, velocity target `v = ε - x₀`, logit-normal `t` sampling (SD3-style), Euler ODE sampler.
3. Reorganizes the codebase into the ML project layout mandated by `coding-style.md` (`src/{data,model,trainer,utils}_module`, `run/{conf,pipeline}`), each file 200–400 lines.
4. Removes all self-conditioning / denoising / imputation auxiliaries — training sees only clean signals.

**Preserved intact**: `SignalEncoder` (multi-kernel conv), `ResidualBlock` (transformer self-attn), per-slot `LayerNorm`, per-slot output heads, Adam+MultiStepLR, CSV logger, 20K/20K random split (random_state=42), BP normalization `(x-100)/50`.

## Target Directory Layout

```
src/
├── data_module/
│   ├── cardiac_dataset.py        # CardiacDataset (returns (3, 500) clean)
│   └── datamodule.py             # build_loaders(cfg) — BP-normalize, 20K/20K split
├── model_module/
│   ├── tasks.py                  # Slot enum + TaskSpec dataclass + TASK_SPECS table
│   ├── attention_masks.py        # build_task_mask(task, L) — lru_cache
│   ├── embeddings.py             # SignalEncoder (preserved), FlowTimeEmbedding (new)
│   ├── residual_block.py         # ResidualBlock (preserved)
│   ├── backbone.py               # UniCardioBackbone — 3-slot version of diff_CSDI
│   └── unicardio_rf.py           # UniCardioRF wrapper — forward(x, t, task)
├── trainer_module/
│   ├── rectified_flow.py         # sample_t_logit_normal, rf_train_step, assemble_x_full
│   ├── sampler.py                # euler_sample (N-step ODE)
│   ├── trainer.py                # train loop, optim, sched, ckpt
│   └── csv_logger.py             # preserved SimpleCSVLogger
└── utils/
    ├── seed.py                   # set_seed (reproducibility rule)
    ├── checkpoint.py             # save/load (epoch, model, optim, cfg, rng_state)
    ├── metrics.py                # RMSE / MAE / KS per task
    └── normalization.py          # bp_normalize / bp_denormalize

run/
├── conf/
│   ├── config.yaml               # Hydra root
│   ├── data/default.yaml
│   ├── model/unicardio_rf.yaml
│   ├── trainer/default.yaml
│   └── sampler/euler8.yaml
├── pipeline/
│   ├── train.py                  # Hydra entrypoint
│   ├── evaluate.py               # loops 5 tasks, Euler sample, metrics
│   └── smoke_test.py             # overfit-1-batch sanity
└── outputs/                      # Hydra run dirs (gitignored)

tests/
├── test_masks.py
├── test_rf_step.py
└── test_sampler.py

data/Final_sig_combined.npy       # moved from base_model/
base_model/                       # deleted after migration
```

## Slot Layout & Task Specification

**Model slot order: ECG=0, PPG=1, ABP=2**. This differs from the on-disk channel order of `Final_sig_combined.npy` (which is PPG, BP, ECG = file indices [0, 1, 2]). The data loader applies a one-time channel permutation `[2, 0, 1]` (file PPG/BP/ECG → model ECG/PPG/ABP) immediately after `np.load`, before any split or normalization logic. BP normalization (`(x-100)/50`) is applied on the ABP channel after permutation, i.e. model slot 2.

The 5 tasks map to:

| Task ID | Cond slots | Target slot |
|---|---|---|
| `ecg2ppg`    | {0}    | 1 |
| `ppg2ecg`    | {1}    | 0 |
| `ecg2abp`    | {0}    | 2 |
| `ppg2abp`    | {1}    | 2 |
| `ecgppg2abp` | {0, 1} | 2 |

Implementation ([src/model_module/tasks.py](src/model_module/tasks.py)):

```python
class Slot(IntEnum):
    ECG = 0; PPG = 1; ABP = 2

@dataclass(frozen=True)
class TaskSpec:
    name: str
    cond_slots: tuple
    target_slot: Slot
    task_id: int
```

Trainer samples one task per batch via `random.choice(TASK_LIST)`.

## Attention Masks ([src/model_module/attention_masks.py](src/model_module/attention_masks.py))

Total token length = `3 * L_slot = 1500`. Slot token ranges: `ECG=[0,500)`, `PPG=[500,1000)`, `ABP=[1000,1500)`. Additive `-inf` mask matching `nn.MultiheadAttention`.

**Rule**: every participating slot attends to itself; target attends to all conditions; conditions attend to each other (matters only for `ecgppg2abp`); non-participating slots are fully blocked.

```python
@lru_cache(maxsize=32)
def build_task_mask(task_name: str, L_slot: int, device: str = "cpu") -> Tensor:
    spec = TASK_SPECS[task_name]
    total = 3 * L_slot
    mask = torch.full((total, total), float("-inf"))
    participants = set(spec.cond_slots) | {spec.target_slot}
    for s in participants:
        _block(mask, s, s, L_slot)                         # self-attention
    for s1, s2 in product(spec.cond_slots, repeat=2):
        if s1 != s2: _block(mask, s1, s2, L_slot)          # cond↔cond
    for c in spec.cond_slots:
        _block(mask, spec.target_slot, c, L_slot)          # target←cond
    return mask.to(device)
```

Per-task row summary (T=target, C=condition, ·=blocked):

| Task | ECG row (slot 0) | PPG row (slot 1) | ABP row (slot 2) |
|---|---|---|---|
| `ecg2ppg`    | C←{0}     | T←{0,1}    | ·          |
| `ppg2ecg`    | T←{0,1}   | C←{1}      | ·          |
| `ecg2abp`    | C←{0}     | ·          | T←{0,2}    |
| `ppg2abp`    | ·         | C←{1}      | T←{1,2}    |
| `ecgppg2abp` | C←{0,1}   | C←{0,1}    | T←{0,1,2}  |

## Rectified Flow Core ([src/trainer_module/rectified_flow.py](src/trainer_module/rectified_flow.py))

```python
def sample_t_logit_normal(B, device, m=0.0, s=1.0):
    return torch.sigmoid(torch.randn(B, device=device) * s + m)

def assemble_x_full(signal_3slot, x_t_target, target_slot, L):
    # (B, 3, L) + (B, 1, L) target → flatten to (B, 1, 3L) with target slot replaced
    B = signal_3slot.size(0)
    flat = signal_3slot.clone().reshape(B, 1, 3 * L)
    s, e = target_slot * L, (target_slot + 1) * L
    flat[:, :, s:e] = x_t_target
    return flat

def rf_train_step(model, batch_signal, task, device):
    B, _, L = batch_signal.shape
    x0 = batch_signal[:, task.target_slot:task.target_slot+1, :].to(device)   # (B,1,L)
    eps = torch.randn_like(x0)
    t = sample_t_logit_normal(B, device)
    t_b = t.view(B, 1, 1)
    x_t = (1.0 - t_b) * x0 + t_b * eps
    v_target = eps - x0
    x_full = assemble_x_full(batch_signal.to(device), x_t, int(task.target_slot), L)
    v_pred = model(x_full, t, task)                                           # (B,1,L)
    return (v_pred - v_target).pow(2).mean()
```

## Model ([src/model_module/unicardio_rf.py](src/model_module/unicardio_rf.py), [src/model_module/backbone.py](src/model_module/backbone.py))

`UniCardioBackbone` mirrors `diff_CSDI` but with 3 slots instead of 4 and `FlowTimeEmbedding(t ∈ [0,1])` in place of `DiffusionEmbedding(integer_step)`. The backbone's `forward(x, t, mask, target_slot)` returns `(B, 1, L)` using the per-slot output head indexed by `target_slot`. No task embedding token is added (attention mask + per-slot head already disambiguate).

```python
class UniCardioRF(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.L = cfg.slot_length
        self.backbone = UniCardioBackbone(cfg)
    def forward(self, x_full, t, task):
        mask = build_task_mask(task.name, self.L, x_full.device)
        return self.backbone(x_full, t, mask, target_slot=int(task.target_slot))
```

## Sampler ([src/trainer_module/sampler.py](src/trainer_module/sampler.py))

Euler ODE, `t=1 → t=0`, default 8 steps:

```python
@torch.no_grad()
def euler_sample(model, conditions, task, n_steps=8, device="cuda"):
    B, _, L = conditions.shape
    x = torch.randn(B, 1, L, device=device)                   # x_1 ~ N(0, I)
    ts = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
    for i in range(n_steps):
        dt = ts[i+1] - ts[i]                                  # negative
        t_b = torch.full((B,), float(ts[i]), device=device)
        x_full = assemble_x_full(conditions.to(device), x, int(task.target_slot), L)
        v = model(x_full, t_b, task)
        x = x + v * dt
    return x                                                  # ≈ x_0
```

`n_samples > 1` via tiling the condition batch.

## Data Pipeline

[src/data_module/datamodule.py](src/data_module/datamodule.py):

1. `np.load(data_path)` → `(N, 3, 500)`, float32. File channel order is PPG=0, BP=1, ECG=2.
2. **Channel permute to model slot order**: `signals = signals[:, [2, 0, 1], :]` (now ECG=0, PPG=1, ABP=2). A constant `FILE_TO_MODEL_PERMUTATION = (2, 0, 1)` in [src/data_module/cardiac_dataset.py](src/data_module/cardiac_dataset.py) documents this mapping.
3. BP normalize on slot 2 (ABP): `signals[:, 2, :] = (signals[:, 2, :] - 100) / 50`.
4. Two-step split matching current `train_original.py`: `train_test_split(test_size=40000, random_state=42)` → 265,581 train + 40,000 temp; `train_test_split(temp, test_size=20000, random_state=42)` → 20,000 val + 20,000 test.
5. `CardiacDataset` returns `(signal,)` with shape `(3, 500)` in model slot order — no `signal_impute`, `signal_noisy`, or `mask`.
6. `DataLoader(batch_size=128, shuffle=True, num_workers=8, pin_memory=True)`.

## Hydra Configs

[run/conf/config.yaml](run/conf/config.yaml):
```yaml
defaults: [data: default, model: unicardio_rf, trainer: default, sampler: euler8, _self_]
seed: 42
device: cuda
output_dir: ${hydra:runtime.output_dir}
```

[run/conf/data/default.yaml](run/conf/data/default.yaml):
```yaml
data_path: ${oc.env:UNICARDIO_DATA, /Volumes/Disk/projects/UniCardio/data/Final_sig_combined.npy}
slot_length: 500
val_size: 20000
test_size: 20000
split_seed: 42
batch_size: 128
num_workers: 8
bp_offset: 100.0
bp_scale: 50.0
```

[run/conf/model/unicardio_rf.yaml](run/conf/model/unicardio_rf.yaml):
```yaml
slot_length: 500
n_slots: 3
channels: 288                # 6 kernels × 48
nheads: 8
n_layers: 5
time_embedding_dim: 256
kernel_sizes: [1, 3, 5, 7, 9, 11]
per_kernel_channels: 48
ffn_dim: 64
```

[run/conf/trainer/default.yaml](run/conf/trainer/default.yaml):
```yaml
epochs: 800
lr: 1.0e-3
weight_decay: 1.0e-6
optimizer: adam
lr_scheduler: {name: multistep, milestones_pct: [0.18, 0.75], gamma: 0.1}
val_every: 10
ckpt_every: 1
t_sampler: {name: logit_normal, mean: 0.0, std: 1.0}
```

[run/conf/sampler/euler8.yaml](run/conf/sampler/euler8.yaml):
```yaml
n_steps: 8
n_samples: 50
```

## File-by-File Migration

| Old file | Action |
|---|---|
| `base_model/diffusion_model_no_compress_final.py` | Extract `SignalEncoder` → `embeddings.py`; `ResidualBlock` → `residual_block.py`; `diff_CSDI` body restructured (3 slots, no `DiffusionEmbedding`) → `backbone.py`. **Drop**: `CSDI_base`, α/β buffers, 7 hard-coded masks, `DiffusionEmbedding`, `one_condition`/`two_conditions`/`three_conditions`, all samplers, `borrow_mode`, `mode==3` / `output_projection4_*`, `input_projection_modality_4`, `noisy_data_update`, `add_scaled_noise`. |
| `base_model/train_original.py` | **Delete.** Data loading → `datamodule.py`; entrypoint → `run/pipeline/train.py`. |
| `base_model/utils_together_original.py` | **Delete.** Salvage `SimpleCSVLogger` → `csv_logger.py`. Drop `thresholder`, `train()`. |
| `base_model/self_process.py` | **Delete entirely.** |
| `base_model/test_final.py` | **Delete.** Replaced by `run/pipeline/evaluate.py`. |
| `base_model/base_no_compress_original.yaml` | **Delete.** Replaced by Hydra configs. |
| `base_model/Final_sig_combined.npy` | **Move** → `data/Final_sig_combined.npy`. |
| `base_model/batch.pth` | **Move** → `data/batch.pth` (keep as optional smoke-test fixture). |
| `base_model/no_compress799.pth` | **Keep in `base_model/legacy_ckpt/`** for reference; new RF weights incompatible. |
| `base_model/` directory | Remove once migration verified. |

## Verification Plan

1. **Mask unit tests** ([tests/test_masks.py](tests/test_masks.py)): for each of the 5 tasks, assert (a) target-row diagonal block is `0`, (b) target-row condition-blocks are `0`, (c) non-participating slot rows are entirely `-inf`, (d) zero-cell count matches hand-computed expected count.

2. **RF step test** ([tests/test_rf_step.py](tests/test_rf_step.py)): assert `rf_train_step` produces finite scalar loss, `v_pred.shape == (B, 1, 500)`, `loss.requires_grad`.

3. **Sampler test** ([tests/test_sampler.py](tests/test_sampler.py)): assert `euler_sample` output has shape `(B, 1, 500)` and no NaN/Inf after 8 steps with randomly-initialized weights.

4. **Smoke test** ([run/pipeline/smoke_test.py](run/pipeline/smoke_test.py)) — ~5 min on laptop:
   - Load 1 batch (B=8).
   - For each task, run 200 steps with `lr=1e-3`. Assert final loss < 0.1× initial loss.
   - Assert grad on output heads of non-target slots is zero (verifies mask + head routing).
   - Run 8-step Euler sampler from overfit weights; assert `MSE(sample, x0_target) < 0.05`.

5. **Integration test**: 2-epoch run on 5K subset, batch_size=64, expect < 10 min on one GPU. Checkpoint round-trip (save → load → identical loss). All 5 tasks log decreasing loss.

6. **Full evaluation** ([run/pipeline/evaluate.py](run/pipeline/evaluate.py)): load best ckpt, for each task run `euler_sample(n_samples=50)` on the 20K test split, report RMSE / MAE / KS (BP in physical units via `bp_denormalize`). Save per-task CSV in `outputs/eval/`.

End-to-end command sequence after implementation:
```bash
python -m pytest tests/                                       # unit tests
python run/pipeline/smoke_test.py                             # sanity
python run/pipeline/train.py                                  # full training
python run/pipeline/evaluate.py +checkpoint=outputs/.../best.pt
```

## Edge Cases & Invariants

- **BP normalization applied exactly once**, in `datamodule.py` after the `[2, 0, 1]` channel permute, on model slot 2. `bp_denormalize` used only in `evaluate.py` for physical-unit metrics.
- **Channel permute is the single source of truth** for the file-vs-model slot mismatch: everything downstream (model, tasks, masks, loss, sampler, metrics) speaks exclusively in model slot indices (ECG=0, PPG=1, ABP=2). Evaluation outputs are re-labeled with human-readable modality names via `Slot(i).name`, not by touching tensor layout.
- **Shape asserts**: top of `UniCardioBackbone.forward` asserts `L_total == 3 * self.L`.
- **Reproducibility**: `set_seed(cfg.seed)` at entrypoints; `torch.backends.cudnn.deterministic` gated behind `cfg.deterministic` (off by default).
- **Checkpoint schema**: `{epoch, model_state, optimizer_state, lr_scheduler_state, config (OmegaConf dict), task_list, rng_state}`. Files: `epoch_{N:04d}.pt`, `latest.pt`, `best.pt` (best = min mean val loss across 5 tasks).
- **DataLoader `worker_init_fn`**: re-seeds numpy per worker.
- **`t` boundary**: `sigmoid(N(0,1))` avoids `t=0` / `t=1` with probability 1; no clipping.
- **Mask cache**: `lru_cache` keyed by `(task_name, L_slot, device_str)`; safe under DDP (per-process cache).
- **GPU wrapping**: default single-GPU; DDP gated behind `cfg.distributed` flag.
- **Macbook debugging**: `device` read from config; setting `device: mps` or `device: cpu` works for local sanity (CLAUDE.md rule).

## Critical Files for Implementation

- [src/model_module/unicardio_rf.py](src/model_module/unicardio_rf.py) — model wrapper
- [src/model_module/backbone.py](src/model_module/backbone.py) — 3-slot network body
- [src/model_module/attention_masks.py](src/model_module/attention_masks.py) — per-task masks
- [src/model_module/tasks.py](src/model_module/tasks.py) — Task spec & enum
- [src/trainer_module/rectified_flow.py](src/trainer_module/rectified_flow.py) — train step
- [src/trainer_module/sampler.py](src/trainer_module/sampler.py) — Euler ODE
- [src/trainer_module/trainer.py](src/trainer_module/trainer.py) — training loop
- [src/data_module/datamodule.py](src/data_module/datamodule.py) — data + splits
- [run/pipeline/train.py](run/pipeline/train.py), [run/pipeline/evaluate.py](run/pipeline/evaluate.py) — entrypoints
- [run/conf/](run/conf/) — Hydra configs

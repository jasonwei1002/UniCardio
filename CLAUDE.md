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

## Benchmark Reference (MD-ViSCo, IEEE JBHI 2026)

Direct point of comparison: **MD-ViSCo** (Meyer et al., KAIST; `paper reference/MD-ViSCo_*.pdf`, arXiv 2506.08357, github `fr-meyer/MD-ViSCo`). UniCardio targets the same multi-directional vital-sign waveform conversion problem on the same PulseDB benchmark, with the same evaluation protocol — only the modeling paradigm differs.

| Axis | MD-ViSCo | UniCardio |
|------|----------|-----------|
| Conversion direction | any-to-any across {ECG, PPG, ABP} | 5 task pairs (3 ABP-target active by default) |
| Modeling paradigm | Deterministic 1D U-Net + Swin Transformer encoder/decoder + AdaIN style injection (target type as style indicator); two-stage approximation + refinement (refinement = PatchTSMixer SBP/DBP regression + linear rescale to mmHg) | Two components, **trained separately but cascaded at inference**. **RF backbone**: Rectified Flow velocity prediction with Lipman/SD3 logit-normal `t` sampling; outputs a **shape-only** ABP waveform in the per-sample min-max `[0, 1]` space (absolute amplitude deliberately discarded). **BP head**: a MD-ViSCo-faithful `VitalEncoder` refinement head (`src/model_module/bp_head.py`) regressing scalar SBP/DBP — see "BP refinement head" below. The RF backbone is **pretrain-only**; only the BP head runs the two stages (pretrain → CalFree finetune). The eval cascade combines both: `pred` mmHg waveform = RF shape × the head's predicted `(SBP, DBP)`; the metric `target` is the **raw on-disk ABP** recovered by exactly inverting the dataset min-max with the per-segment extrema (`abp_minmax`) |
| Slot/channel scheme | Single-channel input + one-hot target-domain selector `d^{(j)}` fed through AdaIN `(γ, β)` | 3-slot concatenated `(ECG, PPG, ABP)` input + per-task attention mask + indexed per-slot output head |
| Datasets | PulseDB (VitalDB + MIMIC-III, ~5.2 M segments, 5 361 patients, 125 Hz, 1 280-pt segments) + UCI (MIMIC-II) | PulseDB only — `data/pulsedb/Train_Subset.npy` (2 506 subjects, 902 160 samples) + `data/pulsedb/CalFree_Test_Subset.npy` (279 subjects, 111 600 samples). **Zero subject overlap between Train_Subset and CalFree.** Slot length L=1 250 (10 s @ 125 Hz). |
| Pretrain protocol | **Patient-level (subject-disjoint) split** — "ensuring that the train and test sets contain no overlapping patients to assess whether the model generalizes to entirely unseen patients" | Stage-1 `mode='pretrain'`: `Train_Subset.npy` 80/20 train/val, `CalFree_Test_Subset.npy` as held-out test. Equivalent to MD-ViSCo's calibration-free pretrain. |
| Finetune protocol | "**Calibration-based** setting where the same patient may appear in both training and test sets" — original test set is re-split into new train/val/test | Stage-2 `mode='finetune'`: `CalFree_Test_Subset.npy` 80/10/10 sample-level split (subjects intentionally shared across train/val/test). Equivalent to MD-ViSCo's calibration-based finetune. |
| Reported metric | MSE + MAE on normalized waveform; AAMI/BHS on ABP in mmHg | RF velocity MSE during training; `evaluate_bp_test` reports per-task SBP/DBP ME+SD in mmHg on the stage-2 test split |

**Implications for code reviewers and future Claude instances:**

1. **The sample-level split inside stage-2 is by design, not a leakage bug.** It mirrors MD-ViSCo's calibration-based protocol and the standard PulseDB CalFree finetune recipe. Do not "fix" it to subject-disjoint without first reading this section.
2. **Stage-2 best.pt is selected on subjects already seen during stage-2 training.** That is the personalized BP setting — absolute mmHg accuracy in clinical workflows requires per-patient calibration, which is what this stage models.
3. **Stage-1 already gives calibration-free generalization signal** because `CalFree_Test_Subset.npy` (subject-disjoint from `Train_Subset`) is the stage-1 test split.
4. **`SignalEncoder` is modality-physics-level** (multi-scale Conv1d on raw ECG/PPG/ABP) and generalizes across subjects by construction; the per-subject mapping lives in the 5 ResidualBlocks and per-slot output heads, which is why `freeze_for_finetune(n_unfrozen_blocks)` exposes the transformer-block freeze depth but keeps the stem frozen by default.

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
# Unit tests (masks, RF step, sampler, BP head, tasks — 65 cases, ~1-3 s on CPU).
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

# Stage-2 RF CalFree finetune (OPTIONAL/legacy — current default is RF
# pretrain-only; calibration-based finetune is done by the BP head instead).
# Loads stage-1 weights, freezes backbone, only last N residual blocks +
# output heads trainable. Auto-runs RF + BP metric eval at the end.
bash finetune.sh run/outputs/<stage1>/checkpoints/best.pt
bash finetune.sh <ckpt> trainer.finetune.n_unfrozen_blocks=1 trainer.lr=5e-5

# Evaluation with a checkpoint. On the GPU server prefer `bash test.sh [ckpt]`,
# which auto-picks the latest `run/outputs/*/checkpoints/best.pt` if no path is given.
python run/pipeline/evaluate.py +checkpoint=run/outputs/<run>/checkpoints/best.pt
# Eval ABP-target tasks in mmHg via a trained BP head:
python run/pipeline/evaluate.py +checkpoint=<rf_ckpt> +bp_head_checkpoint=<bp_head_ckpt>

# BP refinement head — trained separately from the RF backbone, cascaded at eval.
# Pretrain on Train_Subset (calibration-free), then CalFree finetune.
bash train_bp_head.sh                                          # stage-1
bash finetune_bp_head.sh run/outputs/<bp_pretrain>/checkpoints/best.pt   # stage-2 + auto BP-MAE eval
python run/pipeline/train_bp_head.py device=cpu trainer.epochs=2 data.num_workers=0  # CPU debug
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

Both paths apply **per-sample min-max** normalization on model slot 2 (ABP) inside `CardiacDataset.__getitem__` — each segment is mapped to `[0, 1]` by its own `(min, max)`, so the RF backbone learns ABP *shape* only and absolute amplitude is recovered separately (the per-segment extrema are returned as `abp_minmax` for exact inversion at eval; the legacy global `(x - 100) / 50` affine in `src/utils/normalization.py` is deprecated and unused by current runs). ECG/PPG stay at raw scale. Everything downstream speaks exclusively in model slot indices. `split_seed: 42` for reproducibility.

**Dataset returns a 4-tuple** `(signal, sbp_dbp, demographics, abp_minmax)` (see `src/data_module/cardiac_dataset.py`):
- `signal`: `(3, L_slot)` model-slot waveform (as above).
- `sbp_dbp`: `(2,)` scalar SBP/DBP labels read from a **sibling `.csv`** (same dir + stem as the `.npy`, per-cycle mean labeling — *not* per-segment min/max). Missing CSV/columns → zero-filled with a WARNING.
- `demographics`: `(6,) = [age_z, gender, height_z, weight_z, bmi_z, mask]` (z-scored, last column = anthropometric-present mask), also from the CSV.
- `abp_minmax`: `(2,) = (dbp_seg, sbp_seg)` — the per-segment raw mmHg extrema (`abp.min()`/`abp.max()`) used for slot-2 min-max. **`evaluate.py` uses this to exactly invert the normalization and recover the raw on-disk ABP waveform as the metric target** (`reconstruct_mmHg(shape, sbp_seg, dbp_seg)` is the exact inverse), while `pred` = RF shape × the BP head's predicted `(SBP, DBP)`. These two amplitude anchors differ by construction (segment extrema vs per-cycle-mean labels) — a genuine, small part of the end-to-end mmHg error, consistent with MD-ViSCo measuring against the original waveform.

The **RF backbone trainer only consumes `signal`** (`batch[0]`); `sbp_dbp`/`demographics`/`abp_minmax` are ignored there (`abp_minmax` is eval-only). `sbp_dbp`/`demographics` feed the BP head. When `data.bp_label_norm` is set (default `{vmin: 40, vmax: 180}`), `sbp_dbp` is additionally min-max normalized to `[0, 1]` (MD-ViSCo Sec III.D); `BPLabelNorm.from_cfg(cfg.data).denormalize(...)` recovers mmHg at eval. Set it empty for raw-mmHg labels (legacy).

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
│   ├── bp_head.py        # MD-ViSCo VitalEncoder SBP/DBP regressor (cascaded w/ RF at eval)
│   └── unicardio_rf.py   # UniCardioRF wrapper + freeze_for_finetune
├── trainer_module/
│   ├── rectified_flow.py # sample_t_logit_normal, build_rf_inputs, rf_train_step
│   ├── sampler.py        # euler_sample (N-step ODE under Lipman convention)
│   ├── trainer.py        # RF train loop: task sampling, AMP, compile, scheduler, ckpt
│   ├── bp_head_trainer.py# BP-head train loop (L1 regression, single fixed task bp_task, no RF/sampling)
│   ├── bp_metrics.py     # evaluate_bp_test: SBP/DBP ME+SD via trained BP head (ckpt required)
│   └── csv_logger.py     # Append-only CSV logger
└── utils/                # seed, checkpoint (unwraps DP/DDP/OptimizedModule), normalization

run/
├── conf/                 # Hydra configs: data / model / trainer / sampler / root
├── pipeline/
│   ├── train.py          # RF Hydra entrypoint; dispatches stage='pretrain'|'finetune'
│   ├── train_bp_head.py  # BP-head Hydra entrypoint (config_bp_head.yaml)
│   ├── evaluate.py       # per-task RMSE/MAE/Pearson/KS; mmHg vs raw-ABP target w/ BP head, else shape-only [0,1]
│   └── smoke_test.py     # Per-task overfit-one-batch sanity test
└── outputs/              # Hydra run directories (gitignored)

tests/                    # pytest unit tests (masks, rf_step, sampler, bp_head, tasks) — 65 cases
script/                   # Reusable utilities (data conversion, swanlog pull, sample plot)
data/pulsedb/             # Active dataset (gitignored): Train_Subset.npy + CalFree_Test_Subset.npy
train.sh                  # RF Stage-1 pretrain one-shot (GPU server)
finetune.sh <ckpt>        # RF Stage-2 CalFree finetune (loads ckpt, freezes backbone)
test.sh [ckpt]            # 评估入口（默认挑最新 best.pt，可传入 ckpt 路径）
train_bp_head.sh          # BP-head Stage-1 pretrain
finetune_bp_head.sh <ckpt># BP-head Stage-2 CalFree finetune + BP-MAE eval
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

> **Current default: the RF backbone is pretrain-only.** The calibration-based finetune is now delegated to the BP head (see "BP refinement head"). The RF `finetune` path below (`finetune.sh`, `freeze_for_finetune`) still works and is kept as an optional/legacy route, but is not part of the active two-stage plan.

- **`pretrain`** (default, `bash train.sh`) = MD-ViSCo's **calibration-free** stage: full training on `Train_Subset.npy` 80/20 split. Stage-1 test = `CalFree_Test_Subset.npy` (subject-disjoint from Train_Subset, so this is a true generalization-to-unseen-patient evaluation).
- **`finetune`** (`bash finetune.sh <stage1 ckpt>`) = MD-ViSCo's **calibration-based** stage: loads `trainer.init_from` model weights → `UniCardioRF.freeze_for_finetune(n)` freezes the stem (`SignalEncoder` + LayerNorm + FlowTimeEmbedding) and unfreezes the last `trainer.finetune.n_unfrozen_blocks` `ResidualBlock`s plus all `output_heads` → fused Adam is built **after** the freeze, so `param_groups` only see trainable tensors. Data switches to `mode='finetune'` (CalFree 80/10/10 **sample-level**, so subjects are intentionally shared across train/val/test — this is the per-patient calibration model, not a leakage bug). After the last epoch, `best.pt` is reloaded and `bp_metrics.evaluate_bp_test` runs an 8-step Euler sampler on the stage-2 test split, logging per-task SBP/DBP ME+SD (mmHg) to SwanLab + `logs/finetune_bp_metrics.csv`. Set `trainer.finetune.n_unfrozen_blocks=5` to fully unfreeze the transformer (closest to MD-ViSCo's reported full-model finetune); stem stays frozen because it captures modality physics that does not need per-subject adaptation.

### BP refinement head

A **separate training pipeline** from the RF backbone (own model, configs, scripts), but the two are **combined into one evaluation cascade** — see "Feeding RF eval" below. Faithful to MD-ViSCo's `VitalEncoder`/`BPModel` (IEEE JBHI 2026 §6.2.2), it regresses scalar SBP/DBP directly from raw ECG+PPG (+ demographics) — it does **not** touch the RF velocity field, masks, or sampler during its own training.

- **Model** (`src/model_module/bp_head.py`, `BPHead`, ~385M params): per source vital (ECG slot 0, PPG slot 1) an independent stack — HF `PatchTSMixerModel` waveform encoder → `ProjectionHead` → fuse with a demographics embedding as a 2nd channel → second `PatchTSMixerModel` → `MlpBP` SBP/DBP dual head. Per-vital `(SBP, DBP)` are **averaged over the active vitals**. Two intentional deviations from MD-ViSCo's pi=true config (both deferred by the user, not simplifications): demographics come from a numeric MLP instead of DistilBERT, and loss is plain L1 (MD-ViSCo uses L1 + multi-WCL; WCL deferred). Config: `run/conf/model/bp_head.yaml` → `BPHeadConfig`.
- **Output space = training label space**: with `data.bp_label_norm` set (default), the head emits normalized `[0,1]`; `BPLabelNorm.denormalize` recovers mmHg downstream.
- **Training** (`run/pipeline/train_bp_head.py` → `src/trainer_module/bp_head_trainer.py`, config root `run/conf/config_bp_head.yaml` + `run/conf/trainer/bp_head.yaml`): `L1(pred, target)` objective, no time embedding / mask / Euler. **Single fixed task — NOT RF-style per-batch task sampling.** The head trains on one task (`trainer.bp_task`, default `ecgppg2abp`); `cond_slots_to_vitals(bp_task)` sets `active_vitals=["ecg","ppg"]` for every step (both vitals, averaged), matching how `bp_metrics` feeds the multimodal eval cascade. Same two-stage split as the RF backbone (`stage=pretrain` Train_Subset 80/20; `stage=finetune` CalFree 80/10/10). Eval is **per-task** (over the single trained task): val/test report ME±SD in mmHg; `best.pt` selected by the **minimum mean per-task validation L1 loss** (label space, `val_loss_mean`) — mirroring MD-ViSCo's "min val loss" criterion (MD-ViSCo monitors L1 + multi-WCL; WCL deferred here). Scripts: `train_bp_head.sh`, `finetune_bp_head.sh`.
- **Feeding RF eval**: `evaluate_bp_test` (`src/trainer_module/bp_metrics.py`) **requires** a trained BP head — `trainer.bp_head_ckpt` (RF finetune) or `+bp_head_checkpoint=` (evaluate.py) — denormalizing via `bp_norm`; SBP/DBP come straight from the head's scalar regression (no RF sampler, matching MD-ViSCo's scalar refinement metric). RF finetune skips BP eval with a WARNING when no head ckpt is configured. `cond_slots_to_vitals(task)` (`src/model_module/tasks.py`) restricts the head's vital averaging to the task's condition slots (e.g. `ppg2abp → ["ppg"]`, no ECG leakage).

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

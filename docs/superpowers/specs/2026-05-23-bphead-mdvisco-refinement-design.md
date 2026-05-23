# BPHead v3 — MD-ViSCo-faithful refinement head (Path A + B)

Date: 2026-05-23
Status: Approved (design), pending implementation plan

## Problem

In the **calibration-based** stage (stage-2 finetune; CalFree 80/10/10 sample-level,
subjects shared across train/val/test), the current scalar BP head
([src/model/bp_head.py](../../../src/model/bp_head.py)) reports SBP/DBP error
**SD ≫ 8 mmHg** — far above the AAMI ceiling (|ME| ≤ 5, SD ≤ 8). MD-ViSCo reaches
AAMI in this same personalized setting, so this is genuine underfitting of the
refinement head, not an inherent ceiling of the task.

The two-stage MD-ViSCo cascade itself is **already implemented and working** in
UniCardio and is explicitly out of scope:

- RF backbone outputs shape-only ABP — the dataset already per-sample min-max
  normalizes ABP slot 2 to [0,1] ([cardiac_dataset.py:224-227](../../../src/data_module/cardiac_dataset.py)).
- `reconstruct_mmHg = shape*(sbp-dbp)+dbp` ([normalization.py:141](../../../src/utils/normalization.py))
  is the MD-ViSCo `_unscale_waveform` formula.
- The cascade is wired in [evaluate.py:95-105](../../../run/pipeline/evaluate.py).

This work improves **only the scalar BP head accuracy** by making it faithful to
MD-ViSCo's refinement model along the two highest-expected-value axes for reducing
error variance.

## Goals / non-goals

**Goals**
- Reduce calibration-based SBP/DBP error SD toward AAMI (≤ 8 mmHg) by:
  - **Path A**: align the PatchTSMixer encoder hyperparameters to MD-ViSCo (§III.D / §6.2.2).
  - **Path B**: per-vital independent encoders + prediction averaging (ensemble → variance reduction), faithful to MD-ViSCo's `BPModel`.
- Fix a task-semantics inconsistency exposed by per-vital design (single-modality
  tasks currently leak the other modality into the scalar prediction).

**Non-goals (explicitly deferred)**
- WCL (weighted contrastive loss) — Path C; MD-ViSCo's own ablation shows
  architecture > WCL. Revisit only if A+B still fall short of AAMI.
- DistilBERT demographics — UniCardio has a fixed 5-feature PulseDB schema; the
  numeric MLP is the correct tool. DistilBERT is for cross-dataset schema
  agnosticism we do not need.
- RF backbone, dataset normalization, `reconstruct_mmHg`, cascade structure — unchanged.
- Joint/end-to-end RF+head training — MD-ViSCo also trains stages separately.

## Step 0 — non-blocking sanity check (recommended before retraining)

SD ≫ 8 in the calibration-based setting (where MD-ViSCo reaches AAMI) is large
enough to be suspicious of a label/measurement bug rather than pure modeling.
Before spending a server retrain, verify (≈30 min, no code change required):

1. **CSV ↔ npy row alignment** — [cardiac_dataset.py:90-94](../../../src/data_module/cardiac_dataset.py)
   only emits a WARNING on row-count mismatch then proceeds; a misaligned finetune
   split would inflate SD independent of the model.
2. **SBP/DBP not column-swapped**; `BPLabelNorm` round-trip (normalize→denormalize) is identity.
3. **`denormalize_diff`** is applied correctly in the ME/SD computation
   ([bp_head_trainer.py:150](../../../src/trainer_module/bp_head_trainer.py)).
4. **val MAE vs test SD**: val low + test high ⇒ overfit; both high ⇒ underfit/capacity.

This is the user's responsibility to eyeball before retraining; if a bug is found,
A+B may be unnecessary.

## Design

### Module 1 — Path A: PatchTSMixer hyperparameters → MD-ViSCo

Adopt MD-ViSCo §6.2.2 refinement encoder hyperparameters exactly:

| Hyperparam | Current | New (MD-ViSCo) |
|---|---|---|
| `patch_len`   | 50 | **5** |
| `patch_stride`| (= patch_len) | **5** (new field) |
| `dim` (d_model) | 192 | **64** |
| `depth`       | 6  | **15** |
| `mlp_ratio` (expansion_factor) | 2 | **5** |

MD-ViSCo §6.2.2 uses `patch_length=4` at L=1280; we use **patch_len=5** which
divides L=1250 evenly → **250 patches** (no dropped remainder), the closest clean
analogue. Patch count via the general floor formula: `n_patches =
floor((L - patch_len)/patch_stride) + 1 = 250`.

Required code change (small): in [bp_head.py](../../../src/model/bp_head.py)
- add `patch_stride: int = 5` to `BPHeadConfig` (and `from_mapping`);
- change `n_patches` property from the divisibility check to the floor formula
  (general; also handles non-dividing patch/stride if swept later);
- `_patchify` already uses `unfold(-1, patch_len, patch_len)` which drops any
  remainder consistently — generalize the `step` to `patch_stride`.

`pos_emb` shape `(1, n_patches, dim)` and the token-mix MLP (`Linear(n_patches, n_patches*mlp_ratio)`) scale with the new `n_patches=250`; verify memory/throughput on CPU smoke + server.

### Module 2 — Path B: per-vital encoders + averaging

Refactor `BPHead` from one 2-channel encoder to per-vital single-channel encoders,
mirroring MD-ViSCo's `BPModel` + `VitalEncoder`.

- `BPHeadConfig` gains `vitals: tuple[str, ...] = ("ecg", "ppg")`.
- Build `self.encoders = nn.ModuleDict({vital: <PatchTSMixer stack>})`, each with
  `in_channels=1`, sharing the Module-1 hyperparameters.
- **Demographics encoded once** (single shared `demo_encoder`), fused into each
  per-vital head (decision: shared, not per-vital re-encode — fewer params).
- Per-vital head: `concat(wave_emb_v, demo_emb) → MLP → (sbp_v, dbp_v)`.
- Aggregate by **mean over the selected vitals**: `(SBP, DBP) = mean_{v∈active} (sbp_v, dbp_v)`.

Channel→vital mapping uses model slot order ECG=0, PPG=1 (matches `signal[:, :2, :]`).

`forward(ecg_ppg, demographics=None, active_vitals=None) -> Tensor (B, 2)`:
- returns the `(B,2)` aggregate **averaged only over `active_vitals`**;
- `active_vitals`: optional iterable of vital names; default `None` = all configured
  vitals (this is what training uses, so the trainer call site is unchanged);
- with `demographics=None`, the demo branch is bypassed (`demo_emb = 0`) as today.

This single-Tensor return keeps the trainer untouched and makes Module 3 trivial:
the cascade just passes the task's vitals and receives the correctly-averaged
prediction. No per-vital dict / multi-output contract is introduced.

### Module 3 — per-task vital selection in the cascade

`TaskSpec.cond_slots` already encodes the source modalities (ECG=0, PPG=1;
[tasks.py:40-41](../../../src/model_module/tasks.py)). In
[evaluate.py](../../../run/pipeline/evaluate.py), map the task's `cond_slots` to
vital names and pass them as `active_vitals` so:
- `ppg2abp` → averages only the PPG head (no ECG leakage — current bug fixed);
- `ecg2abp` → only ECG head;
- `ecgppg2abp` → mean of both.

Slot.ABP (target) is never a vital input.

### Module 4 — training / metrics / verification

- **Training loop** [bp_head_trainer.py](../../../src/trainer_module/bp_head_trainer.py):
  input remains `signal[:, :2, :]` + demographics, loss remains MSE in normalized
  space. Training always uses both vitals (`active_vitals=None`); per-task selection
  is an eval-time concern. AAMI ME/SD logging unchanged.
- **Config** [bp_head.yaml](../../../run/conf/model/bp_head.yaml): set new defaults
  (Module 1) and add `vitals`, `patch_stride`.
- **Checkpoint compatibility**: param names change (ModuleDict of encoders); old
  bp_head ckpts will NOT load. Retrain stage-1 pretrain + stage-2 finetune. Acceptable.
- **Verification**:
  - CPU: extend/run an overfit-one-batch check on the new `BPHead.forward`
    (loss drops; output shape `(B,2)`).
  - Unit tests under [tests/](../../../tests/): `forward(active_vitals=None)` equals
    the mean of `forward(active_vitals=["ecg"])` and `forward(active_vitals=["ppg"])`
    (averaging correctness); single-vital calls differ from each other; `n_patches`
    floor formula; patchify shape for patch_len=5/stride=5, L=1250 → 250 patches.
  - Server: retrain, compare calibration-based test ME/SD vs the SD≫8 baseline.

## Components & interfaces

- `BPHeadConfig` (frozen dataclass): `+ patch_stride: int`, `+ vitals: tuple[str,...]`;
  `n_patches` → floor formula.
- `BPHead.forward(ecg_ppg, demographics=None, active_vitals=None) -> Tensor (B,2)`:
  returns the aggregate averaged over `active_vitals` (default = all vitals). Single
  Tensor return; no dict / multi-output contract.
- `build_bp_head` factory unchanged in signature.
- Cascade ([evaluate.py](../../../run/pipeline/evaluate.py)): compute `active_vitals`
  from `task.cond_slots`, pass to `bp_head(...)`, use the returned `(B,2)` directly.

## Risks

- **Finer patching cost**: 250 patches × depth 15 raises compute/memory vs current
  25 patches × depth 6. Two per-vital encoders ≈ 2× encoder params. Still small vs
  the 30M RF backbone; verify it fits and `compile.mode=default` still trains.
- **Retrain required** — both stages; expect a server iteration before any accuracy read.
- **Expected gain uncertain**: A (morphology capture) and B (ensemble variance
  reduction) are the strongest available levers for SD, but if Step-0 reveals a label
  bug, modeling changes are moot. Measure after Step 0.

## Success criteria

- New head trains and passes CPU overfit + unit tests.
- Calibration-based test SBP/DBP **SD decreases materially** vs the SD≫8 baseline;
  target trajectory toward AAMI (SD ≤ 8). Exact AAMI attainment is a stretch goal,
  not a gate for this change.
</content>

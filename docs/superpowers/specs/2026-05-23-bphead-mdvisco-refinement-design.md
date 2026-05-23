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

### Module 1 — Path A: replace custom block with HF `PatchTSMixerModel`

**Decision (revised after empirical check):** drop the hand-rolled
`_PatchTSMixerBlock` / `_patchify` and use HuggingFace
`transformers.PatchTSMixerModel` directly, exactly as MD-ViSCo does
([mdvisco.py `WaveformEncoder`](../../../paper%20reference/MD-ViSCo/src/model/mdvisco.py)).
This is the faithful reproduction and avoids re-deriving patch/mixer math.

Encoder config (per vital, MD-ViSCo §6.2.2 hyperparameters):

```python
PatchTSMixerConfig(
    context_length=slot_length,   # 1250
    num_input_channels=1,         # one vital per encoder
    d_model=64,
    num_layers=15,
    expansion_factor=5,
    patch_length=5,               # MD-ViSCo uses 4 @ L=1280; 5 divides 1250 evenly → 250 patches
    patch_stride=5,
)
```

**Measured cost (verified on transformers 5.9.0, CPU):** one such encoder =
**11.0M params**; `last_hidden_state` shape `(B, 1, num_patches=250, d_model=64)`.
Two per-vital encoders ⇒ **~22M** for the head. This is inherent to depth-15 /
patch-5 (the custom block was ~the same); it is **23× the current ~0.95M head**,
accepted as the cost of faithful reproduction. Fits H800 80GB; training
compute/memory rises accordingly.

**New dependency:** `transformers` is **not** currently in
[requirements.txt](../../../requirements.txt) — it must be added.

Pooling: GAP over the patch axis of `last_hidden_state` → `(B, d_model=64)`
`wave_emb`. We deliberately do **not** replicate MD-ViSCo's flatten+`ProjectionHead`
(16000→512 ≈ 8.5M extra per vital); GAP keeps the head at ~22M instead of ~40M+.

### Module 2 — Path B: per-vital encoders + averaging

Refactor `BPHead` from one 2-channel encoder to per-vital single-channel encoders,
mirroring MD-ViSCo's `BPModel` + `VitalEncoder`.

- `BPHeadConfig` gains `vitals: tuple[str, ...] = ("ecg", "ppg")` and HF-aligned
  fields (`d_model`, `num_layers`, `expansion_factor`, `patch_len`, `patch_stride`);
  the old `dim` / `depth` / `mlp_ratio` fields are renamed accordingly.
- Build `self.encoders = nn.ModuleDict({vital: PatchTSMixerModel(...)})`, each
  single-channel, sharing the Module-1 config.
- Per-vital forward: slice `(B,1,L)` → transpose to `(B,L,1)` → encoder →
  `last_hidden_state (B,1,250,64)` → **GAP over patch axis** → `wave_emb (B,64)`.
- **Demographics encoded once** (single shared `demo_encoder`), fused into each
  per-vital head (decision: shared, not per-vital re-encode — fewer params).
- Per-vital head (one per vital in a `ModuleDict`):
  `concat(wave_emb_v, demo_emb) → Linear → GELU → Linear → (sbp_v, dbp_v)`.
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
[tasks.py:40-41](../../../src/model_module/tasks.py)). A shared helper maps
`cond_slots` → vital names and is passed as `active_vitals` so:
- `ppg2abp` → averages only the PPG head (no ECG leakage — current bug fixed);
- `ecg2abp` → only ECG head;
- `ecgppg2abp` → mean of both.

Slot.ABP (target) is never a vital input.

**Two call sites must both be updated** (both currently hard-code
`bp_head(signal[:, :2, :], demographics)` for every task):
- [evaluate.py:96](../../../run/pipeline/evaluate.py) (`_eval_task`);
- [bp_metrics.py:117](../../../src/trainer_module/bp_metrics.py) (`_predict_one_task`,
  used by the stage-2 finetune auto-eval).

The trainer's own training call ([bp_head_trainer.py](../../../src/trainer_module/bp_head_trainer.py))
stays as-is (`active_vitals=None` → both vitals).

### Module 4 — training / metrics / verification

- **Training loop** [bp_head_trainer.py](../../../src/trainer_module/bp_head_trainer.py):
  input remains `signal[:, :2, :]` + demographics, loss remains MSE in normalized
  space. Training always uses both vitals (`active_vitals=None`); per-task selection
  is an eval-time concern. AAMI ME/SD logging unchanged.
- **Config** [bp_head.yaml](../../../run/conf/model/bp_head.yaml): set new HF-aligned
  defaults and add `vitals`, `patch_stride`, `d_model`, `num_layers`, `expansion_factor`.
- **Dependency**: add `transformers` to [requirements.txt](../../../requirements.txt).
- **Checkpoint compatibility**: param names change (ModuleDict of HF encoders); old
  bp_head ckpts will NOT load. Retrain stage-1 pretrain + stage-2 finetune. Acceptable.
- **Verification**:
  - CPU: overfit-one-batch check on the new `BPHead.forward` with a **small** config
    (few layers / small d_model) so it stays fast (loss drops; output shape `(B,2)`).
  - Unit tests under [tests/](../../../tests/): `forward(active_vitals=None)` equals
    the mean of `forward(active_vitals=["ecg"])` and `forward(active_vitals=["ppg"])`
    (averaging correctness); single-vital calls differ; default-config param budget
    in the **15M–30M** band (≈22M); HF encoder output pooled to `(B,64)`.
  - Server: retrain, compare calibration-based test ME/SD vs the SD≫8 baseline.

## Components & interfaces

- `BPHeadConfig` (frozen dataclass): HF-aligned fields `d_model`, `num_layers`,
  `expansion_factor`, `patch_len`, `patch_stride`, `vitals`, `demo_in`, `demo_hidden`,
  `fusion_hidden`.
- `BPHead.forward(ecg_ppg, demographics=None, active_vitals=None) -> Tensor (B,2)`:
  returns the aggregate averaged over `active_vitals` (default = all vitals). Single
  Tensor return; no dict / multi-output contract.
- `build_bp_head` factory unchanged in signature.
- `cond_slots_to_vitals(task) -> list[str]` helper (new): maps `task.cond_slots`
  (ECG/PPG, skipping ABP) to vital names; used by both eval call sites.
- Cascade ([evaluate.py](../../../run/pipeline/evaluate.py),
  [bp_metrics.py](../../../src/trainer_module/bp_metrics.py)): compute `active_vitals`
  from `task.cond_slots`, pass to `bp_head(...)`, use the returned `(B,2)` directly.

## Risks

- **Head size ~22M**: depth-15 / patch-5 HF PatchTSMixer × 2 vitals. ~23× the current
  head; fits H800 80GB but raises training compute/memory. Verify `compile` and AMP
  still train within budget on the server.
- **New `transformers` dependency** pulled into the project for this head.
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

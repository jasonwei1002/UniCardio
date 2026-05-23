# BPHead v3 — MD-ViSCo-faithful refinement head Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve calibration-based SBP/DBP accuracy (SD ≫ 8 today) by rebuilding the Path-A BP head as a faithful MD-ViSCo refinement model: per-vital HuggingFace `PatchTSMixer` encoders + shared demographics MLP + per-vital heads averaged over the task's active vitals.

**Architecture:** Each source vital (ECG slot 0, PPG slot 1) is encoded by its own HF `PatchTSMixerModel` (d_model=64, num_layers=15, patch_length=5/stride=5 over L=1250 → 250 patches, ~11M each). The patch-axis GAP embedding `(B,64)` is concatenated with a single shared demographics embedding and mapped to `(SBP,DBP)` by a per-vital head; predictions are **averaged over the active vitals**. Eval call sites pass the task's source modalities so single-modality tasks (`ppg2abp`) no longer leak the other vital.

**Tech Stack:** PyTorch ≥2.5, HuggingFace `transformers` (new dependency), Hydra/OmegaConf, pytest.

**Spec:** `docs/superpowers/specs/2026-05-23-bphead-mdvisco-refinement-design.md`

---

## File structure

- `requirements.txt` — add `transformers` (modify).
- `src/model_module/bp_head.py` — full rewrite: HF encoders, per-vital, `active_vitals` (modify, ~200 lines).
- `src/model_module/tasks.py` — add `cond_slots_to_vitals` helper (modify).
- `run/conf/model/bp_head.yaml` — HF-aligned defaults + `vitals`/`patch_stride` (modify).
- `run/pipeline/evaluate.py` — pass `active_vitals` in `_eval_task` (modify, line ~96).
- `src/trainer_module/bp_metrics.py` — pass `active_vitals` in `_predict_one_task` (modify, line ~117).
- `tests/test_bp_head.py` — rewrite for new architecture (modify).
- `tests/test_tasks.py` — new: `cond_slots_to_vitals` (create).

Out of scope (do not touch): RF backbone, dataset normalization, `reconstruct_mmHg`, `bp_head_trainer.py` training call (stays `active_vitals=None`).

---

## Task 1: Add `transformers` dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add the dependency line**

In `requirements.txt`, after the `swanlab` line (line 22), add:

```
# src/model_module/bp_head.py 用 HF PatchTSMixer 作为 refinement 波形编码器
transformers>=4.40
```

- [ ] **Step 2: Verify it imports and exposes PatchTSMixer**

Run: `python -c "from transformers import PatchTSMixerModel, PatchTSMixerConfig; print('ok')"`
Expected: prints `ok` (install with `pip install 'transformers>=4.40'` first if missing).

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "build(bp_head): add transformers dependency for HF PatchTSMixer"
```

---

## Task 2: Add `cond_slots_to_vitals` helper

**Files:**
- Modify: `src/model_module/tasks.py`
- Test: `tests/test_tasks.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_tasks.py`:

```python
"""Unit tests for task → vital-name mapping used by the BP-head cascade."""

from __future__ import annotations

from src.model_module.tasks import TASK_SPECS, cond_slots_to_vitals


def test_cond_slots_to_vitals_single_ppg() -> None:
    assert cond_slots_to_vitals(TASK_SPECS["ppg2abp"]) == ["ppg"]


def test_cond_slots_to_vitals_single_ecg() -> None:
    assert cond_slots_to_vitals(TASK_SPECS["ecg2abp"]) == ["ecg"]


def test_cond_slots_to_vitals_multimodal() -> None:
    assert cond_slots_to_vitals(TASK_SPECS["ecgppg2abp"]) == ["ecg", "ppg"]


def test_cond_slots_to_vitals_skips_non_ecg_ppg() -> None:
    # ecg2ppg conditions on ECG only; PPG/ABP targets never appear as vitals.
    assert cond_slots_to_vitals(TASK_SPECS["ecg2ppg"]) == ["ecg"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tasks.py -q`
Expected: FAIL with `ImportError: cannot import name 'cond_slots_to_vitals'`.

- [ ] **Step 3: Implement the helper**

Append to `src/model_module/tasks.py` (after `active_task_pairs`):

```python
# Slot → BP-head vital name. Only ECG/PPG are valid BP-head inputs; ABP is the
# target waveform and is never fed to the scalar head.
_SLOT_TO_VITAL: dict[Slot, str] = {Slot.ECG: "ecg", Slot.PPG: "ppg"}


def cond_slots_to_vitals(task: TaskSpec) -> list[str]:
    """Map a task's condition slots to BP-head vital names (ECG/PPG only).

    Used by the evaluation cascade so the BP head averages only over the
    modalities the task actually conditions on (e.g. ``ppg2abp`` -> ``["ppg"]``,
    no ECG leakage). Order follows ``task.cond_slots``.
    """
    return [_SLOT_TO_VITAL[s] for s in task.cond_slots if s in _SLOT_TO_VITAL]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_tasks.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/model_module/tasks.py tests/test_tasks.py
git commit -m "feat(tasks): add cond_slots_to_vitals for per-task BP-head vital selection"
```

---

## Task 3: Rewrite `BPHead` with HF PatchTSMixer + per-vital + `active_vitals`

**Files:**
- Modify: `src/model_module/bp_head.py` (full rewrite)
- Test: `tests/test_bp_head.py` (full rewrite)

- [ ] **Step 1: Write the failing tests**

Replace the entire contents of `tests/test_bp_head.py` with:

```python
"""Unit tests for :class:`src.model_module.bp_head.BPHead` (Path A v3).

v3 = per-vital HF PatchTSMixer encoders + shared demographics MLP + per-vital
heads averaged over ``active_vitals``. Input contract:
``forward(ecg_ppg: (B, 2, slot_length), demographics: (B, 6) | None,
         active_vitals: Iterable[str] | None)``.
"""

from __future__ import annotations

import pytest
import torch

from src.model_module.bp_head import BPHead, BPHeadConfig, build_bp_head

SLOT_LEN = 1250
DEMO_DIM = 6

# Small config keeps PatchTSMixer init + forward fast on CPU.
TINY = {
    "slot_length": 256,
    "patch_len": 32,
    "patch_stride": 32,
    "d_model": 16,
    "num_layers": 2,
    "expansion_factor": 2,
    "demo_hidden": 16,
    "fusion_hidden": 32,
}


def _tiny() -> BPHead:
    return build_bp_head(TINY)


def test_forward_shape_with_demographics() -> None:
    model = _tiny()
    x = torch.randn(4, 2, TINY["slot_length"])
    d = torch.randn(4, DEMO_DIM)
    out = model(x, d)
    assert out.shape == (4, 2)
    assert out.dtype == torch.float32


def test_forward_shape_without_demographics() -> None:
    model = _tiny()
    x = torch.randn(4, 2, TINY["slot_length"])
    out = model(x)
    assert out.shape == (4, 2)


def test_aggregate_equals_mean_of_single_vitals() -> None:
    """In eval mode (no dropout), the both-vital average equals the mean of
    the two single-vital predictions."""
    model = _tiny().eval()
    x = torch.randn(4, 2, TINY["slot_length"])
    d = torch.randn(4, DEMO_DIM)
    with torch.no_grad():
        both = model(x, d, active_vitals=None)
        ecg = model(x, d, active_vitals=["ecg"])
        ppg = model(x, d, active_vitals=["ppg"])
    assert torch.allclose(both, (ecg + ppg) / 2, atol=1e-5)
    assert not torch.allclose(ecg, ppg)


def test_active_vitals_single_uses_only_that_encoder() -> None:
    model = _tiny().eval()
    x = torch.randn(2, 2, TINY["slot_length"])
    # Zeroing the PPG channel must not change an ECG-only prediction.
    x_zero_ppg = x.clone()
    x_zero_ppg[:, 1, :] = 0.0
    with torch.no_grad():
        a = model(x, active_vitals=["ecg"])
        b = model(x_zero_ppg, active_vitals=["ecg"])
    assert torch.allclose(a, b, atol=1e-6)


def test_backward_grads() -> None:
    model = _tiny()
    x = torch.randn(4, 2, TINY["slot_length"])
    d = torch.randn(4, DEMO_DIM)
    target = torch.tensor([[120.0, 70.0]] * 4)
    loss = ((model(x, d) - target) ** 2).mean()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"missing grad: {name}"


def test_overfit_single_batch() -> None:
    """Tiny BPHead overfits a synthetic batch to MAE < 5 mmHg."""
    torch.manual_seed(0)
    model = _tiny()
    optim = torch.optim.Adam(model.parameters(), lr=3e-3)
    B, L = 8, TINY["slot_length"]
    x = torch.randn(B, 2, L)
    d = torch.randn(B, DEMO_DIM)
    proj = torch.randn(2 * L + DEMO_DIM, 2)
    flat = torch.cat([x.reshape(B, -1), d], dim=-1)
    base = torch.tensor([100.0, 60.0])
    scale = torch.tensor([20.0, 10.0])
    target = base + scale * (flat @ proj / proj.shape[0] ** 0.5)
    for _ in range(400):
        optim.zero_grad(set_to_none=True)
        loss = ((model(x, d) - target) ** 2).mean()
        loss.backward()
        optim.step()
    final_mae = float((model(x, d) - target).abs().mean().item())
    assert final_mae < 5.0, f"final MAE {final_mae:.2f} mmHg (failed to overfit)"


def test_rejects_mismatched_slot_length() -> None:
    model = _tiny()
    bad = torch.randn(2, 2, TINY["slot_length"] + 7)
    with pytest.raises(ValueError, match="BPHead expects"):
        model(bad)


def test_config_from_mapping_defaults_and_override() -> None:
    cfg = BPHeadConfig.from_mapping({"d_model": 32, "num_layers": 3})
    assert cfg.d_model == 32
    assert cfg.num_layers == 3
    assert cfg.slot_length == 1250
    assert cfg.vitals == ("ecg", "ppg")
    assert BPHeadConfig.from_mapping(cfg) is cfg


def test_config_rejects_unknown_vital() -> None:
    with pytest.raises(ValueError, match="unknown vital"):
        build_bp_head({**TINY, "vitals": ["ecg", "abp"]})


def test_default_config_param_budget() -> None:
    """Faithful MD-ViSCo config (depth 15, d_model 64, 2 vitals) ~22M params.

    Instantiation only (no forward), so it stays fast (~1-2s) on CPU.
    """
    model = build_bp_head({})
    n = model.num_parameters()
    assert 15_000_000 < n < 30_000_000, f"param count {n:,} outside 15M-30M"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_bp_head.py -q`
Expected: FAIL (old `BPHead` lacks `active_vitals`, `vitals`, HF encoders, the `num_layers`/`d_model` fields).

- [ ] **Step 3: Rewrite `src/model_module/bp_head.py`**

Replace the entire file with:

```python
"""Path A refinement model — per-vital HF PatchTSMixer + demographics MLP.

Faithful reproduction of MD-ViSCo's refinement model (IEEE JBHI 2026,
Sec III.D / §6.2.2): each source vital (ECG slot 0, PPG slot 1) is encoded by
its own HuggingFace ``PatchTSMixerModel``; the GAP-pooled patch embedding is
fused with a single shared demographics embedding and mapped to ``(SBP, DBP)``
by a per-vital head. Per-vital predictions are averaged over the active vitals.

Intentional differences vs MD-ViSCo:
- demographics via a numeric MLP (PulseDB has a fixed 5-feature schema), not
  DistilBERT (which exists for cross-dataset schema agnosticism we do not need);
- GAP pooling over patches instead of flatten + ProjectionHead (keeps the head
  at ~22M instead of ~40M+);
- no WCL (weighted contrastive loss) — MD-ViSCo's ablation shows architecture >
  WCL; revisit only if this head still falls short of AAMI.

``demographics`` is ``(B, 6) = [age_z, gender, height_z, weight_z, bmi_z, mask]``
(see :mod:`src.data_module.cardiac_dataset`). Pass ``None`` for waveform-only
paths; the demographic branch is then bypassed (``demo_emb = 0``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

import torch
from torch import Tensor, nn
from transformers import PatchTSMixerConfig, PatchTSMixerModel

# Model slot order is ECG=0, PPG=1 (see src/model_module/tasks.py). The BP head
# receives ``signal[:, :2, :]`` so these indices line up directly.
_VITAL_CHANNEL: dict[str, int] = {"ecg": 0, "ppg": 1}


@dataclass(frozen=True)
class BPHeadConfig:
    """Immutable config for :class:`BPHead` (HF PatchTSMixer hyperparameters
    follow MD-ViSCo §6.2.2)."""

    slot_length: int = 1250
    vitals: tuple[str, ...] = ("ecg", "ppg")
    # HF PatchTSMixer encoder hyperparameters.
    patch_len: int = 5            # MD-ViSCo uses 4 @ L=1280; 5 divides 1250 → 250 patches
    patch_stride: int = 5
    d_model: int = 64
    num_layers: int = 15
    expansion_factor: int = 5
    # Demographics + fusion head.
    demo_in: int = 6              # 5 numeric features + 1 missingness mask
    demo_hidden: int = 64
    fusion_hidden: int = 128

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any] | "BPHeadConfig") -> "BPHeadConfig":
        if isinstance(cfg, cls):
            return cfg
        d = cls()
        vitals = cfg.get("vitals", d.vitals)
        return cls(
            slot_length=int(cfg.get("slot_length", d.slot_length)),
            vitals=tuple(str(v).lower() for v in vitals),
            patch_len=int(cfg.get("patch_len", d.patch_len)),
            patch_stride=int(cfg.get("patch_stride", d.patch_stride)),
            d_model=int(cfg.get("d_model", d.d_model)),
            num_layers=int(cfg.get("num_layers", d.num_layers)),
            expansion_factor=int(cfg.get("expansion_factor", d.expansion_factor)),
            demo_in=int(cfg.get("demo_in", d.demo_in)),
            demo_hidden=int(cfg.get("demo_hidden", d.demo_hidden)),
            fusion_hidden=int(cfg.get("fusion_hidden", d.fusion_hidden)),
        )


def _build_encoder(cfg: BPHeadConfig) -> PatchTSMixerModel:
    """One single-channel HF PatchTSMixer encoder per vital."""
    return PatchTSMixerModel(
        PatchTSMixerConfig(
            context_length=cfg.slot_length,
            num_input_channels=1,
            d_model=cfg.d_model,
            num_layers=cfg.num_layers,
            expansion_factor=cfg.expansion_factor,
            patch_length=cfg.patch_len,
            patch_stride=cfg.patch_stride,
        )
    )


class BPHead(nn.Module):
    """(ECG, PPG, demographics) -> (SBP, DBP) mmHg regressor (per-vital averaged)."""

    def __init__(self, config: BPHeadConfig | Mapping[str, Any]) -> None:
        super().__init__()
        self.config = BPHeadConfig.from_mapping(config)
        cfg = self.config
        for v in cfg.vitals:
            if v not in _VITAL_CHANNEL:
                raise ValueError(
                    f"unknown vital '{v}'; BP head supports {sorted(_VITAL_CHANNEL)}"
                )

        self.encoders = nn.ModuleDict({v: _build_encoder(cfg) for v in cfg.vitals})

        self.demo_encoder = nn.Sequential(
            nn.Linear(cfg.demo_in, cfg.demo_hidden),
            nn.GELU(),
            nn.Linear(cfg.demo_hidden, cfg.demo_hidden),
        )

        self.heads = nn.ModuleDict(
            {
                v: nn.Sequential(
                    nn.Linear(cfg.d_model + cfg.demo_hidden, cfg.fusion_hidden),
                    nn.GELU(),
                    nn.Linear(cfg.fusion_hidden, 2),
                )
                for v in cfg.vitals
            }
        )

    def _encode(self, vital: str, slot: Tensor) -> Tensor:
        """``(B, 1, L)`` raw waveform -> ``(B, d_model)`` GAP patch embedding."""
        # HF PatchTSMixer expects (B, seq_len, channels); we hold (B, 1, L).
        hidden = self.encoders[vital](slot.transpose(1, 2)).last_hidden_state
        # last_hidden_state: (B, channels=1, num_patches, d_model). GAP over
        # the patch axis, drop the singleton channel axis -> (B, d_model).
        return hidden.mean(dim=2).squeeze(1)

    def forward(
        self,
        ecg_ppg: Tensor,
        demographics: Optional[Tensor] = None,
        active_vitals: Optional[Iterable[str]] = None,
    ) -> Tensor:
        """Predict ``(SBP, DBP)`` mmHg, averaged over the active vitals.

        Args:
            ecg_ppg: ``(B, 2, slot_length)`` raw ECG (ch 0) + PPG (ch 1).
            demographics: ``(B, demo_in)`` z-scored demographics with missingness
                mask in the last column; ``None`` bypasses the demo branch.
            active_vitals: vital names to average over; ``None`` = all configured
                vitals (training uses this default — call site unchanged).

        Returns:
            ``(B, 2)`` with column 0 = SBP, column 1 = DBP, both in mmHg.
        """
        cfg = self.config
        b, c, length = ecg_ppg.shape
        if length != cfg.slot_length or c < 2:
            raise ValueError(
                f"BPHead expects (B, >=2, {cfg.slot_length}); got (B, {c}, {length})."
            )

        vitals = (
            tuple(str(v).lower() for v in active_vitals)
            if active_vitals is not None
            else cfg.vitals
        )
        if not vitals:
            raise ValueError("active_vitals resolved to an empty set.")

        if demographics is None:
            demo_emb = ecg_ppg.new_zeros(b, cfg.demo_hidden)
        else:
            demo_emb = self.demo_encoder(demographics)

        preds = []
        for v in vitals:
            ch = _VITAL_CHANNEL[v]
            wave_emb = self._encode(v, ecg_ppg[:, ch : ch + 1, :])
            preds.append(self.heads[v](torch.cat([wave_emb, demo_emb.to(wave_emb.dtype)], dim=-1)))
        return torch.stack(preds, dim=0).mean(dim=0)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_bp_head(config: Mapping[str, Any] | BPHeadConfig) -> BPHead:
    """Factory helper mirroring other model-module entry points."""
    return BPHead(config)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_bp_head.py -q`
Expected: PASS (all tests; the ~22M default-config budget test instantiates in ~1-2s).

- [ ] **Step 5: Commit**

```bash
git add src/model_module/bp_head.py tests/test_bp_head.py
git commit -m "feat(bp_head): per-vital HF PatchTSMixer encoders with active_vitals averaging"
```

---

## Task 4: Wire `active_vitals` into the two eval call sites

**Files:**
- Modify: `run/pipeline/evaluate.py` (line ~34 import, ~96 call)
- Modify: `src/trainer_module/bp_metrics.py` (line ~117 call)

- [ ] **Step 1: Update `evaluate.py` import**

In `run/pipeline/evaluate.py`, change the tasks import (line 34) from:

```python
from src.model_module.tasks import Slot, active_task_pairs
```

to:

```python
from src.model_module.tasks import Slot, active_task_pairs, cond_slots_to_vitals
```

- [ ] **Step 2: Pass `active_vitals` in `_eval_task`**

In `run/pipeline/evaluate.py`, inside `_eval_task`, change the BP-head call (line ~96) from:

```python
            bp_pred = bp_head(signal[:, :2, :], demographics).float()
```

to:

```python
            bp_pred = bp_head(
                signal[:, :2, :], demographics,
                active_vitals=cond_slots_to_vitals(task),
            ).float()
```

- [ ] **Step 3: Update `bp_metrics.py`**

In `src/trainer_module/bp_metrics.py`, add the import near the other `..model_module.tasks` import (search for `from ..model_module.tasks`):

```python
from ..model_module.tasks import cond_slots_to_vitals
```

Then in `_predict_one_task`, change the BP-head call (line ~117) from:

```python
            bp_pred = bp_head(signal[:, :2, :], demographics).float()
```

to:

```python
            bp_pred = bp_head(
                signal[:, :2, :], demographics,
                active_vitals=cond_slots_to_vitals(task),
            ).float()
```

- [ ] **Step 4: Verify both modules import cleanly**

Run: `python -c "import run.pipeline.evaluate; import src.trainer_module.bp_metrics; print('ok')"`
Expected: prints `ok` (no ImportError / NameError).

- [ ] **Step 5: Commit**

```bash
git add run/pipeline/evaluate.py src/trainer_module/bp_metrics.py
git commit -m "feat(bp_head): select per-task vitals in eval cascade (fix single-modality leakage)"
```

---

## Task 5: Update `bp_head.yaml` to the faithful MD-ViSCo config

**Files:**
- Modify: `run/conf/model/bp_head.yaml`

- [ ] **Step 1: Replace the config body**

Replace the parameter block of `run/conf/model/bp_head.yaml` (everything after the header comment) with:

```yaml
# BPHead v3 (Path A refinement) — faithful MD-ViSCo §6.2.2 reproduction.
#
# Per-vital HF PatchTSMixer encoders (one per source modality) + shared
# demographics MLP + per-vital heads averaged over the task's active vitals.
# ~22M params (2 × ~11M encoders); inherent to depth-15 / patch-5. Output is
# raw mmHg; loss = MSE in the normalized BP-label space (see data.bp_label_norm).

slot_length: ${data.slot_length}
vitals: [ecg, ppg]          # model slots 0, 1

# HF PatchTSMixer hyperparameters (MD-ViSCo §6.2.2).
patch_len: 5                # 5 divides 1250 → 250 patches (MD-ViSCo uses 4 @ 1280)
patch_stride: 5
d_model: 64
num_layers: 15
expansion_factor: 5

# Demographics + fusion head.
demo_in: 6                  # 5 numeric demographics + 1 missingness mask
demo_hidden: 64
fusion_hidden: 128
```

- [ ] **Step 2: Verify Hydra composes the config and builds the model**

Run:
```bash
python -c "
from hydra import compose, initialize
with initialize(version_base='1.3', config_path='run/conf'):
    cfg = compose(config_name='config_bp_head')
from src.model_module.bp_head import build_bp_head
m = build_bp_head(cfg.model)
print('params(M):', round(m.num_parameters()/1e6, 1))
"
```
Expected: prints `params(M): ~22.0` (15M–30M band), no Hydra interpolation error.

- [ ] **Step 3: Commit**

```bash
git add run/conf/model/bp_head.yaml
git commit -m "feat(bp_head): faithful MD-ViSCo PatchTSMixer config (d_model 64, depth 15, patch 5)"
```

---

## Task 6: Full verification pass

**Files:** none (verification only)

- [ ] **Step 1: Run the full unit-test suite**

Run: `python -m pytest tests/ -q`
Expected: all pass (masks, rf_step, sampler, minmax_norm, bp_head, tasks).

- [ ] **Step 2: Run the existing RF smoke test (regression guard)**

Run: `python run/pipeline/smoke_test.py`
Expected: completes; per-task loss drop + Euler reconstruction (unchanged — RF untouched).

- [ ] **Step 3: CPU mini integration of the BP-head trainer path (synthetic)**

Run:
```bash
python -c "
import torch
from src.model_module.bp_head import build_bp_head
m = build_bp_head({'slot_length':256,'patch_len':32,'patch_stride':32,'d_model':16,'num_layers':2,'expansion_factor':2}).train()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
sig = torch.randn(8,3,256); tgt = torch.rand(8,2); demo = torch.randn(8,6)
l0 = ((m(sig[:,:2,:],demo)-tgt)**2).mean(); l0.backward(); opt.step()
print('trainer-path loss ok:', float(l0.item()) > 0)
"
```
Expected: prints `trainer-path loss ok: True` (mirrors `bp_head_trainer.py`'s `model(signal[:, :2, :], demographics)` call with `active_vitals=None`).

- [ ] **Step 4: Final commit (only if Steps produced changes)**

No code changes expected in Task 6; if verification surfaced a fix, commit it with a descriptive message. Otherwise nothing to commit.

---

## Post-implementation (server, out of plan scope)

These are **manual, on the GPU server** — not part of automated execution:

1. **Step-0 sanity (recommended before retraining):** verify CSV↔npy alignment and SBP/DBP column order on the CalFree finetune split (see spec Step 0). A label bug would make the modeling change moot.
2. Retrain stage-1: `bash train_bp_head.sh` (old ckpts are incompatible — fresh train).
3. Retrain stage-2: `bash finetune_bp_head.sh <stage1 best.pt>`.
4. Compare calibration-based test SBP/DBP **SD** vs the SD ≫ 8 baseline; check `compile.mode=default` + bf16 fit within H800 memory at ~22M head.
```

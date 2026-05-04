---
name: verify-training
description: Run a quick forward pass sanity check on the UniCardio model using a small synthetic batch. Use after modifying model architecture or training code to catch errors before launching a full training run.
user-invocable: true
allowed-tools:
  - Bash
  - Read
---

# Verify Training Skill

Quick architectural sanity checks for UniCardio Rectified Flow. Two layers, both run on CPU in seconds:

1. **Synthetic-batch forward** — exercises `UniCardioRF` + `rf_train_step` + `euler_sample` with random `(B=2, 3, L)` data, no file I/O.
2. **Per-task overfit smoke test** — `python run/pipeline/smoke_test.py` overfits a tiny batch for each active task (~70 s on CPU). The richer of the two checks; use after non-trivial model edits.

> **Layout reminder.** Total token sequence is `3 * slot_length = 1500` for the default
> `slot_length=500`. Slot order is **ECG=0, PPG=1, ABP=2** (model space). The model
> forward signature is `UniCardioRF.forward(x_full, t, mask, target_slot: int) -> (B, 1, L)` —
> tensors + a Python `int`, **never** a `TaskSpec`. RF uses Lipman convention
> (`t=0` noise → `t=1` data, `x_t = (1-t)ε + tx_1`, `v = x_1 - ε`).

## What this checks

1. `UniCardioRF` builds from the resolved Hydra config without error.
2. Forward pass returns the right shape `(B, 1, slot_length)` for each `target_slot` in use.
3. Output is finite (no NaN / Inf at random init).
4. `rf_train_step` produces a scalar loss whose `.backward()` populates parameter grads.
5. (Optional) `euler_sample` runs with `n_steps=4` and returns finite `(B, 1, slot_length)`.

## Layer 1: Synthetic-batch forward

Run from the **repo root** so the absolute config path resolves correctly:

```bash
cd "$CLAUDE_PROJECT_DIR" && python3 - <<'PY'
import os
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.model_module.unicardio_rf import UniCardioRF
from src.model_module.tasks import TASK_SPECS
from src.trainer_module.rectified_flow import rf_train_step
from src.trainer_module.sampler import euler_sample

print("=== UniCardio RF Sanity Check (synthetic) ===")

# initialize_config_dir takes an absolute path (initialize() is relative to
# the caller's __file__, which doesn't exist when invoked via `python3 -`).
with initialize_config_dir(config_dir=os.path.abspath("run/conf"), version_base=None):
    cfg = compose(config_name="config")

L = int(cfg.data.slot_length)
device = torch.device("cpu")  # always CPU for sanity check
torch.manual_seed(0)

# Build model from cfg.model (UniCardioRF accepts a Mapping or BackboneConfig).
model = UniCardioRF(OmegaConf.to_container(cfg.model, resolve=True)).to(device)
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"[OK] UniCardioRF built: {n_params:,} params, slot_length={L}")

# Active tasks (zero weights are filtered out — same logic as the trainer).
active = [name for name, w in cfg.trainer.task_weights.items() if float(w) > 0]
print(f"[OK] Active tasks ({len(active)}): {', '.join(active)}")

# Synthetic (B=2, 3, L) batch — model-space slot order ECG/PPG/ABP, ~unit variance.
B = 2
batch = torch.randn(B, 3, L)

failures = 0
for task_name in active:
    task = TASK_SPECS[task_name]

    # Training step path: rf_train_step builds x_t / mask / target internally.
    loss = rf_train_step(model, batch, task)
    if not torch.isfinite(loss):
        print(f"[FAIL] {task_name}: rf_train_step loss is {loss.item()}")
        failures += 1
        continue

    # Backward populates grads for at least the output head + first conv.
    model.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    if grad_norm == 0.0:
        print(f"[FAIL] {task_name}: zero gradient norm — disconnected graph?")
        failures += 1
        continue

    # Inference path: euler_sample with a tiny step count (CPU-friendly).
    out = euler_sample(model, batch, task, n_steps=4, device=device)
    if out.shape != (B, 1, L):
        print(f"[FAIL] {task_name}: euler_sample shape {tuple(out.shape)} != ({B}, 1, {L})")
        failures += 1
        continue
    if not torch.isfinite(out).all():
        print(f"[FAIL] {task_name}: euler_sample produced non-finite values")
        failures += 1
        continue

    print(f"[OK] {task_name}: loss={loss.item():.4f}, grad_norm={grad_norm:.2e}, "
          f"euler out range=[{out.min().item():.3f}, {out.max().item():.3f}]")

print()
print("=== Sanity Check Complete ===" if failures == 0 else f"=== {failures} failures ===")
PY
```

Expected output:

```
=== UniCardio RF Sanity Check (synthetic) ===
[OK] UniCardioRF built: 4,XXX,XXX params, slot_length=500
[OK] Active tasks (3): ecg2abp, ppg2abp, ecgppg2abp
[OK] ecg2abp: loss=…, grad_norm=…, euler out range=[…, …]
[OK] ppg2abp: …
[OK] ecgppg2abp: …
=== Sanity Check Complete ===
```

Any `[FAIL]` line means the architecture is broken — fix before running real training. The most common causes:
- shape mismatch in `assemble_x_full` (wrong `target_slot` → wrong substitution),
- mask dtype drift (training uses `dtype=torch.bool` for SDPA),
- `task_weights` typo so no tasks are active.

## Layer 2: Per-task overfit smoke test

The repo ships with a richer test that overfits a tiny synthetic batch for each active task and verifies the loss actually drops + Euler reconstruction matches at the end. Slower (~70 s on CPU) but catches optimization-side bugs the synthetic forward misses:

```bash
python run/pipeline/smoke_test.py
```

Use Layer 1 first (fast); fall back to Layer 2 only when Layer 1 passes but training still misbehaves.

## Guardrails

- **Always run from repo root** — Hydra `initialize(config_path="../run/conf")` is relative to the script's location, but the inline heredoc uses `__file__` of `<stdin>`, so anchoring at the repo root is the only way the relative path resolves.
- **CPU only on the laptop** — never start `python run/pipeline/train.py` from this skill. Real training is GPU-server only.
- **No `torch.cuda.amp.GradScaler`** — project is bf16-only on CUDA. If you see a GradScaler reference anywhere, that's a regression.

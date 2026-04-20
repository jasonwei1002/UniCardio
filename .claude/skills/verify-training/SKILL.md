---
name: verify-training
description: Run a quick forward pass sanity check on the UniCardio model using a small synthetic batch. Use after modifying model architecture or training code to catch errors before launching a full training run.
user-invocable: true
allowed-tools:
  - Bash
  - Read
---

# Verify Training Skill

Run a forward pass sanity check using a small **synthetic batch** (2 samples, random noise) — no file I/O, runs in seconds.

Input shape: `(B=2, 1, 2000)` — four signal slots × L=500 concatenated, matching the real data layout.

## What This Checks

1. Model imports and initializes without error
2. Forward pass completes for key `model_flag` combinations
3. Output shapes are correct: `(B, n_samples, 1, 500)`
4. Output values are finite (no NaN/Inf)

## Verification Script

Run this from `base_model/`:

```bash
cd base_model && python3 -c "
import torch
import sys
import os

print('=== UniCardio Forward Pass Verification ===')

# Synthetic input: (B=2, 1, 2000), values in [-1, 1] like real normalized signals
B, L = 2, 500
observed_data = torch.randn(B, 1, 4 * L).clamp(-1, 1)
print(f'[OK] synthetic input shape: {observed_data.shape}')

# Import model
try:
    from diffusion_model_no_compress_final import CSDI_base
    import yaml
    with open('base_no_compress_original.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print(f'[OK] Config loaded: {config.get(\"diffusion\", {}).get(\"layers\", \"?\")} layers')
except Exception as e:
    print(f'[FAIL] Cannot import model: {e}')
    sys.exit(1)

# Initialize model
try:
    device = torch.device('cpu')
    model = CSDI_base(config, device, L=4*L)
    model.eval()
    print(f'[OK] Model initialized')
except Exception as e:
    print(f'[FAIL] Model init failed: {e}')
    sys.exit(1)

# Try loading checkpoint if available
ckpt_files = sorted([f for f in os.listdir('check') if f.startswith('model') and f.endswith('.pth')]) if os.path.exists('check') else []
if ckpt_files:
    try:
        state = torch.load(f'check/{ckpt_files[-1]}', map_location='cpu')
        if isinstance(state, dict) and 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'], strict=False)
        else:
            model.load_state_dict(state, strict=False)
        print(f'[OK] Loaded checkpoint: {ckpt_files[-1]}')
    except Exception as e:
        print(f'[WARN] Could not load checkpoint ({e}), using random weights')
else:
    print('[WARN] No checkpoints found, using random weights')

# Run forward pass tests
test_cases = [
    ('02', 2, 'PPG -> ECG cross-modal'),
    ('03', 2, 'PPG -> slot3 self-conditioning'),
]

B = min(2, observed_data.shape[0])
test_input = observed_data[:B]

with torch.no_grad():
    for model_flag, borrow_mode, description in test_cases:
        try:
            out = model(test_input, n_samples=2, model_flag=model_flag,
                       borrow_mode=borrow_mode, DDIM_flag=1, sample_steps=3, train_gen_flag=1)
            assert out.shape == (B, 2, 1, 500), f'Expected ({B}, 2, 1, 500), got {out.shape}'
            assert torch.isfinite(out).all(), 'Output contains NaN or Inf'
            print(f'[OK] {description}: output shape {out.shape}, range [{out.min():.3f}, {out.max():.3f}]')
        except Exception as e:
            print(f'[FAIL] {description}: {e}')

print()
print('=== Verification Complete ===')
"
```

## Expected Output

```
=== UniCardio Forward Pass Verification ===
[OK] synthetic input shape: torch.Size([2, 1, 2000])
[OK] Config loaded: 5 layers
[OK] Model initialized
[OK] Loaded checkpoint: modelXXX.pth
[OK] PPG -> ECG cross-modal: output shape torch.Size([2, 2, 1, 500]), range [-1.234, 1.456]
[OK] PPG -> slot3 self-conditioning: output shape torch.Size([2, 2, 1, 500]), range [-0.987, 1.123]
=== Verification Complete ===
```

Any `[FAIL]` line indicates a problem that must be fixed before training.

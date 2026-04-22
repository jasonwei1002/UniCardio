---
name: signal-visualize
description: Visualize cardiovascular signals (ECG, PPG, ABP) from UniCardio datasets, model outputs, or synthetic batches. Creates publication-quality plots in the model's 3-slot convention.
user-invocable: true
allowed-tools:
  - Bash
  - Read
  - Glob
---

# Signal Visualize Skill

Generate standardized plots for the three UniCardio modalities. **Always work in model slot order** — the data loader already applied the channel permute and BP normalization.

## Model Slot Convention

> Do not confuse with the on-disk layout. `data/Final_sig_combined.npy` stores `(N, 3, 500)` in PPG=0 / BP=1 / ECG=2 order; `src/data_module/datamodule.py::load_and_preprocess` permutes with `[2, 0, 1]` and applies `(x - 100) / 50` to BP. **All downstream arrays speak the model convention below.**

| Slot | Modality | Color       | Hex       |
|------|----------|-------------|-----------|
| 0    | ECG      | green       | `#4CAF50` |
| 1    | PPG      | blue        | `#2196F3` |
| 2    | ABP      | red         | `#F44336` |

Line conventions:
- **Ground-truth / condition**: solid line, alpha 1.0
- **Model reconstruction**: same color, dashed line
- **Noisy x_t input**: same color, alpha 0.4

Style: white background, no grid, font size 12, figure size `(15, 4)` per signal row.

## Usage

```
/signal-visualize                     # interactive — pick a data source
/signal-visualize dataset [n]         # plot n random samples from the training npy (default 3)
/signal-visualize batch <path>        # plot a saved (B, 3, L) tensor (.pt or .npy)
/signal-visualize compare <task> <ckpt>   # run euler_sample and overlay gt vs prediction for one task
```

`<task>` must be one of the 5 task names registered in `src/model_module/tasks.py`:
`ecg2ppg`, `ppg2ecg`, `ecg2abp`, `ppg2abp`, `ecgppg2abp`.

## Plot 1: Dataset Samples

Read straight from the training npy **after** the loader normalization (re-apply `[2,0,1]` + BP scaling so visualization matches what the model sees).

```bash
cd "$CLAUDE_PROJECT_DIR" && python3 - <<'PY'
import numpy as np
import matplotlib.pyplot as plt

x = np.load("data/Final_sig_combined.npy")          # (N, 3, 500), disk order
x = x[:, [2, 0, 1], :].astype(np.float32)           # model order: ECG, PPG, ABP
x[:, 2, :] = (x[:, 2, :] - 100.0) / 50.0            # BP normalization

idx = np.random.RandomState(0).choice(len(x), 3, replace=False)
names = ["ECG", "PPG", "ABP"]
colors = ["#4CAF50", "#2196F3", "#F44336"]

fig, axes = plt.subplots(3, 3, figsize=(15, 9), sharex=True)
for col, i in enumerate(idx):
    for row in range(3):
        axes[row, col].plot(x[i, row], color=colors[row], linewidth=1.2)
        axes[row, col].set_ylabel(names[row]) if col == 0 else None
        axes[row, col].set_title(f"sample {i}") if row == 0 else None
plt.tight_layout()
plt.savefig("run/outputs/signal_dataset_samples.png", dpi=150)
print("wrote run/outputs/signal_dataset_samples.png")
PY
```

## Plot 2: Reconstruction Comparison

Overlay ground truth target slot against `euler_sample` output for one task.

> **Laptop caveat**: this plot requires a trained checkpoint, which normally lives on the GPU server. Training never happens locally, so the checkpoint must be manually scp'd (e.g. `scp server:~/UniCardio/run/outputs/<run>/checkpoints/best.pt ./run/outputs/<run>/checkpoints/`) before running this block. If no checkpoint is available, stick to Plot 1 (dataset-only) on the laptop.

```bash
cd "$CLAUDE_PROJECT_DIR" && python3 - <<'PY'
import torch, numpy as np, matplotlib.pyplot as plt
from src.model_module.unicardio_rf import UniCardioRF
from src.model_module.tasks import TASK_SPECS, Slot
from src.trainer_module.sampler import euler_sample
from src.utils.checkpoint import load_checkpoint   # adapt to actual loader

TASK = "ecg2ppg"                                     # CHANGE ME
CKPT = "run/outputs/<run>/checkpoints/best.pt"       # CHANGE ME

task = TASK_SPECS[TASK]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, _ = load_checkpoint(CKPT, map_location=device)
model.eval()

x = np.load("data/Final_sig_combined.npy")[:4, [2, 0, 1], :].astype(np.float32)
x[:, 2, :] = (x[:, 2, :] - 100.0) / 50.0
conditions = torch.from_numpy(x).to(device)
pred = euler_sample(model, conditions, task, n_steps=8, device=device).cpu().numpy()
gt = x[:, int(task.target_slot), :]

names = {Slot.ECG: "ECG", Slot.PPG: "PPG", Slot.ABP: "ABP"}
color = {Slot.ECG: "#4CAF50", Slot.PPG: "#2196F3", Slot.ABP: "#F44336"}[task.target_slot]

fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
for i, ax in enumerate(axes):
    ax.plot(gt[i],       color=color, linewidth=1.4, label="ground truth")
    ax.plot(pred[i, 0],  color=color, linewidth=1.4, linestyle="--", label="reconstruction")
    ax.set_ylabel(f"sample {i}: {names[task.target_slot]}")
    ax.legend(loc="upper right")
plt.suptitle(f"{TASK} reconstruction (Euler 8-step)")
plt.tight_layout()
plt.savefig(f"run/outputs/recon_{TASK}.png", dpi=150)
print(f"wrote run/outputs/recon_{TASK}.png")
PY
```

## Guardrails

- **Never hardcode server paths** like `/data2/...`. Always use `$CLAUDE_PROJECT_DIR` or repo-relative paths — this skill runs on both laptop and server.
- **Never plot raw `data/Final_sig_combined.npy`** without the `[2, 0, 1]` permute and BP normalization. The file is in disk order and visualizing it as-is silently mislabels every channel.
- **Never assume checkpoint loading API** without checking `src/utils/checkpoint.py`. Ask the user to confirm the loader signature if unfamiliar.

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

> Do not confuse with the on-disk layout. The repo supports two datasets and the on-disk channel order differs:
>
> | `data.name` | Default? | File(s) | On-disk order | `channel_permutation` |
> |-------------|----------|---------|---------------|------------------------|
> | `pulsedb` | yes | `data/pulsedb/Train_Subset_vitaldb_500.npy` | ECG / PPG / ABP | `[0, 1, 2]` (no-op) |
> | `combined` | legacy | `data/Final_sig_combined.npy` | PPG / BP / ECG | `[2, 0, 1]` |
>
> `CardiacDataset.__getitem__` applies the permutation and `bp_normalize(x) = (x - 100) / 50` on the ABP slot. **Plotting code MUST replicate the same permutation for whichever file it reads** — applying `[2, 0, 1]` to a PulseDB array silently mislabels every channel.

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

Read the dataset selected by Hydra (`run/conf/data/default.yaml::name`). The script auto-resolves which `.npy` to load and which `channel_permutation` to apply, so it stays correct for both `pulsedb` and `combined`. Use `mmap_mode='r'` — these files are 3–13 GB.

```bash
cd "$CLAUDE_PROJECT_DIR" && python3 - <<'PY'
import os
import numpy as np
import matplotlib.pyplot as plt
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

# initialize_config_dir takes an absolute path; initialize() requires a real
# __file__, which doesn't exist when invoked via `python3 -`.
with initialize_config_dir(config_dir=os.path.abspath("run/conf"), version_base=None):
    cfg = compose(config_name="config")

dname = cfg.data.name                                   # "pulsedb" or "combined"
sub = cfg.data[dname]
perm = list(sub.channel_permutation)                    # [0,1,2] for pulsedb, [2,0,1] for combined

# pulsedb has train_path / test_path; combined has data_path. Pick whichever exists.
path = sub.get("train_path") or sub.get("data_path")
print(f"plotting from {dname}: {path}")

x = np.load(path, mmap_mode="r")                        # (N, 3, L), disk order
n = min(3, x.shape[0])
idx = np.random.RandomState(0).choice(x.shape[0], n, replace=False)

# Materialize only the picked rows, then permute to model order ECG/PPG/ABP and normalize ABP.
sample = np.asarray(x[idx])[:, perm, :].astype(np.float32)
sample[:, 2, :] = (sample[:, 2, :] - float(cfg.data.bp_offset)) / float(cfg.data.bp_scale)

names = ["ECG", "PPG", "ABP"]
colors = ["#4CAF50", "#2196F3", "#F44336"]

fig, axes = plt.subplots(3, n, figsize=(15, 9), sharex=True, squeeze=False)
for col in range(n):
    for row in range(3):
        axes[row, col].plot(sample[col, row], color=colors[row], linewidth=1.2)
        if col == 0: axes[row, col].set_ylabel(names[row])
        if row == 0: axes[row, col].set_title(f"sample {idx[col]} ({dname})")
plt.tight_layout()
out = f"run/outputs/signal_dataset_samples_{dname}.png"
plt.savefig(out, dpi=150)
print(f"wrote {out}")
PY
```

## Plot 2: Reconstruction Comparison

Overlay ground truth target slot against `euler_sample` output for one task.

> **Laptop caveat**: this plot requires a trained checkpoint, which normally lives on the GPU server. Training never happens locally, so the checkpoint must be manually scp'd (e.g. `scp server:~/UniCardio/run/outputs/<run>/checkpoints/best.pt ./run/outputs/<run>/checkpoints/`) before running this block. If no checkpoint is available, stick to Plot 1 (dataset-only) on the laptop.

```bash
cd "$CLAUDE_PROJECT_DIR" && python3 - <<'PY'
import os
import torch, numpy as np, matplotlib.pyplot as plt
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir

from src.model_module.unicardio_rf import UniCardioRF
from src.model_module.tasks import TASK_SPECS, Slot
from src.trainer_module.sampler import euler_sample
from src.utils.checkpoint import load_checkpoint

TASK = "ppg2abp"                                     # CHANGE ME — must be an active task
CKPT = "run/outputs/<run>/checkpoints/best.pt"       # CHANGE ME

# Build cfg from saved checkpoint when possible (so plot uses the same dataset
# / slot_length / model dims the run trained with). Fall back to current Hydra
# default if the ckpt has no embedded config.
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg_dict = ck.get("config")
if cfg_dict:
    cfg = OmegaConf.create(cfg_dict)
else:
    with initialize_config_dir(config_dir=os.path.abspath("run/conf"), version_base=None):
        cfg = compose(config_name="config")

task = TASK_SPECS[TASK]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UniCardioRF(OmegaConf.to_container(cfg.model, resolve=True)).to(device)
load_checkpoint(CKPT, model=model, map_location=device)
model.eval()

# Pull conditions from whichever dataset matches the run.
dname = cfg.data.name
sub = cfg.data[dname]
perm = list(sub.channel_permutation)
src_path = sub.get("test_path") or sub.get("data_path") or sub.get("train_path")
x = np.asarray(np.load(src_path, mmap_mode="r")[:4])[:, perm, :].astype(np.float32)
x[:, 2, :] = (x[:, 2, :] - float(cfg.data.bp_offset)) / float(cfg.data.bp_scale)

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
plt.suptitle(f"{TASK} reconstruction (Euler 8-step, {dname})")
plt.tight_layout()
plt.savefig(f"run/outputs/recon_{TASK}_{dname}.png", dpi=150)
print(f"wrote run/outputs/recon_{TASK}_{dname}.png")
PY
```

## Guardrails

- **Never hardcode server paths** like `/data2/...`. Always use `$CLAUDE_PROJECT_DIR` or repo-relative paths — this skill runs on both laptop and server.
- **Never plot raw `data/Final_sig_combined.npy`** without the `[2, 0, 1]` permute and BP normalization. The file is in disk order and visualizing it as-is silently mislabels every channel.
- **Never assume checkpoint loading API** without checking `src/utils/checkpoint.py`. Ask the user to confirm the loader signature if unfamiliar.

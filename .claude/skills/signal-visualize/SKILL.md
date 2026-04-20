---
name: signal-visualize
description: Visualize cardiovascular signals (PPG, BP, ECG) from UniCardio datasets, model outputs, or synthetic batches. Creates publication-quality plots with consistent formatting.
user-invocable: true
allowed-tools:
  - Bash
  - Read
  - Glob
---

# Signal Visualize Skill

Generate standardized cardiovascular signal plots for PPG, BP, and ECG data. Supports training data, model outputs, and batch files.

## Usage

```
/signal-visualize                                # interactive: choose what to plot
/signal-visualize batch                          # visualize the pre-saved test batch
/signal-visualize training [sample_indices]       # plot training data samples (default: first 3)
/signal-visualize compare [model_flag] [borrow_mode]  # compare input vs generated signals
```

## Color & Style Convention

Use these consistent colors across all plots:
- **PPG (slot 0)**: `#2196F3` (blue)
- **BP (slot 1)**: `#F44336` (red)
- **ECG (slot 2)**: `#4CAF50` (green)
- **Generated/Denoised**: same color but dashed line
- **Noisy input**: same color but alpha=0.4

Plot style: white background, no grid, font size 12, figure size (15, 4) per signal row.

## Plot 1: Training Data Samples

```bash
cd /data2/wcn/UniCardio && python3 -c "
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data = np.load('base_model/Final_sig_combined.npy')
names = ['PPG', 'BP', 'ECG']
colors = ['#2196F3', '#F44336', '#4CAF50']
n_samples = 3

fig, axes = plt.subplots(n_samples, 3, figsize=(18, 3 * n_samples))
for i in range(n_samples):
    for j, (name, color) in enumerate(zip(names, colors)):
        ax = axes[i, j]
        signal = data[i, j, :]
        ax.plot(signal, color=color, linewidth=0.8)
        ax.set_title(f'{name} (sample {i})', fontsize=11)
        ax.set_xlim(0, len(signal))
        if j == 0:
            ax.set_ylabel('Amplitude')
        if i == n_samples - 1:
            ax.set_xlabel('Time (samples)')

plt.tight_layout()
plt.savefig('signal_samples.png', dpi=150, bbox_inches='tight')
print('Saved: signal_samples.png')
print(f'Data shape: {data.shape}')
for j, name in enumerate(names):
    print(f'  {name}: [{data[:,j,:].min():.3f}, {data[:,j,:].max():.3f}]')
"
```

## Plot 2: Test Batch Signals

```bash
cd /data2/wcn/UniCardio && python3 -c "
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

batch = torch.load('base_model/batch.pth', map_location='cpu')
names = ['Original', 'Noisy', 'Imputed']
colors = ['#2196F3', '#F44336', '#4CAF50']
slot_names = ['PPG', 'BP', 'ECG', 'Placeholder']

# Plot first sample, all 3 signals × first 3 slots
fig, axes = plt.subplots(3, 3, figsize=(18, 9))
for row, (name, b) in enumerate(zip(names, batch[:3])):
    sig = b[0, 0, :].numpy()  # first sample
    for col in range(3):
        ax = axes[row, col]
        start, end = col * 500, (col + 1) * 500
        ax.plot(sig[start:end], color=colors[col], linewidth=0.8)
        ax.set_title(f'{name} - {slot_names[col]}', fontsize=11)
        if row == 2:
            ax.set_xlabel('Samples')
        if col == 0:
            ax.set_ylabel('Amplitude')

plt.tight_layout()
plt.savefig('batch_signals.png', dpi=150, bbox_inches='tight')
print('Saved: batch_signals.png')
print(f'Batch shapes:')
for i, b in enumerate(batch):
    print(f'  [{i}] {b.shape}')
"
```

## Plot 3: Input vs Generated Comparison

For comparing model input with generated output side by side:

```bash
cd /data2/wcn/UniCardio && python3 -c "
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load batch for demonstration
batch = torch.load('base_model/batch.pth', map_location='cpu')
original = batch[0][0, 0, :].numpy()  # first sample

colors = ['#2196F3', '#F44336', '#4CAF50']
slot_names = ['PPG', 'BP', 'ECG']

fig, axes = plt.subplots(3, 1, figsize=(15, 8))
for i, (name, color) in enumerate(zip(slot_names, colors)):
    ax = axes[i]
    start, end = i * 500, (i + 1) * 500
    signal = original[start:end]
    ax.plot(signal, color=color, linewidth=0.8, label=name)
    ax.set_title(f'{name} (slot {i}, indices {start}:{end})', fontsize=11)
    ax.legend(loc='upper right')
    ax.set_xlim(0, 500)

plt.suptitle('Signal Overview — Sample 0', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('signal_comparison.png', dpi=150, bbox_inches='tight')
print('Saved: signal_comparison.png')
"
```

## Notes

- All plots use `matplotlib.use('Agg')` for headless rendering
- Output PNGs are saved to the working directory
- For model-generated signals, load the checkpoint and run inference first, then pass the output to these plotting scripts
- When comparing denoised/imputed vs original, always overlay on the same subplot with different line styles

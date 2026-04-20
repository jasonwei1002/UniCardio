---
name: experiment-compare
description: Compare experimental results across UniCardio base model and downstream tasks. Use when you need to compare metrics between runs, check training progress, or summarize experiment outcomes.
user-invocable: true
disable-model-invocation: true
allowed-tools:
  - Bash
  - Read
  - Glob
  - Grep
---

# Experiment Compare Skill

Compare and summarize experimental results across UniCardio tasks. Supports base model training logs and all 5 downstream tasks.

## Usage

```
/experiment-compare                          # summary of all available results
/experiment-compare base                     # base model training progress
/experiment-compare ptbxl                    # PTBXL classification metrics
/experiment-compare bp                       # BP estimation metrics
/experiment-compare mimic                    # MIMIC fine-tuning metrics
/experiment-compare af                       # AF detection metrics
/experiment-compare wesad                    # WESAD stress detection metrics
```

## Task 1: Discover Available Results

Scan for result files and checkpoints:

```bash
echo "=== Base Model ==="
if [ -f base_model/check/loss.csv ]; then
    echo "Training log: base_model/check/loss.csv"
    wc -l base_model/check/loss.csv
    ls base_model/check/model*.pth 2>/dev/null | wc -l | xargs echo "Checkpoints:"
else
    echo "No training log found"
fi

echo ""
echo "=== PTBXL ==="
ls down_stream_code/ptbxl/ACC_*.npy down_stream_code/ptbxl/SPC_*.npy down_stream_code/ptbxl/SENS_*.npy 2>/dev/null || echo "No saved metrics"

echo ""
echo "=== Downstream Generated Data ==="
for dir in ptbxl BP MIMIC AF Wesad; do
    count=$(ls down_stream_code/$dir/generated_*.npy down_stream_code/$dir/result_*.npy down_stream_code/$dir/*_generated*.npy 2>/dev/null | wc -l)
    echo "$dir: $count generated files"
done
```

## Task 2: Base Model Training Progress

```bash
cd /data2/wcn/UniCardio && python3 -c "
import csv
try:
    rows = list(csv.DictReader(open('base_model/check/loss.csv')))
    if not rows:
        print('No training data yet')
    else:
        total = len(rows)
        last = rows[-1]
        epoch = int(last.get('epoch', 0))
        stage = int(epoch // 200) + 1 if epoch < 600 else 3

        # Loss trend
        losses = [float(r.get('train_loss', 0)) for r in rows if r.get('train_loss')]
        recent = losses[-20:] if len(losses) >= 20 else losses
        trend = 'decreasing' if recent[-1] < recent[0] else 'increasing'

        # Best loss
        best_loss = min(losses)
        best_epoch = losses.index(best_loss)

        print(f'Epoch: {epoch}/800  Stage: {stage}')
        print(f'Current loss: {losses[-1]:.6f}  Best: {best_loss:.6f} (epoch {best_epoch})')
        print(f'Trend (last 20): {trend} ({recent[0]:.6f} -> {recent[-1]:.6f})')
        print(f'Total checkpoints: {total}')
except FileNotFoundError:
    print('No loss.csv found. Has training been started?')
"
```

## Task 3: PTBXL Metrics

```bash
cd /data2/wcn/UniCardio/down_stream_code/ptbxl && python3 -c "
import numpy as np, os
for prefix, label in [('ACC', 'Accuracy'), ('SPC', 'Specificity'), ('SENS', 'Sensitivity')]:
    files = sorted([f for f in os.listdir('.') if f.startswith(prefix) and f.endswith('.npy')])
    if files:
        for f in files:
            data = np.load(f)
            print(f'{label} ({f}): mean={data.mean():.4f} ± std={data.std():.4f}')
    else:
        print(f'{label}: no saved results')
"
```

## Task 4: Checkpoint Size & Timestamps

```bash
echo "=== Checkpoint Summary ==="
ls -lht base_model/check/model*.pth 2>/dev/null | head -5
echo ""
if [ -f base_model/no_compress799.pth ]; then
    echo "Pretrained: base_model/no_compress799.pth"
    ls -lh base_model/no_compress799.pth
fi
if [ -f base_model/check/final.pth ]; then
    echo "Final: base_model/check/final.pth"
    ls -lh base_model/check/final.pth
fi
```

## Output Format

Always conclude with a summary table:

```
| Task | Status | Key Metric | Value |
|------|--------|------------|-------|
| Base Model | [Training epoch X/800 | Complete | Not started] | Loss | X.XXXXXX |
| PTBXL | [Results available | Not run] | Accuracy | X.XXXX ± X.XXXX |
| BP | ... | RMSE | ... |
| MIMIC | ... | Accuracy | ... |
| AF | ... | HR Error | ... |
| WESAD | ... | HR Error | ... |
```

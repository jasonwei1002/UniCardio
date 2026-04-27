#!/usr/bin/env bash
# UniCardio 可视化一键启动脚本：对 test 集第 0 条样本，绘制每个 active task 的预测 vs 真值。
# 用法：
#   bash plot.sh                              # 自动挑 run/outputs/ 下最新 best.pt
#   bash plot.sh path/to/best.pt              # 指定 checkpoint
set -euo pipefail

cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CHECKPOINT="${1:-}"
if [ -z "$CHECKPOINT" ]; then
    CHECKPOINT="$(ls -t run/outputs/*/checkpoints/best.pt 2>/dev/null | head -1 || true)"
fi
if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
    echo "Error: checkpoint not found." >&2
    echo "Usage: bash plot.sh [checkpoint_path]" >&2
    echo "       default: run/outputs/*/checkpoints/best.pt (latest)" >&2
    exit 1
fi

# 把图写入 checkpoint 所在 run 目录下的 plots/ 子目录，与训练同源。
RUN_DIR="$(dirname "$(dirname "$CHECKPOINT")")"
OUT_DIR="$RUN_DIR/plots"
mkdir -p "$OUT_DIR"
LOG_FILE="$OUT_DIR/plot_$(date +%Y%m%d_%H%M%S).log"
echo "Plotting checkpoint: $CHECKPOINT" | tee "$LOG_FILE"
echo "Writing artifacts to: $OUT_DIR" | tee -a "$LOG_FILE"

python script/plot_first_sample.py \
    +checkpoint="$CHECKPOINT" \
    hydra.run.dir="$OUT_DIR" \
    device=cuda \
    data.batch_size=1 \
    data.num_workers=0 \
    2>&1 | tee -a "$LOG_FILE"

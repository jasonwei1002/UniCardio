#!/usr/bin/env bash
# UniCardio 评估一键启动脚本（仅波形模型 / RF backbone）
#
# 只评估 RF backbone 出的归一化波形：ABP 输出留在 shape-only [0, 1] 空间，
# 报 RMSE / MAE / Pearson / KS（其中 Pearson 物理可比，RMSE/MAE 为无量纲量）。
# 不加载 BP head，不做 mmHg 幅值还原。
#
# 用法：
#   bash test.sh [rf_ckpt]
#     rf_ckpt 省略时自动挑选最新的 run/outputs/*/checkpoints/best.pt
set -euo pipefail

cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [ "$#" -ge 1 ]; then
    CHECKPOINT="$1"
else
    CHECKPOINT="$(ls -t run/outputs/*/checkpoints/best.pt 2>/dev/null | head -n 1 || true)"
    if [ -z "$CHECKPOINT" ]; then
        echo "Error: no checkpoint given and no run/outputs/*/checkpoints/best.pt found." >&2
        echo "Usage: bash test.sh [rf_ckpt]" >&2
        exit 1
    fi
    echo "No checkpoint given; using latest: $CHECKPOINT"
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: RF checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

# 把 eval 产物写入 RF checkpoint 所在 run 目录下的 eval/ 子目录，与训练同源。
RUN_DIR="$(dirname "$(dirname "$CHECKPOINT")")"
OUT_DIR="$RUN_DIR/eval"
mkdir -p "$OUT_DIR"
LOG_FILE="$OUT_DIR/eval_$(date +%Y%m%d_%H%M%S).log"
echo "RF checkpoint:        $CHECKPOINT" | tee "$LOG_FILE"
echo "Writing artifacts to: $OUT_DIR" | tee -a "$LOG_FILE"

python run/pipeline/evaluate.py \
    +checkpoint="$CHECKPOINT" \
    hydra.run.dir="$OUT_DIR" \
    device=cuda \
    data.batch_size=512 \
    data.num_workers=8 \
    2>&1 | tee -a "$LOG_FILE"

#!/usr/bin/env bash
# UniCardio 评估一键启动脚本
#
# 评估「完整 mmHg 幅值」：RF backbone 出归一化波形形状，BP head 出标量 SBP/DBP，
# 二者经 reconstruct_mmHg 组合成完整 ABP 波形后再算 RMSE/MAE/Pearson/KS。
# 两个 checkpoint 路径都必须显式传入（不自动挑选）。
#
# 用法：
#   bash test.sh <rf_ckpt> <bp_head_ckpt>
set -euo pipefail

cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [ "$#" -ne 2 ]; then
    echo "Usage: bash test.sh <rf_ckpt> <bp_head_ckpt>" >&2
    echo "  rf_ckpt:      RF backbone checkpoint (run/outputs/<rf_run>/checkpoints/best.pt)" >&2
    echo "  bp_head_ckpt: BP head checkpoint (run/outputs/<bp_head_run>/checkpoints/best.pt)" >&2
    exit 1
fi

CHECKPOINT="$1"
BP_HEAD_CKPT="$2"
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: RF checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi
if [ ! -f "$BP_HEAD_CKPT" ]; then
    echo "Error: BP head checkpoint not found: $BP_HEAD_CKPT" >&2
    exit 1
fi

# 把 eval 产物写入 RF checkpoint 所在 run 目录下的 eval/ 子目录，与训练同源。
RUN_DIR="$(dirname "$(dirname "$CHECKPOINT")")"
OUT_DIR="$RUN_DIR/eval"
mkdir -p "$OUT_DIR"
LOG_FILE="$OUT_DIR/eval_$(date +%Y%m%d_%H%M%S).log"
echo "RF checkpoint:      $CHECKPOINT" | tee "$LOG_FILE"
echo "BP head checkpoint: $BP_HEAD_CKPT" | tee -a "$LOG_FILE"
echo "Writing artifacts to: $OUT_DIR" | tee -a "$LOG_FILE"

python run/pipeline/evaluate.py \
    +checkpoint="$CHECKPOINT" \
    +bp_head_checkpoint="$BP_HEAD_CKPT" \
    hydra.run.dir="$OUT_DIR" \
    device=cuda \
    data.batch_size=512 \
    data.num_workers=8 \
    2>&1 | tee -a "$LOG_FILE"

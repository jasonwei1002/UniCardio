#!/usr/bin/env bash
# UniCardio fine-tune 一键启动脚本：从 ckpt 加载**仅模型权重**作为初始化，
# optimizer / scheduler / epoch 全新（=新实验从这个 ckpt 起步，不是续训）。
# Hydra 会创建新的 run/outputs/<timestamp>/，不污染原训练目录。
#
# 用法：
#   bash finetune.sh <ckpt>                              # 用默认 cfg fine-tune
#   bash finetune.sh <ckpt> trainer.lr=1e-4              # 改 lr
#   bash finetune.sh <ckpt> trainer.lr=1e-4 trainer.epochs=50 trainer.warmup_pct=0.0
#
# 提示：fine-tune 通常用更低 lr（1e-4 起步比较保守），cosine schedule 也会
# 重新算。Adam 动量会从零累起，前几百 step 可能略震荡，正常现象。
set -euo pipefail

cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CHECKPOINT="${1:-}"
if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
    echo "Error: fine-tune init checkpoint required and must exist." >&2
    echo "Usage: bash finetune.sh <ckpt> [hydra overrides...]" >&2
    exit 1
fi
shift  # 把 $1 移走，剩下都作为 Hydra overrides 透传

mkdir -p logs
LOG_FILE="logs/finetune_$(date +%Y%m%d_%H%M%S).log"
echo "Fine-tuning from: $CHECKPOINT" | tee "$LOG_FILE"
echo "Extra overrides: $*" | tee -a "$LOG_FILE"

python run/pipeline/train.py \
    trainer.init_from="$CHECKPOINT" \
    device=cuda \
    data.batch_size=512 \
    data.num_workers=8 \
    "$@" \
    2>&1 | tee -a "$LOG_FILE"

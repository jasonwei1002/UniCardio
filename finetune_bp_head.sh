#!/usr/bin/env bash
# Path A Stream 2 — BP head 阶段二（CalFree calibration-based finetune）。
#
# 行为：
#   - 加载阶段一 ckpt（仅模型权重，optimizer/scheduler/epoch 全新）。
#   - 数据集走 mode='finetune'：CalFree_Test_Subset.npy 内部 80/10/10
#     sample-level 划分（subject 共享 = MD-ViSCo calibration-based 协议）。
#   - 训练完成后自动在 10% test split 上跑一次 BP MAE 评估。
#
# 用法：
#   bash finetune_bp_head.sh <pretrain_ckpt>
#   bash finetune_bp_head.sh <pretrain_ckpt> trainer.lr=5e-5 trainer.epochs=80
set -euo pipefail

cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export BP_HEAD_STAGE=finetune

CHECKPOINT="${1:-}"
if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
    echo "Error: BP head finetune requires a pretrain checkpoint that exists." >&2
    echo "Usage: bash finetune_bp_head.sh <ckpt> [hydra overrides...]" >&2
    exit 1
fi
shift

mkdir -p logs
LOG_FILE="logs/bp_head_finetune_$(date +%Y%m%d_%H%M%S).log"
echo "BP head finetune from: $CHECKPOINT" | tee "$LOG_FILE"
echo "Extra overrides: $*" | tee -a "$LOG_FILE"

python run/pipeline/train_bp_head.py \
    trainer.stage=finetune \
    trainer.init_from="$CHECKPOINT" \
    trainer.lr=1.0e-4 \
    trainer.epochs=100 \
    device=cuda \
    data.batch_size=512 \
    data.num_workers=8 \
    swanlab.experiment_name=bp_head_finetune_calfree \
    "$@" \
    2>&1 | tee -a "$LOG_FILE"

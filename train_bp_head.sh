#!/usr/bin/env bash
# Path A Stream 2 — BP head 阶段一（pretrain）一键启动脚本。
#
# 行为：
#   - 数据集走 mode='pretrain'：Train_Subset.npy 80/20 train/val；
#     CalFree_Test_Subset 作为 held-out test（subject-disjoint，
#     calibration-free 评估基线）。
#   - 单一 MSE(pred_mmHg, target_mmHg) 回归目标，BPHead 全量训练。
#   - 训练结束后保存 best.pt（按 val/mae_mean 选）。
#
# 用法：
#   bash train_bp_head.sh
#   bash train_bp_head.sh trainer.lr=5e-4 trainer.epochs=50
set -euo pipefail

cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export BP_HEAD_STAGE=pretrain

mkdir -p logs
LOG_FILE="logs/bp_head_pretrain_$(date +%Y%m%d_%H%M%S).log"
echo "BP head pretrain on Train_Subset" | tee "$LOG_FILE"
echo "Extra overrides: $*" | tee -a "$LOG_FILE"

python run/pipeline/train_bp_head.py \
    trainer.stage=pretrain \
    trainer.lr=1.0e-3 \
    trainer.epochs=50 \
    device=cuda \
    data.batch_size=512 \
    data.num_workers=12 \
    swanlab.experiment_name=bp_head_pretrain \
    "$@" \
    2>&1 | tee -a "$LOG_FILE"

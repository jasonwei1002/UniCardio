#!/usr/bin/env bash
# BP head 阶段一（pretrain）一键启动脚本。
#
# 行为：
#   - 数据集走 mode='pretrain'：Train_Subset.npy 80/20 train/val；
#     CalFree_Test_Subset 作为 held-out test（subject-disjoint，
#     calibration-free 评估基线）。
#   - loss_mode=l1+wcl（trainer.wcl.enabled=true + pretrain_contrastive_only=false，
#     由本脚本下方 override 固定）：L_ref = L1 + 0.2·WCL，从头联合训练完整 BP
#     回归器（含 fusion + MlpBP 头）。L1 监督直接逼 encoder 学出可判别 BP 的
#     embedding，从根上防止对比塌缩——故 WCL 回归 MD-ViSCo raw 原配
#     （normalize_embeddings=false → 点积 + 各项原始 temperature），撤掉之前为救
#     wcl_only 才加的 L2-norm/temp hack。
#   - best.pt 按 mean 每任务 val L1 选（MD-ViSCo：min val loss）；stage 末在
#     CalFree test（subject-disjoint）出 calibration-free SBP/DBP SD。
#   - 切回旧路径：pretrain_contrastive_only=true → wcl_only 纯对比预训练；
#     trainer.wcl.enabled=false → 纯 L1 监督。
#
# 超参锚点（MD-ViSCo, IEEE JBHI 2026, 正文 Sec III 末）：
#   batch size = 2048, lr = 1e-3, scheduler patience 3, early-stop 5, max 30K steps。
#   ⚠️ l1+wcl 训的是完整模型（含 fusion + MlpBP，比 wcl_only 重很多），B=2048 显存
#   可能吃紧；若 OOM 退 data.batch_size=1024。lr 保持 1e-3 不要随 batch 上调
#   （原文就是 2048+1e-3）。
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
    trainer.epochs=100 \
    device=cuda \
    data.batch_size=2048 \
    data.num_workers=12 \
    trainer.wcl.pretrain_contrastive_only=false \
    trainer.wcl.normalize_embeddings=false \
    swanlab.experiment_name=bp_head_pretrain \
    "$@" \
    2>&1 | tee -a "$LOG_FILE"

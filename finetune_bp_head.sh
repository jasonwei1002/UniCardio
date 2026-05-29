#!/usr/bin/env bash
# BP head 阶段二（CalFree calibration-based finetune）。
#
# 行为：
#   - 加载阶段一对比预训练 ckpt（仅模型权重，optimizer/scheduler/epoch 全新）。
#   - 数据集走 mode='finetune'：CalFree_Test_Subset.npy 内部 80/10/10
#     sample-level 划分（subject 共享 = MD-ViSCo calibration-based 协议）。
#   - loss_mode=l1+wcl（stage=finetune 默认）：L_ref = L_MAE + multi-WCL
#     （MD-ViSCo §6.2.2）。best.pt 按 val mean-L1 选。
#   - 训练完成后自动在 10% test split 上跑一次 BP MAE 评估
#     （目标按 data.bp_label_source，默认 segment_minmax = 对齐 MD-ViSCo）。
#
# 超参锚点（MD-ViSCo, IEEE JBHI 2026, 正文 Sec III 末）：batch size = 2048。
#   注意：stage-2 走全模型路径（fusion + MlpBP 头都参与），比 stage-1 的 wcl_only
#   重，2048 更易 OOM——若 OOM 退 data.batch_size=1024。lr 用 1e-4（finetune，
#   低于 pretrain 的 1e-3，属正常微调档）。
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
    trainer.lr=5.0e-4 \
    trainer.epochs=100 \
    device=cuda \
    data.batch_size=1024 \
    data.num_workers=8 \
    swanlab.experiment_name=bp_head_finetune_calfree \
    "$@" \
    2>&1 | tee -a "$LOG_FILE"

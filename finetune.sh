#!/usr/bin/env bash
# UniCardio 阶段二（CalFree finetune）一键启动脚本。
#
# 行为：
#   - 加载阶段一 ckpt 仅模型权重（optimizer/scheduler/epoch 全新）
#   - 冻结 backbone，仅微调最后 2 个 ResidualBlock + 全部 output_heads
#   - 数据集走 mode='finetune'：CalFree_Test_Subset.npy 内部 80/10/10 划分
#   - 训练完成后自动在 10% test 上跑一次 RF loss 评估并写 SwanLab + CSV
#
# 用法：
#   bash finetune.sh <ckpt>                              # 默认 cfg
#   bash finetune.sh <ckpt> trainer.lr=5e-5              # 改 lr
#   bash finetune.sh <ckpt> trainer.finetune.n_unfrozen_blocks=1
#   bash finetune.sh <ckpt> trainer.epochs=80
#
# 命令行透传的 overrides 会覆盖脚本里的默认值（Hydra 后到先得）。
set -euo pipefail

cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Hydra `hydra.run.dir` 通过 ${oc.env:UNICARDIO_STAGE,pretrain} 取这个变量做后缀。
export UNICARDIO_STAGE=finetune

CHECKPOINT="${1:-}"
if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
    echo "Error: stage-2 finetune requires an init checkpoint that exists." >&2
    echo "Usage: bash finetune.sh <ckpt> [hydra overrides...]" >&2
    exit 1
fi
shift  # 把 $1 移走，剩下都作为 Hydra overrides 透传

mkdir -p logs
LOG_FILE="logs/finetune_$(date +%Y%m%d_%H%M%S).log"
echo "Stage-2 finetune from: $CHECKPOINT" | tee "$LOG_FILE"
echo "Extra overrides: $*" | tee -a "$LOG_FILE"

# trainer.lr=1e-4：pretrain 是 1e-3；微调小数据 + 部分参数，调小一档。
# trainer.epochs=50：经验值，按 CalFree 80% 训练集大小自行调。
python run/pipeline/train.py \
    trainer.stage=finetune \
    trainer.init_from="$CHECKPOINT" \
    trainer.finetune.n_unfrozen_blocks=2 \
    trainer.lr=1.0e-4 \
    trainer.epochs=50 \
    device=cuda \
    data.batch_size=256 \
    data.num_workers=8 \
    trainer.task_weights.ecg2ppg=0.0 \
    trainer.task_weights.ppg2ecg=0.0 \
    trainer.task_weights.ecg2abp=1.0 \
    trainer.task_weights.ppg2abp=1.0 \
    trainer.task_weights.ecgppg2abp=1.0 \
    swanlab.experiment_name=finetune_calfree \
    "$@" \
    2>&1 | tee -a "$LOG_FILE"

#!/usr/bin/env bash
# UniCardio Rectified Flow 训练一键启动脚本
# 用法：放在仓库根目录，`bash train.sh`
set -euo pipefail

cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Hydra `hydra.run.dir` 通过 ${oc.env:UNICARDIO_STAGE,pretrain} 取这个变量做后缀。
export UNICARDIO_STAGE=pretrain

mkdir -p logs
LOG_FILE="logs/train_$(date +%Y%m%d_%H%M%S).log"

python run/pipeline/train.py \
    device=cuda \
    data.batch_size=256 \
    data.num_workers=12 \
    trainer.val_every=1 \
    2>&1 | tee "$LOG_FILE"

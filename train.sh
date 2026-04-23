#!/usr/bin/env bash
# UniCardio Rectified Flow 训练一键启动脚本
# 用法：放在仓库根目录，`bash train.sh`
set -euo pipefail

cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p logs
LOG_FILE="logs/train_$(date +%Y%m%d_%H%M%S).log"

python run/pipeline/train.py \
    device=cuda \
    data.batch_size=512 \
    data.num_workers=8 \
    trainer.lr=2.0e-3 \
    trainer.weight_decay=1.0e-6 \
    trainer.amp.enabled=true \
    trainer.compile.enabled=true \
    trainer.compile.mode=default \
    trainer.warmup_steps=519 \
    trainer.val_every=1 \
    2>&1 | tee "$LOG_FILE"

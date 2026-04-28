#!/usr/bin/env bash
# UniCardio Rectified Flow 训练一键启动脚本
# 用法：放在仓库根目录，`bash train.sh`
# 续训用 resume.sh，不在这里混。
set -euo pipefail

cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p logs
LOG_FILE="logs/train_$(date +%Y%m%d_%H%M%S).log"

python run/pipeline/train.py \
    device=cuda \
    data.batch_size=256 \
    data.num_workers=8 \
    model.downsample_factor=1 \
    trainer.amp.enabled=true \
    trainer.compile.enabled=true \
    trainer.compile.mode=default \
    trainer.val_every=1 \
    2>&1 | tee "$LOG_FILE"

#!/usr/bin/env bash
# UniCardio 续训一键启动脚本：从某个 checkpoint 接着练，沿用 ckpt 所在的 run 目录
# （新 checkpoints / loss.csv / swanlog 全部追加到原目录里）。
#
# 用法：
#   bash resume.sh                              # 自动挑 run/outputs/ 下最新 latest.pt
#   bash resume.sh path/to/latest.pt            # 指定 checkpoint
#   bash resume.sh path/to/best.pt trainer.epochs=600   # 续训 + 加 Hydra overrides
set -euo pipefail

cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CHECKPOINT="${1:-}"
if [ -z "$CHECKPOINT" ]; then
    CHECKPOINT="$(ls -t run/outputs/*/checkpoints/latest.pt 2>/dev/null | head -1 || true)"
elif [ -f "$CHECKPOINT" ]; then
    shift  # 显式传了 ckpt 路径才 shift；剩下的作为 Hydra overrides 透传
fi
if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
    echo "Error: resume checkpoint not found." >&2
    echo "Usage: bash resume.sh [checkpoint_path] [hydra overrides...]" >&2
    echo "       default: run/outputs/*/checkpoints/latest.pt (latest)" >&2
    exit 1
fi

# 沿用 ckpt 所在的 run 目录。trainer.resume_from 会同时恢复
# model / optimizer / scheduler / epoch；csv_logger 是 append 模式，
# loss.csv 不会重写表头。
RUN_DIR="$(dirname "$(dirname "$CHECKPOINT")")"
echo "Resuming from: $CHECKPOINT"
echo "Reusing run dir: $RUN_DIR"

mkdir -p logs
LOG_FILE="logs/resume_$(date +%Y%m%d_%H%M%S).log"

python run/pipeline/train.py \
    trainer.resume_from="$CHECKPOINT" \
    hydra.run.dir="$RUN_DIR" \
    device=cuda \
    data.batch_size=512 \
    data.num_workers=8 \
    "$@" \
    2>&1 | tee "$LOG_FILE"

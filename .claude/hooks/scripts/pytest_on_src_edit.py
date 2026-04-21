#!/usr/bin/env python3
"""
PostToolUse hook: run fast unit tests after edits to src/.

Rationale: tests/ 套件约 1 秒完成，核心 RF / mask / sampler 逻辑若被改坏会在
mask 一致性、rf_step 速度场、Euler 采样维度检查上立刻暴露。
Hook 是非阻塞的 —— 测试失败只会打印警告，不阻止编辑本身。
"""

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = PROJECT_DIR / "src"


def main() -> None:
    try:
        raw = sys.stdin.read().strip()
        if not raw:
            sys.exit(0)
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        sys.exit(0)

    tool_input = data.get("tool_input", {}) or {}
    file_path = tool_input.get("file_path", "")
    if not file_path:
        sys.exit(0)

    if not file_path.endswith(".py"):
        sys.exit(0)

    try:
        edited = Path(file_path).resolve()
    except (OSError, ValueError):
        sys.exit(0)

    if not edited.is_relative_to(SRC_DIR):
        sys.exit(0)

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "--no-header", "--tb=line"],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=12,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        sys.exit(0)

    if result.returncode == 0:
        sys.exit(0)

    tail = "\n".join(result.stdout.strip().splitlines()[-15:])
    print(
        f"[UniCardio] pytest regression after editing {edited.relative_to(PROJECT_DIR)}:\n{tail}",
        flush=True,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()

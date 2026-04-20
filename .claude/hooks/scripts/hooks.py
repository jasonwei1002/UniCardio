#!/usr/bin/env python3
"""
UniCardio Claude Code Hook Handler

SessionStart: prints the last few training epochs from check/loss.csv
              so every conversation starts with awareness of training state.

PostToolUse (Edit on model files): warns if attention mask buffers may be affected.
"""

import sys
import json
import os
from pathlib import Path


PROJECT_DIR = Path(__file__).parent.parent.parent  # .claude/hooks/scripts/ -> project root

MASK_KEYWORDS = ["mask1", "mask2", "mask3", "mask12", "mask13", "mask23", "mask123", "borrow_mode"]
CORE_MODEL_FILES = {
    "diffusion_model_no_compress_final.py",
    "train_original.py",
    "utils_together_original.py",
}


def handle_session_start():
    """Print last training epochs on session start."""
    loss_csv = PROJECT_DIR / "base_model" / "check" / "loss.csv"
    if not loss_csv.exists():
        print("[UniCardio] No training log found at base_model/check/loss.csv", flush=True)
        return

    try:
        import csv
        with open(loss_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            print("[UniCardio] Training log is empty.", flush=True)
            return

        last = rows[-1]
        epoch = last.get("epoch", "?")
        loss = last.get("train_loss", "?")
        stage = last.get("stage", "?")
        lr = last.get("lr", "?")

        # Count checkpoints
        check_dir = PROJECT_DIR / "base_model" / "check"
        ckpts = list(check_dir.glob("model*.pth")) if check_dir.exists() else []

        print(f"[UniCardio] Training status — epoch {epoch}/800 | stage {stage} | loss {loss} | lr {lr} | {len(ckpts)} checkpoints saved", flush=True)

        # Warn if loss has been stagnant (last 10 epochs)
        if len(rows) >= 10:
            recent_losses = []
            for r in rows[-10:]:
                try:
                    recent_losses.append(float(r.get("train_loss", "")))
                except (ValueError, TypeError):
                    pass
            if recent_losses and len(recent_losses) >= 5:
                delta = abs(recent_losses[-1] - recent_losses[0])
                if delta < 1e-5:
                    print("[UniCardio] WARNING: Loss has not changed in the last 10 epochs — possible training stall.", flush=True)

    except Exception as e:
        print(f"[UniCardio] Could not read loss.csv: {e}", flush=True)


def handle_post_tool_use(data):
    """Warn if edits to core model files may affect attention masks."""
    tool_name = data.get("tool_name", "")
    if tool_name not in ("Edit", "Write"):
        return

    tool_input = data.get("tool_input", {})
    file_path = tool_input.get("file_path", "")

    filename = Path(file_path).name
    if filename not in CORE_MODEL_FILES:
        return

    # Check if the edit touches mask-related code
    new_string = tool_input.get("new_string", "") or tool_input.get("content", "")
    old_string = tool_input.get("old_string", "")
    changed_text = (old_string + " " + new_string).lower()

    hit_keywords = [kw for kw in MASK_KEYWORDS if kw in changed_text]
    if hit_keywords:
        print(
            f"[UniCardio] ATTENTION: Edit to {filename} touches mask/borrow logic: {hit_keywords}. "
            f"Run /verify-training to confirm forward pass is still correct.",
            flush=True
        )


def main():
    try:
        stdin_content = sys.stdin.read().strip()
        if not stdin_content:
            sys.exit(0)

        data = json.loads(stdin_content)
        event = data.get("hook_event_name", "")

        if event == "SessionStart":
            handle_session_start()
        elif event == "PostToolUse":
            handle_post_tool_use(data)

        sys.exit(0)

    except json.JSONDecodeError:
        sys.exit(0)
    except Exception:
        sys.exit(0)


if __name__ == "__main__":
    main()

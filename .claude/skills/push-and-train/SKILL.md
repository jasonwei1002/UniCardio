---
name: push-and-train
description: Push local changes to GitHub and emit the exact command to paste on the GPU server to pull + start training. Use when the user wants to launch a training run after making code changes locally.
user-invocable: true
disable-model-invocation: true
allowed-tools:
  - Bash
  - Read
---

# Push-and-Train Skill

Automates the one-way laptop → GitHub → GPU server training launch described in CLAUDE.md.

> **Why a skill?** The laptop cannot SSH into the server, so the flow is always:
> push from laptop, then manually run `git pull && bash train.sh` on the server.
> This skill packages the laptop half and prints a ready-to-paste command for the server half.

## Workflow

When invoked, perform these steps **in order** and stop at the first failure:

1. **Show local state** — run `git status --short` and `git log --oneline -5`. If the working tree is dirty, ask the user whether to commit or stash. Do **not** auto-commit without confirmation.
2. **Confirm branch** — run `git branch --show-current`. Warn if not on `main` (training server pulls `main`).
3. **Push** — run `git push` (plain, no `--force`). If the push is rejected, stop and surface the error; never force-push on the user's behalf.
4. **Print server command** — print, in a copyable block, the exact command to paste on the server:
   ```
   cd <server UniCardio path> && git pull --ff-only && bash train.sh
   ```
   The server path is not known from the laptop; ask the user once and save it to `.claude/.push-and-train-server-path` (git-ignored by default since `.claude/` itself is committed — add a `.gitignore` exclusion if the user wants that cached).
5. **Optional follow-up** — suggest `/swanlog` after training starts to pull the latest SwanLab run log.

## Guardrails

- Never `git push --force` or `--force-with-lease` without explicit user instruction.
- Never amend already-pushed commits.
- Never run training commands directly — the laptop is CPU-only and would take hours on the 3.4 GB dataset (see CLAUDE.md "Development Environment").
- If `git status` shows changes under `data/`, `run/outputs/`, or `logs/`, warn the user: these are `.gitignore`'d, so they will **not** reach the server via git.

## Arguments

No positional arguments. Optional named hints the user may provide in the invocation:

- `message=<str>` — commit message to use if the skill proposes committing dirty changes.
- `branch=<str>` — push to a non-default branch (skill will still warn that the server pulls `main`).

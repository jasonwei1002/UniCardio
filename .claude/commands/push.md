---
description: 把本地改动推到 GitHub（查 status/log、确认分支、git push），不碰训练也不提示服务器命令
allowed-tools: Bash
---

把当前分支推到 GitHub。**只推送**，不做别的。

## 执行步骤

按顺序执行，遇到任何一步出错就停下来告诉用户原因：

1. **看本地状态** — 跑 `git status --short` 和 `git log --oneline -5`
   - 工作区有未提交改动时，**问用户**是 commit 还是 stash。**不要自动 commit**
   - 如果 `git status` 里有 `data/`、`run/outputs/`、`logs/` 下的改动，提醒一句这些路径是 `.gitignore` 的，不会被推走
2. **确认分支** — 跑 `git branch --show-current`
   - 分支名看起来不寻常时（比如 throwaway / 很旧的 feature 分支），问一句是不是确定推这个
3. **推送** — 跑 `git push`（裸命令，**不加** `--force` / `--force-with-lease`）
   - 被远端拒收时，把错误原样给用户，**绝不**在未授权下 force push
   - 推成功后把 `refspec: old..new` 那一行简短汇报

## 禁止事项

- 未经明确指示不得 `git push --force` 或 `--force-with-lease`
- 不 amend 已经推出去的 commit
- 不做任何训练相关动作，不提示服务器命令，不建议 `/swanlog`

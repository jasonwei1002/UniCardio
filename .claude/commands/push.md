---
description: 把本地改动推到 GitHub：工作区脏了自动 commit，然后 git push（裸命令，不带 force）
allowed-tools: Bash, Read
---

把当前分支推到 GitHub。**有未提交改动时自动 commit + push；没有就只 push。**

## 执行步骤

按顺序执行，遇到任何一步出错就停下来告诉用户原因：

1. **看本地状态** — 跑 `git status --short` 和 `git log --oneline -5`
   - 工作区干净时跳到第 4 步直接 `git push`。
   - 如果 `git status` 里有 `data/`、`run/outputs/`、`logs/`、`*.npy`、`*.pt` 之类的路径，提醒一句它们应当被 `.gitignore` 屏蔽（出现在 status 里说明 .gitignore 漏了，**让用户决定要不要推**，不要继续）。

2. **安全扫描** — 工作区脏时，先 `git status --short` + `git diff --stat HEAD` 看一眼改动文件名 / 内容里是否触碰敏感物：
   - 文件名：`.env`、`.env.*`、`*.pem`、`*.key`、`credentials*`、`settings.json`、`*_secret*`、`*_token*`、`*.sqlite`、`*.db`
   - 内容：`sk-`、`ghp_`、`AKIA`、`-----BEGIN PRIVATE KEY-----` 等显式凭据片段
   - 命中任何一条 → **立即停下问用户**，不要 commit、不要 push。

3. **自动 commit** —
   - `git add -A` 把跟踪的修改和新增的非忽略文件全部入暂存。
   - 跑 `git diff --cached --stat` 和 `git log --oneline -10` 摸清当前改动范围与本仓库的 commit 风格（Conventional Commits：`feat / fix / docs / refactor / perf / test / chore / tune` + scope）。
   - 起一条简洁、聚焦"为什么"的 commit message：标题 ≤72 字符，必要时加正文段落，**用 HEREDOC 喂给 `git commit -m`**，并在末尾保留：
     ```
     Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
     ```
   - **绝不**用 `git commit --amend`、`--no-verify`、`--no-gpg-sign`。
   - pre-commit hook 失败时不要 amend，修好之后**新建**一个 commit。

4. **确认分支** — 跑 `git branch --show-current`。
   - 分支名看起来不寻常（throwaway / 很旧的 feature 分支 / 名字奇怪）时，问一句是不是确定推这个。

5. **推送** — 跑 `git push`（裸命令，**不加** `--force` / `--force-with-lease`）。
   - 被远端拒收时，把错误原样给用户，**绝不**在未授权下 force push。
   - 推成功后把 `refspec: old..new` 那一行简短汇报；若本步包含了第 3 步的新 commit，也带一句 commit 标题。

## 禁止事项

- 未经明确指示不得 `git push --force` 或 `--force-with-lease`。
- 不 amend 已经推出去的 commit。
- 不跳过 hook（`--no-verify` 等）。
- 不做任何训练相关动作，不提示服务器命令，不建议 `/swanlog`。
- 命中安全扫描的文件，绝不"先 commit 再说"——必须先停下问用户。

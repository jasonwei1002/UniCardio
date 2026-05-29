---
description: 一键拉取 SwanLab 最新一次实验的完整日志到 run/outputs/swanlog_<exp_id>/
argument-hint: "[--exp-id <id>] [--extra-keys 'k1,k2'] (默认 --latest)"
allowed-tools: Bash, Read
---

从 SwanLab 云端拉取一次实验的全量数据（metrics / config / metadata / requirements / run_info）到本地。trainer 无关 — RF backbone、BP head 以及未来新增的 trainer 都按同一脚本拉取。

## 执行步骤

1. **运行拉取脚本**
   - 无参数默认拉 `--latest`
   - 用户传的参数（`--exp-id <id>` / `--extra-keys "<csv>"` 等）原样透传
   - 命令（在仓库根目录执行）：`python script/fetch_swanlog.py $ARGUMENTS`；`$ARGUMENTS` 为空时改为 `python script/fetch_swanlog.py --latest`
   - 用 Bash 工具执行，捕获完整 stdout/stderr

2. **如果拉取失败**，把 stderr 最后 10 行原样给用户看，然后停止。常见失败：
   - 未登录（`~/.swanlab/.netrc` 不存在 / 过期）→ 提示 `swanlab login`
   - 项目不存在 / 404 → 提示云端还没有训练上传
   - 网络错误 → 原样报告

3. **拉取成功后**，读取输出目录里的 `run_info.json`，向用户简报：
   - **run**: `<name>`（id: `<id>`）
   - **state**: `<state>`（RUNNING / FINISHED / CRASHED / ABORTED）
   - **created_at** / **finished_at**
   - **url**（WebUI 直达链接）

4. **如果 `metrics.csv` 存在**，给用户看：
   - 总行数（`wc -l`）
   - 该 run 实际命中的非空 value 列（脚本日志里的 "value cols" 行已列出，可直接复用）
   - 最末一行非空 value 列的数值（用 Pandas 一行读取即可：每个 value 列的 `df[col].dropna().iloc[-1]`，跳过空列）

   不要再固定看 `epoch/avg_loss` / `val/loss_mean` — 这两个只是 RF backbone 才有。BP head run 的 eval 指标是**按 task 嵌套**的 `val/<task>/mae_mean` / `test/<task>/mae_sbp` / `test/loss_mean` 等（task 名夹在中间，如 `val/ecgppg2abp/mae_mean`），外加每任务 `epoch/train_loss_<task>`；以脚本输出的 value cols 为准。

5. **如果用户传了 `--extra-keys`**，简短复述追加了哪些 key（脚本日志里有 "appended N extra keys" 行）。

6. 最后一句告诉用户输出目录绝对路径。

## 注意

- 这个命令是**幂等覆盖**的：同一 exp_id 重复调用会刷新 `run/outputs/swanlog_<exp_id>/` 里的文件，不会累积垃圾。
- 不要启动训练 / 改动配置，命令只负责拉数据。
- 当 SwanLab 上出现脚本 `_metric_keys()` 没列出的新 metric 时，用 `--extra-keys "<csv>"` 一次性追加；长期改动应在 `script/fetch_swanlog.py::_metric_keys()` 里登记一段新 trainer 的注释 + key 列表（参考已有的 RF backbone / BP head 两段）。

---
description: 一键拉取 SwanLab 最新一次实验的完整日志到 run/outputs/swanlog_<exp_id>/
argument-hint: "[--exp-id <id>] (默认 --latest)"
allowed-tools: Bash, Read
---

从 SwanLab 云端拉取一次实验的全量数据（metrics / config / metadata / requirements / run_info）到本地。

## 执行步骤

1. **运行拉取脚本**
   - 无参数默认拉 `--latest`
   - 若用户传了 `--exp-id <id>` 就透传过去
   - 命令（在仓库根目录执行）：`python script/fetch_swanlog.py $ARGUMENTS` 或 `python script/fetch_swanlog.py --latest`（无参数时）
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

4. **如果 `metrics.csv` 存在**，用 Bash 读最后几行，给用户看：
   - 总行数（`wc -l`）
   - 最新一行的 `epoch/avg_loss` 和 `val/loss_mean`（若有）

5. 最后一句告诉用户输出目录绝对路径。

## 注意

- 这个命令是**幂等覆盖**的：同一 exp_id 重复调用会刷新 `run/outputs/swanlog_<exp_id>/` 里的文件，不会累积垃圾。
- 不要启动训练 / 改动配置，命令只负责拉数据。

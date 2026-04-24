"""从 SwanLab 云端拉取一次训练的指标 + 完整 profile（config、metadata、requirements）。

输出目录结构::

    run/outputs/swanlog_<YYYY-MM-DD_HH-MM-SS>/     # created_at（Asia/Singapore, UTC+8）
    ├── metrics.csv         # 全部指标的 pandas DataFrame
    ├── config.yaml         # 当次实验的 Hydra resolved config
    ├── metadata.json       # swanlab 采集的系统/硬件/git 信息
    ├── requirements.txt    # 训练环境的 pip freeze
    └── run_info.json       # 实验名/id/状态/时间戳

用法::

    # 本地 swanlab login 后可自动识别账号；--user 不填就用登录账号
    python script/fetch_swanlog.py --exp-id <experiment_id>
    python script/fetch_swanlog.py --latest
    python script/fetch_swanlog.py --latest -o run/outputs/swanlog_latest

要求：
    - swanlab>=0.7.15
    - 本地已 ``swanlab login``（凭据在 ``~/.swanlab/.netrc``）
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402
import swanlab  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from swanlab.error import ApiError  # noqa: E402

from src.model_module.tasks import TASK_LIST  # noqa: E402

logger = logging.getLogger(__name__)


_LOCAL_TZ = ZoneInfo("Asia/Singapore")


def _format_created_at(created_at: Any) -> str:
    """把 swanlab 的 created_at（UTC ISO8601 字符串或 datetime）转成
    Asia/Singapore 时区下文件系统安全、字典序即时间序的时间戳，形如
    ``2026-04-23_14-33-12``。
    """
    if isinstance(created_at, datetime):
        dt = created_at
    else:
        s = str(created_at).rstrip("Z").split(".")[0]
        dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)  # swanlab 时间戳统一视作 UTC
    return dt.astimezone(_LOCAL_TZ).strftime("%Y-%m-%d_%H-%M-%S")


def _metric_keys() -> list[str]:
    """与 trainer.py 里 swanlab.log 记录的 key 对齐。"""
    task_names = [t.name for t in TASK_LIST]
    return [
        "train/loss",
        "train/lr",
        *[f"train/loss_{n}" for n in task_names],
        "epoch/avg_loss",
        "epoch/lr",
        "epoch/time_s",
        *[f"epoch/loss_{n}" for n in task_names],
        "val/loss_mean",
        *[f"val/loss_{n}" for n in task_names],
    ]


def _fetch_metrics(run: Any, keys: list[str]) -> pd.DataFrame:
    """按 key 逐个拉取 metrics，遇到 404（该 key 不存在于此实验）就跳过。

    任务权重为 0 的任务不会被训练循环记录到 SwanLab，批量请求会因为单个
    404 整体失败；逐 key 请求并合并可以兼容 3 任务 / 5 任务两种 run。
    """
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for key in keys:
        try:
            df = run.metrics(keys=[key], x_axis="step")
        except ApiError as exc:
            if "404" in str(exc):
                missing.append(key)
                continue
            raise
        if df is not None and not df.empty:
            frames.append(df)
    if missing:
        logger.info("skipped %d missing keys: %s", len(missing), missing)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)


def _resolve_run(
    api: "swanlab.Api",
    user: str,
    project: str,
    exp_id: str | None,
    latest: bool,
):
    if exp_id:
        return api.run(path=f"{user}/{project}/{exp_id}")
    runs = list(api.runs(path=f"{user}/{project}"))
    if not runs:
        raise RuntimeError(f"No experiments in {user}/{project}.")
    if latest:
        return max(runs, key=lambda r: r.created_at)
    raise ValueError("Must pass --exp-id or --latest.")


def _dump_profile(profile: Any, out_dir: Path) -> list[str]:
    """把 profile 里的各块分别落到 out_dir；返回已写文件名列表。

    SwanLab 0.7.x 的 ``run.profile`` 是 ``Profile`` 对象而不是 dict，
    暴露 ``config`` / ``metadata`` / ``requirements`` / ``conda`` 属性。
    """
    def _get(key: str) -> Any:
        if profile is None:
            return None
        if isinstance(profile, dict):
            return profile.get(key)
        return getattr(profile, key, None)

    written: list[str] = []
    if config := _get("config"):
        cfg_path = out_dir / "config.yaml"
        cfg_path.write_text(OmegaConf.to_yaml(OmegaConf.create(config)))
        written.append(cfg_path.name)
    if metadata := _get("metadata"):
        meta_path = out_dir / "metadata.json"
        meta_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, default=str)
        )
        written.append(meta_path.name)
    if requirements := _get("requirements"):
        req_text = (
            "\n".join(requirements)
            if isinstance(requirements, list)
            else str(requirements)
        )
        req_path = out_dir / "requirements.txt"
        req_path.write_text(req_text)
        written.append(req_path.name)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--user", default=None,
        help="SwanLab 用户名；默认用本地 swanlab login 的账号",
    )
    parser.add_argument("--project", default="UniCardio", help="项目名")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--exp-id", help="实验 ID（WebUI URL 里可复制）")
    group.add_argument(
        "--latest", action="store_true", help="挑选项目下最新一次实验"
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help=(
            "输出目录；默认 run/outputs/swanlog_<YYYY-MM-DD_HH-MM-SS>/"
            "（Asia/Singapore 时区）"
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    api = swanlab.Api()
    user = args.user or api.user().username
    run = _resolve_run(api, user, args.project, args.exp_id, args.latest)
    logger.info(
        "run: %s id=%s state=%s url=%s",
        run.name, run.id, run.state, run.url,
    )

    out_dir = (
        Path(args.output)
        if args.output
        else REPO_ROOT / "run" / "outputs"
        / f"swanlog_{_format_created_at(run.created_at)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _fetch_metrics(run, _metric_keys())
    metrics_path = out_dir / "metrics.csv"
    df.to_csv(metrics_path, index=True)
    logger.info(
        "metrics: %d rows × %d cols -> %s",
        len(df), len(df.columns), metrics_path.name,
    )

    (out_dir / "run_info.json").write_text(
        json.dumps(
            {
                "id": run.id,
                "name": run.name,
                "description": run.description,
                "state": run.state,
                "created_at": run.created_at,
                "finished_at": run.finished_at,
                "url": run.url,
                "user": getattr(run.user, "username", str(run.user)),
            },
            indent=2, ensure_ascii=False, default=str,
        )
    )

    written = _dump_profile(run.profile or {}, out_dir)
    logger.info("profile files: %s", written or "(profile 为空)")
    logger.info("output dir: %s", out_dir)


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


class SimpleCSVLogger:
    """append-only 的 CSV 日志，首次打开时会写入表头。

    Args:
        filepath: 目标 CSV 路径（不存在则自动创建）。
        fieldnames: 有序列名列表，同时作为 CSV 表头。
    """

    def __init__(
        self, filepath: str | Path, fieldnames: Sequence[str]
    ) -> None:
        self.filepath = Path(filepath)
        self.fieldnames = list(fieldnames)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        if not self.filepath.exists():
            with self.filepath.open("w", newline="") as f:
                csv.writer(f).writerow(self.fieldnames)

    def log(self, **row: Any) -> None:
        """追加一行记录，缺失字段会写成空字符串。"""
        values = [row.get(name, "") for name in self.fieldnames]
        with self.filepath.open("a", newline="") as f:
            csv.writer(f).writerow(values)

    def log_mapping(self, row: Mapping[str, Any]) -> None:
        """接收 dict 的版本（功能同 :meth:`log`）。"""
        values = [row.get(name, "") for name in self.fieldnames]
        with self.filepath.open("a", newline="") as f:
            csv.writer(f).writerow(values)

"""Lightweight append-only CSV logger.

The original ``SimpleCSVLogger`` in ``base_model/utils_together_original.py``
used a fixed 9-column schema tailored to diffusion staged training. We keep
the class name for continuity but accept an arbitrary ordered field list so
the RF trainer can log per-task losses without rewriting the schema.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


class SimpleCSVLogger:
    """Append-only CSV logger that writes a header on first open.

    Args:
        filepath: Destination CSV (created if missing).
        fieldnames: Ordered list of column names; becomes the CSV header.
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
        """Append one row. Missing fields become empty strings."""
        values = [row.get(name, "") for name in self.fieldnames]
        with self.filepath.open("a", newline="") as f:
            csv.writer(f).writerow(values)

    def log_mapping(self, row: Mapping[str, Any]) -> None:
        """Variant that takes a dict instead of keyword args."""
        values = [row.get(name, "") for name in self.fieldnames]
        with self.filepath.open("a", newline="") as f:
            csv.writer(f).writerow(values)

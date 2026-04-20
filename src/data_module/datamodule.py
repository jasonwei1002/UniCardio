"""Dataloader factory: load, permute channels, normalize BP, split, wrap.

This is the single place in the codebase that touches the file-vs-model slot
mismatch. After :func:`load_and_preprocess`, every tensor is in model slot
order ``(ECG=0, PPG=1, ABP=2)`` and the ABP channel (slot 2) is in
normalized units ``(x - 100) / 50``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from ..utils.normalization import bp_normalize
from ..utils.seed import worker_init_fn
from .cardiac_dataset import FILE_TO_MODEL_PERMUTATION, CardiacDataset

logger = logging.getLogger(__name__)


def load_and_preprocess(data_path: str | Path) -> np.ndarray:
    """Load the raw ``.npy`` file, permute channels, normalize ABP.

    Args:
        data_path: Path to ``Final_sig_combined.npy`` (shape ``(N, 3, L)``).

    Returns:
        ``float32`` array of shape ``(N, 3, L)`` in model slot order with
        BP already normalized.
    """
    arr = np.load(Path(data_path))
    if arr.ndim != 3 or arr.shape[1] != 3:
        raise ValueError(
            f"Expected (N, 3, L) data, got shape {arr.shape}"
        )
    arr = arr.astype(np.float32, copy=False)

    # File order (PPG, BP, ECG) -> model order (ECG, PPG, ABP).
    arr = arr[:, list(FILE_TO_MODEL_PERMUTATION), :]

    # BP is now at slot 2 (ABP). Normalize to model units (x - 100) / 50.
    arr[:, 2, :] = bp_normalize(arr[:, 2, :])
    return arr


def _split_three_way(
    signals: np.ndarray,
    val_size: int,
    test_size: int,
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the two-step split from ``base_model/train_original.py``."""
    train, temp = train_test_split(
        signals, test_size=val_size + test_size, random_state=split_seed
    )
    val, test = train_test_split(
        temp, test_size=test_size, random_state=split_seed
    )
    return train, val, test


def build_loaders(
    cfg: Mapping[str, Any],
    *,
    num_workers_override: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Construct ``(train, val, test)`` loaders from a config mapping.

    Args:
        cfg: Data config with keys ``data_path``, ``val_size``, ``test_size``,
            ``split_seed``, ``batch_size``, ``num_workers``, ``pin_memory``.
        num_workers_override: If provided, overrides ``cfg['num_workers']``.
            Useful when smoke-testing on CPU where workers add startup lag.

    Returns:
        Tuple of three :class:`DataLoader` instances.
    """
    signals = load_and_preprocess(cfg["data_path"])
    val_size = int(cfg.get("val_size", 20_000))
    test_size = int(cfg.get("test_size", 20_000))
    split_seed = int(cfg.get("split_seed", 42))
    train, val, test = _split_three_way(signals, val_size, test_size, split_seed)

    logger.info(
        "Data splits — train %d, val %d, test %d (slot_length=%d)",
        train.shape[0],
        val.shape[0],
        test.shape[0],
        train.shape[-1],
    )

    batch_size = int(cfg.get("batch_size", 128))
    num_workers = (
        num_workers_override
        if num_workers_override is not None
        else int(cfg.get("num_workers", 8))
    )
    pin_memory = bool(cfg.get("pin_memory", True))

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": worker_init_fn if num_workers > 0 else None,
    }

    train_loader = DataLoader(
        CardiacDataset(train), shuffle=True, **loader_kwargs
    )
    val_loader = DataLoader(
        CardiacDataset(val), shuffle=False, **loader_kwargs
    )
    test_loader = DataLoader(
        CardiacDataset(test), shuffle=False, **loader_kwargs
    )
    return train_loader, val_loader, test_loader

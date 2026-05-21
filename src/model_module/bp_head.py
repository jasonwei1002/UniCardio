"""Scalar SBP/DBP regression head — Path A refinement model.

Path A decomposes BP estimation into (a) per-sample min-max ABP shape via RF
and (b) scalar (SBP, DBP) via this :class:`BPHead`. They are trained
independently and combined only at inference time
(:func:`src.utils.normalization.reconstruct_mmHg`).

The head is intentionally small (~25 k params): a 3-stage strided Conv1d
stack on ECG+PPG, global-average pool, then a 2-layer MLP to a 2-D output
``(sbp_mmHg, dbp_mmHg)``. Output is raw mmHg — no further denormalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class BPHeadConfig:
    """Immutable config for :class:`BPHead`."""

    in_channels: int = 2  # ECG + PPG (slots 0 and 1)
    hidden: int = 64
    depth: int = 3
    kernel_size: int = 15
    stride: int = 4
    mlp_hidden: int = 64

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any] | "BPHeadConfig") -> "BPHeadConfig":
        if isinstance(cfg, cls):
            return cfg
        return cls(
            in_channels=int(cfg.get("in_channels", 2)),
            hidden=int(cfg.get("hidden", 64)),
            depth=int(cfg.get("depth", 3)),
            kernel_size=int(cfg.get("kernel_size", 15)),
            stride=int(cfg.get("stride", 4)),
            mlp_hidden=int(cfg.get("mlp_hidden", 64)),
        )


class BPHead(nn.Module):
    """``(B, in_channels, L) → (B, 2)`` regressor for ``(SBP, DBP)`` in mmHg."""

    def __init__(self, config: BPHeadConfig | Mapping[str, Any]) -> None:
        super().__init__()
        self.config = BPHeadConfig.from_mapping(config)
        cfg = self.config

        layers: list[nn.Module] = []
        in_c = cfg.in_channels
        for i in range(cfg.depth):
            # First stage doubles channels to a small ramp; subsequent stages
            # keep channel count at `hidden` to stay parameter-cheap.
            out_c = cfg.hidden // 2 if i == 0 else cfg.hidden
            layers.append(
                nn.Conv1d(
                    in_c,
                    out_c,
                    kernel_size=cfg.kernel_size,
                    stride=cfg.stride,
                    padding=cfg.kernel_size // 2,
                )
            )
            layers.append(nn.GELU())
            in_c = out_c
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_c, cfg.mlp_hidden),
            nn.GELU(),
            nn.Linear(cfg.mlp_hidden, 2),
        )

    def forward(self, ecg_ppg: Tensor) -> Tensor:
        """Predict ``(SBP, DBP)`` in mmHg.

        Args:
            ecg_ppg: ``(B, in_channels, L)`` raw (un-normalized) ECG + PPG.

        Returns:
            ``(B, 2)`` with column 0 = SBP, column 1 = DBP, both in mmHg.
        """
        h = self.conv(ecg_ppg)
        h = self.pool(h).squeeze(-1)
        return self.mlp(h)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_bp_head(config: Mapping[str, Any] | BPHeadConfig) -> BPHead:
    """Factory helper mirroring other model-module entry points."""
    return BPHead(config)

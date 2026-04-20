"""3-slot Rectified-Flow backbone for UniCardio.

This replaces the 4-slot ``diff_CSDI`` module from the original diffusion
codebase. Key differences:

* 3 slots (ECG=0, PPG=1, ABP=2) — the zero-padded placeholder slot and the
  associated ``borrow_mode`` output-head routing have been removed.
* Continuous time conditioning via :class:`FlowTimeEmbedding` (``t ∈ [0, 1]``)
  instead of integer-step ``DiffusionEmbedding``.
* Mask is passed in from the caller (built per task elsewhere) rather than
  being selected from a hand-coded set of 7 pre-registered masks.

Architectural pieces preserved verbatim: :class:`SignalEncoder`,
:class:`ResidualBlock`, per-slot :class:`LayerNorm`, per-slot 2-stage output
head, skip-sum aggregation with ``1 / sqrt(n_layers)`` scaling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .embeddings import (
    DEFAULT_CHANNELS_PER_KERNEL,
    DEFAULT_KERNELS,
    FlowTimeEmbedding,
    SignalEncoder,
    conv1d_kaiming,
)
from .residual_block import ResidualBlock

N_SLOTS: int = 3


@dataclass(frozen=True)
class BackboneConfig:
    """Plain config holder so the backbone can be instantiated without Hydra."""

    slot_length: int = 500
    channels: int = 288
    n_layers: int = 5
    nheads: int = 8
    time_embedding_dim: int = 256
    kernel_sizes: tuple[int, ...] = DEFAULT_KERNELS
    per_kernel_channels: int = DEFAULT_CHANNELS_PER_KERNEL
    ffn_dim: int = 64

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> "BackboneConfig":
        """Build from a plain dict (or OmegaConf DictConfig)."""
        return cls(
            slot_length=int(cfg["slot_length"]),
            channels=int(cfg["channels"]),
            n_layers=int(cfg.get("n_layers", cfg.get("layers", 5))),
            nheads=int(cfg["nheads"]),
            time_embedding_dim=int(cfg["time_embedding_dim"]),
            kernel_sizes=tuple(cfg.get("kernel_sizes", DEFAULT_KERNELS)),
            per_kernel_channels=int(
                cfg.get("per_kernel_channels", DEFAULT_CHANNELS_PER_KERNEL)
            ),
            ffn_dim=int(cfg.get("ffn_dim", 64)),
        )


class _OutputHead(nn.Module):
    """Per-slot 2-stage output head (``C -> C -> 1``) with ReLU between."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.proj1 = conv1d_kaiming(channels, channels, 1)
        self.proj2 = conv1d_kaiming(channels, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj2(F.relu(self.proj1(x)))


class UniCardioBackbone(nn.Module):
    """Transformer backbone with per-slot encoders and a single task head.

    Forward signature: ``forward(x, t, mask, target_slot)`` where

    * ``x``: ``(B, 1, 3 * L_slot)`` — concatenated slot tensor.
    * ``t``: ``(B,)`` — continuous time in ``[0, 1]``.
    * ``mask``: ``(3 * L_slot, 3 * L_slot)`` — additive attention mask.
    * ``target_slot``: ``int`` — which slot's output head to use.
    """

    def __init__(self, config: Mapping[str, Any] | BackboneConfig) -> None:
        super().__init__()
        cfg = (
            config
            if isinstance(config, BackboneConfig)
            else BackboneConfig.from_mapping(config)
        )
        self.cfg = cfg
        self.L = cfg.slot_length
        self.channels = cfg.channels
        self.n_layers = cfg.n_layers

        # Expected feature dim from the encoder bank (must match self.channels).
        encoded_channels = cfg.per_kernel_channels * len(cfg.kernel_sizes)
        if encoded_channels != cfg.channels:
            raise ValueError(
                f"channels ({cfg.channels}) must equal "
                f"per_kernel_channels * len(kernel_sizes) "
                f"({cfg.per_kernel_channels} * {len(cfg.kernel_sizes)} "
                f"= {encoded_channels})."
            )

        self.input_encoders = nn.ModuleList(
            [
                SignalEncoder(
                    in_channels=1,
                    kernel_sizes=cfg.kernel_sizes,
                    per_kernel_channels=cfg.per_kernel_channels,
                )
                for _ in range(N_SLOTS)
            ]
        )
        self.input_norms = nn.ModuleList(
            [nn.LayerNorm([self.channels, self.L]) for _ in range(N_SLOTS)]
        )

        self.time_embedding = FlowTimeEmbedding(
            embedding_dim=cfg.time_embedding_dim
        )

        total_length = self.L * N_SLOTS
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    channels=self.channels,
                    time_embedding_dim=cfg.time_embedding_dim,
                    nheads=cfg.nheads,
                    length=total_length,
                    ffn_dim=cfg.ffn_dim,
                )
                for _ in range(self.n_layers)
            ]
        )

        self.output_heads = nn.ModuleList(
            [_OutputHead(self.channels) for _ in range(N_SLOTS)]
        )

    @property
    def total_length(self) -> int:
        return self.L * N_SLOTS

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        mask: Tensor,
        target_slot: int,
    ) -> Tensor:
        """Predict the velocity for the target slot only.

        Returns a tensor of shape ``(B, 1, L_slot)``.
        """
        B, input_dim, L_total = x.shape
        if L_total != self.total_length:
            raise ValueError(
                f"Expected L_total={self.total_length}, got {L_total}"
            )
        if input_dim != 1:
            raise ValueError(f"Expected input channel dim 1, got {input_dim}")
        if not 0 <= target_slot < N_SLOTS:
            raise ValueError(
                f"target_slot must be in [0, {N_SLOTS}), got {target_slot}"
            )

        # Per-slot encoding -> per-slot LayerNorm.
        slot_feats: list[Tensor] = []
        for i in range(N_SLOTS):
            start, end = i * self.L, (i + 1) * self.L
            slot_in = x[:, :, start:end]
            encoded = self.input_encoders[i](slot_in)
            encoded = encoded.reshape(B, self.channels, self.L)
            encoded = self.input_norms[i](encoded)
            slot_feats.append(encoded)

        h = torch.cat(slot_feats, dim=-1)  # (B, C, 3 * L)

        time_emb = self.time_embedding(t)  # (B, time_embedding_dim)

        skip_list: list[Tensor] = []
        for layer in self.residual_layers:
            h, skip = layer(h, time_emb, mask)
            skip_list.append(skip)

        h = torch.stack(skip_list, dim=0).sum(dim=0) / math.sqrt(self.n_layers)
        # Extract target-slot activations and decode with the matching head.
        start = target_slot * self.L
        end = start + self.L
        target_feat = h[:, :, start:end]
        return self.output_heads[target_slot](target_feat)

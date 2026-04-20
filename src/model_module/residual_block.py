"""Residual transformer block used by :class:`UniCardioBackbone`.

Preserved structure from the original UniCardio code:

* 1-layer ``nn.TransformerEncoder`` (``d_model=channels``, GELU, ``dim_feedforward=64``).
* Sinusoidal position embedding along the token dimension (registered as a
  non-persistent buffer so it follows ``.to(device)``).
* Gated activation ``sigmoid(gate) * tanh(filter)`` on the mid-projection.
* Residual is scaled by ``1 / sqrt(2)``; skip is returned separately for
  aggregation by the backbone.

The only behavioural change vs. the diffusion version: the block no longer
takes a ``device`` argument. Placement is controlled externally via
``module.to(device)``, which matches idiomatic PyTorch and avoids duplicate
tensor-movement logic.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from .embeddings import conv1d_kaiming


def _transformer_layer(
    channels: int,
    nheads: int,
    *,
    ffn_dim: int = 64,
    dropout: float = 0.0,
) -> nn.TransformerEncoder:
    """Mirror of ``get_torch_trans`` from the original code."""
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels,
        nhead=nheads,
        dim_feedforward=ffn_dim,
        dropout=dropout,
        activation="gelu",
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=1)


def _sinusoidal_position_embedding(length: int, d_model: int) -> Tensor:
    """Standard sinusoidal positional encoding, shape ``(length, d_model)``."""
    pe = torch.zeros(length, d_model)
    position = torch.arange(length).unsqueeze(1).float()
    div_term = 1.0 / torch.pow(
        10000.0, torch.arange(0, d_model, 2).float() / d_model
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class ResidualBlock(nn.Module):
    """One gated transformer block with flow-time conditioning.

    Args:
        channels: Feature dimension ``C`` of the token sequence.
        time_embedding_dim: Dimension of the time embedding vector; a linear
            projection maps it to ``channels`` before broadcast-adding.
        nheads: Number of attention heads.
        length: Total token sequence length (``3 * L_slot`` for this project).
        ffn_dim: Transformer FFN dimension; matches the original code's 64.
    """

    def __init__(
        self,
        channels: int,
        time_embedding_dim: int,
        nheads: int,
        length: int,
        *,
        ffn_dim: int = 64,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.length = length
        self.time_projection = nn.Linear(time_embedding_dim, channels)
        self.cond_projection = conv1d_kaiming(channels, 2 * channels, 1)
        self.mid_projection = conv1d_kaiming(channels, 2 * channels, 1)
        self.output_projection = conv1d_kaiming(channels, 2 * channels, 1)
        self.time_layer = _transformer_layer(channels, nheads, ffn_dim=ffn_dim)
        self.register_buffer(
            "pe",
            _sinusoidal_position_embedding(length, channels),
            persistent=False,
        )

    def forward(
        self, x: Tensor, time_emb: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x: ``(B, C, L)`` input sequence (C = ``channels``).
            time_emb: ``(B, time_embedding_dim)`` flow-time embedding.
            mask: ``(L, L)`` additive attention mask (0 / -inf).

        Returns:
            ``(x_out, skip)`` both of shape ``(B, C, L)``.
        """
        B, C, L = x.shape
        base_shape = x.shape

        # Broadcast-add the time embedding to every spatial position.
        t_emb = self.time_projection(time_emb).unsqueeze(-1)  # (B, C, 1)
        y = x + t_emb

        # Transformer self-attention along the token axis with the task mask.
        y_seq = y.permute(2, 0, 1)  # (L, B, C)
        y_seq = self.time_layer(y_seq, mask=mask)
        y = y_seq.permute(1, 2, 0).reshape(B, C, L)

        # Gated mid-projection with positional bias.
        y = self.mid_projection(y)  # (B, 2C, L)
        pos_bias = (
            self.pe.unsqueeze(0)
            .expand(B, -1, -1)
            .permute(0, 2, 1)
            .contiguous()
        )  # (B, C, L)
        y = y + self.cond_projection(pos_bias)

        gate, filt = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)  # (B, C, L)

        y = self.output_projection(y)  # (B, 2C, L)
        residual, skip = torch.chunk(y, 2, dim=1)

        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip

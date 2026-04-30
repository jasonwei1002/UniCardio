""":class:`UniCardioBackbone` 使用的残差 Transformer block。

"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .embeddings import conv1d_kaiming


class _SdpaAttentionFFN(nn.Module):

    def __init__(
        self,
        channels: int,
        nheads: int,
        *,
        ffn_dim: int = 64,
    ) -> None:
        super().__init__()
        if channels % nheads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by nheads ({nheads})"
            )
        self.channels = channels
        self.nheads = nheads
        self.head_dim = channels // nheads

        self.qkv_proj = nn.Linear(channels, 3 * channels)
        self.out_proj = nn.Linear(channels, channels)
        self.ln1 = nn.LayerNorm(channels)
        self.ln2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, channels),
        )

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        """``(B, L, C)`` 输入输出；``attn_mask`` 形状 ``(L, L)`` bool。"""
        B, L, C = x.shape
        qkv = self.qkv_proj(x)  # (B, L, 3C)
        q, k, v = qkv.chunk(3, dim=-1)
        # (B, L, C) -> (B, H, L, D)
        q = q.view(B, L, self.nheads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.nheads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.nheads, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=False
        )  # (B, H, L, D)
        y = y.transpose(1, 2).contiguous().view(B, L, C)
        y = self.out_proj(y)

        x = self.ln1(x + y)
        x = self.ln2(x + self.ffn(x))
        return x


def _sinusoidal_position_embedding(length: int, d_model: int) -> Tensor:
    """标准正弦位置编码，形状为 ``(length, d_model)``。"""
    pe = torch.zeros(length, d_model)
    position = torch.arange(length).unsqueeze(1).float()
    div_term = 1.0 / torch.pow(
        10000.0, torch.arange(0, d_model, 2).float() / d_model
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class ResidualBlock(nn.Module):
    """一个带 flow-time 条件的门控 Transformer block。

    Args:
        channels: token 序列的特征维 ``C``。
        time_embedding_dim: 时间 embedding 的维度；通过一个线性层映射到
            ``channels`` 后再广播相加。
        nheads: 注意力头数。
        length: token 序列总长度（本项目为 ``3 * L_slot``）。
        ffn_dim: Transformer FFN 维度。
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
        self.time_layer = _SdpaAttentionFFN(
            channels=channels, nheads=nheads, ffn_dim=ffn_dim
        )
        self.register_buffer(
            "pe",
            _sinusoidal_position_embedding(length, channels),
            persistent=False,
        )

    def forward(
        self, x: Tensor, time_emb: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """前向传播。

        Args:
            x: ``(B, C, L)`` 输入序列（C = ``channels``）。
            time_emb: ``(B, time_embedding_dim)`` 的 flow-time embedding。
            mask: ``(L, L)`` 的 **bool** 注意力 mask，``True`` 为允许注意力。

        Returns:
            ``(x_out, skip)``，两者形状均为 ``(B, C, L)``。
        """
        B, C, L = x.shape
        base_shape = x.shape

        # 将时间 embedding 广播加到每个时空位置。
        t_emb = self.time_projection(time_emb).unsqueeze(-1)  # (B, C, 1)
        y = x + t_emb

        # Transformer self-attn + FFN，batch_first 布局、bool mask。
        y_seq = y.permute(0, 2, 1)  # (B, L, C)
        y_seq = self.time_layer(y_seq, attn_mask=mask)
        y = y_seq.permute(0, 2, 1).reshape(B, C, L)

        # 带位置偏置的门控 mid-projection。
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

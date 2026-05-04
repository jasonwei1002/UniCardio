""":class:`UniCardioBackbone` 使用的残差 Transformer block。

方向 A 改造后版本：标准 pre-norm Transformer block（self-attn + GELU FFN），
不再含 WaveNet 风格的 sigmoid×tanh 门控分支。位置编码由 backbone 在
residual stack 之前一次性注入，本块不再持有 PE buffer。

为保留 backbone 的 skip-sum 聚合（``UniCardioBackbone.forward`` 把每层
``skip`` 求和后做 ``/ sqrt(n_layers)`` 归一化），每层在残差流上额外取一份
``LN(h)`` 作为 skip 输出。
"""

from __future__ import annotations

import math

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class _PreNormBlock(nn.Module):
    """``(B, L, C)`` 上的 pre-norm self-attn + GELU FFN block。

    采用 LLaMA / GPT-NeoX 等现代 Transformer 的标准 pre-norm 结构：
    ``x = x + sdpa(LN(x))``，``x = x + ffn(LN(x))``。
    """

    def __init__(
        self,
        channels: int,
        nheads: int,
        *,
        ffn_dim: int,
    ) -> None:
        super().__init__()
        if channels % nheads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by nheads ({nheads})"
            )
        self.nheads = nheads
        self.head_dim = channels // nheads

        self.ln_attn = nn.LayerNorm(channels)
        self.ln_ffn = nn.LayerNorm(channels)
        self.qkv_proj = nn.Linear(channels, 3 * channels)
        self.out_proj = nn.Linear(channels, channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, channels),
        )

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        """``(B, L, C)`` 输入输出；``attn_mask`` 形状 ``(L, L)`` bool。"""
        B, L, C = x.shape

        qkv = self.qkv_proj(self.ln_attn(x))
        q, k, v = qkv.chunk(3, dim=-1)
        # (B, L, C) -> (B, H, L, D)
        q = q.view(B, L, self.nheads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.nheads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.nheads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=False
        )
        attn = attn.transpose(1, 2).contiguous().view(B, L, C)
        x = x + self.out_proj(attn)
        x = x + self.ffn(self.ln_ffn(x))
        return x


class ResidualBlock(nn.Module):
    """一个带 flow-time 条件的 pre-norm Transformer block。

    Args:
        channels: token 序列的特征维 ``C``。
        time_embedding_dim: 时间 embedding 的维度；通过一个线性层映射到
            ``channels`` 后再广播相加。
        nheads: 注意力头数。
        ffn_dim: Transformer FFN 中间维度（建议 4 × ``channels``）。
    """

    def __init__(
        self,
        channels: int,
        time_embedding_dim: int,
        nheads: int,
        *,
        ffn_dim: int,
    ) -> None:
        super().__init__()
        self.time_projection = nn.Linear(time_embedding_dim, channels)
        self.block = _PreNormBlock(
            channels=channels, nheads=nheads, ffn_dim=ffn_dim
        )
        self.ln_skip = nn.LayerNorm(channels)

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
        t_emb = self.time_projection(time_emb).unsqueeze(-1)  # (B, C, 1)
        y_seq = (x + t_emb).permute(0, 2, 1)                  # (B, L, C)
        y_seq = self.block(y_seq, attn_mask=mask)
        skip = self.ln_skip(y_seq).permute(0, 2, 1)           # (B, C, L)
        y = y_seq.permute(0, 2, 1)                            # (B, C, L)

        # sqrt(2) 归一化补偿 residual sum 的方差膨胀（CSDI 主干的历史约定）。
        return (x + y) / math.sqrt(2.0), skip

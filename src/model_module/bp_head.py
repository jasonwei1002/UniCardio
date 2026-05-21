"""Path A refinement model — PatchTSMixer waveform encoder + demographics MLP.

Mirrors MD-ViSCo's refinement model (IEEE JBHI 2026, Sec III.D):

    Source x^(i): (B, in_channels=2, L)
        |
        v  PatchTSMixer encoder (patch + token-mix + channel-mix × depth)
        |
    wave_emb: (B, dim)
                     +
    Demographics: (B, demo_in=6) = [age_z, gender, height_z, weight_z, bmi_z, mask]
        |
        v  Demographic encoder (Linear-GELU-Linear)
        |
    demo_emb: (B, demo_hidden)
                     concat
                     |
                     v  Fusion MLP (Linear-GELU-Linear)
                     |
                 (B, 2) = (SBP_mmHg, DBP_mmHg)

Key differences vs MD-ViSCo:
- We do NOT use DistilBERT for demographics — PulseDB demographics are 5
  numeric features with a fixed schema, so a plain Linear-MLP is the right
  tool. DistilBERT in MD-ViSCo is for cross-dataset schema agnosticism,
  which we do not need.
- We do NOT include the WCL (weighted contrastive loss) pretraining. The
  ablation in the paper shows architecture > WCL in contribution; WCL is a
  follow-up option if BPHead v2 alone falls short of AAMI.
- ``mask`` (6th demographic dim) is 1.0 when anthropometric values
  (height/weight/bmi) are present in PulseDB and 0.0 otherwise. ~48% of
  PulseDB samples (MIMIC-III subset) lack these and are imputed to z=0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class BPHeadConfig:
    """Immutable config for :class:`BPHead`."""

    in_channels: int = 2          # ECG + PPG
    slot_length: int = 1250       # 10 s @ 125 Hz
    patch_len: int = 50           # 50 samples = 0.4 s per patch; 25 patches
    dim: int = 128                # token / channel dim
    depth: int = 6                # number of PatchTSMixer blocks
    mlp_ratio: int = 2            # hidden expansion factor in MLPs
    demo_in: int = 6              # 5 features + 1 missingness mask
    demo_hidden: int = 64
    fusion_hidden: int = 128

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any] | "BPHeadConfig") -> "BPHeadConfig":
        if isinstance(cfg, cls):
            return cfg
        defaults = cls()
        return cls(
            in_channels=int(cfg.get("in_channels", defaults.in_channels)),
            slot_length=int(cfg.get("slot_length", defaults.slot_length)),
            patch_len=int(cfg.get("patch_len", defaults.patch_len)),
            dim=int(cfg.get("dim", defaults.dim)),
            depth=int(cfg.get("depth", defaults.depth)),
            mlp_ratio=int(cfg.get("mlp_ratio", defaults.mlp_ratio)),
            demo_in=int(cfg.get("demo_in", defaults.demo_in)),
            demo_hidden=int(cfg.get("demo_hidden", defaults.demo_hidden)),
            fusion_hidden=int(cfg.get("fusion_hidden", defaults.fusion_hidden)),
        )

    @property
    def n_patches(self) -> int:
        if self.slot_length % self.patch_len != 0:
            raise ValueError(
                f"slot_length ({self.slot_length}) must be divisible by "
                f"patch_len ({self.patch_len})."
            )
        return self.slot_length // self.patch_len


class _PatchTSMixerBlock(nn.Module):
    """Single PatchTSMixer block: token-mixing MLP + channel-mixing MLP.

    Mirrors Ekambaram et al. 2023 ("PatchTSMixer", KDD). Token mixing
    operates along the patch axis (cross-time interaction); channel
    mixing operates along the feature axis (cross-feature interaction).
    Both are MLPs with residual + LayerNorm.
    """

    def __init__(self, n_patches: int, dim: int, mlp_ratio: int) -> None:
        super().__init__()
        token_hidden = max(n_patches * mlp_ratio, 4)
        channel_hidden = dim * mlp_ratio
        self.norm_token = nn.LayerNorm(dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(n_patches, token_hidden),
            nn.GELU(),
            nn.Linear(token_hidden, n_patches),
        )
        self.norm_channel = nn.LayerNorm(dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, channel_hidden),
            nn.GELU(),
            nn.Linear(channel_hidden, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, n_patches, dim)
        h = self.norm_token(x)
        h = h.transpose(1, 2)        # (B, dim, n_patches)
        h = self.token_mlp(h)
        h = h.transpose(1, 2)        # (B, n_patches, dim)
        x = x + h
        h = self.norm_channel(x)
        h = self.channel_mlp(h)
        return x + h


class BPHead(nn.Module):
    """(ECG+PPG, demographics) -> (SBP, DBP) mmHg regressor."""

    def __init__(self, config: BPHeadConfig | Mapping[str, Any]) -> None:
        super().__init__()
        self.config = BPHeadConfig.from_mapping(config)
        cfg = self.config
        n_patches = cfg.n_patches

        # Patch embedding: each patch is (in_channels * patch_len) flattened,
        # then projected to `dim`. + learnable positional embedding.
        self.patch_proj = nn.Linear(cfg.in_channels * cfg.patch_len, cfg.dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_patches, cfg.dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        self.blocks = nn.ModuleList(
            [
                _PatchTSMixerBlock(n_patches, cfg.dim, cfg.mlp_ratio)
                for _ in range(cfg.depth)
            ]
        )
        self.final_norm = nn.LayerNorm(cfg.dim)

        # Demographic encoder. Operates on a 6-dim vector:
        #   [age_z, gender, height_z, weight_z, bmi_z, has_anthropometrics]
        # All z-scored (except gender ∈ {0, 1}) at the dataset boundary.
        self.demo_encoder = nn.Sequential(
            nn.Linear(cfg.demo_in, cfg.demo_hidden),
            nn.GELU(),
            nn.Linear(cfg.demo_hidden, cfg.demo_hidden),
        )

        # Fusion + regression head.
        self.head = nn.Sequential(
            nn.Linear(cfg.dim + cfg.demo_hidden, cfg.fusion_hidden),
            nn.GELU(),
            nn.Linear(cfg.fusion_hidden, 2),
        )

    def _patchify(self, ecg_ppg: Tensor) -> Tensor:
        """``(B, C, L)`` -> ``(B, n_patches, C * patch_len)``."""
        B, C, L = ecg_ppg.shape
        cfg = self.config
        if L != cfg.slot_length or C != cfg.in_channels:
            raise ValueError(
                f"BPHead expects (B, {cfg.in_channels}, {cfg.slot_length}); "
                f"got (B, {C}, {L})."
            )
        # unfold returns (B, C, n_patches, patch_len)
        patches = ecg_ppg.unfold(-1, cfg.patch_len, cfg.patch_len)
        # -> (B, n_patches, C, patch_len) -> flatten last two dims
        patches = patches.permute(0, 2, 1, 3).contiguous()
        return patches.reshape(B, cfg.n_patches, cfg.in_channels * cfg.patch_len)

    def forward(
        self,
        ecg_ppg: Tensor,
        demographics: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict ``(SBP, DBP)`` in mmHg.

        Args:
            ecg_ppg: ``(B, in_channels, slot_length)`` raw ECG + PPG.
            demographics: ``(B, demo_in)`` z-scored demographic vector with
                missingness mask in the last column (see
                :mod:`src.data_module.cardiac_dataset`). Pass ``None`` only
                for waveform-only inference paths (e.g. unit tests, legacy
                loaders without CSV) — in that case the demographic branch
                is bypassed (``demo_emb = 0``), which is **NOT** the same
                as feeding a real per-row "anthropometrics missing" vector
                ``[age_z, gender, 0, 0, 0, 0]`` (which still produces a
                non-zero demo embedding through the encoder).

        Returns:
            ``(B, 2)`` with column 0 = SBP, column 1 = DBP, both in mmHg.
        """
        h = self._patchify(ecg_ppg)
        h = self.patch_proj(h) + self.pos_emb
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)
        wave_emb = h.mean(dim=1)            # global average pool over patches

        if demographics is None:
            demo_emb = wave_emb.new_zeros(wave_emb.shape[0], self.config.demo_hidden)
        else:
            demo_emb = self.demo_encoder(demographics.to(wave_emb.dtype))

        return self.head(torch.cat([wave_emb, demo_emb], dim=-1))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_bp_head(config: Mapping[str, Any] | BPHeadConfig) -> BPHead:
    """Factory helper mirroring other model-module entry points."""
    return BPHead(config)

"""Weighted Contrastive Loss (WCL) — faithful port of MD-ViSCo's multi-WCL.

MD-ViSCo (IEEE JBHI 2026, §6.2.2) trains its BP refinement model with
``L_ref = L_MAE + L_WCL``, where ``L_WCL`` is a sum of six weighted
contrastive terms. WCL is a metric-learning regularizer: it pulls together
embeddings of samples whose *clinical label* (SBP/DBP/age/gender) is similar
and pushes apart dissimilar ones, structuring the embedding space so that the
downstream scalar BP regression generalizes with lower variance.

This module ports the exact math from the upstream
``src/criterions/weighted_contrastive_loss.py::_compute_loss`` and the
six-term composition from ``conf/criterion/multi_wcl_loss.yaml`` (see
``plan/bp_head_wcl_unify/notes.md`` for the extracted spec). It is a pure,
dependency-free criterion — no Hydra, no batch-dict plumbing — so the BP-head
trainer can call it with explicit ``embeddings`` / ``weights`` tensor dicts.

Single term, given embeddings ``E[B, d]`` and a scalar weight ``w[B]``::

    W_ij     = exp(-|w_i - w_j| / temperature_weight)        # [B, B]
    W        = where(W >= threshold, W, 0)                    # threshold filter
    W_norm_i = sum_j W_ij                                     # [B]
    S_ij     = (E @ E^T)_ij / temperature_embeddings          # [B, B]
    logp     = log_softmax(S, dim=-1)
    loss_i   = -sum_j (W_ij * logp_ij) / (W_norm_i + 1e-8)    # [B]
    loss     = scale_factor * mean_i(loss_i)                  # scalar

The upstream config's ``temperature`` (main) field is stored but unused by
``_compute_loss`` — only ``temperature_weight`` and ``temperature_embeddings``
enter the math. WCL weights are **raw** (mmHg SBP/DBP, years age, 0/1 gender),
*not* normalized labels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

_EPS = 1e-8


@dataclass(frozen=True)
class WCLTerm:
    """One weighted-contrastive term (MD-ViSCo ``multi_wcl_loss.yaml`` entry)."""

    name: str
    embedding_key: str          # which embedding to contrast (e.g. "ecg_embeddings")
    weight_key: str             # which raw label weights the similarity (e.g. "y_sbp_raw")
    temperature_embeddings: float = 1.0
    temperature_weight: float = 1.0
    threshold: float = 0.0
    scale_factor: float = 1.0


# MD-ViSCo's six-term multi-WCL (conf/criterion/multi_wcl_loss.yaml).
# 4 waveform terms (ECG/PPG × SBP/DBP, scale 1e-3) + 2 patient-info terms
# (gender threshold 1.0 → same-gender pairs only, age threshold 0.0235).
DEFAULT_WCL_TERMS: tuple[WCLTerm, ...] = (
    WCLTerm("ecg_sbp", "ecg_embeddings", "y_sbp_raw", 1.0, 1.0, 0.0, 1e-3),
    WCLTerm("ppg_sbp", "ppg_embeddings", "y_sbp_raw", 1.0, 1.0, 0.0, 1e-3),
    WCLTerm("ecg_dbp", "ecg_embeddings", "y_dbp_raw", 1.0, 1.0, 0.0, 1e-3),
    WCLTerm("ppg_dbp", "ppg_embeddings", "y_dbp_raw", 1.0, 1.0, 0.0, 1e-3),
    WCLTerm("text_gender_wcl", "text_embeddings", "gender_raw", 4.0, 1.0, 1.0, 1e-2),
    WCLTerm("text_age_wcl", "text_embeddings", "age_raw", 4.0, 4.0, 0.0235, 1e-2),
)


def weighted_contrastive_loss(
    embeddings: Tensor,
    weights: Tensor,
    *,
    temperature_embeddings: float = 1.0,
    temperature_weight: float = 1.0,
    threshold: float = 0.0,
    scale_factor: float = 1.0,
) -> Tensor:
    """Single WCL term, mean-reduced scalar (MD-ViSCo ``_compute_loss``).

    Args:
        embeddings: ``(B, d)`` feature embeddings (NOT L2-normalized — faithful
            to upstream, which uses the raw dot product).
        weights: ``(B,)`` or ``(B, 1)`` raw scalar labels driving the similarity.
        temperature_embeddings: divides the embedding dot-product similarity.
        temperature_weight: divides the ``|w_i - w_j|`` weight similarity.
        threshold: weight-similarity entries ``< threshold`` are zeroed.
        scale_factor: multiplies the final loss.

    Returns:
        Scalar tensor (mean over the batch).

    Raises:
        ValueError: if ``embeddings`` is not 2-D or batch sizes mismatch.
    """
    if embeddings.dim() != 2:
        raise ValueError(f"embeddings must be 2-D (B, d); got {tuple(embeddings.shape)}")
    if weights.dim() == 1:
        weights = weights.unsqueeze(-1)             # [B] -> [B, 1]
    if weights.dim() != 2 or weights.shape[-1] != 1:
        raise ValueError(f"weights must be (B,) or (B, 1); got {tuple(weights.shape)}")
    if embeddings.size(0) != weights.size(0):
        raise ValueError(
            f"batch size mismatch: embeddings {embeddings.size(0)} vs weights {weights.size(0)}"
        )

    # WCL math is sensitive to fp16/bf16 underflow in the B×B softmax; compute
    # in fp32 regardless of the surrounding autocast dtype.
    weights = weights.float()
    emb = embeddings.float()

    weight_similarity = torch.exp(-(torch.abs(weights - weights.T) / temperature_weight))
    weight_similarity = torch.where(
        weight_similarity >= threshold,
        weight_similarity,
        torch.zeros_like(weight_similarity),
    )
    weight_norm = weight_similarity.sum(dim=-1, keepdim=True)         # [B, 1]

    emb_similarity = (emb @ emb.T) / temperature_embeddings           # [B, B]
    log_prob = F.log_softmax(emb_similarity, dim=-1)                  # [B, B]

    loss = -torch.sum(weight_similarity * log_prob, dim=-1) / (
        weight_norm.squeeze(-1) + _EPS
    )                                                                  # [B]
    return (loss * scale_factor).mean()


def multi_wcl(
    embeddings: Mapping[str, Tensor],
    weights: Mapping[str, Tensor],
    terms: tuple[WCLTerm, ...] = DEFAULT_WCL_TERMS,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Sum the active WCL terms; skip any term whose embedding/weight is absent.

    A term is skipped (contributes 0) when its ``embedding_key`` is missing
    from ``embeddings`` (e.g. a vital not active for the task) or its
    ``weight_key`` is missing from ``weights`` (e.g. no demographics) — so the
    same criterion serves multimodal ``ecgppg2abp``, single-vital tasks, and
    waveform-only batches without reconfiguration.

    Args:
        embeddings: ``{key: (B, d)}`` (e.g. ``ecg_embeddings``, ``text_embeddings``).
        weights: ``{key: (B,) | (B, 1)}`` raw label tensors.
        terms: term configs to sum (default = MD-ViSCo's six-term multi-WCL).

    Returns:
        ``(total_loss, per_term)`` where ``total_loss`` is a scalar tensor (0.0
        tensor if no term is active) and ``per_term`` maps active term name ->
        its **detached scalar tensor** (kept on-device so callers can accumulate
        without a per-batch ``.item()`` sync; ``.item()`` once when logging).
    """
    total: Tensor | None = None
    per_term: dict[str, Tensor] = {}
    for t in terms:
        emb = embeddings.get(t.embedding_key)
        w = weights.get(t.weight_key)
        if emb is None or w is None:
            continue
        term_loss = weighted_contrastive_loss(
            emb, w,
            temperature_embeddings=t.temperature_embeddings,
            temperature_weight=t.temperature_weight,
            threshold=t.threshold,
            scale_factor=t.scale_factor,
        )
        per_term[t.name] = term_loss.detach()
        total = term_loss if total is None else total + term_loss

    if total is None:
        # No active term — return a 0 tensor on a sensible device for the caller
        # to add to L1 without a graph break.
        any_emb = next(iter(embeddings.values()), None)
        device = any_emb.device if any_emb is not None else torch.device("cpu")
        total = torch.zeros((), device=device)
    return total, per_term

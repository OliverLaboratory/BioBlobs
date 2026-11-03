"""Residue scores read out from a trained BioBlobs model (Section 3.3).

Two readouts rank residues by how strongly the model's blobs use them:
``max_k m_ki`` (maximum membership) and ``sum_k alpha_k m_ki`` (attention-weighted membership).
"""

from __future__ import annotations

import torch


def max_membership(memberships: torch.Tensor, blob_mask: torch.Tensor) -> torch.Tensor:
    """Maximum membership over valid blobs, ``max_k m_ki``.

    Args:
        memberships: Soft memberships, ``[B, K, N]``.
        blob_mask: Valid blobs, ``[B, K]``.

    Returns:
        Residue scores, ``[B, N]``.
    """
    masked = memberships.masked_fill(~blob_mask.unsqueeze(-1), float("-inf"))
    scores = masked.max(dim=1).values
    return torch.where(torch.isfinite(scores), scores, torch.zeros_like(scores))


def weighted_membership(memberships: torch.Tensor, importance: torch.Tensor) -> torch.Tensor:
    """Attention-weighted membership ``sum_k alpha_k m_ki``.

    Args:
        memberships: Soft memberships, ``[B, K, N]``.
        importance: MIL importance scores alpha, ``[B, K]`` (zero on invalid blobs).

    Returns:
        Residue scores, ``[B, N]``.
    """
    return (importance.unsqueeze(-1) * memberships).sum(dim=1)

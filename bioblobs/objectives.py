"""Training objectives (Section 3.4).

The total loss is ``L_task + lambda_H L_Hoyer`` (Eq. objective), plus, where enabled, the
RVQ commitment term and seed alignment for site-supervised VenusX models.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def hoyer_sparsity(
    memberships: torch.Tensor,
    candidate_mask: torch.Tensor,
    blob_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean Hoyer-Square sparseness of the valid blobs (Eq. hoyer).

    ``HS_k = (||m_k||_1 / ||m_k||_2)^2 / n_k`` with ``n_k`` the candidate-set size; it is 1
    for a uniform membership vector and ``1 / n_k`` for a one-hot one.

    Args:
        memberships: Soft memberships, ``[B, K, N]``.
        candidate_mask: Candidate sets, ``[B, K, N]``.
        blob_mask: Validly seeded blobs, ``[B, K]``.
        eps: Floor on the L2 norm.

    Returns:
        Scalar ``L_Hoyer``.
    """
    l1 = memberships.sum(dim=-1)
    l2 = memberships.pow(2).sum(dim=-1).clamp_min(eps).sqrt()
    n_candidates = candidate_mask.float().sum(dim=-1).clamp_min(1.0)
    hoyer = (l1 / l2).pow(2) / n_candidates
    valid = blob_mask.float()
    return (hoyer * valid).sum() / valid.sum().clamp_min(1.0)


def seed_alignment(
    seed_logits: torch.Tensor,
    site_mask: torch.Tensor,
    residue_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Seed alignment: ``-log`` of the seed-softmax mass on annotated site residues.

    Used only for the site-supervised VenusX models. Proteins without site residues are
    skipped.

    Args:
        seed_logits: Seed scores, ``[B, N]``.
        site_mask: Annotated site residues, ``[B, N]``.
        residue_mask: Valid residues, ``[B, N]``.
        eps: Floor inside the log.

    Returns:
        Scalar loss, averaged over proteins with at least one site residue.
    """
    probs = F.softmax(seed_logits.masked_fill(~residue_mask, float("-inf")), dim=-1)
    target = site_mask.float()
    annotated = target.sum(dim=-1) > 0
    if not annotated.any():
        return seed_logits.new_zeros(()).requires_grad_(True)
    mass = (probs * target).sum(dim=-1).clamp_min(eps)
    return -torch.log(mass[annotated]).mean()


def blob_alignment(
    memberships: torch.Tensor,
    site_mask: torch.Tensor,
    residue_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Blob alignment ``-log(sum_i t_i max_k m_ki / sum_i t_i)``, averaged over annotated proteins.

    Args:
        memberships: Soft memberships, ``[B, K, N]``.
        site_mask: Annotated site residues, ``[B, N]``.
        residue_mask: Valid residues, ``[B, N]``.
        eps: Floor inside the log.

    Returns:
        Scalar loss.
    """
    covered = memberships.max(dim=1).values * residue_mask.float()
    target = site_mask.float()
    annotated = target.sum(dim=-1) > 0
    if not annotated.any():
        return memberships.new_zeros(()).requires_grad_(True)
    recall = ((covered * target).sum(-1) / target.sum(-1).clamp_min(1.0)).clamp_min(eps)
    return -torch.log(recall[annotated]).mean()


def class_balanced_weights(counts: list[int] | np.ndarray, beta: float) -> torch.Tensor:
    """Effective-number class weights ``(1 - beta) / (1 - beta^n_c)``, normalized to mean 1.

    Args:
        counts: Training examples per class, in class-index order.
        beta: Effective-number hyperparameter (for example 0.9999).

    Returns:
        Float tensor of per-class weights.
    """
    n = np.maximum(np.asarray(counts, dtype=np.int64), 1)
    denom = np.maximum(-np.expm1(n * np.log(beta)), 1e-12)
    weights = (1.0 - beta) / denom
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)

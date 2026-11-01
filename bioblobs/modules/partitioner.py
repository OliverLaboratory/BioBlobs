"""Neural blob partitioner (Section 3.2): seed selection, blob expansion and membership."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Partition:
    """Output of a partitioner for a padded batch of B proteins with up to N residues.

    Attributes:
        memberships: Soft membership ``M`` of each residue in each blob, ``[B, K, N]``.
            Zero outside each blob's candidate set and at padded residues.
        blob_mask: True for blobs with a real seed, ``[B, K]``.
        residue_mask: True for real residues with a finite C-alpha, ``[B, N]``.
        candidate_mask: Candidate set ``C^(k)`` of each blob, ``[B, K, N]``.
        seed_logits: Seed score ``s_i`` of every residue, ``[B, N]``.
        seed_indices: Residue index of each seed, ``-1`` for empty slots, ``[B, K]``.
        blob_features: Explicit blob embeddings ``[B, K, D]`` for partitioners whose blobs are
            not membership-weighted means (TopK / SAGPool); None otherwise.
        regularizer: Weighted structural regularizer of the partitioner (for example
            DiffPool's entropy and link terms), a scalar or None.
        metrics: Detached diagnostics for logging.
    """

    memberships: torch.Tensor
    blob_mask: torch.Tensor
    residue_mask: torch.Tensor
    candidate_mask: torch.Tensor | None = None
    seed_logits: torch.Tensor | None = None
    seed_indices: torch.Tensor | None = None
    blob_features: torch.Tensor | None = None
    regularizer: torch.Tensor | None = None
    metrics: dict[str, torch.Tensor] = field(default_factory=dict)


def annealed_temperature(epoch: int, init: float, minimum: float, decay: float) -> float:
    """Temperature ``max(minimum, init * decay**epoch)`` shared by seeds and memberships."""
    return max(minimum, init * decay**epoch)


class NeuralBlobPartitioner(nn.Module):
    """Partitions each protein into K soft, spatially local blobs.

    Seed selection scores every residue with a one-hidden-layer MLP and picks K seeds
    sequentially without replacement (straight-through argmax of a temperature-scaled
    softmax). Blob expansion gives every residue within ``seed_radius`` of a seed a soft
    membership ``sigmoid((a_ki + proximity_bias * rho_ki) / tau_mem)``; the seed itself is
    clamped to membership 1.

    Args:
        input_dim: Residue embedding width D.
        num_blobs: Number of blobs K per protein.
        seed_hidden_dim: Hidden width d_h of the seed scoring network.
        attn_dim: Projection width d_a of the seed-residue attention score.
        seed_radius: Candidate radius r in angstroms.
        proximity_bias: Weight lambda_rho of the distance prior.
        seed_tau: ``(init, min, decay)`` of the seed temperature schedule.
        membership_tau: ``(init, min, decay)`` of the membership temperature schedule.
        eps: Numerical floor.
    """

    def __init__(
        self,
        input_dim: int,
        num_blobs: int = 8,
        seed_hidden_dim: int = 128,
        attn_dim: int = 128,
        seed_radius: float = 12.0,
        proximity_bias: float = 0.5,
        seed_tau: tuple[float, float, float] = (1.0, 0.25, 0.95),
        membership_tau: tuple[float, float, float] = (1.0, 0.25, 0.95),
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.num_blobs = num_blobs
        self.attn_dim = attn_dim
        self.seed_radius = seed_radius
        self.proximity_bias = proximity_bias
        self.seed_tau = tuple(seed_tau)
        self.membership_tau = tuple(membership_tau)
        self.eps = eps
        self.seed_scorer = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, seed_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(seed_hidden_dim, 1),
        )
        self.seed_query = nn.Linear(input_dim, attn_dim, bias=False)
        self.residue_key = nn.Linear(input_dim, attn_dim, bias=False)
        self.query_norm = nn.RMSNorm(attn_dim)
        self.key_norm = nn.RMSNorm(attn_dim)
        self.register_buffer("epoch", torch.tensor(0, dtype=torch.long))

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch that drives the temperature schedules."""
        self.epoch.fill_(max(int(epoch), 0))

    def temperatures(self) -> tuple[float, float]:
        """Current ``(tau_seed, tau_mem)``."""
        epoch = int(self.epoch.item())
        return annealed_temperature(epoch, *self.seed_tau), annealed_temperature(
            epoch, *self.membership_tau
        )

    def select_seeds(
        self, seed_logits: torch.Tensor, residue_mask: torch.Tensor, tau: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Seed selection: K straight-through top-1 picks without replacement.

        Args:
            seed_logits: Seed scores, ``[B, N]``.
            residue_mask: Valid residues, ``[B, N]``.
            tau: Seed temperature.

        Returns:
            One-hot seed assignments with softmax gradients ``[B, K, N]``, seed residue
            indices ``[B, K]`` (``-1`` when a protein has fewer than K residues) and the
            seed validity mask ``[B, K]``.
        """
        batch_size = seed_logits.shape[0]
        device = seed_logits.device
        seed_indices = torch.full((batch_size, self.num_blobs), -1, dtype=torch.long, device=device)
        seed_valid = torch.zeros((batch_size, self.num_blobs), dtype=torch.bool, device=device)
        available = residue_mask.clone()
        tau = max(float(tau), 1e-6)
        fill_value = torch.finfo(seed_logits.dtype).min
        assignments = []
        for slot in range(self.num_blobs):
            hard = torch.zeros_like(seed_logits)
            soft = torch.zeros_like(seed_logits)
            rows = available.any(dim=-1)
            if rows.any():
                logits = seed_logits.masked_fill(~available, fill_value)
                soft[rows] = F.softmax(logits[rows] / tau, dim=-1)
                picked = logits[rows].argmax(dim=-1)
                row_idx = rows.nonzero(as_tuple=True)[0]
                hard[row_idx, picked] = 1.0
                seed_indices[row_idx, slot] = picked
                seed_valid[row_idx, slot] = True
                available[row_idx, picked] = False
            assignments.append(hard + (soft - soft.detach()))
        return torch.stack(assignments, dim=1), seed_indices, seed_valid

    def forward(
        self, residue_embeddings: torch.Tensor, coords: torch.Tensor, residue_mask: torch.Tensor
    ) -> Partition:
        """Partition a padded batch.

        Args:
            residue_embeddings: Residue embeddings Z, ``[B, N, D]``.
            coords: C-alpha coordinates X, ``[B, N, 3]``; non-finite rows are allowed where
                ``residue_mask`` is False.
            residue_mask: Valid residues, ``[B, N]``.

        Returns:
            The partition, with memberships ``[B, K, N]``.
        """
        batch_size, num_res, _ = residue_embeddings.shape
        tau_seed, tau_mem = self.temperatures()
        seed_logits = self.seed_scorer(residue_embeddings).squeeze(-1)
        seed_onehot, seed_indices, seed_valid = self.select_seeds(
            seed_logits, residue_mask, tau_seed
        )
        safe_indices = seed_indices.clamp_min(0)
        batch_idx = torch.arange(batch_size, device=coords.device).unsqueeze(-1)
        seed_embeddings = torch.einsum("bkn,bnd->bkd", seed_onehot, residue_embeddings)
        # Empty seed slots carry index -1 (clamped to 0); reading their coordinates from the
        # masked tensor keeps a missing C-alpha at residue 0 from turning them into NaN.
        masked_coords = torch.where(residue_mask.unsqueeze(-1), coords, torch.zeros_like(coords))
        seed_coords = masked_coords[batch_idx, safe_indices] * seed_valid.unsqueeze(-1)

        queries = self.query_norm(self.seed_query(seed_embeddings))
        keys = self.key_norm(self.residue_key(residue_embeddings))
        attention = torch.einsum("bkh,bnh->bkn", queries, keys) / math.sqrt(float(self.attn_dim))
        distances = torch.cdist(seed_coords, masked_coords, compute_mode="use_mm_for_euclid_dist")
        seed_onehot_mask = F.one_hot(safe_indices, num_classes=num_res).to(
            torch.bool
        ) & seed_valid.unsqueeze(-1)
        candidate_mask = (
            residue_mask.unsqueeze(1) & seed_valid.unsqueeze(-1) & (distances <= self.seed_radius)
        ) | seed_onehot_mask
        proximity = (1.0 - distances / max(float(self.seed_radius), self.eps)).clamp_min(0.0)
        logits = attention + self.proximity_bias * proximity
        memberships = torch.sigmoid(logits / max(tau_mem, 1e-6)) * candidate_mask.float()
        memberships = torch.where(seed_onehot_mask, torch.ones_like(memberships), memberships)
        memberships = memberships * residue_mask.unsqueeze(1).float()

        return Partition(
            memberships=memberships,
            blob_mask=seed_valid,
            residue_mask=residue_mask,
            candidate_mask=candidate_mask,
            seed_logits=seed_logits,
            seed_indices=seed_indices,
            metrics=_blob_metrics(memberships, candidate_mask, seed_valid, residue_mask),
        )


@torch.no_grad()
def _blob_metrics(
    memberships: torch.Tensor,
    candidate_mask: torch.Tensor,
    seed_valid: torch.Tensor,
    residue_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Blob size and coverage diagnostics (logged, never optimized)."""
    valid = seed_valid.float()
    n_valid = valid.sum().clamp_min(1.0)
    hard_size = ((memberships >= 0.5) & residue_mask.unsqueeze(1)).float().sum(-1)
    covered = (candidate_mask.any(dim=1) & residue_mask).float().sum(-1)
    coverage = covered / residue_mask.float().sum(-1).clamp_min(1.0)
    return {
        "blob_avg_mass": memberships.sum(-1)[seed_valid].mean()
        if seed_valid.any()
        else memberships.new_zeros(()),
        "blob_avg_size": (hard_size * valid).sum() / n_valid,
        "residue_coverage": coverage.mean(),
    }

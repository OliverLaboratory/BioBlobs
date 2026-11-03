"""Multiple instance learning predictor (Section 3.3).

Blobs are the instances of a protein bag. Each blob embedding ``b_k`` (membership-weighted
mean of residue embeddings) passes through a feature transform ``u_k = ReLU(W_phi b_k + c_phi)``
and an attention gate ``alpha_k = softmax_k(w_a^T b_k + c_a)`` over valid blobs. The bag
representation ``sum_k alpha_k u_k`` goes to an MLP prediction head.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def blob_embeddings(
    memberships: torch.Tensor, residue_embeddings: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Membership-weighted mean of residue embeddings (Eq. blob_embedding).

    Args:
        memberships: Soft memberships, ``[B, K, N]``.
        residue_embeddings: Residue embeddings, ``[B, N, D]``.
        eps: Floor on the blob mass; blobs with no mass get a zero embedding.

    Returns:
        Blob embeddings, ``[B, K, D]``.
    """
    mass = memberships.sum(dim=-1, keepdim=True)
    pooled = torch.einsum("bkn,bnd->bkd", memberships, residue_embeddings)
    pooled = pooled / mass.clamp_min(eps)
    return torch.where(mass > eps, pooled, torch.zeros_like(pooled))


class PredictionHead(nn.Module):
    """MLP prediction head, optionally preceded by ``Linear + LayerNorm`` to ``proj_dim``.

    Hidden widths are ``multiplier * width`` for each entry of ``hidden_multipliers``, where
    ``width`` is ``proj_dim`` if set and ``input_dim`` otherwise.

    Args:
        input_dim: Input width.
        num_classes: Number of output logits.
        hidden_multipliers: Hidden-layer width multipliers.
        dropout: Dropout after each hidden layer.
        proj_dim: Optional projection width.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_multipliers: Sequence[int] = (4, 2),
        dropout: float = 0.1,
        proj_dim: int | None = None,
    ) -> None:
        super().__init__()
        if proj_dim is None:
            self.projection: nn.Module = nn.Identity()
            width = input_dim
        else:
            self.projection = nn.Sequential(nn.Linear(input_dim, proj_dim), nn.LayerNorm(proj_dim))
            width = proj_dim
        layers: list[nn.Module] = []
        current = width
        for multiplier in hidden_multipliers:
            hidden = int(width * multiplier)
            layers += [nn.Linear(current, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            current = hidden
        layers.append(nn.Linear(current, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map ``[..., input_dim]`` features to ``[..., num_classes]`` logits."""
        return self.mlp(self.projection(features))


class BlobSelfAttention(nn.Module):
    """Optional single-head self-attention across the blobs of a protein.

    Applied to the transformed blob features before aggregation when
    ``use_blob_interaction`` is set; output is ``LayerNorm(u + W_o attn(u))``.
    """

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.scale = dim**-0.5
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Attend across blobs.

        Args:
            x: Blob features, ``[B, K, D]``.
            mask: Valid blobs, ``[B, K]``.

        Returns:
            Updated features ``[B, K, D]`` and blob-blob attention ``[B, K, K]``.
        """
        scores = torch.bmm(self.q(x), self.k(x).transpose(1, 2)) * self.scale
        scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        attn = F.softmax(scores, dim=-1) * mask.unsqueeze(-1).float()
        context = torch.bmm(self.dropout(attn), self.v(x))
        return self.norm(x + self.proj(context)), attn


class MILPredictor(nn.Module):
    """Attention-based MIL predictor over blob embeddings.

    Blob embeddings are first projected to a shared width ``mil_dim`` (Linear + LayerNorm)
    so the head has the same capacity for every encoder.

    Args:
        input_dim: Blob embedding width D.
        num_classes: Number of output logits.
        mil_dim: Shared width of the feature transform, attention gate and head input.
        dropout: Dropout in the feature transform, blob self-attention and head.
        hidden_multipliers: Hidden-layer multipliers of the prediction head.
        use_blob_interaction: Apply blob self-attention after the feature transform.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        mil_dim: int = 512,
        dropout: float = 0.1,
        hidden_multipliers: Sequence[int] = (4, 2),
        use_blob_interaction: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_projection = nn.Sequential(nn.Linear(input_dim, mil_dim), nn.LayerNorm(mil_dim))
        self.feature_transform = nn.Linear(mil_dim, mil_dim)
        self.blob_interaction = (
            BlobSelfAttention(mil_dim, dropout=dropout) if use_blob_interaction else None
        )
        self.attention_gate = nn.Linear(mil_dim, 1)
        self.prediction_head = PredictionHead(mil_dim, num_classes, hidden_multipliers, dropout)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.feature_transform.weight)
        nn.init.zeros_(self.feature_transform.bias)
        nn.init.xavier_uniform_(self.attention_gate.weight)
        nn.init.zeros_(self.attention_gate.bias)

    def forward(
        self, blobs: torch.Tensor, blob_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict from a bag of blobs.

        Args:
            blobs: Blob embeddings, ``[B, K, D]``.
            blob_mask: Valid blobs, ``[B, K]``.

        Returns:
            Logits ``[B, C]`` and importance scores alpha ``[B, K]`` (zero on invalid blobs,
            summing to one over valid blobs).
        """
        b = self.input_projection(blobs)
        u = self.dropout(F.relu(self.feature_transform(b)))
        if self.blob_interaction is not None:
            u, _ = self.blob_interaction(u, blob_mask)
        gate = self.attention_gate(b).squeeze(-1).masked_fill(~blob_mask, float("-inf"))
        alpha = F.softmax(gate, dim=-1) * blob_mask.float()
        alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        bag = torch.einsum("bk,bkd->bd", alpha, u)
        return self.prediction_head(bag), alpha

    def blob_logits(self, blobs: torch.Tensor, blob_mask: torch.Tensor) -> torch.Tensor:
        """Per-blob logits: the prediction head on each transformed blob, ``[B, K, C]``."""
        b = self.input_projection(blobs)
        u = self.dropout(F.relu(self.feature_transform(b)))
        if self.blob_interaction is not None:
            u, _ = self.blob_interaction(u, blob_mask)
        return self.prediction_head(u) * blob_mask.unsqueeze(-1).float()

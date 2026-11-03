"""Comparators for the neural blob partitioner.

Residue-level heads (no blobs): mean pooling and attention pooling, each followed by the
same MLP prediction head as BioBlobs. Learned-pooling partitioners that feed the MIL
predictor in place of the neural blob partitioner: DiffPool (soft clusters) and TopK / SAGPool
(selected residues as instances).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, SAGPooling, TopKPooling
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    scatter,
    subgraph,
    to_dense_adj,
    to_dense_batch,
)

from bioblobs.modules.mil_predictor import PredictionHead
from bioblobs.modules.partitioner import Partition


class MeanPooling(nn.Module):
    """Mean of residue embeddings followed by an MLP prediction head.

    Args:
        input_dim: Residue embedding width D.
        num_classes: Number of output logits.
        input_layernorm: Apply LayerNorm(D) to residue embeddings before pooling.
        masked: Average only over residues with a finite C-alpha (otherwise all residues).
        proj_dim: Optional ``Linear + LayerNorm`` projection before the MLP.
        hidden_multipliers: Hidden-layer multipliers of the head.
        dropout: Head dropout.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        input_layernorm: bool = False,
        masked: bool = False,
        proj_dim: int | None = None,
        hidden_multipliers: Sequence[int] = (4, 2),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.masked = masked
        self.input_norm = nn.LayerNorm(input_dim) if input_layernorm else nn.Identity()
        self.prediction_head = PredictionHead(
            input_dim, num_classes, hidden_multipliers, dropout, proj_dim
        )

    def forward(
        self, features: torch.Tensor, batch: torch.Tensor, residue_mask: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        """Pool flat residue features ``[N_total, D]`` per protein; returns ``[B, C]`` logits."""
        h = self.input_norm(features)
        weight = residue_mask.float() if self.masked else torch.ones_like(batch, dtype=h.dtype)
        total = scatter(h * weight.unsqueeze(-1), batch, dim=0, reduce="sum")
        count = scatter(weight, batch, dim=0, reduce="sum").clamp_min(1.0)
        return self.prediction_head(total / count.unsqueeze(-1)), None


class AttentionPooling(nn.Module):
    """Attention pooling ``z = sum_i softmax_i(w^T h_i) h_i`` over valid residues + MLP head.

    A single linear scoring vector, without the gating branch of Ilse et al. (2018).

    Args:
        input_dim: Residue embedding width D.
        num_classes: Number of output logits.
        input_layernorm: Apply LayerNorm(D) before scoring and pooling.
        proj_dim: Optional ``Linear + LayerNorm`` projection before the MLP.
        hidden_multipliers: Hidden-layer multipliers of the head.
        dropout: Head dropout.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        input_layernorm: bool = True,
        proj_dim: int | None = None,
        hidden_multipliers: Sequence[int] = (4, 2),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim) if input_layernorm else nn.Identity()
        self.attention_gate = nn.Linear(input_dim, 1)
        self.prediction_head = PredictionHead(
            input_dim, num_classes, hidden_multipliers, dropout, proj_dim
        )

    def forward(
        self, features: torch.Tensor, batch: torch.Tensor, residue_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool flat residue features ``[N_total, D]``.

        Returns:
            Logits ``[B, C]`` and residue attention ``[B, N]`` (zero on invalid residues).
        """
        h, slots = to_dense_batch(self.input_norm(features), batch)
        valid = slots & to_dense_batch(residue_mask.bool(), batch)[0]
        logits = self.attention_gate(h).squeeze(-1).masked_fill(~valid, float("-inf"))
        attention = F.softmax(logits, dim=-1)
        pooled = (h * attention.unsqueeze(-1)).sum(dim=1)
        return self.prediction_head(pooled), attention


class DiffPool(nn.Module):
    """Single-stage DiffPool (Ying et al., 2018) as a partitioner.

    Two GraphSAGE layers on the residue graph give each valid residue a softmax assignment
    over ``min(max_clusters, n_residues)`` clusters. Clusters are the MIL instances, with
    embeddings equal to the assignment-weighted mean of residue embeddings.

    Args:
        input_dim: Residue embedding width D.
        max_clusters: Maximum number of clusters per protein.
        hidden_dim: GraphSAGE width.
        temperature: Softmax temperature of the assignments.
        entropy_weight: Weight of the assignment-entropy regularizer.
        link_weight: Weight of the link-prediction regularizer.
        eps: Numerical floor.
    """

    def __init__(
        self,
        input_dim: int,
        max_clusters: int = 32,
        hidden_dim: int = 128,
        temperature: float = 1.0,
        entropy_weight: float = 0.1,
        link_weight: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.max_clusters = max_clusters
        self.temperature = temperature
        self.entropy_weight = entropy_weight
        self.link_weight = link_weight
        self.eps = eps
        self.assign_conv1 = SAGEConv(input_dim, hidden_dim)
        self.assign_conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.assign_proj = nn.Linear(hidden_dim, max_clusters)

    def forward(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        residue_mask: torch.Tensor,
        num_res: int,
    ) -> Partition:
        """Assign the valid residues of each protein to clusters; memberships are ``[B, K, N]``."""
        num_graphs = int(batch.max().item()) + 1
        memberships = features.new_zeros((num_graphs, self.max_clusters, num_res))
        blobs = features.new_zeros((num_graphs, self.max_clusters, features.size(-1)))
        blob_mask = torch.zeros(
            (num_graphs, self.max_clusters), dtype=torch.bool, device=features.device
        )
        entropies, links = [], []
        for g in range(num_graphs):
            graph_nodes = (batch == g).nonzero(as_tuple=True)[0]
            local = torch.arange(graph_nodes.numel(), device=batch.device)
            keep = residue_mask[graph_nodes].bool()
            nodes, local = graph_nodes[keep], local[keep]
            if nodes.numel() == 0:
                blob_mask[g, 0] = True
                continue
            edges, _ = subgraph(nodes, edge_index, relabel_nodes=True, num_nodes=features.size(0))
            edges = remove_self_loops(edges)[0]
            mp_edges = add_self_loops(edges, num_nodes=nodes.numel())[0]
            hidden = F.relu(self.assign_conv1(features[nodes], mp_edges))
            hidden = F.relu(self.assign_conv2(hidden, mp_edges))
            n_clusters = min(self.max_clusters, nodes.numel())
            logits = self.assign_proj(hidden)[:, :n_clusters]
            s = torch.softmax(logits / max(self.temperature, self.eps), dim=-1)
            memberships[g, :n_clusters, local] = s.t()
            # Clusters keep a mean embedding even when their mass falls below eps.
            mass = s.sum(dim=0).clamp_min(self.eps).unsqueeze(-1)
            blobs[g, :n_clusters] = s.t() @ features[nodes] / mass
            blob_mask[g, :n_clusters] = True
            entropies.append(-(s * (s + self.eps).log()).sum(dim=-1).mean())
            adjacency = to_dense_adj(edges, max_num_nodes=nodes.numel())[0].to(s.dtype)
            adjacency = torch.maximum(adjacency, adjacency.t()).clamp_(0.0, 1.0)
            links.append(
                torch.norm(adjacency - s @ s.t(), p="fro") / float(max(nodes.numel() ** 2, 1))
            )
        zero = features.new_zeros(())
        entropy = torch.stack(entropies).mean() if entropies else zero
        link = torch.stack(links).mean() if links else zero
        return Partition(
            memberships=memberships,
            blob_mask=blob_mask,
            residue_mask=to_dense_batch(residue_mask.bool(), batch)[0],
            blob_features=blobs,
            regularizer=self.entropy_weight * entropy + self.link_weight * link,
            metrics={"diffpool_entropy": entropy.detach(), "diffpool_link": link.detach()},
        )


class SelectionPool(nn.Module):
    """TopK (Gao and Ji, 2019) or SAGPool (Lee et al., 2019) residue selection as a partitioner.

    Keeps the top ``ratio`` fraction of residues per protein; each kept residue, gated by its
    selection score, is one MIL instance.

    Args:
        input_dim: Residue embedding width D.
        method: ``"topk"`` (learned projection score) or ``"sagpool"`` (graph-convolution score).
        ratio: Fraction of residues kept.
    """

    def __init__(self, input_dim: int, method: str = "topk", ratio: float = 0.5) -> None:
        super().__init__()
        if method not in {"topk", "sagpool"}:
            raise ValueError(f"method must be 'topk' or 'sagpool', got {method!r}")
        if not 0.0 < ratio <= 1.0:
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")
        self.pool = (TopKPooling if method == "topk" else SAGPooling)(input_dim, ratio=ratio)

    def forward(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        residue_mask: torch.Tensor,
        num_res: int,
    ) -> Partition:
        """Select residues; returns a partition with explicit blob features ``[B, K, D]``."""
        kept, _, _, kept_batch, perm, score = self.pool(features, edge_index, batch=batch)
        num_graphs = int(batch.max().item()) + 1
        blobs, blob_mask = to_dense_batch(kept, kept_batch, batch_size=num_graphs)
        res_offsets = _offsets(batch, num_graphs)
        slot = torch.arange(kept_batch.numel(), device=batch.device)
        slot = slot - _offsets(kept_batch, num_graphs)[kept_batch]
        memberships = features.new_zeros((num_graphs, blobs.shape[1], num_res))
        memberships[kept_batch, slot, perm - res_offsets[kept_batch]] = score.to(features.dtype)
        return Partition(
            memberships=memberships,
            blob_mask=blob_mask,
            residue_mask=to_dense_batch(residue_mask.bool(), batch)[0],
            blob_features=blobs,
        )


def _offsets(index: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Start position of each graph in a sorted batch vector."""
    counts = torch.bincount(index, minlength=num_graphs)
    return torch.cumsum(counts, 0) - counts

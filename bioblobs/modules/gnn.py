"""Residue-level GNN comparators (GCN, GIN, GAT) on frozen protein-encoder embeddings.

The GNN refines residue embeddings over the residue contact graph; the refined embeddings
are mean-pooled and classified by the same MLP prediction head as the other baselines.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, GINConv


class ResidueGNN(nn.Module):
    """Stack of message-passing layers with LayerNorm, ReLU, dropout and residuals.

    Args:
        conv_type: ``"gcn"``, ``"gin"`` or ``"gat"``.
        input_dim: Width of the frozen residue embeddings.
        hidden_dim: Width of every layer and of the output.
        num_layers: Number of message-passing layers.
        dropout: Dropout after each layer.
        heads: GAT heads (``hidden_dim`` must be divisible by ``heads``).
        residual: Add a skip connection on hidden-to-hidden layers.
        layer_norm: Apply LayerNorm after each layer.
        add_self_loops: Self loops for GCN and GAT.
    """

    def __init__(
        self,
        conv_type: str = "gcn",
        input_dim: int = 640,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        heads: int = 4,
        residual: bool = True,
        layer_norm: bool = True,
        add_self_loops: bool = True,
    ) -> None:
        super().__init__()
        if conv_type not in {"gcn", "gin", "gat"}:
            raise ValueError(f"conv_type must be gcn, gin or gat, got {conv_type!r}")
        if conv_type == "gat" and hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim {hidden_dim} must be divisible by heads {heads}")
        self.output_dim = hidden_dim
        self.residual = residual
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer in range(num_layers):
            in_dim = input_dim if layer == 0 else hidden_dim
            self.convs.append(_conv(conv_type, in_dim, hidden_dim, heads, add_self_loops))
            self.norms.append(nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity())
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Refine flat residue features ``[N_total, D_in]`` to ``[N_total, hidden_dim]``."""
        x = features.float()
        for conv, norm in zip(self.convs, self.norms, strict=True):
            h = self.dropout(self.activation(norm(conv(x, edge_index))))
            x = h + x if self.residual and h.shape == x.shape else h
        return x


def _conv(conv_type: str, in_dim: int, out_dim: int, heads: int, self_loops: bool) -> nn.Module:
    if conv_type == "gcn":
        return GCNConv(in_dim, out_dim, add_self_loops=self_loops)
    if conv_type == "gat":
        return GATConv(
            in_dim, out_dim // heads, heads=heads, concat=True, add_self_loops=self_loops
        )
    mlp = nn.Sequential(
        nn.Linear(in_dim, out_dim), nn.ReLU(inplace=True), nn.Linear(out_dim, out_dim)
    )
    return GINConv(mlp, train_eps=True)

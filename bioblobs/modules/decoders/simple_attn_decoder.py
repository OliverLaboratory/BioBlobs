"""Simple attention-pool decoder.

Faithful port of ``GatedAttentionMLP`` from BioBlobs_VenusX
(``train_baseline_attn.py``): the minimal residue-level attention head, with
only ``D + 1`` pool params. Sits alongside ``AttentionPoolDecoder`` (the
fuller Ilse 2018 V/U/w sandwich) and ``LightAttentionDecoder`` (the Stärk
2021 Conv1d head).

Per protein, given residue features ``h_i ∈ R^D`` (i = 1..L)::

    h_i  = LayerNorm(h_i)                   # optional
    s_i  = w^T h_i                          # single learned scoring vector
    α_i  = softmax_i(s_i) over valid i      # masked over padded / invalid Cα
    z    = Σ_i α_i h_i                      # pooled protein vector ∈ R^D
    ŷ    = MLPDecoder(z)

This is the simplest learnable attention pool: one ``Linear(D → 1)``, no
``tanh``, no ``sigmoid`` gate, no auxiliary projection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from .mlp_decoder import MLPDecoder


class SimpleAttnDecoder(nn.Module):
    """Single-vector attention pooling + MLP classifier."""

    consumes_batch_data: bool = True

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        input_layernorm: bool = True,
        hidden_multipliers: list[int] | tuple[int, ...] = (4, 2),
        drop_rate: float = 0.1,
        proj_dim: int | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.proj_dim = proj_dim

        self.input_norm = (
            nn.LayerNorm(input_dim) if input_layernorm else nn.Identity()
        )
        self.attention_gate = nn.Linear(input_dim, 1)

        self.classifier = MLPDecoder(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_multipliers=list(hidden_multipliers),
            drop_rate=drop_rate,
            proj_dim=proj_dim,
        )

    def forward(self, batch_data) -> tuple[torch.Tensor, dict]:
        h = self.input_norm(batch_data.node_features)             # [N, D]

        h_dense, slot_mask = to_dense_batch(h, batch_data.batch)  # [B, L, D], [B, L]
        node_mask = getattr(batch_data, "mask", None)
        if node_mask is None:
            valid_mask = slot_mask
        else:
            valid_dense, _ = to_dense_batch(node_mask.float(), batch_data.batch)
            valid_mask = slot_mask & (valid_dense > 0.5)          # [B, L]

        attn_logits = self.attention_gate(h_dense).squeeze(-1)    # [B, L]
        attn_logits = attn_logits.masked_fill(~valid_mask, float("-inf"))
        attn = F.softmax(attn_logits, dim=-1)                     # [B, L]

        pooled = (h_dense * attn.unsqueeze(-1)).sum(dim=1)        # [B, D]

        logits = self.classifier(pooled)                          # [B, num_classes]

        extra = {
            "attention_weights_per_graph": [
                attn[b][valid_mask[b]].detach()
                for b in range(attn.size(0))
            ]
        }
        return logits, extra

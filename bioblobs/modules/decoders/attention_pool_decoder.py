"""Basic attention-pool decoder.

A minimal learned attention-pooling baseline over residue embeddings, in the
Ilse et al. (2018) "Attention-based Deep MIL" formulation but applied at the
residue level rather than the blob level.

Per protein, given residue features ``h_i ∈ R^D`` (i = 1..L)::

    u_i = tanh(V h_i)                       # additive scoring (Bahdanau)
    g_i = sigmoid(U h_i)                    # optional gate (Ilse 2018)
    s_i = w^T (u_i ⊙ g_i)                   # scalar score per residue
    α_i = softmax_i(s_i) over valid i       # masked over padded / invalid Cα
    z   = Σ_i α_i h_i                       # pooled protein vector ∈ R^D
    ŷ   = MLP(z)

This sits between mean-pool + MLP and the convolutional Light-Attention head:
no Conv1d residue mixing, no max-pool branch, just a single learned scoring
function over residues — the standard "attention pool" baseline.

Like ``MILDecoder`` and ``LightAttentionDecoder``, this decoder consumes the
full ``batch_data`` (it needs ``node_features``, ``batch``, and ``mask``)
rather than a pre-pooled ``graph_features`` tensor, so the framework pipeline
skips the explicit pooling step.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from .mlp_decoder import MLPDecoder


class AttentionPoolDecoder(nn.Module):
    """Basic attention pooling + MLP classifier."""

    consumes_batch_data: bool = True

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        attn_hidden_dim: int | None = None,
        gated: bool = True,
        input_layernorm: bool = True,
        hidden_multipliers: list[int] | tuple[int, ...] = (4, 2),
        drop_rate: float = 0.1,
        proj_dim: int | None = None,
    ):
        """
        Args:
            input_dim: Encoder output dim (residue feature width D).
            num_classes: Number of output classes.
            attn_hidden_dim: Width of the attention scoring projection
                ``V`` (and gate ``U`` when ``gated=True``). ``None`` defaults
                to ``input_dim`` (Ilse et al. default).
            gated: If True, multiply ``tanh(V h)`` by ``sigmoid(U h)`` before
                the final scoring linear (Ilse et al. 2018, "gated attention").
                If False, use plain additive attention (Bahdanau-style).
            input_layernorm: Apply LayerNorm(D) on residue embeddings before
                attention. Cheap (~2·D params); helps when fp16-cached PLM
                features have non-unit variance.
            hidden_multipliers: MLP head hidden widths as multipliers of the
                MLP input dim (which is D — the pooled vector lives in
                input_dim, unlike LightAttention's 2·D concat).
            drop_rate: Dropout in the MLP head.
            proj_dim: Optional projection from D → proj_dim → LayerNorm
                applied before the MLP head. Mirrors
                ``MILDecoder.input_proj`` so this baseline can match the
                BioBlobs MIL classifier capacity.
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.gated = gated
        self.proj_dim = proj_dim

        attn_hidden_dim = attn_hidden_dim or input_dim
        self.attn_hidden_dim = attn_hidden_dim

        self.input_norm = (
            nn.LayerNorm(input_dim) if input_layernorm else nn.Identity()
        )
        self.V = nn.Linear(input_dim, attn_hidden_dim)
        self.U = nn.Linear(input_dim, attn_hidden_dim) if gated else None
        self.w = nn.Linear(attn_hidden_dim, 1)

        self.classifier = MLPDecoder(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_multipliers=list(hidden_multipliers),
            drop_rate=drop_rate,
            proj_dim=proj_dim,
        )

    def forward(self, batch_data) -> tuple[torch.Tensor, dict]:
        """
        Args:
            batch_data: PyG Batch with:
                - node_features: [N, D] residue embeddings
                - batch:         [N] protein-id assignment
                - mask:          [N] per-residue valid mask (False = invalid Cα)
        Returns:
            logits: [B, num_classes]
            extra:  dict with `attention_weights_per_graph` — list of
                    [L_p] residue-level attention weights, one per protein.
        """
        h = self.input_norm(batch_data.node_features)            # [N, D]

        h_dense, slot_mask = to_dense_batch(h, batch_data.batch)  # [B, L, D], [B, L]
        node_mask = getattr(batch_data, "mask", None)
        if node_mask is None:
            valid_mask = slot_mask
        else:
            valid_dense, _ = to_dense_batch(node_mask.float(), batch_data.batch)
            valid_mask = slot_mask & (valid_dense > 0.5)          # [B, L]

        u = torch.tanh(self.V(h_dense))                           # [B, L, H]
        if self.U is not None:
            u = u * torch.sigmoid(self.U(h_dense))                # gated attention
        attn_logits = self.w(u).squeeze(-1)                       # [B, L]

        attn_logits = attn_logits.masked_fill(~valid_mask, float("-inf"))
        attn = F.softmax(attn_logits, dim=-1)                     # [B, L]

        # Attention-weighted sum across residues.
        pooled = (h_dense * attn.unsqueeze(-1)).sum(dim=1)        # [B, D]

        logits = self.classifier(pooled)                          # [B, num_classes]

        extra = {
            "attention_weights_per_graph": [
                attn[b][valid_mask[b]].detach()
                for b in range(attn.size(0))
            ]
        }
        return logits, extra

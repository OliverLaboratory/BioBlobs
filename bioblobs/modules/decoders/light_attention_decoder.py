"""Light Attention decoder.

Faithful port of the Light Attention head from Stärk et al., 2021
("Light Attention Predicts Protein Location from the Language of Life",
Bioinformatics Advances). Originally trained on top of frozen residue PLM
embeddings as a sequence-classification baseline; ported here so it can sit
in the BioBlobs framework alongside mean-pool + MLP and the BioBlobs MIL
head.

Architecture (per protein):
    (optional) LayerNorm(D)
      → two parallel Conv1d(D, D, kernel_size) over the residue axis
      → attention branch is masked-softmaxed across residues per channel
      → value branch is multiplied by those weights and summed across
        residues (giving a [B, D] attention-pooled vector) and also
        masked-max-pooled across residues (another [B, D] vector)
      → concat to [B, 2D] and feed to MLPDecoder.

Like ``MILDecoder``, this decoder consumes the full ``batch_data`` (it needs
``node_features``, ``batch``, and ``mask``) rather than a pre-pooled
``graph_features`` tensor, so the framework pipeline skips the explicit
pooling step and routes the batch object straight to ``forward``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from .mlp_decoder import MLPDecoder


class LightAttentionDecoder(nn.Module):
    """Light-attention pooling + MLP classifier."""

    # Pipeline marker: the framework should pass the full batch_data to this
    # decoder (and skip the explicit pooling step), as with MILDecoder.
    consumes_batch_data: bool = True

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        kernel_size: int = 9,
        input_layernorm: bool = True,
        hidden_multipliers: list[int] | tuple[int, ...] = (2, 1),
        drop_rate: float = 0.1,
        proj_dim: int | None = None,
    ):
        """
        Args:
            input_dim: Encoder output dim (residue feature width D).
            num_classes: Number of output classes.
            kernel_size: Conv1d kernel along the residue axis. Must be odd
                so that symmetric padding preserves length. Stärk et al.
                default is 9 (Tab. 2).
            input_layernorm: LayerNorm(D) on residue embeddings before the
                attention/value convs. Cheap (~2·D params); helps when
                fp16-cached PLM features have non-unit variance.
            hidden_multipliers: MLP head hidden widths as multipliers of the
                MLP input dim. Default (2, 1) keeps hidden layer widths
                identical to the mean-pool baseline (D → 4D → 2D → C);
                here the head input is 2·D so the projection is
                2D → 4D → 2D → C.
            drop_rate: Dropout in the MLP head.
            proj_dim: Optional projection from 2·D → proj_dim → LayerNorm
                applied before the MLP head. Set to mirror the
                ``MILDecoder.input_proj`` bottleneck for fair-comparison
                runs. ``None`` keeps the original head (no bottleneck).
        """
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be odd to preserve length under symmetric "
                f"padding, got {kernel_size}"
            )
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.proj_dim = proj_dim

        pad = kernel_size // 2
        self.input_norm = (
            nn.LayerNorm(input_dim) if input_layernorm else nn.Identity()
        )
        self.attn_conv = nn.Conv1d(input_dim, input_dim, kernel_size, padding=pad)
        self.value_conv = nn.Conv1d(input_dim, input_dim, kernel_size, padding=pad)

        self.classifier = MLPDecoder(
            input_dim=2 * input_dim,
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
                    [L_p] residue-level mean attention weights, one per
                    protein in the batch.
        """
        h = self.input_norm(batch_data.node_features)            # [N, D]

        # Densify to [B, L, D] and build a per-residue valid mask.
        x_dense, slot_mask = to_dense_batch(h, batch_data.batch)  # [B, L, D], [B, L]
        node_mask = getattr(batch_data, "mask", None)
        if node_mask is None:
            valid_mask = slot_mask
        else:
            valid_dense, _ = to_dense_batch(node_mask.float(), batch_data.batch)
            valid_mask = slot_mask & (valid_dense > 0.5)          # [B, L]

        # Conv1d expects [B, D, L].
        x_t = x_dense.transpose(1, 2)                             # [B, D, L]
        attn_logits = self.attn_conv(x_t)                         # [B, D, L]
        values = self.value_conv(x_t)                             # [B, D, L]

        mask_b1l = valid_mask.unsqueeze(1)                        # [B, 1, L]
        attn_logits = attn_logits.masked_fill(~mask_b1l, float("-inf"))
        attn = F.softmax(attn_logits, dim=-1)                     # [B, D, L]

        # Attention-weighted sum across residues, per channel.
        attended = (values * attn).sum(dim=-1)                    # [B, D]
        # Masked max-pool across residues, per channel.
        values_masked = values.masked_fill(~mask_b1l, float("-inf"))
        max_pooled = values_masked.amax(dim=-1)                   # [B, D]

        feat = torch.cat([attended, max_pooled], dim=-1)          # [B, 2D]
        logits = self.classifier(feat)                            # [B, num_classes]

        # Per-residue saliency for downstream interpretability.
        attn_mean = attn.mean(dim=1)                              # [B, L]
        extra = {
            "attention_weights_per_graph": [
                attn_mean[b][valid_mask[b]].detach()
                for b in range(attn_mean.size(0))
            ]
        }
        return logits, extra

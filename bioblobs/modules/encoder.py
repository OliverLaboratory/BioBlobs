"""Protein encoder (Section 3.1): frozen ESM2 or SaProt residue embeddings.

Embeddings are computed once by ``preprocessing/prepare_plm_cache.py`` and read from an
on-disk cache, so the encoder module at training time only checks and returns them.
"""

from __future__ import annotations

import torch
import torch.nn as nn

KNOWN_DIMS = {
    "facebook/esm2_t30_150M_UR50D": 640,
    "facebook/esm2_t33_650M_UR50D": 1280,
    "westlake-repl/SaProt_650M_AF2": 1280,
}


class ProteinEncoder(nn.Module):
    """Frozen protein encoder whose residue embeddings are precomputed.

    Args:
        name: ``"esm2_static"`` or ``"saprot_static"``.
        model_name: Hugging Face model id used to build the cache.
        output_dim: Embedding width D (inferred from ``model_name`` when None).
    """

    def __init__(self, name: str, model_name: str, output_dim: int | None = None) -> None:
        super().__init__()
        self.name = name
        self.model_name = model_name
        if output_dim is None:
            if model_name not in KNOWN_DIMS:
                raise ValueError(f"Set encoders.output_dim for unknown model {model_name!r}")
            output_dim = KNOWN_DIMS[model_name]
        self.output_dim = int(output_dim)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """Return cached residue embeddings ``[N_total, D]`` as float32."""
        if node_features.shape[-1] != self.output_dim:
            raise ValueError(
                f"Cached embeddings have width {node_features.shape[-1]}, "
                f"expected {self.output_dim} for {self.model_name}"
            )
        return node_features.float()

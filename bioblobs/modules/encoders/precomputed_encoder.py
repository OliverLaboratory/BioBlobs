"""Pass-through encoder for precomputed node features."""

from __future__ import annotations

from typing import List

from bioblobs.datasets.esm2_cache import infer_esm2_output_dim

from .base_encoder import ProteinEncoder


class PrecomputedNodeFeatureEncoder(ProteinEncoder):
    """Encoder shim for cached residue embeddings."""

    def __init__(
        self,
        model_name: str = "facebook/esm2_t30_150M_UR50D",
        output_dim: int | None = None,
        repr_layer: int = -1,
        window_size: int = 1022,
        window_overlap: int = 128,
        max_batch_tokens: int = 8192,
        cache_dir: str = "esm2_static_cache",
        cache_dtype: str = "float16",
        precompute_missing: bool = True,
        overwrite_cache: bool = False,
        precompute_device: str = "cpu",
        foldseek_bin: str | None = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.output_dim = infer_esm2_output_dim(model_name, output_dim)
        self.repr_layer = repr_layer
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.max_batch_tokens = max_batch_tokens
        self.cache_dir = cache_dir
        self.cache_dtype = cache_dtype
        self.precompute_missing = precompute_missing
        self.overwrite_cache = overwrite_cache
        self.precompute_device = precompute_device
        self.foldseek_bin = foldseek_bin

    def forward(self, batch_data):
        self._validate_batch_data(batch_data)
        return batch_data

    def get_output_dim(self):
        return self.output_dim

    def get_required_features(self) -> List[str]:
        return ["node_features"]

    def get_num_parameters(self) -> int:
        return 0

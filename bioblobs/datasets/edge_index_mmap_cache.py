"""Memory-mapped per-protein ``edge_index`` cache.

Companion to :class:`MmapPlmCache`. The on-disk layout is two files under one
directory keyed by edge construction parameters (e.g. ``eps_8``, ``knn_30``):

    <root>/edges.bin    raw int64, shape ``[2, total_edges_across_all_proteins]``,
                        written contiguously by ``scripts/build_mmap_edge_cache.py``.
    <root>/meta.pt      dict with: names (list[str]), offsets (LongTensor[N+1]),
                        edge_method (str), edge_value (int|float),
                        edge_types (str), total_edges (int).

The featurizer falls back to recomputing ``edge_index`` per sample via
``torch_cluster`` if the cache is absent, so this is a pure performance
optimization.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class EdgeIndexMmapCache:
    """Read-only memory-mapped per-protein edge_index cache."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        meta_path = self.root / "meta.pt"
        bin_path = self.root / "edges.bin"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.pt not found in {self.root}")
        if not bin_path.exists():
            raise FileNotFoundError(f"edges.bin not found in {self.root}")

        meta = torch.load(meta_path, weights_only=False)
        self.names: list[str] = list(meta["names"])
        offsets = meta["offsets"]
        if isinstance(offsets, torch.Tensor):
            offsets = offsets.cpu().numpy()
        self.offsets: np.ndarray = np.asarray(offsets, dtype=np.int64)
        self.edge_method: str = str(meta["edge_method"])
        self.edge_value = meta["edge_value"]
        self.edge_types: str = str(meta.get("edge_types", f"{self.edge_method}_{self.edge_value}"))
        self.total_edges = int(meta["total_edges"])

        if len(self.offsets) != len(self.names) + 1:
            raise ValueError(
                f"offsets length {len(self.offsets)} != names length {len(self.names)} + 1"
            )

        self._name_to_idx: dict[str, int] = {n: i for i, n in enumerate(self.names)}

        self._mmap: np.memmap = np.memmap(
            bin_path,
            dtype=np.int64,
            mode="r",
            shape=(2, self.total_edges),
        )

    def get(self, name: str) -> torch.Tensor:
        """Return the ``[2, num_edges]`` long tensor for ``name``.

        Returns an empty ``[2, 0]`` tensor if the protein had no valid edges
        (e.g. all coordinates non-finite). Always returns a copy because the
        consumer (``torch_geometric.data.Batch.from_data_list``) needs a
        contiguous, writable tensor for index shifting.
        """
        idx = self._name_to_idx.get(name)
        if idx is None:
            raise KeyError(f"protein {name!r} not in edge cache (root: {self.root})")
        start = int(self.offsets[idx])
        end = int(self.offsets[idx + 1])
        if end <= start:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.from_numpy(self._mmap[:, start:end].copy())

    def __contains__(self, name: str) -> bool:
        return name in self._name_to_idx

    def __len__(self) -> int:
        return len(self.names)

    def __repr__(self) -> str:
        return (
            f"EdgeIndexMmapCache(root={self.root}, n_proteins={len(self.names)}, "
            f"total_edges={self.total_edges:,}, edge_types={self.edge_types})"
        )

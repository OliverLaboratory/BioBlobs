"""Memory-mapped cache for precomputed PLM (ESM2 / SaProt) residue embeddings.

The on-disk layout is two files under one directory:

    <root>/embeddings.bin   raw float16 (or float32) tensor, shape
                            ``[total_residues_across_all_proteins, hidden_dim]``,
                            written contiguously by `scripts/build_mmap_plm_cache.py`.
    <root>/meta.pt          dict with: names (list[str]), offsets (LongTensor[N+1]),
                            hidden_dim (int), dtype (str), total_rows (int).

A ``MmapPlmCache`` instance opens ``embeddings.bin`` via ``numpy.memmap`` (no RAM
cost at open time — the OS page cache lazily faults pages as ``get(name)`` is
called) and resolves ``name -> torch.Tensor`` in O(1).

Multiple concurrent processes that mmap the same file share physical RAM via the
OS page cache, which is the key advantage over an in-memory preload when running
several training jobs on the same node.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


_SUPPORTED_DTYPES = {"float16": np.float16, "float32": np.float32}


class MmapPlmCache:
    """Read-only memory-mapped PLM embedding cache."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        meta_path = self.root / "meta.pt"
        bin_path = self.root / "embeddings.bin"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.pt not found in {self.root}")
        if not bin_path.exists():
            raise FileNotFoundError(f"embeddings.bin not found in {self.root}")

        meta = torch.load(meta_path, weights_only=False)
        self.names: list[str] = list(meta["names"])
        offsets = meta["offsets"]
        if isinstance(offsets, torch.Tensor):
            offsets = offsets.cpu().numpy()
        self.offsets: np.ndarray = np.asarray(offsets, dtype=np.int64)
        self.hidden_dim = int(meta["hidden_dim"])
        self.dtype_str = str(meta["dtype"])
        self.total_rows = int(meta["total_rows"])

        if self.dtype_str not in _SUPPORTED_DTYPES:
            raise ValueError(
                f"Unsupported dtype {self.dtype_str!r} (supported: {sorted(_SUPPORTED_DTYPES)})"
            )
        if len(self.offsets) != len(self.names) + 1:
            raise ValueError(
                f"offsets length {len(self.offsets)} != names length {len(self.names)} + 1"
            )

        # O(1) name -> index lookup.
        self._name_to_idx: dict[str, int] = {n: i for i, n in enumerate(self.names)}

        np_dtype = _SUPPORTED_DTYPES[self.dtype_str]
        self._mmap: np.memmap = np.memmap(
            bin_path,
            dtype=np_dtype,
            mode="r",
            shape=(self.total_rows, self.hidden_dim),
        )

    def get(self, name: str) -> torch.Tensor:
        """Return the ``[num_residues, hidden_dim]`` tensor for ``name``.

        The returned tensor is a view onto the mmap region (zero-copy). Downstream
        code should treat it as read-only; ``.to(...)`` later in the pipeline
        produces an independent tensor.
        """
        idx = self._name_to_idx.get(name)
        if idx is None:
            raise KeyError(f"protein {name!r} not in mmap cache (root: {self.root})")
        start = int(self.offsets[idx])
        end = int(self.offsets[idx + 1])
        # np.memmap opened with mode='r' yields a read-only view; torch.from_numpy
        # warns on non-writable arrays. .copy() on a tiny per-protein slice is
        # cheap (a few hundred KB) and silences the warning without affecting throughput.
        return torch.from_numpy(self._mmap[start:end].copy())

    def __contains__(self, name: str) -> bool:
        return name in self._name_to_idx

    def __len__(self) -> int:
        return len(self.names)

    def __repr__(self) -> str:
        return (
            f"MmapPlmCache(root={self.root}, n_proteins={len(self.names)}, "
            f"total_rows={self.total_rows:,}, hidden_dim={self.hidden_dim}, "
            f"dtype={self.dtype_str})"
        )

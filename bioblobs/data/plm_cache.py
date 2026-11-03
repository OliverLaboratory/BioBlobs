"""Memory-mapped caches of residue embeddings.

Embedding cache layout (one directory per dataset and encoder)::

    <root>/embeddings.bin   float16 or float32 array [total_residues, hidden_dim]
    <root>/meta.pt          {names, offsets [n+1], hidden_dim, dtype, total_rows}

Reading maps the file lazily, so several training processes on one host share the page cache.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch

DTYPES = {"float16": np.float16, "float32": np.float32}


class _MmapIndex:
    def __init__(self, root: str | Path, data_file: str) -> None:
        self.root = Path(root)
        if not (self.root / "meta.pt").exists() or not (self.root / data_file).exists():
            raise FileNotFoundError(f"No {data_file} + meta.pt under {self.root}")
        self.meta = torch.load(self.root / "meta.pt", weights_only=True)
        self.names: list[str] = list(self.meta["names"])
        self.offsets = np.asarray(self.meta["offsets"], dtype=np.int64)
        self._index = {name: i for i, name in enumerate(self.names)}

    def _span(self, name: str) -> tuple[int, int]:
        if name not in self._index:
            raise KeyError(f"{name!r} not in cache {self.root}")
        i = self._index[name]
        return int(self.offsets[i]), int(self.offsets[i + 1])

    def __contains__(self, name: str) -> bool:
        return name in self._index

    def __len__(self) -> int:
        return len(self.names)


class EmbeddingCache(_MmapIndex):
    """Read-only memory-mapped residue embeddings, ``name -> [N, D]``."""

    def __init__(self, root: str | Path) -> None:
        super().__init__(root, "embeddings.bin")
        self.hidden_dim = int(self.meta["hidden_dim"])
        self.dtype = str(self.meta["dtype"])
        self._data = np.memmap(
            self.root / "embeddings.bin",
            dtype=DTYPES[self.dtype],
            mode="r",
            shape=(int(self.meta["total_rows"]), self.hidden_dim),
        )

    def get(self, name: str) -> torch.Tensor:
        """Embeddings of one protein, ``[N, D]`` (a copy, in the cache dtype)."""
        start, end = self._span(name)
        return torch.from_numpy(self._data[start:end].copy())


def write_embedding_cache(
    root: str | Path,
    items: Iterable[tuple[str, torch.Tensor]],
    hidden_dim: int,
    dtype: str = "float16",
) -> int:
    """Stream ``(name, embeddings [N, D])`` pairs into an embedding cache.

    Args:
        root: Output directory (created if missing).
        items: Embeddings per protein, in the order they should be stored.
        hidden_dim: Embedding width D.
        dtype: ``"float16"`` or ``"float32"``.

    Returns:
        Number of proteins written.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    names, offsets = [], [0]
    with (root / "embeddings.bin").open("wb") as handle:
        for name, emb in items:
            if emb.ndim != 2 or emb.shape[1] != hidden_dim:
                raise ValueError(f"{name}: expected [N, {hidden_dim}], got {tuple(emb.shape)}")
            handle.write(emb.detach().cpu().numpy().astype(DTYPES[dtype]).tobytes())
            names.append(name)
            offsets.append(offsets[-1] + emb.shape[0])
    meta = {
        "names": names,
        "offsets": torch.tensor(offsets),
        "hidden_dim": hidden_dim,
        "dtype": dtype,
        "total_rows": offsets[-1],
    }
    torch.save(meta, root / "meta.pt")
    return len(names)

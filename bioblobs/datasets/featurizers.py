"""Featurizer strategy pattern for encoder-agnostic protein dataset.

Each featurizer converts a raw protein dict into a torch_geometric.data.Data
object.  Adding support for a new encoder only requires adding a new
``BaseFeaturizer`` subclass and registering it in ``FEATURIZER_REGISTRY``.
"""

import math
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch_cluster
import torch_geometric.data

from ..modules.encoders.config import resolve_encoder_name

# ---------------------------------------------------------------------------
# Custom Data subclass for correct PyG batching of line-graph attributes
# ---------------------------------------------------------------------------

class ProteinData(torch_geometric.data.Data):
    """torch_geometric Data with correct batching for line_edge_index.

    PyG's default __inc__ increments any ``*_index`` attribute by num_nodes.
    line_edge_index indexes into the *edge* space, so it must be incremented
    by num_edges instead.
    """

    def __inc__(self, key, value, *args, **kwargs):
        if key == "line_edge_index":
            return self.num_edges
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "line_edge_index":
            return -1  # concatenate along columns, same as edge_index
        return super().__cat_dim__(key, value, *args, **kwargs)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

LETTER_TO_NUM = {
    "C": 4,
    "D": 3,
    "S": 15,
    "Q": 5,
    "K": 11,
    "I": 9,
    "P": 14,
    "T": 16,
    "F": 13,
    "A": 0,
    "G": 7,
    "H": 8,
    "E": 6,
    "L": 10,
    "R": 1,
    "W": 17,
    "V": 19,
    "N": 2,
    "Y": 18,
    "M": 12,
    "X": 20,  # Unknown amino acid
}

NUM_AMINO_ACIDS = 21  # 20 standard amino acids + X for unknown


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _label_to_tensor(label, *, device: str):
    """Convert scalar or dense multi-label targets to the appropriate tensor dtype."""
    if isinstance(label, np.ndarray):
        if label.ndim == 0:
            return torch.tensor(label.item(), dtype=torch.long, device=device)
        return torch.tensor(label.tolist(), dtype=torch.float32, device=device)

    if isinstance(label, (list, tuple)):
        return torch.tensor(label, dtype=torch.float32, device=device)

    return torch.tensor(label, dtype=torch.long, device=device)


def _attach_optional_fields(data, protein: dict, *, device: str) -> None:
    """Pass through task-specific metadata that downstream code may inspect."""
    passthrough_fields = (
        "sample_id",
        "protein_uid",
        "interpro_id",
        "source_interpro_label",
        "fragment_start",
        "fragment_end",
        "source_split",
        "target",
        "split_strategy",
    )

    for field_name in passthrough_fields:
        if field_name in protein:
            setattr(data, field_name, protein[field_name])

    residue_target = protein.get("residue_target")
    if residue_target is not None:
        data.residue_target = torch.tensor(
            residue_target,
            dtype=torch.float32,
            device=device,
        )

def _parse_edge_types(edge_types: str) -> tuple:
    """Parse an edge_types string (e.g. 'knn_30') into (method, value)."""
    parts = edge_types.split("_")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid edge_types format: {edge_types}. "
            "Expected format: 'method_value' (e.g., 'knn_30', 'eps_16')"
        )
    method, value_str = parts
    if method not in ("knn", "eps"):
        raise ValueError(f"Invalid edge method: {method}. Must be 'knn' or 'eps'")
    try:
        value = int(value_str)
    except ValueError:
        raise ValueError(f"Invalid edge value: {value_str}. Must be an integer")
    return method, value


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseFeaturizer(ABC):
    """Abstract featurizer: converts a protein dict to a PyG Data object."""

    @abstractmethod
    def featurize(self, protein: dict) -> torch_geometric.data.Data:
        """Featurize a single protein dict."""
        ...

    @classmethod
    @abstractmethod
    def from_config(cls, ds_cfg, enc_cfg=None) -> "BaseFeaturizer":
        """Construct featurizer from dataset config (and optional encoder config)."""
        ...


# ---------------------------------------------------------------------------
# Static ESM2 graph featurizer
# ---------------------------------------------------------------------------

# Datasets that share the same proteins (identical names + coords) can share a
# single mmap cache per family. When present, the featurizer prefers the shared
# location over the per-dataset location.
_SHARED_MMAP_FAMILY_DIR: dict[str, str] = {
    "scop_fam_proteinshake":               "_shared_mmap/scop",
    "scop_sf_proteinshake":                "_shared_mmap/scop",
    "go_molecular_function_proteinshake":  "_shared_mmap/go",
    "go_cellular_component_proteinshake":  "_shared_mmap/go",
}


class ESM2StaticGraphFeaturizer(BaseFeaturizer):
    """Featurizer for cached PLM node features on the structural graph.

    Works with both ESM2 and SaProt static caches — the ``cache_path_key``
    parameter selects which cache path key to read from the protein dict.
    """

    def __init__(self, edge_method, edge_value, device, cache_path_key="esm2_cache_path"):
        self.edge_method = edge_method
        self.edge_value = edge_value
        self.device = device
        self.cache_path_key = cache_path_key
        # Lazy-loaded mmap caches, initialized from the first protein's cache path.
        self._mmap_cache = None
        self._mmap_attempted = False
        self._edge_mmap_cache = None
        self._edge_mmap_attempted = False

    def _ensure_mmap_loaded(self, sample_cache_path: str) -> None:
        """Try to load a mmap PLM cache that lives next to the per-protein .pt cache.

        Layout assumption (matches build_mmap_plm_cache.py):
            <dataset>/<encoder>_cache/<key>/embeddings/<name>.pt   (existing)
            <dataset>/<encoder>_cache_mmap/embeddings.bin          (mmap)
            <dataset>/<encoder>_cache_mmap/meta.pt
        """
        if self._mmap_attempted:
            return
        self._mmap_attempted = True
        try:
            from pathlib import Path
            cache_path = Path(sample_cache_path)
            # walk up: <name>.pt → embeddings → <key> → <encoder>_cache → <dataset>
            cache_root = cache_path.parent.parent.parent  # <dataset>/<encoder>_cache
            dataset_dir = cache_root.parent                # data/<dataset>_proteinshake
            data_root = dataset_dir.parent                 # data/
            mmap_name = f"{cache_root.name}_mmap"

            # Prefer shared location for dataset families with identical proteins;
            # fall back to per-dataset location.
            candidates = []
            family_sub = _SHARED_MMAP_FAMILY_DIR.get(dataset_dir.name)
            if family_sub is not None:
                candidates.append(data_root / family_sub / mmap_name)
            candidates.append(dataset_dir / mmap_name)

            mmap_root = next(
                (c for c in candidates
                 if (c / "embeddings.bin").exists() and (c / "meta.pt").exists()),
                None,
            )
            if mmap_root is not None:
                from .mmap_plm_cache import MmapPlmCache
                from loguru import logger
                self._mmap_cache = MmapPlmCache(mmap_root)
                logger.info(
                    "Featurizer using mmap PLM cache at {} ({} proteins, {} rows, dtype={})",
                    mmap_root, len(self._mmap_cache), self._mmap_cache.total_rows,
                    self._mmap_cache.dtype_str,
                )
        except Exception as exc:  # noqa: BLE001
            from loguru import logger
            logger.debug("mmap cache load skipped ({}); falling back to per-protein torch.load", exc)
            self._mmap_cache = None

    def _ensure_edge_mmap_loaded(self, sample_cache_path: str) -> None:
        """Try to load a mmap edge_index cache for this featurizer's edge config.

        Layout assumption (matches build_mmap_edge_cache.py):
            <dataset>/edge_cache_mmap/<edge_types>/edges.bin
            <dataset>/edge_cache_mmap/<edge_types>/meta.pt
        """
        if self._edge_mmap_attempted:
            return
        self._edge_mmap_attempted = True
        try:
            from pathlib import Path
            cache_path = Path(sample_cache_path)
            # walk up: <name>.pt → embeddings → <key> → <encoder>_cache → <dataset>
            dataset_dir = cache_path.parent.parent.parent.parent
            data_root = dataset_dir.parent
            # Stringify edge_value: int for knn, raw for eps (matches edge_types convention).
            if self.edge_method == "knn":
                edge_types = f"knn_{int(self.edge_value)}"
            else:
                # Drop trailing .0 for nice integer-looking dirs ("eps_8" not "eps_8.0").
                ev = self.edge_value
                if isinstance(ev, float) and ev.is_integer():
                    ev = int(ev)
                edge_types = f"eps_{ev}"

            # Prefer shared location for dataset families with identical coords.
            candidates = []
            family_sub = _SHARED_MMAP_FAMILY_DIR.get(dataset_dir.name)
            if family_sub is not None:
                candidates.append(data_root / family_sub / "edge_cache_mmap" / edge_types)
            candidates.append(dataset_dir / "edge_cache_mmap" / edge_types)

            edge_root = next(
                (c for c in candidates
                 if (c / "edges.bin").exists() and (c / "meta.pt").exists()),
                None,
            )
            if edge_root is not None:
                from .edge_index_mmap_cache import EdgeIndexMmapCache
                from loguru import logger
                self._edge_mmap_cache = EdgeIndexMmapCache(edge_root)
                logger.info(
                    "Featurizer using mmap edge cache at {} ({} proteins, {} edges)",
                    edge_root, len(self._edge_mmap_cache), self._edge_mmap_cache.total_edges,
                )
        except Exception as exc:  # noqa: BLE001
            from loguru import logger
            logger.debug("edge mmap load skipped ({}); falling back to torch_cluster knn_graph", exc)
            self._edge_mmap_cache = None

    @classmethod
    def from_config(cls, ds_cfg, enc_cfg=None) -> "ESM2StaticGraphFeaturizer":
        edge_types = (
            ds_cfg.get("edge_types", "knn_30")
            if hasattr(ds_cfg, "get")
            else getattr(ds_cfg, "edge_types", "knn_30")
        )
        edge_method, edge_value = _parse_edge_types(edge_types)
        device = (
            ds_cfg.get("device", "cpu")
            if hasattr(ds_cfg, "get")
            else getattr(ds_cfg, "device", "cpu")
        )
        cache_path_key = "esm2_cache_path"
        if enc_cfg is not None:
            encoder_name = (
                enc_cfg.get("name", "")
                if hasattr(enc_cfg, "get")
                else getattr(enc_cfg, "name", "")
            )
            if "saprot" in encoder_name:
                cache_path_key = "saprot_cache_path"
        return cls(edge_method, edge_value, device, cache_path_key)

    def _build_edge_index(
        self,
        x_ca: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        valid_idx = mask.nonzero(as_tuple=True)[0]
        if valid_idx.numel() <= 1:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)

        valid_x_ca = x_ca[valid_idx]
        if self.edge_method == "knn":
            k = min(self.edge_value, valid_idx.numel() - 1)
            if k <= 0:
                return torch.empty((2, 0), dtype=torch.long, device=self.device)
            edge_index_local = torch_cluster.knn_graph(valid_x_ca, k=k)
        elif self.edge_method == "eps":
            edge_index_local = torch_cluster.radius_graph(valid_x_ca, r=self.edge_value)
        else:
            raise ValueError(f"Unknown edge method: {self.edge_method}")

        return valid_idx[edge_index_local]

    def featurize(self, protein: dict) -> torch_geometric.data.Data:
        name = protein["name"]
        cache_path = protein.get(self.cache_path_key)
        if cache_path is None:
            raise KeyError(
                f"Protein {name!r} is missing {self.cache_path_key!r}. "
                "Run static PLM cache preparation before featurization."
            )

        with torch.no_grad():
            coords = torch.as_tensor(
                protein["coords"], device=self.device, dtype=torch.float32
            )
            seq = torch.as_tensor(
                [LETTER_TO_NUM.get(a, 20) for a in protein["seq"]],
                device=self.device,
                dtype=torch.long,
            )

            assert len(seq) == coords.shape[0], (
                f"Sequence length {len(seq)} doesn't match coords {coords.shape[0]} for {name}. "
                f"This should be handled by extract_backbone_coords()."
            )

            self._ensure_mmap_loaded(cache_path)
            if self._mmap_cache is not None and name in self._mmap_cache:
                node_features = self._mmap_cache.get(name)
            else:
                node_features = torch.load(cache_path, map_location="cpu", weights_only=False)
            if not torch.is_tensor(node_features):
                raise TypeError(
                    f"Static ESM2 cache for {name!r} must be a tensor, got {type(node_features)}"
                )
            if node_features.ndim != 2:
                raise ValueError(
                    f"Static ESM2 cache for {name!r} must be 2D, got shape {tuple(node_features.shape)}"
                )
            if node_features.size(0) != len(seq):
                raise ValueError(
                    f"Static ESM2 cache length mismatch for {name!r}: "
                    f"expected {len(seq)}, got {node_features.size(0)}"
                )
            node_features = node_features.to(device=self.device, dtype=torch.float32)

            mask = torch.isfinite(coords.sum(dim=(1, 2)))
            coords[~mask] = np.inf
            x_ca = coords[:, 1]

            # Edge index: prefer mmap cache, fall back to recomputation.
            self._ensure_edge_mmap_loaded(cache_path)
            if self._edge_mmap_cache is not None and name in self._edge_mmap_cache:
                edge_index = self._edge_mmap_cache.get(name).to(device=self.device)
            else:
                edge_index = self._build_edge_index(x_ca, mask)

            label = protein.get("label", 0)
            y = _label_to_tensor(label, device=self.device)

        data = ProteinData(
            x=x_ca,
            seq=seq,
            name=name,
            resnum=protein["resnum"],
            edge_index=edge_index,
            mask=mask,
            node_features=node_features,
            num_nodes=len(seq),
            y=y,
        )

        if "partition" in protein and protein["partition"] is not None:
            data.partition = protein["partition"]

        _attach_optional_fields(data, protein, device=self.device)

        return data


# ---------------------------------------------------------------------------
# Registry and factory
# ---------------------------------------------------------------------------

FEATURIZER_REGISTRY = {
    "esm2_static": ESM2StaticGraphFeaturizer,
    "saprot_static": ESM2StaticGraphFeaturizer,  # same featurizer, different cache_path_key
}


def build_featurizer(cfg) -> BaseFeaturizer:
    """Build the appropriate featurizer from a full config object.

    Args:
        cfg: Config with ``cfg.datasets`` and ``cfg.encoders.name``.

    Returns:
        A ``BaseFeaturizer`` instance ready to call ``.featurize(protein)``.
    """
    ds_cfg = cfg.datasets
    enc_cfg = cfg.get("encoders", {}) if hasattr(cfg, "get") else getattr(cfg, "encoders", {})
    encoder_name = resolve_encoder_name(enc_cfg)
    if encoder_name not in FEATURIZER_REGISTRY:
        raise ValueError(
            f"Unknown encoder '{encoder_name}'. Supported: {list(FEATURIZER_REGISTRY)}"
        )
    cls = FEATURIZER_REGISTRY[encoder_name]
    return cls.from_config(ds_cfg, enc_cfg)

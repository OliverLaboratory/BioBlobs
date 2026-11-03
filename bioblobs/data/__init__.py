"""Datasets: ProteinShake tasks, VenusX site-fragment classification and SwissProt EC-L3."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import torch
from omegaconf import DictConfig

from bioblobs.data.dataset import ProteinGraphDataset, make_loader, open_graph_cache
from bioblobs.data.plm_cache import EmbeddingCache

SOURCES = {
    "ec": "proteinshake",
    "pfam": "proteinshake",
    "go_mf": "proteinshake",
    "scop_fam": "proteinshake",
    "scop_sf": "proteinshake",
    "venusx_site_fragment": "venusx",
    "swissprot_ec": "swissprot",
}


GRAPH_PARTITIONERS = {"diffpool", "topk", "sagpool"}
SPLITS = {"swissprot": ("train", "val")}


@dataclass
class Splits:
    """Train, validation and test datasets of one task."""

    train: ProteinGraphDataset | None
    val: ProteinGraphDataset | None
    test: ProteinGraphDataset | None
    num_classes: int
    class_counts: list[int] | None = None


def load_splits(cfg: DictConfig, names: tuple[str, ...] = ("train", "val", "test")) -> Splits:
    """Load the datasets named by ``cfg.datasets.dataset_name``.

    Args:
        cfg: Composed config (``datasets``, ``encoders``, ``tasks`` and ``objective`` groups).
        names: Splits to load.

    Returns:
        The datasets and the number of classes.
    """
    ds = cfg.datasets
    source = import_module(f"bioblobs.data.{SOURCES[ds.dataset_name]}")
    root = Path(source.dataset_root(ds))
    cache_dir = ds.get("embedding_cache") or root / f"{cfg.encoders.name}_cache_mmap"
    embeddings = EmbeddingCache(cache_dir)
    needs_graph = cfg.encoders.get("gnn") is not None or cfg.partitioners.name in GRAPH_PARTITIONERS
    edge_types = ds.get("edge_types") if needs_graph else None
    graphs = open_graph_cache(root, edge_types)
    key = ds.get("cache_key_field", "name")

    def load(split: str) -> tuple[ProteinGraphDataset, int]:
        if ds.dataset_name == "venusx_site_fragment":
            site = bool(ds.get("site_supervised", False))
            structures, n = source.load_split(cfg, split, site_supervised=site)
        else:
            structures, n = source.load_split(cfg, split)
        if ds.get("max_samples"):
            structures = structures[: int(ds.max_samples)]
        return ProteinGraphDataset(structures, embeddings, key, edge_types, graphs), n

    available = SPLITS.get(SOURCES[ds.dataset_name], ("train", "val", "test"))
    loaded, num_classes = {}, 0
    for name in (n for n in names if n in available):
        loaded[name], num_classes = load(name)
    counts = source.class_counts(cfg) if hasattr(source, "class_counts") else None
    return Splits(loaded.get("train"), loaded.get("val"), loaded.get("test"), num_classes, counts)


def class_weights(counts: list[int] | None, beta: float | None) -> torch.Tensor | None:
    """Class-balanced weights when ``beta`` is set, else None."""
    if counts is None or not beta:
        return None
    from bioblobs.objectives import class_balanced_weights

    return class_balanced_weights(counts, beta)


__all__ = ["ProteinGraphDataset", "Splits", "class_weights", "load_splits", "make_loader"]

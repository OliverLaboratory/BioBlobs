"""SwissProt EC-L3 pretraining data.

Reproduces the train/val split, labels and ``max_len`` filter used to train the
released SwissProt EC-L3 checkpoints, over the public ``bioblobs.data.structure``
convention. The split CSVs, structure cache, token map and class priors are
produced upstream by ``preprocessing/swissprot_ec/build_structures.py`` and
``preprocessing/swissprot_ec/build_val_split.py``; this module only reads
those prepared artifacts.

Prepared dataset directory (``dataset_root``)::

    <data_dir>/<dataset_name>/cache/structures.pt        structure dicts (name, seq, coords, resnum)
    <data_dir>/<dataset_name>/train_split_train.csv      id, label (+ seq, ec_number, ...)
    <data_dir>/<dataset_name>/train_split_val.csv        id, label
    <data_dir>/<dataset_name>/token_map.json             {"a.b.c": class_index}
    <data_dir>/<dataset_name>/class_priors.json          {"N", "counts", "priors"}
    <data_dir>/<dataset_name>/esm2_static_cache_mmap/    residue embedding cache (built separately)

The embedding cache is keyed by ``structure["name"]`` (``bioblobs.data.plm_cache.EmbeddingCache``
reads it with ``key_field="name"``, the ``ProteinGraphDataset`` default).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from loguru import logger

from bioblobs.data.structure import Structure

DEFAULT_DATASET_NAME = "swissprot_ec"
DEFAULT_MAX_LEN = 1000


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` off a dict-like, OmegaConf, or attribute-style config."""
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _dataset_cfg(cfg: Any) -> Any:
    """Accept either the dataset sub-config or a full config with a ``datasets`` field."""
    nested = _get(cfg, "datasets", None)
    return cfg if nested is None else nested


def dataset_root(ds_cfg: Any) -> Path:
    """Resolve the prepared SwissProt EC-L3 dataset directory.

    Holds the structure cache, train/val split CSVs, token map, class priors
    and (once built separately) the ``esm2_static_cache_mmap`` embedding
    cache.

    Args:
        ds_cfg: Dataset config with ``data_dir`` (default ``"data"``) and
            ``dataset_name`` (default ``"swissprot_ec"``).

    Returns:
        Absolute path to the prepared dataset directory.
    """
    ds_cfg = _dataset_cfg(ds_cfg)
    data_dir = Path(_get(ds_cfg, "data_dir", "data"))
    name = _get(ds_cfg, "dataset_name", DEFAULT_DATASET_NAME)
    return (data_dir / name).resolve()


def _load_token_map(root: Path) -> dict[str, int]:
    path = root / "token_map.json"
    with path.open() as fh:
        return json.load(fh)


def _split_csv(ds_cfg: Any, root: Path, split: str) -> Path:
    if split == "train":
        override = _get(ds_cfg, "train_csv", None)
        return Path(override).resolve() if override else root / "train_split_train.csv"
    if split == "val":
        return root / "train_split_val.csv"
    raise ValueError(f"Unknown split {split!r}; expected 'train' or 'val'")


def load_split(cfg: Any, split: str) -> tuple[list[Structure], int]:
    """Load one split's structures with labels attached.

    Proteins are kept in the CSV row order (the last row wins on a duplicate
    id), then filtered to ``len(seq) <= max_len`` in place, preserving order.
    This reproduces the exact train/val protein sets, order and labels used
    to train the released checkpoints.

    Args:
        cfg: The SwissProt EC dataset config (or a full config with a
            ``datasets`` field); see :func:`dataset_root`. Optional fields:
            ``max_len`` (default 1000, drops longer proteins) and
            ``train_csv`` (override path for the train split only, e.g. a
            homology-excluded variant).
        split: ``"train"`` or ``"val"``.

    Returns:
        ``(structures, num_classes)``. Each structure has ``name``, ``seq``,
        ``coords``, ``resnum`` (from the structure cache) and an added
        integer ``label``.
    """
    ds_cfg = _dataset_cfg(cfg)
    root = dataset_root(ds_cfg)
    token_map = _load_token_map(root)
    num_classes = len(token_map)

    structures_pt = root / "cache" / "structures.pt"
    logger.info("Loading structure cache: {}", structures_pt)
    raw = torch.load(structures_pt, weights_only=False, mmap=True)
    by_name = {s["name"]: s for s in raw}

    csv_path = _split_csv(ds_cfg, root, split)
    df = pd.read_csv(csv_path, usecols=["id", "label"])
    id_to_label = dict(zip(df["id"].astype(str), df["label"].astype(int), strict=True))

    structures: list[Structure] = []
    missing = 0
    for name, label in id_to_label.items():
        s = by_name.get(name)
        if s is None:
            missing += 1
            continue
        structure = dict(s)
        structure["label"] = int(label)
        structures.append(structure)
    if missing:
        logger.info("{}: {:,} ids missing from structure cache (skipped)", csv_path.name, missing)

    max_len = _get(ds_cfg, "max_len", DEFAULT_MAX_LEN)
    if max_len is not None:
        before = len(structures)
        structures = [s for s in structures if len(s["seq"]) <= max_len]
        if len(structures) < before:
            logger.info(
                "max_len {}: {} -> {:,} ({} dropped)",
                max_len,
                before,
                len(structures),
                before - len(structures),
            )

    logger.info("{}: {:,} proteins, {} classes", split, len(structures), num_classes)
    return structures, num_classes


def class_counts(cfg: Any) -> list[int]:
    """Per-class training counts in class-index order, for class-balanced loss weights.

    The class index of each EC-L3 label is its position in
    ``sorted(class_priors.counts)``, which is identical to ``token_map.json``'s
    index for every label in the prepared dataset.

    Args:
        cfg: The SwissProt EC dataset config (or a full config with a
            ``datasets`` field); see :func:`dataset_root`.

    Returns:
        ``counts[class_index] = n_train_proteins_with_that_label``.
    """
    root = dataset_root(cfg)
    with (root / "class_priors.json").open() as fh:
        priors = json.load(fh)
    counts: dict[str, int] = priors["counts"]
    sorted_names = sorted(counts)
    return [int(counts[name]) for name in sorted_names]

"""VenusX site-fragment dataset: Act / BindI / Evo / Motif InterPro classification.

Each sample is one (protein, InterPro annotation) pair: ``sample_id`` is
``"<protein_uid>__<interpro_id>"`` and ``label`` is the dense, zero-based class
index of ``interpro_id`` within the target. Several samples can share one
``protein_uid`` (the same protein annotated for different InterPro families),
so ``protein_uid`` -- not ``sample_id`` -- is the embedding-cache key.

Two protein sets coexist for this benchmark, mirroring the two training
regimes used to produce the paper's numbers (``site_supervised`` selects
between them):

* ``site_supervised=False`` -- label-free classification models. Every sample
  whose protein has an entry in *both* the ESM2 and SaProt embedding caches is
  kept; the residue-annotation column is never read to decide protein
  membership or for training. A ``residue_target`` is still attached when the
  annotation can be projected onto the resolved residue numbers, which is how test
  localization is scored; samples whose projection fails carry no ``residue_target``.
* ``site_supervised=True`` -- seed-aligned/site-supervised models and
  baselines. A sample needs a successfully parsed structure *and* a
  successfully projected residue annotation to be kept at all; either failure
  drops the sample from the protein set.

Both read the same prepared-directory layout, produced by
``preprocessing/prepare_venusx.py``::

    <data_dir>/<dataset_name>/<target>/<split_strategy>/
        samples.csv                                  one row per (protein, InterPro) sample
        token_maps.json                              {"1": {interpro_id: dense_label}}
        {train,val,test}_split.csv                   sample_id manifests
        pdb/<protein_uid>.pdb
        cache/uid_structures_min_completion_0p95.pt  per-protein backbone geometry
        esm2_static_cache_mmap/, saprot_static_cache_mmap/   residue embedding caches
        metadata.json
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig

from bioblobs.data.structure import load_structures

TARGETS = ("Act", "BindI", "Evo", "Motif", "Dom")
SPLIT_STRATEGIES = ("MF50", "MF70", "MF90")
SPLITS = ("train", "val", "test")
ENCODER_CACHES = ("esm2_static_cache_mmap", "saprot_static_cache_mmap")


def dataset_root(ds_cfg: DictConfig) -> Path:
    """Prepared VenusX directory: ``<data_dir>/<dataset_name>/<target>/<split_strategy>``."""
    return (
        Path(ds_cfg.data_dir)
        / str(ds_cfg.dataset_name)
        / str(ds_cfg.target)
        / str(ds_cfg.split_strategy)
    )


def load_split(cfg: DictConfig, split: str, site_supervised: bool) -> tuple[list[dict], int]:
    """Load one VenusX split as structure dicts ready for ``structure.to_graph``.

    Args:
        cfg: Composed config; reads ``cfg.datasets`` (see the module docstring
            for the required fields and the on-disk layout).
        split: ``"train"``, ``"val"`` or ``"test"``.
        site_supervised: Use the seed-aligned protein set (every sample has a
            projected residue annotation) when True; the label-free protein
            set (every sample is embedding-cache-covered) when False.

    Returns:
        ``(structures, num_classes)``. Each structure has ``name``,
        ``protein_uid``, ``seq``, ``coords``, ``resnum``, ``label`` and,
        when the site annotation was available, ``residue_target``.
    """
    if split not in SPLITS:
        raise ValueError(f"Unknown VenusX split {split!r}; expected one of {SPLITS}")
    ds_cfg = cfg.datasets
    _validate_target(ds_cfg)
    if site_supervised:
        return _load_site_supervised(ds_cfg, split)
    return _load_labelfree(ds_cfg, split)


def _validate_target(ds_cfg: DictConfig) -> None:
    if str(ds_cfg.target) not in TARGETS:
        raise ValueError(f"Unsupported VenusX target {ds_cfg.target!r}; expected one of {TARGETS}")
    if str(ds_cfg.split_strategy) not in SPLIT_STRATEGIES:
        raise ValueError(
            f"Unsupported VenusX split strategy {ds_cfg.split_strategy!r}; "
            f"expected one of {SPLIT_STRATEGIES}"
        )


def _class_map(root: Path) -> dict[str, int]:
    token_maps_path = root / "token_maps.json"
    if not token_maps_path.exists():
        raise FileNotFoundError(
            f"No prepared VenusX dataset at {root}; run preprocessing/prepare_venusx.py first"
        )
    raw = json.loads(token_maps_path.read_text())["1"]
    class_map = {str(k): int(v) for k, v in raw.items()}
    if set(class_map.values()) != set(range(len(class_map))):
        raise ValueError(f"VenusX class vocabulary at {root} is not dense and zero-based")
    return class_map


def _read_split_ids(path: Path) -> list[str]:
    ids = pd.read_csv(path)["sample_id"].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate sample IDs in {path}")
    return ids


def _check_disjoint_and_complete(split_ids: dict[str, list[str]], all_ids: set[str]) -> None:
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(split_ids[a]) & set(split_ids[b]):
            raise ValueError(f"Overlapping VenusX sample IDs between {a!r} and {b!r}")
    if set().union(*(set(v) for v in split_ids.values())) != all_ids:
        raise ValueError("Split manifests and samples.csv disagree on the VenusX sample set")


def _cache_names(root: Path, subdir: str) -> set[str]:
    meta_path = root / subdir / "meta.pt"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No {meta_path}; build the embedding cache with preprocessing/prepare_plm_cache.py"
        )
    meta = torch.load(meta_path, map_location="cpu", weights_only=False)
    return set(meta["names"])


def _geometry_cache_path(root: Path, min_completion: float) -> Path:
    tag = str(min_completion).replace(".", "p")
    return root / "cache" / f"uid_structures_min_completion_{tag}.pt"


def _load_geometry(
    root: Path, min_completion: float, requested_uids: set[str], num_workers: int
) -> dict[str, dict]:
    """Per-protein backbone geometry, parsing any PDB not yet in the cache."""
    cache_path = _geometry_cache_path(root, min_completion)
    geometry: dict[str, dict] = {}
    if cache_path.exists():
        geometry = torch.load(cache_path, map_location="cpu", weights_only=False)
    missing = sorted(uid for uid in requested_uids if uid not in geometry)
    if not missing:
        return geometry
    logger.warning(
        "{} VenusX protein structures missing from {}; parsing PDBs", len(missing), cache_path
    )
    tasks = [(uid, root / "pdb" / f"{uid}.pdb", None, min_completion) for uid in missing]
    for structure in load_structures(tasks, num_workers=num_workers, desc="venusx-geometry"):
        geometry[structure["name"]] = {
            "protein_uid": structure["name"], "seq": structure["seq"],
            "coords": structure["coords"], "resnum": structure["resnum"],
        }  # fmt: skip
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(geometry, cache_path)
    return geometry


def _project_residue_target(
    raw_residue_label: str, resnum: list[int], seq_len: int
) -> tuple[list[int] | None, str | None]:
    """Project a full-protein-length binary site annotation onto resolved residue numbers.

    ``raw_residue_label`` is the JSON- (or ``repr``-) encoded 0/1 list over the full
    InterPro-annotated sequence; ``resnum`` are the residue numbers actually resolved
    in the structure (1-based, possibly a strict subset of the full sequence).
    """
    try:
        labels = json.loads(raw_residue_label)
    except (json.JSONDecodeError, TypeError):
        try:
            labels = ast.literal_eval(raw_residue_label)
        except (ValueError, SyntaxError) as exc:
            return None, f"unparsable_residue_label:{exc}"
    if not isinstance(labels, (list, tuple)) or not all(v in (0, 1) for v in labels):
        return None, "invalid_binary_annotation"
    if any(r < 1 or r > len(labels) for r in resnum):
        return None, "residue_number_out_of_annotation_range"
    if len(resnum) != seq_len:
        return None, "residue_sequence_length_mismatch"
    return [int(labels[r - 1]) for r in resnum], None


def _load_labelfree(ds_cfg: DictConfig, split: str) -> tuple[list[dict], int]:
    """Label-free protein set: every embedding-cache-covered sample; site labels never select."""
    root = dataset_root(ds_cfg)
    samples_path = root / "samples.csv"
    if not samples_path.exists():
        raise FileNotFoundError(
            f"No prepared VenusX dataset at {root}; run preprocessing/prepare_venusx.py first"
        )
    samples = pd.read_csv(
        samples_path,
        dtype={"sample_id": str, "protein_uid": str, "interpro_id": str},
        usecols=["sample_id", "protein_uid", "interpro_id", "interpro_label", "residue_label"],
    )
    if samples["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample IDs in {samples_path}")
    by_id = samples.set_index("sample_id", drop=False).to_dict("index")

    class_map = _class_map(root)
    for row in by_id.values():
        if class_map.get(row["interpro_id"]) != int(row["interpro_label"]):
            raise ValueError(f"Unknown or inconsistent protein label: {row['sample_id']}")

    split_ids = {s: _read_split_ids(root / f"{s}_split.csv") for s in SPLITS}
    _check_disjoint_and_complete(split_ids, set(by_id))

    esm2_names = _cache_names(root, ENCODER_CACHES[0])
    if esm2_names != _cache_names(root, ENCODER_CACHES[1]):
        raise ValueError("ESM2 and SaProt embedding-cache protein coverage differs")
    all_uids = {row["protein_uid"] for row in by_id.values()}
    missing_uids = all_uids - esm2_names
    logger.info(
        "VenusX {}: {} of {} proteins are absent from the embedding cache and excluded "
        "from every split", ds_cfg.target, len(missing_uids), len(all_uids),
    )  # fmt: skip

    needed_uids = {by_id[sid]["protein_uid"] for sid in split_ids[split]} - missing_uids
    min_completion = float(ds_cfg.get("min_backbone_completion", 0.95))
    geometry = _load_geometry(root, min_completion, needed_uids, int(ds_cfg.get("num_workers", 4)))

    structures, unprojected = [], 0
    for sample_id in split_ids[split]:
        row = by_id[sample_id]
        uid = row["protein_uid"]
        if uid in missing_uids:
            continue
        item = geometry.get(uid)
        if item is None:
            raise ValueError(f"Missing geometry for protein {uid} ({sample_id})")
        if len(item["seq"]) != len(item["coords"]) or len(item["resnum"]) != len(item["seq"]):
            raise ValueError(f"Geometry/sequence length mismatch: {sample_id}")
        if not np.isfinite(np.asarray(item["coords"])).all(axis=(1, 2)).any():
            raise ValueError(f"No valid geometry: {sample_id}")
        structure = {
            "name": sample_id, "protein_uid": uid, "seq": item["seq"],
            "coords": item["coords"], "resnum": item["resnum"], "label": int(row["interpro_label"]),
        }  # fmt: skip
        target, _reason = _project_residue_target(
            row["residue_label"], item["resnum"], len(item["seq"])
        )
        if target is not None:
            structure["residue_target"] = target
        else:
            unprojected += 1
        structures.append(structure)
    if not structures:
        raise ValueError(f"Empty eligible split {split!r} for VenusX target {ds_cfg.target}")
    logger.info(
        "VenusX {} {}: {} samples ({} without a projected residue annotation)",
        ds_cfg.target,
        split,
        len(structures),
        unprojected,
    )
    return structures, len(class_map)


def _load_site_supervised(ds_cfg: DictConfig, split: str) -> tuple[list[dict], int]:
    """Site-supervised protein set: samples with a parsed structure and a projected annotation."""
    root = dataset_root(ds_cfg)
    samples_path = root / "samples.csv"
    if not samples_path.exists():
        raise FileNotFoundError(
            f"No prepared VenusX dataset at {root}; run preprocessing/prepare_venusx.py first"
        )
    samples = pd.read_csv(samples_path, dtype={"sample_id": str, "protein_uid": str})
    by_id = samples.set_index("sample_id", drop=False).to_dict("index")
    class_map = _class_map(root)

    split_ids = _read_split_ids(root / f"{split}_split.csv")
    needed_uids = {by_id[sid]["protein_uid"] for sid in split_ids}
    min_completion = float(ds_cfg.get("min_backbone_completion", 0.95))
    geometry = _load_geometry(root, min_completion, needed_uids, int(ds_cfg.get("num_workers", 4)))
    strict = bool(ds_cfg.get("strict_alignment", False))

    structures, skipped_structure, skipped_projection = [], 0, 0
    for sample_id in split_ids:
        row = by_id[sample_id]
        uid = row["protein_uid"]
        item = geometry.get(uid)
        if item is None:
            skipped_structure += 1
            if strict:
                raise ValueError(f"Failed to parse VenusX structure for {uid} ({sample_id})")
            continue
        target, reason = _project_residue_target(
            row["residue_label"], item["resnum"], len(item["seq"])
        )
        if target is None:
            skipped_projection += 1
            if strict:
                raise ValueError(f"{sample_id}: {reason}")
            continue
        structures.append({
            "name": sample_id, "protein_uid": uid, "seq": item["seq"], "coords": item["coords"],
            "resnum": item["resnum"], "label": int(row["interpro_label"]), "residue_target": target,
        })  # fmt: skip
    logger.info(
        "VenusX {} {} (site-supervised): {} samples ({} without structure, "
        "{} failed annotation projection)",
        ds_cfg.target,
        split,
        len(structures),
        skipped_structure,
        skipped_projection,
    )
    return structures, len(class_map)

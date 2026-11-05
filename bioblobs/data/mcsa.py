"""M-CSA catalytic-site evaluation data.

Loads the curated single-chain M-CSA (Mechanism and Catalytic Site Atlas) enzymes
used for the catalytic-residue-localization evaluation reported in Figure 2.

Inputs (shipped under ``data/mcsa/`` unless noted otherwise):
    mcsa_single_chain.json   one chain per M-CSA entry: pdb_id, chain_used,
                             ec_number, catalytic_residues[{resid, ...}]
    heldout_ids.txt          79 pdb_chain ids with no SwissProt EC-L3 training
                             protein in the same MMseqs2 cluster, one per cluster
    <feat_cache>.pt          NOT shipped (large): {pdb_chain: {"nf", "ca", "resnums"}},
                             a cached ESM2 feature file. Path is required via
                             ``cfg.feat_cache``.

Two feature caches exist with different chain-length policies (both windowed ESM2,
``window_size=1022``, ``window_overlap=128``, ``repr_layer=-1``,
``facebook/esm2_t30_150M_UR50D``):
    feat_cache.pt              chains filtered at length <= 1022 (874 of the 886 unique
                                pdb_chain keys in mcsa_single_chain.json, 892 raw records).
                                Used for the 74-protein evaluation cohort reported in
                                Figure 2.
    feat_cache_maxlen1500.pt   chains filtered at length <= 1500 (883 of 886), matching
                                the training max_len. Used for a full-cohort check over
                                every cached chain.
Pick one via ``cfg.feat_cache``; this module does not build either cache (see
``preprocessing/mcsa/README.md``).

Catalytic residues are read only as data here (mapped onto the cached residue order
by residue number) and never inspected to select, filter or order proteins.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from bioblobs.data.structure import BACKBONE, Structure
from bioblobs.data.structure import to_graph as _to_graph

_PACKAGE_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "mcsa"
DEFAULT_LABELS = _PACKAGE_DATA_DIR / "mcsa_single_chain.json"
DEFAULT_HELDOUT_IDS = _PACKAGE_DATA_DIR / "heldout_ids.txt"


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` off a dict-like, OmegaConf, or attribute-style config."""
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _read_id_list(path: Path) -> set[str]:
    """Read a newline-separated id file, ignoring blank lines and ``#`` comments."""
    ids = set()
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.add(line)
    return ids


def _load_curated_entries(labels_path: Path) -> dict[str, dict]:
    """One curated entry per (pdb_id, chain_used); later entries win on a duplicate key."""
    raw = json.loads(labels_path.read_text())
    by_key: dict[str, dict] = {}
    for entry in raw:
        by_key[f"{entry['pdb_id']}_{entry['chain_used']}"] = entry
    return by_key


def catalytic_indices(entry: dict, resnums: list[int]) -> list[int]:
    """Map an entry's annotated catalytic residue numbers onto cached residue indices.

    A residue id with no match in ``resnums`` (outside the cached window, or an
    insertion-code mismatch) is silently dropped.

    Args:
        entry: A curated M-CSA entry with ``catalytic_residues: [{"resid": int, ...}]``.
        resnums: Cached author residue numbers, in residue order.

    Returns:
        Residue indices (into ``resnums``) of the mapped catalytic residues.
    """
    resnum_to_idx = {int(r): i for i, r in enumerate(resnums)}
    return [
        resnum_to_idx[int(r["resid"])]
        for r in entry["catalytic_residues"]
        if int(r["resid"]) in resnum_to_idx
    ]


def load_mcsa(cfg: Any) -> list[Structure]:
    """Load the curated single-chain M-CSA set as structure dicts.

    Residue order and Cα coordinates come straight from the cached feature
    file (not re-derived from PDB files). The cache only holds the Cα atom; a
    residue is valid when its cached Cα coordinate is finite
    (``isfinite(ca.sum(-1))``), not by a full 4-atom backbone check. To reuse
    ``bioblobs.data.structure.to_graph`` unchanged, Cα is replicated across
    all 4 backbone slots: ``to_graph``'s ``x`` (the "CA" slot) and ``mask``
    (``isfinite`` of the same 3 numbers, summed 4 times) then equal the
    cached Cα and the Cα-only validity check exactly; N/C/O are not real atoms.

    Args:
        cfg: Config with ``feat_cache`` (path to ``feat_cache.pt`` or
            ``feat_cache_maxlen1500.pt``; required — see the module docstring),
            and optional ``labels`` (default: the shipped
            ``data/mcsa/mcsa_single_chain.json``) and ``heldout_ids`` (default:
            the shipped ``data/mcsa/heldout_ids.txt``).

    Returns:
        One structure dict per cached (pdb_id, chain), each with ``name``
        (``"{pdb_id}_{chain}"``), ``seq`` (``"X"`` padding; only its length is
        meaningful), ``coords`` ``[N][4][3]``, ``resnum``, ``residue_target``
        (0.0/1.0 catalytic labels), ``held_out`` (bool), ``embedding`` (cached
        ESM2 features, for :func:`to_graph`), ``pdb_id``, ``chain_used`` and
        ``ec_number``.
    """
    feat_cache_path = _get(cfg, "feat_cache", None)
    if feat_cache_path is None:
        raise ValueError(
            "cfg.feat_cache is required: path to feat_cache.pt or "
            "feat_cache_maxlen1500.pt (see bioblobs/data/mcsa.py and "
            "preprocessing/mcsa/README.md)."
        )
    labels_path = Path(_get(cfg, "labels", DEFAULT_LABELS))
    heldout_path = Path(_get(cfg, "heldout_ids", DEFAULT_HELDOUT_IDS))

    entries = _load_curated_entries(labels_path)
    heldout = _read_id_list(heldout_path)
    cache: dict[str, dict] = torch.load(
        Path(feat_cache_path), map_location="cpu", weights_only=False
    )
    logger.info(
        "M-CSA feature cache {}: {:,} chains; {:,} curated entries",
        feat_cache_path,
        len(cache),
        len(entries),
    )

    structures: list[Structure] = []
    skipped = 0
    for key, entry in entries.items():
        feat = cache.get(key)
        if feat is None:
            skipped += 1
            continue
        ca = feat["ca"].float()
        resnums = [int(r) for r in feat["resnums"]]
        n = ca.shape[0]

        cat_idx = catalytic_indices(entry, resnums)
        residue_target = [0.0] * n
        for i in cat_idx:
            residue_target[i] = 1.0

        structures.append(
            {
                "name": key,
                "seq": "X" * n,
                "coords": ca.unsqueeze(1).expand(n, len(BACKBONE), 3).tolist(),
                "resnum": resnums,
                "residue_target": residue_target,
                "held_out": key in heldout,
                "embedding": feat["nf"].float(),
                "pdb_id": entry["pdb_id"],
                "chain_used": entry["chain_used"],
                "ec_number": entry.get("ec_number", ""),
            }
        )
    if skipped:
        logger.info("{:,} curated entries missing from the feature cache (skipped)", skipped)
    n_held = sum(s["held_out"] for s in structures)
    logger.info("M-CSA: {:,} structures loaded ({:,} held-out)", len(structures), n_held)
    return structures


def to_graph(structure: Structure):
    """Build the PyG ``Data`` for one M-CSA structure loaded by :func:`load_mcsa`.

    Thin wrapper around ``bioblobs.data.structure.to_graph`` using the ESM2
    features attached by :func:`load_mcsa` under ``structure["embedding"]``.
    """
    return _to_graph(structure, structure["embedding"])

#!/usr/bin/env python
"""Download and prepare one VenusX site-fragment target (Act/BindI/Evo/Motif/Dom).

Reproduces, for one ``(target, split_strategy)`` pair, the directory ``bioblobs.data.venusx``
reads (see that module's docstring for the full layout):

1. Downloads ``AI4Protein/VenusX_Res_<target>_<split_strategy>`` (a HF ``DatasetDict`` with
   ``train``/``validation``/``test`` splits) and normalizes every row into one sample keyed by
   ``"<protein_uid>__<interpro_id>"``, with a dense, zero-based InterPro class index.
2. Stages one AlphaFold2 PDB per unique protein from ``AI4Protein/VenusX_<target>_AlphaFold2_PDB``
   (``preprocessing/venusx_pdb.py``).
3. Writes ``samples.csv``, ``token_maps.json``, ``{train,val,test}_split.csv`` and
   ``metadata.json``.
4. Parses every staged PDB's backbone geometry into ``cache/uid_structures_min_completion_*.pt``
   (``bioblobs.data.structure.load_structures``), the same per-protein cache ``load_split`` reads.

The residue-embedding caches (``esm2_static_cache_mmap``, ``saprot_static_cache_mmap``) are built
separately by ``preprocessing/prepare_plm_cache.py`` once this has run.

Already-complete output is left untouched unless ``--force`` is passed (no network access then).

Usage:
    .venv/bin/python -B preprocessing/prepare_venusx.py --target Act --split-strategy MF50 \\
        --data-dir data
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from loguru import logger
from omegaconf import OmegaConf

# The repo isn't installed as a package (pyproject.toml: package = false), so `bioblobs` is
# only importable once the repo root is on sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bioblobs.data.structure import load_structures  # noqa: E402
from bioblobs.data.venusx import SPLIT_STRATEGIES, TARGETS, dataset_root  # noqa: E402
from preprocessing.venusx_pdb import stage_pdb_files  # noqa: E402

HF_TO_LOCAL_SPLIT = {"train": "train", "validation": "val", "test": "test"}
REQUIRED_COLUMNS = (
    "uid", "interpro_id", "seq_full", "seq_fragment", "start", "end", "label", "interpro_label",
)  # fmt: skip


def parse_args() -> argparse.Namespace:
    """CLI arguments for preparing one VenusX ``(target, split_strategy)`` pair."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--target", required=True, choices=TARGETS)
    p.add_argument("--split-strategy", required=True, choices=SPLIT_STRATEGIES)
    p.add_argument("--data-dir", required=True, type=Path, help="Root data directory.")
    p.add_argument("--dataset-name", default="venusx_site_fragment")
    p.add_argument(
        "--source-repo-id", default=None, help="Default: AI4Protein/VenusX_Res_<target>_<split>."
    )
    p.add_argument(
        "--pdb-repo-id", default=None, help="Default: AI4Protein/VenusX_<target>_AlphaFold2_PDB."
    )
    p.add_argument("--hf-cache-dir", type=Path, default=None, help="Default: <data-dir>/hf_cache.")
    p.add_argument("--min-backbone-completion", type=float, default=0.95)
    p.add_argument("--num-workers", type=int, default=4, help="PDB-parsing worker processes.")
    p.add_argument("--limit", type=int, default=None, help="Cap rows per HF split (smoke testing).")
    p.add_argument("--force", action="store_true", help="Re-download and rebuild even if complete.")
    return p.parse_args()


def resolve_source_repo_id(target: str, split_strategy: str, explicit: str | None) -> str:
    """HF repo holding the per-sample rows, unless ``explicit`` overrides it."""
    return explicit or f"AI4Protein/VenusX_Res_{target}_{split_strategy}"


def resolve_pdb_repo_id(target: str, explicit: str | None) -> str:
    """HF repo holding the AlphaFold2 structures, unless ``explicit`` overrides it."""
    return explicit or f"AI4Protein/VenusX_{target}_AlphaFold2_PDB"


def _parse_fragment_positions(field_name: str, value: Any) -> list[int]:
    if isinstance(value, int):
        return [value]
    if isinstance(value, float) and value.is_integer():
        return [int(value)]
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string or integer, received {value!r}")
    parts = [part.strip() for part in value.split("|") if part.strip()]
    if not parts:
        raise ValueError(f"{field_name} cannot be empty")
    try:
        return [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be '|'-separated integers, received {value!r}"
        ) from exc


def _parse_residue_label(raw_value: Any) -> list[int]:
    """Binary site annotation as ``list[int]``, from HF's native list or a JSON/``repr`` string."""
    parsed = raw_value
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if not stripped:
            raise ValueError("label cannot be empty")
        parsed = None
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(stripped)
                break
            except (SyntaxError, ValueError, TypeError, json.JSONDecodeError):
                continue
        if parsed is None:
            raise ValueError(f"Unable to parse residue label payload: {raw_value!r}")
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"Residue label must be list-like, received {type(parsed)!r}")
    residue_label = []
    for index, item in enumerate(parsed):
        if isinstance(item, bool):
            value = int(item)
        elif isinstance(item, int):
            value = item
        elif isinstance(item, float) and item.is_integer():
            value = int(item)
        else:
            raise ValueError(f"Residue label index {index} is not binary-compatible: {item!r}")
        if value not in (0, 1):
            raise ValueError(f"Residue label index {index} must be 0 or 1, received {value}")
        residue_label.append(value)
    return residue_label


def _normalize_row(
    raw_row: dict[str, Any], *, local_split: str, target: str, split_strategy: str,
    source_repo_id: str, pdb_repo_id: str,
) -> dict[str, Any]:  # fmt: skip
    """One validated, flattened sample row from a raw HF VenusX record."""
    missing = [c for c in REQUIRED_COLUMNS if c not in raw_row]
    if missing:
        raise ValueError(f"VenusX row is missing required columns: {missing}")
    protein_uid = str(raw_row["uid"]).strip()
    interpro_id = str(raw_row["interpro_id"]).strip()
    seq_full = str(raw_row["seq_full"]).strip()
    seq_fragment = str(raw_row["seq_fragment"]).strip()
    if not (protein_uid and interpro_id and seq_full and seq_fragment):
        raise ValueError(f"Empty uid/interpro_id/seq_full/seq_fragment for uid={protein_uid!r}")

    starts = _parse_fragment_positions("start", raw_row["start"])
    ends = _parse_fragment_positions("end", raw_row["end"])
    fragments = [f.strip() for f in seq_fragment.split("|") if f.strip()]
    if not (len(starts) == len(ends) == len(fragments)):
        raise ValueError(f"start/end/seq_fragment segment counts disagree for uid={protein_uid}")
    for start, end, fragment in zip(starts, ends, fragments, strict=True):
        if end < start or len(fragment) != end - start + 1:
            raise ValueError(f"Fragment span/length mismatch for uid={protein_uid}, {interpro_id}")

    interpro_label = int(raw_row["interpro_label"])
    if interpro_label < 0:
        raise ValueError(f"interpro_label must be non-negative for uid={protein_uid}")
    residue_label = _parse_residue_label(raw_row["label"])
    if len(seq_full) != len(residue_label):
        raise ValueError(
            f"seq_full/residue-label length mismatch for uid={protein_uid}, {interpro_id}"
        )

    return {
        "sample_id": f"{protein_uid}__{interpro_id}", "protein_uid": protein_uid,
        "split": local_split, "target": target, "split_strategy": split_strategy,
        "source_repo_id": source_repo_id, "pdb_repo_id": pdb_repo_id, "interpro_id": interpro_id,
        "interpro_label": interpro_label, "source_interpro_label": interpro_label,
        "seq_full": seq_full, "seq_fragment": seq_fragment,
        "fragment_start": "|".join(map(str, starts)), "fragment_end": "|".join(map(str, ends)),
        "residue_label": residue_label,
    }  # fmt: skip


def load_and_normalize_rows(
    args: argparse.Namespace, source_repo_id: str, pdb_repo_id: str
) -> list[dict]:
    """Download the HF source repo and normalize every row of every split."""
    from datasets import load_dataset

    dataset_dict = load_dataset(source_repo_id, cache_dir=str(args.hf_cache_dir))
    missing_splits = [s for s in HF_TO_LOCAL_SPLIT if s not in dataset_dict]
    if missing_splits:
        raise ValueError(f"{source_repo_id!r} is missing required splits: {missing_splits}")

    rows, skipped, examples = [], 0, []
    for hf_split, local_split in HF_TO_LOCAL_SPLIT.items():
        split = dataset_dict[hf_split]
        if args.limit is not None:
            split = split.select(range(min(len(split), args.limit)))
        for raw_row in split:
            try:
                rows.append(_normalize_row(
                    dict(raw_row), local_split=local_split, target=args.target,
                    split_strategy=args.split_strategy, source_repo_id=source_repo_id,
                    pdb_repo_id=pdb_repo_id,
                ))  # fmt: skip
            except ValueError as exc:
                skipped += 1
                if len(examples) < 5:
                    examples.append(str(exc))
    if skipped:
        logger.warning("Skipped {} malformed VenusX rows. Examples: {}", skipped, examples)
    return rows


def validate_interpro_mapping(rows: list[dict]) -> tuple[dict[str, dict[str, int]], dict[str, Any]]:
    """Dense, zero-based ``interpro_id -> class index`` map, sorted by the source label.

    Rewrites each row's ``interpro_label`` in place to the dense index.
    """
    interpro_to_source: dict[str, int] = {}
    source_to_interpro: dict[int, str] = {}
    for row in rows:
        interpro_id, source_label = str(row["interpro_id"]), int(row["source_interpro_label"])
        if interpro_to_source.setdefault(interpro_id, source_label) != source_label:
            raise ValueError(f"interpro_id {interpro_id!r} maps to multiple class indices")
        if source_to_interpro.setdefault(source_label, interpro_id) != interpro_id:
            raise ValueError(f"interpro_label {source_label} maps to multiple interpro_id values")

    ordered = sorted(interpro_to_source.items(), key=lambda item: item[1])
    dense_map = {interpro_id: dense for dense, (interpro_id, _) in enumerate(ordered)}
    for row in rows:
        row["interpro_label"] = dense_map[str(row["interpro_id"])]

    source_labels = sorted(source_to_interpro)
    dense_zero_based = source_labels == list(range(len(source_labels)))
    summary = {
        "source_interpro_labels_are_dense_zero_based": dense_zero_based,
        "source_interpro_label_count": len(source_labels),
        "source_interpro_label_min": source_labels[0] if source_labels else None,
        "source_interpro_label_max": source_labels[-1] if source_labels else None,
    }
    return {"1": dense_map}, summary


def check_sample_ids_unique(rows: list[dict]) -> None:
    """Raise if any ``sample_id`` repeats (a protein annotated twice for the same InterPro id)."""
    seen: set[str] = set()
    duplicates = []
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in seen:
            duplicates.append(sample_id)
        seen.add(sample_id)
    if duplicates:
        raise ValueError(f"Duplicate VenusX sample_id values: {duplicates[:5]}")


def write_split_manifests(root: Path, rows: list[dict]) -> None:
    """Write ``{train,val,test}_split.csv``, each one ``sample_id`` column, sorted."""
    for split_name in ("train", "val", "test"):
        ids = sorted(str(r["sample_id"]) for r in rows if str(r["split"]) == split_name)
        pd.DataFrame({"sample_id": ids}).to_csv(root / f"{split_name}_split.csv", index=False)


def _is_prepared(root: Path, source_repo_id: str, pdb_repo_id: str) -> bool:
    """Whether ``root`` already holds a complete, matching preparation."""
    manifests = [root / f"{s}_split.csv" for s in ("train", "val", "test")]
    required = (root / "samples.csv", root / "token_maps.json", root / "metadata.json", *manifests)
    if not all(p.exists() for p in required):
        return False
    try:
        metadata = json.loads((root / "metadata.json").read_text())
        samples = pd.read_csv(root / "samples.csv", usecols=["protein_uid"])
    except (json.JSONDecodeError, ValueError):
        return False
    repos_match = (
        metadata.get("source_repo_id") == source_repo_id
        and metadata.get("pdb_repo_id") == pdb_repo_id
    )
    if not repos_match:
        return False
    uids = samples["protein_uid"].astype(str).unique()
    return len(uids) > 0 and all((root / "pdb" / f"{u}.pdb").exists() for u in uids)


def build_geometry_cache(
    root: Path, uids: list[str], min_completion: float, num_workers: int
) -> Path:
    """Parse every staged PDB's backbone into ``cache/uid_structures_min_completion_*.pt``.

    This is the same per-protein cache ``bioblobs.data.venusx.load_split`` reads; building it
    here means loading never has to fall back to parsing PDBs on the fly.
    """
    tag = str(min_completion).replace(".", "p")
    cache_path = root / "cache" / f"uid_structures_min_completion_{tag}.pt"
    tasks = [(uid, root / "pdb" / f"{uid}.pdb", None, min_completion) for uid in uids]
    structures = load_structures(tasks, num_workers=num_workers, desc="venusx-geometry")
    geometry = {
        s["name"]: {
            "protein_uid": s["name"],
            "seq": s["seq"],
            "coords": s["coords"],
            "resnum": s["resnum"],
        }
        for s in structures
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(geometry, cache_path)
    logger.info(
        "Cached backbone geometry for {}/{} proteins at {}", len(geometry), len(uids), cache_path
    )
    return cache_path


def prepare(args: argparse.Namespace) -> Path:
    """Run the full VenusX preparation pipeline for one ``(target, split_strategy)`` pair."""
    ds_cfg = OmegaConf.create({
        "data_dir": str(args.data_dir), "dataset_name": args.dataset_name,
        "target": args.target, "split_strategy": args.split_strategy,
    })  # fmt: skip
    root = dataset_root(ds_cfg)
    source_repo_id = resolve_source_repo_id(args.target, args.split_strategy, args.source_repo_id)
    pdb_repo_id = resolve_pdb_repo_id(args.target, args.pdb_repo_id)

    if not args.force and _is_prepared(root, source_repo_id, pdb_repo_id):
        logger.info("VenusX {}/{} already prepared at {}", args.target, args.split_strategy, root)
        return root

    logger.warning("Preparing VenusX {}/{} at {}", args.target, args.split_strategy, root)
    (root / "pdb").mkdir(parents=True, exist_ok=True)
    hf_cache_dir = args.hf_cache_dir or (args.data_dir / "hf_cache")

    rows = sorted(
        load_and_normalize_rows(args, source_repo_id, pdb_repo_id),
        key=lambda r: (str(r["split"]), str(r["sample_id"])),
    )
    check_sample_ids_unique(rows)
    token_maps, mapping_summary = validate_interpro_mapping(rows)

    required_uids = sorted({str(r["protein_uid"]) for r in rows})
    staged = stage_pdb_files(required_uids, pdb_repo_id, root / "pdb", hf_cache_dir)

    encoded = [{**r, "residue_label": json.dumps(r["residue_label"])} for r in rows]
    pd.DataFrame(encoded).to_csv(root / "samples.csv", index=False)
    write_split_manifests(root, rows)
    (root / "token_maps.json").write_text(json.dumps(token_maps, indent=2, sort_keys=True))

    split_names = ("train", "val", "test")
    split_sizes = {s: sum(1 for r in rows if str(r["split"]) == s) for s in split_names}
    metadata = {
        "dataset_name": args.dataset_name, "target": args.target,
        "split_strategy": args.split_strategy, "split_sizes": split_sizes,
        "source_repo_id": source_repo_id, "pdb_repo_id": pdb_repo_id,
        "class_count": len(token_maps["1"]),
        "validation_summary": {
            "row_count": len(rows), "unique_proteins": len(required_uids),
            "unique_interpro_ids": len(token_maps["1"]), **mapping_summary,
        },
        "download_summary": {"staged_pdb_count": staged},
    }  # fmt: skip
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    build_geometry_cache(root, required_uids, args.min_backbone_completion, args.num_workers)
    logger.success("Prepared VenusX {}/{} at {}", args.target, args.split_strategy, root)
    return root


def main() -> None:
    prepare(parse_args())


if __name__ == "__main__":
    main()

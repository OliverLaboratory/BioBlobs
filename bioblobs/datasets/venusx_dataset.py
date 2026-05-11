"""Preparation and loading helpers for the VenusX site-fragment dataset."""

from __future__ import annotations

import ast
import fcntl
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile

import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from .parallel import _process_single_pdb


VENUSX_TARGETS = ("Act", "BindI", "Evo", "Motif", "Dom")
VENUSX_SPLIT_STRATEGIES = ("MF50", "MF70", "MF90")
HF_TO_LOCAL_SPLIT = {
    "train": "train",
    "validation": "val",
    "test": "test",
}
REQUIRED_COLUMNS = (
    "uid",
    "interpro_id",
    "seq_full",
    "seq_fragment",
    "start",
    "end",
    "label",
    "interpro_label",
)
TEST_MODE_PREPARE_LIMIT = 100
TEST_MODE_LOAD_LIMIT = 10


def get_prepared_dataset_root(ds_cfg) -> Path:
    return Path(ds_cfg.data_dir) / ds_cfg.dataset_name / ds_cfg.target / ds_cfg.split_strategy


def resolve_source_repo_id(ds_cfg) -> str:
    explicit_repo = ds_cfg.get("source_repo_id")
    if explicit_repo:
        return str(explicit_repo)
    return f"AI4Protein/VenusX_Res_{ds_cfg.target}_{ds_cfg.split_strategy}"


def resolve_pdb_repo_id(ds_cfg) -> str:
    explicit_repo = ds_cfg.get("pdb_repo_id")
    if explicit_repo:
        return str(explicit_repo)
    return f"AI4Protein/VenusX_{ds_cfg.target}_AlphaFold2_PDB"


def _resolve_hf_cache_dir(ds_cfg) -> Path:
    configured = ds_cfg.get("hf_cache_dir")
    if configured:
        return Path(configured)
    return Path(ds_cfg.data_dir) / "hf_cache"


def _load_hf_dataset(repo_id: str, *, cache_dir: str | None, download_mode: str | None):
    from datasets import load_dataset

    kwargs: dict[str, object] = {}
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    if download_mode is not None:
        kwargs["download_mode"] = download_mode
    return load_dataset(repo_id, **kwargs)


def _list_repo_files(repo_id: str, *, cache_dir: str | None):
    del cache_dir
    from huggingface_hub import list_repo_files

    return list_repo_files(repo_id=repo_id, repo_type="dataset")


def _hf_hub_download(repo_id: str, *, filename: str, cache_dir: str | None):
    from huggingface_hub import hf_hub_download

    kwargs: dict[str, object] = {
        "repo_id": repo_id,
        "filename": filename,
        "repo_type": "dataset",
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    return hf_hub_download(**kwargs)


def _num_workers_from_cfg(ds_cfg) -> int:
    preprocessing = ds_cfg.get("preprocessing", {})
    if hasattr(preprocessing, "get"):
        return int(preprocessing.get("num_workers", 4))
    return int(getattr(preprocessing, "num_workers", 4))


def _min_completion_from_cfg(ds_cfg) -> float:
    preprocessing = ds_cfg.get("preprocessing", {})
    if hasattr(preprocessing, "get"):
        return float(preprocessing.get("min_backbone_completion", 0.95))
    return float(getattr(preprocessing, "min_backbone_completion", 0.95))


def _strict_alignment_from_cfg(ds_cfg) -> bool:
    preprocessing = ds_cfg.get("preprocessing", {})
    if hasattr(preprocessing, "get"):
        return bool(preprocessing.get("strict_alignment", False))
    return bool(getattr(preprocessing, "strict_alignment", False))


def _validate_target_and_split(ds_cfg) -> None:
    if ds_cfg.target not in VENUSX_TARGETS:
        raise ValueError(
            f"Unsupported VenusX target {ds_cfg.target!r}. "
            f"Expected one of: {', '.join(VENUSX_TARGETS)}"
        )
    if ds_cfg.split_strategy not in VENUSX_SPLIT_STRATEGIES:
        raise ValueError(
            f"Unsupported VenusX split strategy {ds_cfg.split_strategy!r}. "
            f"Expected one of: {', '.join(VENUSX_SPLIT_STRATEGIES)}"
        )


def _parse_int_field(field_name: str, value) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer, received {value!r}") from exc
    return parsed


def _parse_fragment_positions(field_name: str, value) -> list[int]:
    if isinstance(value, int):
        return [value]

    if isinstance(value, float) and value.is_integer():
        return [int(value)]

    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string or integer, received {value!r}")

    parts = [part.strip() for part in value.split("|") if part.strip()]
    if not parts:
        raise ValueError(f"{field_name} cannot be empty")

    positions: list[int] = []
    for part in parts:
        try:
            positions.append(int(part))
        except ValueError as exc:
            raise ValueError(
                f"{field_name} must contain integer positions separated by '|', received {value!r}"
            ) from exc
    return positions


def _parse_residue_label(raw_value) -> list[int]:
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

    residue_label: list[int] = []
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
    raw_row: dict[str, object],
    *,
    local_split: str,
    target: str,
    split_strategy: str,
    source_repo_id: str,
    pdb_repo_id: str,
) -> dict[str, object]:
    missing = [column for column in REQUIRED_COLUMNS if column not in raw_row]
    if missing:
        raise ValueError(f"VenusX row is missing required columns: {missing}")

    protein_uid = str(raw_row["uid"]).strip()
    interpro_id = str(raw_row["interpro_id"]).strip()
    seq_full = str(raw_row["seq_full"]).strip()
    seq_fragment = str(raw_row["seq_fragment"]).strip()
    if not protein_uid:
        raise ValueError("uid cannot be empty")
    if not interpro_id:
        raise ValueError("interpro_id cannot be empty")
    if not seq_full:
        raise ValueError(f"seq_full cannot be empty for uid={protein_uid}")
    if not seq_fragment:
        raise ValueError(f"seq_fragment cannot be empty for uid={protein_uid}")

    fragment_starts = _parse_fragment_positions("start", raw_row["start"])
    fragment_ends = _parse_fragment_positions("end", raw_row["end"])
    fragment_sequences = [fragment.strip() for fragment in seq_fragment.split("|") if fragment.strip()]

    if len(fragment_starts) != len(fragment_ends):
        raise ValueError(
            f"start and end must contain the same number of segments for "
            f"uid={protein_uid}, interpro_id={interpro_id}"
        )
    if len(fragment_sequences) != len(fragment_starts):
        raise ValueError(
            f"seq_fragment segment count does not match start/end for "
            f"uid={protein_uid}, interpro_id={interpro_id}"
        )

    for start_pos, end_pos, fragment_seq in zip(fragment_starts, fragment_ends, fragment_sequences):
        if end_pos < start_pos:
            raise ValueError(
                f"Fragment end must be >= start for uid={protein_uid}, interpro_id={interpro_id}"
            )
        expected_length = (end_pos - start_pos) + 1
        if len(fragment_seq) != expected_length:
            raise ValueError(
                f"Fragment sequence length {len(fragment_seq)} does not match "
                f"start/end span {expected_length} for uid={protein_uid}, interpro_id={interpro_id}"
            )

    interpro_label = _parse_int_field("interpro_label", raw_row["interpro_label"])
    if interpro_label < 0:
        raise ValueError(
            f"interpro_label must be non-negative for uid={protein_uid}, interpro_id={interpro_id}"
        )

    residue_label = _parse_residue_label(raw_row["label"])
    if len(seq_full) != len(residue_label):
        raise ValueError(
            f"seq_full length {len(seq_full)} does not match residue label length "
            f"{len(residue_label)} for uid={protein_uid}, interpro_id={interpro_id}"
        )

    sample_id = f"{protein_uid}__{interpro_id}"
    return {
        "sample_id": sample_id,
        "protein_uid": protein_uid,
        "split": local_split,
        "target": target,
        "split_strategy": split_strategy,
        "source_repo_id": source_repo_id,
        "pdb_repo_id": pdb_repo_id,
        "interpro_id": interpro_id,
        "interpro_label": interpro_label,
        "source_interpro_label": interpro_label,
        "seq_full": seq_full,
        "seq_fragment": seq_fragment,
        "fragment_start": "|".join(str(position) for position in fragment_starts),
        "fragment_end": "|".join(str(position) for position in fragment_ends),
        "residue_label": residue_label,
    }


def _iter_split_rows(dataset_split, *, test_mode: bool) -> Iterable[dict[str, object]]:
    if test_mode:
        limit = min(len(dataset_split), TEST_MODE_PREPARE_LIMIT)
        if hasattr(dataset_split, "select"):
            dataset_split = dataset_split.select(range(limit))
        else:
            dataset_split = list(dataset_split)[:limit]
    return dataset_split


def _load_and_normalize_rows(ds_cfg) -> list[dict[str, object]]:
    source_repo_id = resolve_source_repo_id(ds_cfg)
    dataset_dict = _load_hf_dataset(
        source_repo_id,
        cache_dir=str(_resolve_hf_cache_dir(ds_cfg)),
        download_mode=ds_cfg.get("download_mode"),
    )

    missing_splits = [split_name for split_name in HF_TO_LOCAL_SPLIT if split_name not in dataset_dict]
    if missing_splits:
        raise ValueError(
            f"VenusX dataset {source_repo_id!r} is missing required splits: {missing_splits}"
        )

    rows: list[dict[str, object]] = []
    skipped = 0
    skip_examples: list[str] = []
    for hf_split, local_split in HF_TO_LOCAL_SPLIT.items():
        for raw_row in _iter_split_rows(dataset_dict[hf_split], test_mode=bool(ds_cfg.get("test_mode", False))):
            try:
                rows.append(
                    _normalize_row(
                        dict(raw_row),
                        local_split=local_split,
                        target=ds_cfg.target,
                        split_strategy=ds_cfg.split_strategy,
                        source_repo_id=source_repo_id,
                        pdb_repo_id=resolve_pdb_repo_id(ds_cfg),
                    )
                )
            except ValueError as exc:
                skipped += 1
                if len(skip_examples) < 5:
                    skip_examples.append(str(exc))
    if skipped > 0:
        logger.warning(
            "VenusX target {}: skipped {} malformed rows during normalization. Examples: {}",
            ds_cfg.target, skipped, skip_examples,
        )
    return rows


def _validate_interpro_mapping(
    rows: list[dict[str, object]]
) -> tuple[dict[str, dict[str, int]], dict[str, object]]:
    interpro_to_source_label: dict[str, int] = {}
    source_label_to_interpro: dict[int, str] = {}
    source_labels: set[int] = set()

    for row in rows:
        interpro_id = str(row["interpro_id"])
        source_interpro_label = int(row["source_interpro_label"])
        source_labels.add(source_interpro_label)

        existing_label = interpro_to_source_label.get(interpro_id)
        if existing_label is not None and existing_label != source_interpro_label:
            raise ValueError(
                f"interpro_id {interpro_id!r} maps to multiple class indices: "
                f"{existing_label} and {source_interpro_label}"
            )
        interpro_to_source_label[interpro_id] = source_interpro_label

        existing_interpro = source_label_to_interpro.get(source_interpro_label)
        if existing_interpro is not None and existing_interpro != interpro_id:
            raise ValueError(
                f"interpro_label {source_interpro_label} maps to multiple interpro_id values: "
                f"{existing_interpro!r} and {interpro_id!r}"
            )
        source_label_to_interpro[source_interpro_label] = interpro_id

    sorted_interpro_items = sorted(
        interpro_to_source_label.items(),
        key=lambda item: item[1],
    )
    dense_token_map = {
        interpro_id: dense_label
        for dense_label, (interpro_id, _source_label) in enumerate(sorted_interpro_items)
    }

    for row in rows:
        row["interpro_label"] = dense_token_map[str(row["interpro_id"])]

    sorted_source_labels = sorted(source_labels)
    expected_dense_source_labels = list(range(len(sorted_source_labels)))
    return {
        "1": dense_token_map
    }, {
        "source_interpro_labels_are_dense_zero_based": (
            sorted_source_labels == expected_dense_source_labels
        ),
        "source_interpro_label_min": sorted_source_labels[0] if sorted_source_labels else None,
        "source_interpro_label_max": sorted_source_labels[-1] if sorted_source_labels else None,
        "source_interpro_label_count": len(sorted_source_labels),
    }


def _encode_rows_for_csv(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    encoded_rows: list[dict[str, object]] = []
    for row in rows:
        encoded = dict(row)
        encoded["residue_label"] = json.dumps(row["residue_label"])
        encoded_rows.append(encoded)
    return encoded_rows


def _check_sample_ids_unique(rows: list[dict[str, object]]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in seen:
            duplicates.append(sample_id)
        seen.add(sample_id)
    if duplicates:
        raise ValueError(f"Duplicate VenusX sample_id values found: {duplicates[:5]}")


def _split_manifest_paths(base_path: Path) -> dict[str, Path]:
    return {
        split_name: base_path / f"{split_name}_split.csv"
        for split_name in ("train", "val", "test")
    }


def _write_split_manifests(base_path: Path, rows: list[dict[str, object]]) -> None:
    for split_name, manifest_path in _split_manifest_paths(base_path).items():
        sample_ids = sorted(
            str(row["sample_id"]) for row in rows if str(row["split"]) == split_name
        )
        pd.DataFrame({"sample_id": sample_ids}).to_csv(manifest_path, index=False)


def _build_remote_pdb_map(repo_files: Iterable[str]) -> tuple[dict[str, str], dict[str, str]]:
    exact_map: dict[str, str] = {}
    lower_map: dict[str, str] = {}

    for repo_path in repo_files:
        if not repo_path.lower().endswith(".pdb"):
            continue

        stem = Path(repo_path).stem
        if stem in exact_map and exact_map[stem] != repo_path:
            raise ValueError(f"Duplicate PDB basename in VenusX repo: {stem!r}")
        exact_map[stem] = repo_path

        lower_stem = stem.lower()
        if lower_stem in lower_map and lower_map[lower_stem] != repo_path:
            raise ValueError(f"Case-insensitive duplicate PDB basename in VenusX repo: {stem!r}")
        lower_map[lower_stem] = repo_path

    if not exact_map:
        raise ValueError("No .pdb files found in the VenusX AlphaFold2 repo")

    return exact_map, lower_map


def _resolve_remote_pdb_path(
    uid: str,
    *,
    exact_map: dict[str, str],
    lower_map: dict[str, str],
) -> str:
    if uid in exact_map:
        return exact_map[uid]

    lower_uid = uid.lower()
    if lower_uid in lower_map:
        return lower_map[lower_uid]

    raise FileNotFoundError(f"Missing PDB for VenusX uid {uid!r}")


def _stage_pdb_files(ds_cfg, rows: list[dict[str, object]], pdb_dir: Path) -> int:
    pdb_repo_id = resolve_pdb_repo_id(ds_cfg)
    repo_files = _list_repo_files(
        pdb_repo_id,
        cache_dir=str(_resolve_hf_cache_dir(ds_cfg)),
    )
    required_uids = sorted({str(row["protein_uid"]) for row in rows})
    pdb_repo_files = [path for path in repo_files if path.lower().endswith(".pdb")]
    if pdb_repo_files:
        exact_map, lower_map = _build_remote_pdb_map(pdb_repo_files)

        for uid in tqdm(required_uids, desc="Staging VenusX PDBs"):
            remote_path = _resolve_remote_pdb_path(uid, exact_map=exact_map, lower_map=lower_map)
            cached_path = Path(
                _hf_hub_download(
                    pdb_repo_id,
                    filename=remote_path,
                    cache_dir=str(_resolve_hf_cache_dir(ds_cfg)),
                )
            )
            output_path = pdb_dir / f"{uid}.pdb"
            if not output_path.exists():
                shutil.copy2(cached_path, output_path)
        return len(required_uids)

    repo_name = pdb_repo_id.split("/")[-1]
    archive_name = f"{repo_name}.zip"
    if archive_name not in repo_files:
        raise ValueError(
            f"No direct .pdb files or expected archive {archive_name!r} found in {pdb_repo_id!r}"
        )

    archive_path = Path(
        _hf_hub_download(
            pdb_repo_id,
            filename=archive_name,
            cache_dir=str(_resolve_hf_cache_dir(ds_cfg)),
        )
    )
    with ZipFile(archive_path) as archive:
        member_candidates = _build_archive_member_candidates(archive.namelist())
        for uid in tqdm(required_uids, desc="Extracting VenusX PDBs"):
            output_path = pdb_dir / f"{uid}.pdb"
            if output_path.exists():
                continue
            member_name = _resolve_archive_member(uid, member_candidates)
            with archive.open(member_name) as source_handle, output_path.open("wb") as output_handle:
                shutil.copyfileobj(source_handle, output_handle)

    return len(required_uids)


def _build_archive_member_candidates(member_names: Iterable[str]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    exact_map: dict[str, list[str]] = {}
    lower_map: dict[str, list[str]] = {}

    for member_name in member_names:
        if not member_name.lower().endswith(".pdb"):
            continue

        stem = Path(member_name).stem
        uid = stem.rsplit("_", 1)[-1]
        exact_map.setdefault(uid, []).append(member_name)
        lower_map.setdefault(uid.lower(), []).append(member_name)

    if not exact_map:
        raise ValueError("No .pdb members found inside the VenusX AlphaFold2 archive")
    return exact_map, lower_map


def _resolve_archive_member(
    uid: str,
    member_candidates: tuple[dict[str, list[str]], dict[str, list[str]]],
) -> str:
    exact_map, lower_map = member_candidates
    candidates = exact_map.get(uid) or lower_map.get(uid.lower())
    if not candidates:
        raise FileNotFoundError(f"Missing archived PDB for VenusX uid {uid!r}")
    return sorted(candidates)[0]


def _metadata_matches(path: Path, *, expected_source_repo: str, expected_pdb_repo: str) -> bool:
    if not path.exists():
        return False
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return (
        metadata.get("source_repo_id") == expected_source_repo
        and metadata.get("pdb_repo_id") == expected_pdb_repo
    )


def _prepared_dataset_is_complete(
    *,
    base_path: Path,
    samples_path: Path,
    expected_source_repo: str,
    expected_pdb_repo: str,
) -> bool:
    if not (
        samples_path.exists()
        and (base_path / "token_maps.json").exists()
        and all(path.exists() for path in _split_manifest_paths(base_path).values())
        and _metadata_matches(
            base_path / "metadata.json",
            expected_source_repo=expected_source_repo,
            expected_pdb_repo=expected_pdb_repo,
        )
    ):
        return False

    try:
        samples_df = pd.read_csv(samples_path, usecols=["protein_uid"])
    except ValueError:
        return False

    if samples_df.empty:
        return False

    pdb_dir = base_path / "pdb"
    return all((pdb_dir / f"{uid}.pdb").exists() for uid in samples_df["protein_uid"].astype(str).unique())


def ensure_prepared_dataset(cfg) -> Path:
    ds_cfg = cfg.datasets
    _validate_target_and_split(ds_cfg)

    base_path = get_prepared_dataset_root(ds_cfg)
    pdb_dir = base_path / "pdb"
    samples_path = base_path / "samples.csv"
    token_map_path = base_path / "token_maps.json"
    metadata_path = base_path / "metadata.json"

    source_repo_id = resolve_source_repo_id(ds_cfg)
    pdb_repo_id = resolve_pdb_repo_id(ds_cfg)

    if _prepared_dataset_is_complete(
        base_path=base_path,
        samples_path=samples_path,
        expected_source_repo=source_repo_id,
        expected_pdb_repo=pdb_repo_id,
    ):
        logger.info("Found prepared VenusX dataset at {}", base_path)
        return base_path

    logger.warning("Prepared VenusX dataset missing or incomplete at {}. Creating it now...", base_path)
    pdb_dir.mkdir(parents=True, exist_ok=True)
    (base_path / "cache").mkdir(parents=True, exist_ok=True)

    rows = _load_and_normalize_rows(ds_cfg)
    rows = sorted(rows, key=lambda row: (str(row["split"]), str(row["sample_id"])))
    _check_sample_ids_unique(rows)
    token_maps, mapping_summary = _validate_interpro_mapping(rows)
    staged_pdb_count = _stage_pdb_files(ds_cfg, rows, pdb_dir)

    pd.DataFrame(_encode_rows_for_csv(rows)).to_csv(samples_path, index=False)
    _write_split_manifests(base_path, rows)
    token_map_path.write_text(json.dumps(token_maps, indent=2, sort_keys=True), encoding="utf-8")

    split_sizes = {
        split_name: sum(1 for row in rows if str(row["split"]) == split_name)
        for split_name in ("train", "val", "test")
    }
    metadata = {
        "dataset_name": ds_cfg.dataset_name,
        "target": ds_cfg.target,
        "split_strategy": ds_cfg.split_strategy,
        "source_repo_id": source_repo_id,
        "pdb_repo_id": pdb_repo_id,
        "split_sizes": split_sizes,
        "class_count": len(token_maps["1"]),
        "validation_summary": {
            "required_columns": list(REQUIRED_COLUMNS),
            "schema_validated": True,
            "interpro_mapping_validated": True,
            "row_count": len(rows),
            "unique_proteins": len({str(row["protein_uid"]) for row in rows}),
            "unique_interpro_ids": len(token_maps["1"]),
            **mapping_summary,
        },
        "download_summary": {
            "staged_pdb_count": staged_pdb_count,
        },
        "load_summary": {
            "loaded_samples": 0,
            "skipped_structure_samples": 0,
            "skipped_projection_samples": 0,
            "skip_reasons": [],
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    logger.success("Prepared VenusX dataset at {}", base_path)
    return base_path


def _read_samples(samples_path: Path) -> list[dict[str, object]]:
    rows = pd.read_csv(samples_path).to_dict(orient="records")
    normalized_rows: list[dict[str, object]] = []
    for row in rows:
        sample = dict(row)
        sample["interpro_label"] = int(sample["interpro_label"])
        sample["source_interpro_label"] = int(
            sample.get("source_interpro_label", sample["interpro_label"])
        )
        sample["fragment_start"] = str(sample["fragment_start"])
        sample["fragment_end"] = str(sample["fragment_end"])
        sample["residue_label"] = _parse_residue_label(sample["residue_label"])
        normalized_rows.append(sample)
    return normalized_rows


def _uid_structure_cache_path(base_path: Path, *, min_completion: float) -> Path:
    min_completion_tag = str(min_completion).replace(".", "p")
    return base_path / "cache" / f"uid_structures_min_completion_{min_completion_tag}.pt"


def _load_uid_structure_cache(
    base_path: Path,
    *,
    requested_uids: list[str],
    min_completion: float,
    num_workers: int,
    strict_alignment: bool,
) -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    cache_path = _uid_structure_cache_path(base_path, min_completion=min_completion)
    uid_structures: dict[str, dict[str, object]] = {}
    if cache_path.exists():
        uid_structures = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=False,
        )

    missing_uids = [uid for uid in requested_uids if uid not in uid_structures]
    if not missing_uids:
        return uid_structures, {}

    tasks = [
        (uid, base_path / "pdb" / f"{uid}.pdb", {"protein_uid": uid}, min_completion)
        for uid in missing_uids
    ]
    failures: dict[str, str] = {}

    if num_workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_process_single_pdb, task): task[0]
                for task in tasks
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Parsing VenusX PDBs",
            ):
                uid = futures[future]
                structure, error = future.result()
                if structure is None:
                    failures[uid] = error or "unknown error"
                    continue
                uid_structures[uid] = {
                    "protein_uid": uid,
                    "seq": structure["seq"],
                    "coords": structure["coords"],
                    "resnum": structure["resnum"],
                }
    else:
        for task in tqdm(tasks, desc="Parsing VenusX PDBs"):
            uid = task[0]
            structure, error = _process_single_pdb(task)
            if structure is None:
                failures[uid] = error or "unknown error"
                continue
            uid_structures[uid] = {
                "protein_uid": uid,
                "seq": structure["seq"],
                "coords": structure["coords"],
                "resnum": structure["resnum"],
            }

    if failures and strict_alignment:
        first_uid, first_error = next(iter(failures.items()))
        raise ValueError(f"Failed to parse VenusX structure for {first_uid}: {first_error}")

    torch.save(uid_structures, cache_path)
    return uid_structures, failures


def _project_residue_target(
    *,
    protein_uid: str,
    interpro_id: str,
    residue_label: list[int],
    resnum: list[int],
    seq: str,
) -> list[int]:
    residue_target: list[int] = []
    for residue_number in resnum:
        if residue_number < 1 or residue_number > len(residue_label):
            raise ValueError(
                f"Parsed residue index {residue_number} falls outside residue label range "
                f"for uid={protein_uid}, interpro_id={interpro_id}"
            )
        residue_target.append(int(residue_label[residue_number - 1]))

    if len(seq) != len(resnum) or len(seq) != len(residue_target):
        raise ValueError(
            f"Projected residue target length mismatch for uid={protein_uid}, interpro_id={interpro_id}: "
            f"len(seq)={len(seq)}, len(resnum)={len(resnum)}, len(residue_target)={len(residue_target)}"
        )
    return residue_target


def _limit_split_sample_ids(sample_ids: list[str], *, test_mode: bool) -> list[str]:
    if test_mode:
        return sample_ids[:TEST_MODE_LOAD_LIMIT]
    return sample_ids


def _update_metadata_load_summary(base_path: Path, *, loaded: int, skipped_structure: int, skipped_projection: int, reasons: list[str]) -> None:
    """Update metadata.json with the latest load summary.

    Concurrency-safe: holds an exclusive ``flock`` on a sidecar ``.lock`` file
    while doing the read-modify-write, then commits via atomic rename. Multiple
    training processes running concurrently on the same prepared target would
    otherwise truncate each other's metadata.json mid-read (the original bug
    surfaced as ``JSONDecodeError: Expecting value: line 1 column 1``).

    The whole update is best-effort — if locking or rewriting fails for any
    reason it is logged and silently skipped, since the load summary is
    diagnostic-only and not load-bearing for downstream training.
    """
    metadata_path = base_path / "metadata.json"
    lock_path = base_path / "metadata.json.lock"
    try:
        with lock_path.open("a") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata["load_summary"] = {
                    "loaded_samples": loaded,
                    "skipped_structure_samples": skipped_structure,
                    "skipped_projection_samples": skipped_projection,
                    "skip_reasons": reasons[:25],
                }
                tmp_path = metadata_path.with_suffix(".json.tmp")
                tmp_path.write_text(
                    json.dumps(metadata, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                os.replace(tmp_path, metadata_path)  # atomic
            finally:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to update load_summary in {} ({}); continuing.",
            metadata_path, exc,
        )


def load_prepared_structures(cfg) -> tuple[dict[str, list[dict[str, object]]], int]:
    ds_cfg = cfg.datasets
    base_path = ensure_prepared_dataset(cfg)
    samples = _read_samples(base_path / "samples.csv")
    rows_by_sample_id = {str(row["sample_id"]): row for row in samples}

    token_maps = json.loads((base_path / "token_maps.json").read_text(encoding="utf-8"))
    num_classes = len(token_maps["1"])

    split_sample_ids: dict[str, list[str]] = {}
    requested_uids: set[str] = set()
    for split_name, manifest_path in _split_manifest_paths(base_path).items():
        sample_ids = pd.read_csv(manifest_path)["sample_id"].astype(str).tolist()
        sample_ids = _limit_split_sample_ids(sample_ids, test_mode=bool(ds_cfg.get("test_mode", False)))
        split_sample_ids[split_name] = sample_ids
        for sample_id in sample_ids:
            requested_uids.add(str(rows_by_sample_id[sample_id]["protein_uid"]))

    uid_structures, structure_failures = _load_uid_structure_cache(
        base_path,
        requested_uids=sorted(requested_uids),
        min_completion=_min_completion_from_cfg(ds_cfg),
        num_workers=_num_workers_from_cfg(ds_cfg),
        strict_alignment=_strict_alignment_from_cfg(ds_cfg),
    )

    structures: dict[str, list[dict[str, object]]] = {}
    skipped_structure_samples = 0
    skipped_projection_samples = 0
    skip_reasons: list[str] = []
    strict_alignment = _strict_alignment_from_cfg(ds_cfg)

    for split_name in ("train", "val", "test"):
        split_structures: list[dict[str, object]] = []
        for sample_id in split_sample_ids[split_name]:
            row = rows_by_sample_id[sample_id]
            protein_uid = str(row["protein_uid"])
            base_structure = uid_structures.get(protein_uid)

            if base_structure is None:
                skipped_structure_samples += 1
                reason = structure_failures.get(protein_uid, "missing structure cache entry")
                formatted_reason = (
                    f"{sample_id}: skipped because structure for {protein_uid} could not be loaded ({reason})"
                )
                if strict_alignment:
                    raise ValueError(formatted_reason)
                skip_reasons.append(formatted_reason)
                continue

            try:
                residue_target = _project_residue_target(
                    protein_uid=protein_uid,
                    interpro_id=str(row["interpro_id"]),
                    residue_label=list(row["residue_label"]),
                    resnum=list(base_structure["resnum"]),
                    seq=str(base_structure["seq"]),
                )
            except ValueError as exc:
                skipped_projection_samples += 1
                if strict_alignment:
                    raise
                skip_reasons.append(f"{sample_id}: {exc}")
                continue

            split_structures.append(
                {
                    "name": sample_id,
                    "sample_id": sample_id,
                    "protein_uid": protein_uid,
                    "seq": base_structure["seq"],
                    "coords": base_structure["coords"],
                    "resnum": base_structure["resnum"],
                    "label": int(row["interpro_label"]),
                    "source_interpro_label": int(row["source_interpro_label"]),
                    "residue_target": residue_target,
                    "interpro_id": str(row["interpro_id"]),
                    "fragment_start": str(row["fragment_start"]),
                    "fragment_end": str(row["fragment_end"]),
                    "source_split": split_name,
                    "target": ds_cfg.target,
                    "split_strategy": ds_cfg.split_strategy,
                }
            )

        structures[split_name] = split_structures
        logger.info(
            "Loaded VenusX {} split: {} samples", split_name, len(split_structures)
        )

    _update_metadata_load_summary(
        base_path,
        loaded=sum(len(split_items) for split_items in structures.values()),
        skipped_structure=skipped_structure_samples,
        skipped_projection=skipped_projection_samples,
        reasons=skip_reasons,
    )
    return structures, num_classes

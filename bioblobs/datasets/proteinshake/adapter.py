"""Preparation and loading helpers for ProteinShake GO and Pfam tasks."""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from ..parallel import _save_single_protein, load_structures_from_pdb
from .task_registry import GO_ALIAS_TO_BRANCH, GO_BRANCHES, SCOP_LEVELS, build_task_kwargs, get_dataset_spec


def get_prepared_dataset_root(ds_cfg) -> Path:
    go_branch = ds_cfg.get("go_branch") if ds_cfg.dataset_name == "go" else None
    spec = get_dataset_spec(ds_cfg.dataset_name, go_branch)
    return Path(ds_cfg.data_dir) / spec.prepared_root


def _load_task(dataset_name: str, **kwargs):
    from proteinshake.tasks import GeneOntologyTask, ProteinFamilyTask, StructuralClassTask

    task_map = {
        "go": GeneOntologyTask,
        "go_mf": GeneOntologyTask,
        "go_bp": GeneOntologyTask,
        "go_cc": GeneOntologyTask,
        "pfam": ProteinFamilyTask,
        "scop_fam": StructuralClassTask,
        "scop_sf": StructuralClassTask,
    }
    return task_map[dataset_name](**kwargs)


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


def _extract_target_payload(spec, protein, token_map, *, go_branch: str | None = None) -> dict[str, object]:
    protein_info = protein["protein"]

    # GO datasets (original "go" name and aliases)
    if spec.dataset_name in GO_ALIAS_TO_BRANCH:
        branch = GO_ALIAS_TO_BRANCH[spec.dataset_name]
        labels = sorted({int(token_map[label]) for label in protein_info[branch]})
        return {"label_ids": labels}
    if spec.dataset_name == "go":
        branch = go_branch or "molecular_function"
        labels = sorted({int(token_map[label]) for label in protein_info[branch]})
        return {"label_ids": labels}

    # SCOP datasets
    if spec.dataset_name in SCOP_LEVELS:
        scop_level_key = SCOP_LEVELS[spec.dataset_name]
        scop_label = protein_info[scop_level_key]
        return {"label_id": int(token_map[scop_label])}

    # Pfam
    pfam_label = protein_info["Pfam"][0]
    return {"label_id": int(token_map[pfam_label])}


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _save_pdb_files(tasks, num_workers: int) -> None:
    if num_workers > 1 and len(tasks) > 1:
        failures: list[tuple[str, str]] = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_save_single_protein, task): task[2]
                for task in tasks
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Saving PDB files",
            ):
                protein_id = futures[future]
                try:
                    _, success, error = future.result()
                    if not success:
                        failures.append((protein_id, error or "unknown error"))
                except Exception as exc:  # pragma: no cover - defensive
                    failures.append((protein_id, str(exc)))
        if failures:
            logger.warning("{} proteins failed to save", len(failures))
            for protein_id, error in failures[:5]:
                logger.warning("  {}: {}", protein_id, error)
        return

    for task in tqdm(tasks, desc="Saving PDB files"):
        _save_single_protein(task)


def ensure_prepared_dataset(cfg) -> Path:
    ds_cfg = cfg.datasets
    spec = get_dataset_spec(ds_cfg.dataset_name, ds_cfg.get("go_branch"))

    if cfg.tasks.problem_type != spec.problem_type:
        raise ValueError(
            f"Task config problem_type={cfg.tasks.problem_type!r} does not match "
            f"dataset {spec.dataset_name!r} ({spec.problem_type!r})."
        )

    output_dir = get_prepared_dataset_root(ds_cfg)
    pdb_dir = output_dir / "pdb"
    split_dir = output_dir / ds_cfg.split
    targets_path = output_dir / "targets.jsonl"
    token_map_path = output_dir / "token_map.json"
    info_path = output_dir / "dataset_info.json"

    # Check shared data (PDB files, targets, token map) separately from
    # per-split CSVs.  Shared data is independent of the split type and
    # should only be generated once.
    pdb_has_files = pdb_dir.exists() and any(pdb_dir.iterdir())
    shared_ready = (
        pdb_has_files
        and targets_path.exists()
        and token_map_path.exists()
        and info_path.exists()
    )
    split_ready = split_dir.exists() and any(split_dir.glob("*.csv"))

    if shared_ready and split_ready:
        logger.info("Found prepared ProteinShake dataset at {}", output_dir)
        return output_dir

    pdb_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cache").mkdir(parents=True, exist_ok=True)

    task_kwargs = build_task_kwargs(ds_cfg)
    task = _load_task(ds_cfg.dataset_name, **task_kwargs)
    dataset = task.dataset
    token_map = {str(label): int(index) for label, index in task.token_map.items()}
    train_index = set(int(index) for index in task.train_index)
    val_index = set(int(index) for index in task.val_index)
    test_index = set(int(index) for index in task.test_index)

    test_mode = bool(ds_cfg.get("test_mode", False))
    num_workers = _num_workers_from_cfg(ds_cfg)
    dataset_name = ds_cfg.dataset_name
    if dataset_name in GO_ALIAS_TO_BRANCH:
        go_branch = GO_ALIAS_TO_BRANCH[dataset_name]
    else:
        go_branch = ds_cfg.get("go_branch")

    split_assignments: dict[str, str] = {}
    target_rows: list[dict[str, object]] = []
    pdb_tasks = []

    logger.info(
        "Preparing {} dataset (split={}, test_mode={}, shared_ready={})",
        ds_cfg.dataset_name,
        ds_cfg.split,
        test_mode,
        shared_ready,
    )
    for idx, protein in enumerate(tqdm(dataset.proteins(resolution="atom"), desc="Loading proteins")):
        if test_mode and idx >= 100:
            logger.warning("TEST MODE: limiting ProteinShake preparation to 100 proteins")
            break

        protein_id = protein["protein"]["ID"].lower()

        # Track split assignment for this split type.
        if idx in train_index:
            split_assignments[protein_id] = "train"
        elif idx in val_index:
            split_assignments[protein_id] = "val"
        elif idx in test_index:
            split_assignments[protein_id] = "test"

        # Always record targets and PDB tasks for ALL proteins so that
        # targets.jsonl is complete and reusable across split types.
        if not shared_ready:
            target_payload = _extract_target_payload(
                spec,
                protein,
                token_map,
                go_branch=go_branch,
            )
            target_rows.append({"pdb_id": protein_id, **target_payload})
            pdb_tasks.append((protein, pdb_dir / f"{protein_id}.pdb", protein_id))

    # Write shared data only if it wasn't already present.
    if not shared_ready:
        _save_pdb_files(pdb_tasks, num_workers)
        _write_jsonl(targets_path, sorted(target_rows, key=lambda row: row["pdb_id"]))
        with token_map_path.open("w", encoding="utf-8") as handle:
            json.dump(token_map, handle, indent=2, sort_keys=True)
        is_go = dataset_name == "go" or dataset_name in GO_ALIAS_TO_BRANCH
        with info_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "dataset_name": ds_cfg.dataset_name,
                    "go_branch": go_branch if is_go else None,
                    "problem_type": spec.problem_type,
                    "target_format": spec.target_format,
                    "num_classes": len(token_map),
                    "valid_go_branches": list(GO_BRANCHES) if is_go else None,
                    "scop_level": SCOP_LEVELS.get(dataset_name) if dataset_name in SCOP_LEVELS else None,
                },
                handle,
                indent=2,
                sort_keys=True,
            )

    # Always write split CSVs (this is the per-split part).
    for split_name in ("train", "val", "test"):
        split_ids = sorted(
            pdb_id for pdb_id, assignment in split_assignments.items() if assignment == split_name
        )
        pd.DataFrame({"pdb_id": split_ids}).to_csv(
            split_dir / f"{split_name}_split.csv",
            index=False,
        )

    logger.success("Prepared ProteinShake dataset at {}", output_dir)
    return output_dir


def _load_target_map(targets_path: Path) -> dict[str, dict[str, object]]:
    return {row["pdb_id"]: row for row in _read_jsonl(targets_path)}


def _materialize_label(payload: dict[str, object], *, problem_type: str, num_classes: int):
    if problem_type == "multi_class":
        return int(payload["label_id"])

    label = np.zeros(num_classes, dtype=np.float32)
    for index in payload.get("label_ids", []):
        label[int(index)] = 1.0
    return label.tolist()


def load_prepared_structures(cfg) -> tuple[dict[str, list[dict[str, object]]], int]:
    ds_cfg = cfg.datasets
    spec = get_dataset_spec(ds_cfg.dataset_name, ds_cfg.get("go_branch"))
    base_path = ensure_prepared_dataset(cfg)

    pdb_dir = base_path / "pdb"
    split_dir = base_path / ds_cfg.split
    cache_dir = base_path / "cache"
    cache_dir.mkdir(exist_ok=True)

    with (base_path / "token_map.json").open("r", encoding="utf-8") as handle:
        token_map = json.load(handle)
    target_map = _load_target_map(base_path / "targets.jsonl")
    num_classes = len(token_map)

    test_mode = bool(ds_cfg.get("test_mode", False))
    split_similarity_threshold = ds_cfg.get("split_similarity_threshold", 0.7)
    edge_types = ds_cfg.edge_types
    min_completion = _min_completion_from_cfg(ds_cfg)
    num_workers = _num_workers_from_cfg(ds_cfg)

    structures: dict[str, list[dict[str, object]]] = {}
    for split_name in ("train", "val", "test"):
        split_csv = split_dir / f"{split_name}_split.csv"
        pdb_ids = pd.read_csv(split_csv)["pdb_id"].tolist()
        if test_mode:
            pdb_ids = pdb_ids[:10]
            logger.warning("TEST MODE: limiting {} split to {} samples", split_name, len(pdb_ids))

        cache_path = cache_dir / (
            f"{ds_cfg.split}_{split_name}_{edge_types}_{split_similarity_threshold}_cache.pt"
        )
        if cache_path.exists() and not test_mode:
            logger.info("Loading {} split from cache {}", split_name, cache_path)
            start_time = time.time()
            split_structures = torch.load(
                cache_path,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            logger.info(
                "  {}: {} structures loaded from cache in {:.2f}s",
                split_name,
                len(split_structures),
                time.time() - start_time,
            )
        else:
            tasks = []
            missing_pdb = 0
            for pdb_id in pdb_ids:
                pdb_path = pdb_dir / f"{pdb_id}.pdb"
                if not pdb_path.exists():
                    logger.warning("PDB file not found: {}", pdb_path)
                    missing_pdb += 1
                    continue
                tasks.append((pdb_id, pdb_path, target_map[pdb_id], min_completion))

            split_structures, skipped = load_structures_from_pdb(tasks, num_workers, split_name)
            skipped += missing_pdb
            logger.info(
                "{}: {} structures loaded ({} skipped)",
                split_name,
                len(split_structures),
                skipped,
            )
            if not test_mode:
                torch.save(split_structures, cache_path)

        for structure in split_structures:
            structure["label"] = _materialize_label(
                structure["target_payload"],
                problem_type=spec.problem_type,
                num_classes=num_classes,
            )
        structures[split_name] = split_structures

    return structures, num_classes

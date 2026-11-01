"""ProteinShake function-prediction tasks: EC, Pfam, GO-MF, SCOP-FAM and SCOP-SF.

``prepare`` downloads a task with the ``proteinshake`` package and writes, under
``<data_dir>/<task>_proteinshake/``::

    pdb/<id>.pdb                      one structure per protein
    <split>/{train,val,test}_split.csv
    labels.csv + token_maps.json      (EC: full EC numbers and per-level vocabularies)
    targets.jsonl + token_map.json    (other tasks)

``load_structures`` parses the PDB files of one split once and caches them under ``cache/``.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from bioblobs.data.structure import Structure, load_structures, write_pdb


@dataclass(frozen=True)
class Task:
    """ProteinShake task metadata."""

    root: str
    problem_type: str
    task_class: str
    label_key: str


TASKS = {
    "ec": Task("ec_proteinshake", "multi_class", "EnzymeClassTask", "EC"),
    "pfam": Task("pfam_proteinshake", "multi_class", "ProteinFamilyTask", "Pfam"),
    "go_mf": Task(
        "go_molecular_function_proteinshake",
        "multi_label",
        "GeneOntologyTask",
        "molecular_function",
    ),
    "scop_fam": Task("scop_fam_proteinshake", "multi_class", "StructuralClassTask", "SCOP-FA"),
    "scop_sf": Task("scop_sf_proteinshake", "multi_class", "StructuralClassTask", "SCOP-SF"),
}


def dataset_root(ds_cfg: DictConfig) -> Path:
    """Prepared-data directory of a task."""
    return Path(ds_cfg.data_dir) / TASKS[ds_cfg.dataset_name].root


def prepare(ds_cfg: DictConfig) -> Path:
    """Download a task and write PDB files, labels and split lists (skipped if present).

    Args:
        ds_cfg: ``datasets`` config with ``dataset_name``, ``data_dir``, ``split`` and
            ``split_similarity_threshold``.

    Returns:
        The prepared-data directory.
    """
    name, task_meta = ds_cfg.dataset_name, TASKS[ds_cfg.dataset_name]
    root = dataset_root(ds_cfg)
    split_dir = root / ds_cfg.split
    label_file = root / ("labels.csv" if name == "ec" else "targets.jsonl")
    pdb_dir = root / "pdb"
    if label_file.exists() and pdb_dir.is_dir() and any(split_dir.glob("*_split.csv")):
        return root
    import proteinshake.tasks

    kwargs = {
        "split": ds_cfg.split,
        "root": ds_cfg.data_dir,
        "split_similarity_threshold": ds_cfg.get("split_similarity_threshold", 0.7),
    }
    if name == "go_mf":
        kwargs["branch"] = task_meta.label_key
    elif name.startswith("scop"):
        kwargs["scop_level"] = task_meta.label_key
    task = getattr(proteinshake.tasks, task_meta.task_class)(**kwargs)
    splits = {
        int(i): s
        for s, idx in (
            ("train", task.train_index),
            ("val", task.val_index),
            ("test", task.test_index),
        )
        for i in idx
    }
    for directory in (pdb_dir, split_dir, root / "cache"):
        directory.mkdir(parents=True, exist_ok=True)

    labels, assignment, to_write = {}, {}, []
    for idx, protein in enumerate(tqdm(task.dataset.proteins(resolution="atom"), desc=name)):
        pid = protein["protein"]["ID"].lower()
        labels[pid] = protein["protein"][task_meta.label_key]
        if idx in splits:
            assignment[pid] = splits[idx]
        to_write.append((protein, pdb_dir / f"{pid}.pdb"))
    with ProcessPoolExecutor(max_workers=int(ds_cfg.get("num_workers", 8))) as pool:
        list(pool.map(_write_missing, to_write, chunksize=64))

    if name == "ec":
        _write_ec_labels(root, labels)
    else:
        token_map = {str(k): int(v) for k, v in task.token_map.items()}
        (root / "token_map.json").write_text(json.dumps(token_map, indent=2, sort_keys=True))
        with (root / "targets.jsonl").open("w") as handle:
            for pid in sorted(labels):
                handle.write(
                    json.dumps(
                        {"pdb_id": pid, **_target(name, labels[pid], token_map)}, sort_keys=True
                    )
                    + "\n"
                )
    for split in ("train", "val", "test"):
        ids = sorted(pid for pid, s in assignment.items() if s == split)
        pd.DataFrame({"pdb_id": ids}).to_csv(split_dir / f"{split}_split.csv", index=False)
    logger.success("Prepared {} at {}", name, root)
    return root


def _write_missing(item: tuple[dict, Path]) -> None:
    protein, path = item
    if not path.exists():
        write_pdb(protein, path)


def _write_ec_labels(root: Path, labels: dict[str, str]) -> None:
    """EC vocabularies per level: level L classes are the distinct L-th EC components."""
    token_maps = {}
    for level in range(4):
        values = {ec.split(".")[level] for ec in set(labels.values()) if len(ec.split(".")) > level}
        token_maps[level + 1] = {v: i for i, v in enumerate(sorted(values))}
    (root / "token_maps.json").write_text(json.dumps(token_maps, indent=2))
    frame = pd.DataFrame(sorted(labels.items()), columns=["pdb_id", "label"])
    frame.to_csv(root / "labels.csv", index=False)


def _target(name: str, value, token_map: dict[str, int]) -> dict:
    if name == "go_mf":
        return {"label_ids": sorted({int(token_map[v]) for v in value})}
    if name == "pfam":
        value = value[0]
    return {"label_id": int(token_map[value])}


def load_split(cfg: DictConfig, split: str) -> tuple[list[Structure], int]:
    """Load the structures and labels of one split.

    Args:
        cfg: Composed config (``datasets`` and ``tasks`` groups).
        split: ``"train"``, ``"val"`` or ``"test"``.

    Returns:
        Structures with a ``label`` field and the number of classes.
    """
    ds = cfg.datasets
    root = prepare(ds)
    ids = pd.read_csv(root / ds.split / f"{split}_split.csv")["pdb_id"].tolist()
    if ds.get("test_mode", False):
        ids = ids[:10]
    min_completion = float(ds.get("min_backbone_completion", 0.95))
    if ds.dataset_name == "ec":
        level = int(ds.ec_level)
        token_map = json.loads((root / "token_maps.json").read_text())[str(level)]
        targets = dict(pd.read_csv(root / "labels.csv").itertuples(index=False, name=None))
    else:
        token_map = json.loads((root / "token_map.json").read_text())
        rows = [json.loads(line) for line in (root / "targets.jsonl").read_text().splitlines()]
        targets = {row["pdb_id"]: row for row in rows if line_ok(row)}
    num_classes = len(token_map)

    cache = (
        root
        / "cache"
        / f"{ds.split}_{split}_{ds.edge_types}_{ds.split_similarity_threshold}_cache.pt"
    )
    if cache.exists() and not ds.get("test_mode", False):
        structures = torch.load(cache, map_location="cpu", weights_only=False, mmap=True)
    else:
        tasks = [
            (pid, root / "pdb" / f"{pid}.pdb", targets[pid], min_completion)
            for pid in ids
            if (root / "pdb" / f"{pid}.pdb").exists()
        ]
        structures = load_structures(tasks, int(ds.get("num_workers", 4)), desc=split)
        if not ds.get("test_mode", False):
            torch.save(structures, cache)

    if ds.dataset_name == "ec":
        structures = [s for s in structures if len(_ec(s).split(".")) >= level]
        for s in structures:
            s["label"] = token_map[_ec(s).split(".")[level - 1]]
    elif cfg.tasks.problem_type == "multi_label":
        for s in structures:
            label = np.zeros(num_classes, dtype=np.float32)
            label[s["target_payload"]["label_ids"]] = 1.0
            s["label"] = label.tolist()
    else:
        for s in structures:
            s["label"] = int(s["target_payload"]["label_id"])
    return structures, num_classes


def _ec(structure: Structure) -> str:
    return structure.get("target_payload") or structure["ec_string"]


def line_ok(row: dict) -> bool:
    """Whether a targets.jsonl row has a protein id."""
    return "pdb_id" in row

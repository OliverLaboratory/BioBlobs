"""Protein structures: PDB parsing, backbone extraction and conversion to graph samples."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch_cluster
from loguru import logger
from torch_geometric.data import Data
from tqdm import tqdm

THREE_TO_ONE = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}
ONE_TO_THREE = {v: k for k, v in THREE_TO_ONE.items()}
BACKBONE = ("N", "CA", "C", "O")
MISSING = [math.inf, math.inf, math.inf]

Structure = dict[str, Any]


def parse_backbone(pdb_path: str | Path) -> tuple[str, list, list[int]]:
    """Read the sequence and N/CA/C/O coordinates of every residue in a PDB file.

    Residues are keyed by residue number (chains and insertion codes are not separated);
    missing backbone atoms are ``inf``. The sequence is truncated or padded with ``X`` to
    the number of residues with atoms.

    Args:
        pdb_path: Path to a PDB file.

    Returns:
        ``(sequence, coords [N][4][3], residue_numbers)``.
    """
    from biopandas.pdb import PandasPdb

    atoms = PandasPdb().read_pdb(str(pdb_path)).df["ATOM"]
    first = atoms.drop_duplicates(subset=["residue_number"]).sort_values("residue_number")
    sequence = "".join(THREE_TO_ONE.get(name, "X") for name in first["residue_name"])
    residues: dict[int, dict[str, list[float]]] = defaultdict(dict)
    for row in atoms.itertuples():
        if row.atom_name in BACKBONE and row.atom_name not in residues[row.residue_number]:
            residues[row.residue_number][row.atom_name] = [row.x_coord, row.y_coord, row.z_coord]
    resnums = sorted(residues)
    coords = [[residues[r].get(atom, MISSING) for atom in BACKBONE] for r in resnums]
    sequence = sequence[: len(coords)].ljust(len(coords), "X")
    return sequence, coords, resnums


def backbone_completion(coords: Sequence) -> float:
    """Fraction of residues with all four backbone atoms."""
    complete = sum(all(atom != MISSING for atom in residue) for residue in coords)
    return complete / len(coords) if len(coords) else 0.0


def _load_one(task: tuple[str, Path, Any, float]) -> tuple[Structure | None, str | None]:
    name, path, target, min_completion = task
    try:
        sequence, coords, resnums = parse_backbone(path)
    except Exception as exc:  # noqa: BLE001 - any parse failure skips the protein
        return None, str(exc)
    completion = backbone_completion(coords)
    if completion < min_completion:
        return None, f"low backbone completion {completion:.2%}"
    return {
        "name": name,
        "seq": sequence,
        "coords": coords,
        "resnum": resnums,
        "target_payload": target,
    }, None


def load_structures(
    tasks: Sequence[tuple[str, Path, Any, float]], num_workers: int = 4, desc: str = "parse"
) -> list[Structure]:
    """Parse PDB files in parallel, keeping proteins that pass the completion filter.

    Args:
        tasks: ``(name, pdb_path, target_payload, min_backbone_completion)`` tuples.
        num_workers: Worker processes (1 parses in-process).
        desc: Progress-bar label.

    Returns:
        Structure dicts sorted by name.
    """
    structures, skipped = [], 0
    if num_workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = [pool.submit(_load_one, task) for task in tasks]
            results = [
                f.result() for f in tqdm(as_completed(futures), total=len(futures), desc=desc)
            ]
    else:
        results = [_load_one(task) for task in tqdm(tasks, desc=desc)]
    for structure, _ in results:
        if structure is None:
            skipped += 1
        else:
            structures.append(structure)
    logger.info("{}: {} structures parsed, {} skipped", desc, len(structures), skipped)
    return sorted(structures, key=lambda s: s["name"])


def write_pdb(protein: dict, path: str | Path) -> None:
    """Write a ProteinShake atom-resolution protein dict to a PDB file (residues from 1)."""
    df = pd.DataFrame(protein["atom"])
    df["atom_number"] = range(1, len(df) + 1)
    df["residue_number"] = df["residue_number"] - df["residue_number"].min() + 1
    chain = df["chain_id"] if "chain_id" in df.columns else ["A"] * len(df)
    lines = [
        "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}"
        "          {:>2s}{:2s}".format(
            "ATOM",
            r.atom_number,
            r.atom_type,
            " ",
            ONE_TO_THREE[r.residue_type],
            c,
            r.residue_number,
            " ",
            r.x,
            r.y,
            r.z,
            1.0,
            20.0,
            r.atom_type[:1],
            "  ",
        )
        for r, c in zip(df.itertuples(), chain, strict=True)
    ]
    Path(path).write_text("\n".join(lines))


def residue_graph(ca: torch.Tensor, mask: torch.Tensor, edge_types: str) -> torch.Tensor:
    """Residue contact graph over valid residues: ``knn_<k>`` or ``eps_<radius>``.

    Args:
        ca: C-alpha coordinates, ``[N, 3]``.
        mask: Valid residues, ``[N]``.
        edge_types: Graph spec, for example ``"knn_30"`` or ``"eps_8"``.

    Returns:
        ``edge_index`` over residue indices, ``[2, E]``.
    """
    method, value = edge_types.split("_")
    valid = mask.nonzero(as_tuple=True)[0]
    if valid.numel() <= 1:
        return torch.empty((2, 0), dtype=torch.long)
    if method == "knn":
        k = min(int(value), valid.numel() - 1)
        local = torch_cluster.knn_graph(ca[valid], k=k)
    elif method == "eps":
        local = torch_cluster.radius_graph(ca[valid], r=float(value))
    else:
        raise ValueError(f"Unknown edge_types {edge_types!r}")
    return valid[local]


def to_graph(
    structure: Structure,
    embeddings: torch.Tensor,
    edge_index: torch.Tensor | None = None,
) -> Data:
    """Build a graph sample from a structure and its residue embeddings.

    A residue is valid when all four backbone atoms are present; ``x`` holds C-alpha
    coordinates (``inf`` for invalid residues).

    Args:
        structure: Dict with ``name``, ``seq``, ``coords`` ``[N][4][3]``, ``resnum`` and
            optionally ``label`` and ``residue_target``.
        embeddings: Residue embeddings, ``[N, D]``.
        edge_index: Optional residue graph.

    Returns:
        ``Data`` with ``x``, ``mask``, ``node_features``, ``y`` and optional fields.
    """
    coords = torch.as_tensor(np.asarray(structure["coords"]), dtype=torch.float32)
    if embeddings.shape[0] != coords.shape[0]:
        raise ValueError(
            f"{structure['name']}: {embeddings.shape[0]} embeddings for {coords.shape[0]} residues"
        )
    mask = torch.isfinite(coords.sum(dim=(1, 2)))
    coords[~mask] = math.inf
    label = structure.get("label", 0)
    if isinstance(label, (list, tuple, np.ndarray)) and np.ndim(label) > 0:
        y = torch.tensor(np.asarray(label).tolist(), dtype=torch.float32)
    else:
        y = torch.tensor(int(label), dtype=torch.long)
    data = Data(
        x=coords[:, 1],
        mask=mask,
        node_features=embeddings.float(),
        y=y,
        name=structure["name"],
        num_nodes=coords.shape[0],
    )
    if edge_index is not None:
        data.edge_index = edge_index
    if structure.get("residue_target") is not None:
        data.residue_target = torch.tensor(structure["residue_target"], dtype=torch.float32)
    return data

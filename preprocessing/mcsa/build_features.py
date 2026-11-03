#!/usr/bin/env python
"""Build the M-CSA ESM2 feature cache.

Writes a ``{pdb_chain: {"nf", "ca", "resnums"}}`` torch ``.pt`` file: ``nf`` is the
per-residue ESM2 embedding, ``ca`` the Cα coordinates (``inf`` for a missing atom) and
``resnums`` the author residue numbers, in residue order. Consumed by
``bioblobs.data.mcsa.load_mcsa`` via ``cfg.feat_cache``.

For each curated entry (``data/mcsa/mcsa_single_chain.json``), parses the named chain's
backbone from its PDB file, keeps chains with sequence length ``<= --max-len`` and
backbone completeness ``>= --min-backbone``, and embeds the sequence with ESM2
(``bioblobs.modules.plm.embed_esm2``; sliding-window, ``window=1022``, ``overlap=128``,
``repr_layer=-1``, ``facebook/esm2_t30_150M_UR50D``).

``--max-len`` is an admission filter, not a truncation: longer chains are dropped
entirely, shorter ones are windowed in full. This release documents two caches built
with the same embedding settings but different filters (see
``bioblobs/data/mcsa.py``'s module docstring for which evaluation used which):
    --max-len 1022   -> feat_cache.pt
    --max-len 1500   -> feat_cache_maxlen1500.pt

``--existing`` reads a prior cache (default: ``--out`` itself, if it exists) and copies
its entries through unchanged, embedding only the chains still missing — so raising
``--max-len`` against a lower-``--max-len`` cache only computes the newly admitted
chains. Pass ``--rebuild`` to re-embed every chain regardless.

Usage::

    .venv/bin/python preprocessing/mcsa/build_features.py \\
        --labels data/mcsa/mcsa_single_chain.json \\
        --pdb-dir data/mcsa_raw/pdbs \\
        --out data/mcsa_raw/feat_cache.pt --max-len 1022

    .venv/bin/python preprocessing/mcsa/build_features.py \\
        --labels data/mcsa/mcsa_single_chain.json \\
        --pdb-dir data/mcsa_raw/pdbs \\
        --existing data/mcsa_raw/feat_cache.pt \\
        --out data/mcsa_raw/feat_cache_maxlen1500.pt --max-len 1500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

# Allow `python preprocessing/mcsa/build_features.py ...` from any working directory:
# the repo isn't installed as a package (pyproject.toml: package = false), so
# `bioblobs` is only importable once the repo root is on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
from loguru import logger  # noqa: E402

from bioblobs.data.structure import (  # noqa: E402
    BACKBONE,
    MISSING,
    THREE_TO_ONE,
    backbone_completion,
)
from bioblobs.modules.plm import embed_esm2  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--labels", type=Path, required=True, help="data/mcsa/mcsa_single_chain.json.")
    p.add_argument("--pdb-dir", type=Path, required=True, help="Directory of <pdb_id>.pdb files.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--max-len",
        type=int,
        required=True,
        help="Admission filter: drop chains longer than this (1022 or 1500 for the two "
        "caches this release documents).",
    )
    p.add_argument("--min-backbone", type=float, default=0.9, help="Minimum backbone completeness.")
    p.add_argument("--model-name", default="facebook/esm2_t30_150M_UR50D")
    p.add_argument("--repr-layer", type=int, default=-1)
    p.add_argument("--window", type=int, default=1022)
    p.add_argument("--overlap", type=int, default=128)
    p.add_argument("--max-batch-tokens", type=int, default=8192)
    p.add_argument("--cache-dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--existing",
        type=Path,
        default=None,
        help="Prior cache to extend (default: --out itself, if it already exists).",
    )
    p.add_argument(
        "--rebuild", action="store_true", help="Re-embed every chain, ignoring any prior cache."
    )
    p.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only use the local Hugging Face cache (default: download weights when missing).",
    )
    return p.parse_args()


def load_entries(labels_path: Path) -> dict[str, dict]:
    """One curated entry per (pdb_id, chain_used); later entries win on a duplicate key."""
    raw = json.loads(labels_path.read_text())
    by_key: dict[str, dict] = {}
    for entry in raw:
        by_key[f"{entry['pdb_id']}_{entry['chain_used']}"] = entry
    return by_key


def chain_backbone(pdb_path: Path, chain: str) -> tuple[str, list, list[int]] | None:
    """``(sequence, coords [N][4][3], resnums)`` for one chain, or ``None`` if absent.

    ``sequence`` and ``coords`` are aligned to the same residue order and length;
    missing backbone atoms are ``inf``.
    """
    from biopandas.pdb import PandasPdb

    atoms = PandasPdb().read_pdb(str(pdb_path)).df["ATOM"]
    atoms = atoms[atoms["chain_id"] == chain]
    if atoms.empty:
        return None
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


def main() -> None:
    args = parse_args()
    entries = load_entries(args.labels)
    logger.info("{:,} curated entries", len(entries))

    source = args.existing if args.existing is not None else args.out
    cache: dict[str, dict] = {}
    if not args.rebuild and source.exists():
        cache = torch.load(source, map_location="cpu", weights_only=False)
        logger.info("extending {} ({:,} chains already cached)", source, len(cache))

    todo = []
    for key, entry in entries.items():
        if key in cache:
            continue
        pdb_path = args.pdb_dir / f"{entry['pdb_id'].lower()}.pdb"
        if not pdb_path.exists():
            continue
        parsed = chain_backbone(pdb_path, entry["chain_used"])
        if parsed is None:
            continue
        sequence, coords, resnums = parsed
        if len(sequence) == 0 or len(sequence) > args.max_len:
            continue
        if backbone_completion(coords) < args.min_backbone:
            continue
        todo.append((key, sequence, coords, resnums))
    logger.info(
        "max-len {}, min-backbone {}: {:,} new chains to embed ({:,} curated entries total)",
        args.max_len, args.min_backbone, len(todo), len(entries),
    )  # fmt: skip

    if todo:
        t0 = time.time()
        embeddings = embed_esm2(
            [t[1] for t in todo],
            model_name=args.model_name,
            repr_layer=args.repr_layer,
            window=args.window,
            overlap=args.overlap,
            max_batch_tokens=args.max_batch_tokens,
            device=args.device,
            local_files_only=args.local_files_only,
        )
        logger.info("embedded {:,} chains in {:.0f}s", len(todo), time.time() - t0)
        cache_dtype = torch.float16 if args.cache_dtype == "float16" else torch.float32
        for (key, _sequence, coords, resnums), nf in zip(todo, embeddings, strict=True):
            ca = torch.as_tensor(coords, dtype=torch.float32)[:, 1, :]
            cache[key] = {
                "nf": torch.nan_to_num(nf).to(cache_dtype).cpu(),
                "ca": ca.cpu(),
                "resnums": [int(r) for r in resnums],
            }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    torch.save(cache, tmp)
    tmp.rename(args.out)
    logger.success("wrote {:,} chains -> {}", len(cache), args.out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Builds the common M-CSA input file for the three localization baselines.

Reads a curated single-chain M-CSA JSON file -- a list of records with
``pdb_id``, ``chain_used``, ``ec_number`` and ``catalytic_residues`` -- plus
the matching PDB structure files, and writes one record per chain in the
common input format consumed by ``proteinfer/score.py``,
``deepfri/score.py`` and ``protdetr/score.py`` (see ``README.md``):
``{id, seq, ca, ec}``.

The sequence and its aligned C-alpha coordinates are read directly from
each chain's ``ATOM``/``HETATM`` records, not from the catalytic-residue
list, so ``seq`` is exactly the residue ordering the baselines' per-residue
scores must align to. Catalytic-residue labels are not part of this
baseline-input format -- they are only needed for evaluation, which is out
of scope for this release's baseline wrappers.

Usage:
    python make_inputs.py \\
        --mcsa-json /path/to/mcsa_single_chain.json \\
        --pdb-dir /path/to/pdbs \\
        --out mcsa_inputs.jsonl

``--pdb-dir`` must contain one file per structure, named
``<pdb_id, lowercase>.pdb``.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LOGGER = logging.getLogger("make_inputs")

# Standard residues, plus selenomethionine (MSE), a common crystallographic
# substitution for methionine that PDB files store as HETATM.
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
}


def parse_chain(pdb_path: Path, chain_id: str) -> Tuple[str, List[List[float]]]:
    """Reads one chain's sequence and C-alpha coordinates from a PDB file.

    Only the first ``MODEL`` block is read (for multi-model/NMR files).
    Only residues with a resolved C-alpha atom are kept, so the returned
    sequence and coordinates are aligned 1:1, in author residue order --
    the same order the M-CSA catalytic-residue indices (``resid``) refer
    to.

    Args:
        pdb_path: Path to a PDB-format structure file.
        chain_id: The author chain identifier to extract (PDB column 22).

    Returns:
        A tuple ``(seq, ca)`` where ``seq`` is a one-letter amino-acid
        string and ``ca`` is a list of ``[x, y, z]`` coordinates, one per
        residue of ``seq``.
    """
    residues: Dict[Tuple[int, str], Dict[str, Optional[object]]] = {}
    order: List[Tuple[int, str]] = []
    seen_models = 0
    with open(pdb_path) as fh:
        for line in fh:
            record = line[:6].strip()
            if record == "MODEL":
                seen_models += 1
                if seen_models > 1:
                    break
                continue
            if record == "ENDMDL":
                break
            if record not in ("ATOM", "HETATM"):
                continue
            if len(line) < 27 or line[21] != chain_id:
                continue
            resname = line[17:20].strip()
            if record == "HETATM" and resname != "MSE":
                continue
            if resname not in THREE_TO_ONE:
                continue
            atom_name = line[12:16].strip()
            alt_loc = line[16]
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            icode = line[26]
            key = (resseq, icode)
            if key not in residues:
                residues[key] = {"resname": resname, "ca": None}
                order.append(key)
            if atom_name == "CA" and residues[key]["ca"] is None and alt_loc in (" ", "A"):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                residues[key]["ca"] = [x, y, z]

    seq_chars: List[str] = []
    coords: List[List[float]] = []
    for key in order:
        res = residues[key]
        if res["ca"] is None:
            continue
        seq_chars.append(THREE_TO_ONE[str(res["resname"])])
        coords.append(res["ca"])  # type: ignore[arg-type]
    return "".join(seq_chars), coords


def build_records(mcsa_json: Path, pdb_dir: Path) -> List[dict]:
    """Builds the common-format input records for every M-CSA chain.

    Args:
        mcsa_json: Path to the curated single-chain M-CSA JSON file.
        pdb_dir: Directory of ``<pdb_id>.pdb`` structure files
            (lowercase, e.g. ``1b73.pdb``).

    Returns:
        A list of ``{id, seq, ca, ec}`` records.
    """
    entries = json.loads(mcsa_json.read_text())
    records: List[dict] = []
    n_missing_pdb = 0
    n_parse_fail = 0
    n_too_short = 0
    for entry in entries:
        pdb_id = entry["pdb_id"]
        chain = entry["chain_used"]
        rec_id = "%s_%s" % (pdb_id, chain)
        pdb_path = pdb_dir / ("%s.pdb" % pdb_id.lower())
        if not pdb_path.exists():
            LOGGER.warning("%s: no structure file at %s, skipping", rec_id, pdb_path)
            n_missing_pdb += 1
            continue
        try:
            seq, ca = parse_chain(pdb_path, chain)
        except Exception as exc:  # noqa: BLE001 - log and skip malformed files
            LOGGER.warning("%s: failed to parse %s (%s), skipping", rec_id, pdb_path, exc)
            n_parse_fail += 1
            continue
        if len(seq) < 10:
            LOGGER.warning("%s: resolved sequence too short (%d residues), skipping", rec_id, len(seq))
            n_too_short += 1
            continue
        ec = entry.get("ec_number")
        records.append({
            "id": rec_id,
            "seq": seq,
            "ca": ca,
            "ec": str(ec) if ec is not None else None,
        })
    LOGGER.info(
        "built %d/%d records (%d missing structure, %d parse failures, %d too short)",
        len(records), len(entries), n_missing_pdb, n_parse_fail, n_too_short,
    )
    return records


def write_records(records: List[dict], out_path: Path) -> None:
    """Writes records as a JSON array, or as JSON-lines if out_path ends in .jsonl."""
    if out_path.suffix == ".jsonl":
        with open(out_path, "w") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")
    else:
        with open(out_path, "w") as fh:
            json.dump(records, fh)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mcsa-json", required=True, type=Path,
                         help="Curated single-chain M-CSA JSON (pdb_id, chain_used, ec_number, catalytic_residues).")
    parser.add_argument("--pdb-dir", required=True, type=Path,
                         help="Directory of <pdb_id>.pdb structure files (lowercase filenames).")
    parser.add_argument("--out", required=True, type=Path,
                         help="Output path; written as JSON-lines if it ends in .jsonl, else a JSON array.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    records = build_records(args.mcsa_json, args.pdb_dir)
    write_records(records, args.out)
    LOGGER.info("wrote %s", args.out)


if __name__ == "__main__":
    main()

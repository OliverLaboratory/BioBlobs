#!/usr/bin/env python
"""Label-free reference scores: burial (-rSASA) and uniform random.

``burial``: negative relative solvent accessibility. Shrake-Rupley SASA of the deposited structure
(first model, every atom including bound ligands and waters, which therefore occlude), divided by
the theoretical maximum accessibility of Tien et al. (2013). Residues without a value are NaN and
are dropped by ``evaluate_scores.py``. Needs Biopython.

``random``: ``numpy.random.default_rng(0).random(N)`` drawn chain by chain in the order of
``data/mcsa/mcsa_single_chain.json``, over every chain of the feature file that has catalytic and
non-catalytic residues (the draw order of the paper's run).

Usage::

    python baselines/mcsa/heuristics.py burial --features data/mcsa/features.pt \\
        --pdb-dir data/mcsa_raw/pdbs --out scores/burial.json
    python baselines/mcsa/heuristics.py random --features data/mcsa/features.pt \\
        --out scores/random.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from inhouse import DEFAULT_IDS, DEFAULT_LABELS, load_features, read_ids, site_labels, write_scores

# Theoretical maximum residue accessibility in A^2 (Tien et al., 2013).
MAX_ASA = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0,
    "GLN": 225.0, "GLU": 223.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0,
    "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, "PRO": 159.0,
    "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
}  # fmt: skip


def burial(pdb_path: str, chain: str, resnums: list[int]) -> list[float]:
    """``-rSASA`` per residue number of ``chain`` (NaN where unavailable)."""
    from Bio.PDB import PDBParser
    from Bio.PDB.SASA import ShrakeRupley

    structure = PDBParser(QUIET=True).get_structure("s", pdb_path)
    ShrakeRupley().compute(structure, level="R")
    model = next(iter(structure))
    rsasa = {}
    for residue in model[chain] if chain in model else next(iter(model)):
        if residue.id[0] == " " and residue.resname in MAX_ASA and hasattr(residue, "sasa"):
            rsasa[residue.id[1]] = residue.sasa / MAX_ASA[residue.resname]
    return [-rsasa[r] if r in rsasa else float("nan") for r in resnums]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("method", choices=["burial", "random"])
    p.add_argument("--features", required=True, help="M-CSA feature file (residue order)")
    p.add_argument("--pdb-dir", help="PDB files (burial)")
    p.add_argument("--ids", default=str(DEFAULT_IDS), help="chains to score (burial)")
    p.add_argument("--labels", default=str(DEFAULT_LABELS))
    p.add_argument("--out", required=True)
    args = p.parse_args()

    features = load_features(args.features)
    raw_entries = json.load(open(args.labels))
    scores = {}
    if args.method == "burial":
        entries = {f"{e['pdb_id']}_{e['chain_used']}": e for e in raw_entries}
        for key in read_ids(args.ids):
            entry, resnums = entries[key], [int(r) for r in features[key]["resnums"]]
            pdb = f"{args.pdb_dir}/{entry['pdb_id'].lower()}.pdb"
            scores[key] = burial(pdb, entry["chain_used"], resnums)
    else:
        rng = np.random.default_rng(0)
        for entry in raw_entries:  # file order, duplicates included, as in the paper's run
            key = f"{entry['pdb_id']}_{entry['chain_used']}"
            if key in features:
                y = site_labels(entry, features[key]["resnums"])
                if 0 < y.sum() < len(y):
                    scores[key] = rng.random(len(y)).tolist()
    write_scores(args.out, scores)
    print(f"wrote {len(scores)} chains to {args.out}")


if __name__ == "__main__":
    main()

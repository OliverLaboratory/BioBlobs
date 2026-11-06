"""Predict functions, blobs and residue scores for your own PDB files.

Usage:
    # Released M-CSA model (EC-L3 classes), a directory of PDB files
    python inference.py --model path/to/bioblobs-mcsa-rvq-k8 --pdb-dir my_pdbs --out results

    # A model trained with train_bioblobs.py, one chain of one structure
    python inference.py --model outputs/<run>/seed0/best.ckpt --pdb 1abc.pdb --chain A --out results

Outputs: results/predictions.csv and results/proteins/<name>.json with the top classes, the
blobs ranked by importance score (seed and member residue numbers) and per-residue scores.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from bioblobs.inference import featurize_pdbs, load_pretrained, predict, write_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="released model directory or .ckpt")
    parser.add_argument("--pdb", nargs="*", default=[], help="PDB files")
    parser.add_argument("--pdb-dir", default=None, help="directory of .pdb files")
    parser.add_argument("--chain", default=None, help="chain to use (default: all residues)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--foldseek-bin", default=None, help="Foldseek binary (SaProt models)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    paths = [Path(p) for p in args.pdb]
    if args.pdb_dir:
        paths += sorted(Path(args.pdb_dir).glob("*.pdb"))
    if not paths:
        parser.error("give --pdb or --pdb-dir")
    model, cfg, classes = load_pretrained(args.model, args.device)
    graphs = featurize_pdbs(paths, cfg, args.device, [args.chain] * len(paths), args.foldseek_bin)
    multi_label = cfg.get("tasks", {}).get("problem_type") == "multi_label"
    write_results(predict(model, graphs, classes, args.top_k, multi_label=multi_label), args.out)


if __name__ == "__main__":
    main()


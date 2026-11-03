"""Evaluate a trained model: test classification and residue localization.

Usage:
    # VenusX / ProteinShake test split, attention-weighted membership readout
    python evaluate.py --checkpoint outputs/venusx/act_esm2_labelfree/seed1/best.ckpt \
        --score weighted_membership --out results/act_esm2_labelfree_seed1

    # M-CSA catalytic residues with the released model (max-membership readout)
    python evaluate.py --checkpoint bioblobs-mcsa-rvq-k8 --mcsa-features data/mcsa/features.pt \
        --mcsa-ids data/mcsa/evaluation_ids.txt --score max_membership --out results/mcsa

    # Override config entries of the training run, e.g. the data location
    python evaluate.py --checkpoint ... --set datasets.data_dir=/data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from bioblobs.data import load_splits
from bioblobs.evaluation import SCORES, evaluate, evaluate_mcsa
from bioblobs.inference import load_pretrained


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help=".ckpt or released model directory")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--score", default=None, choices=SCORES)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mcsa-features", default=None, help="M-CSA feature file (M-CSA mode)")
    parser.add_argument("--mcsa-ids", default=None, help="file of M-CSA ids to evaluate")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    args = parser.parse_args()

    model, cfg, _ = load_pretrained(args.checkpoint, args.device)
    if args.mcsa_features:
        from bioblobs.data.mcsa import load_mcsa

        structures = load_mcsa({"feat_cache": args.mcsa_features})
        ids = None
        if args.mcsa_ids:
            lines = Path(args.mcsa_ids).read_text().splitlines()
            ids = {line.strip() for line in lines if line.strip() and not line.startswith("#")}
        evaluate_mcsa(model, structures, ids, args.score or "max_membership", args.out)
        return
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.set))
    splits = load_splits(cfg, (args.split,))
    dataset = splits.test if args.split == "test" else splits.val
    evaluate(model, dataset, cfg, splits.num_classes, args.score, args.out, args.batch_size)


if __name__ == "__main__":
    main()

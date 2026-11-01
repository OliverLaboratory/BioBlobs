"""Evaluate a trained checkpoint: test classification and residue localization.

Usage:
    # Classification + localization with the attention-weighted membership readout
    python evaluate.py --checkpoint outputs/venusx/act_esm2_labelfree/seed1/best.ckpt \
        --score weighted_membership --out results/act_esm2_labelfree_seed1

    # Baselines on a pooling model: attention weights, integrated gradients, burial, random
    python evaluate.py --checkpoint <attention_pool run>/best.ckpt --score attention --out ...

    # Override config entries of the training run, e.g. the data location
    python evaluate.py --checkpoint ... --set datasets.data_dir=/data
"""

from __future__ import annotations

import argparse

import torch
from omegaconf import OmegaConf

from bioblobs.data import load_splits
from bioblobs.evaluation import SCORES, evaluate
from bioblobs.training.trainer import load_module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--score", default=None, choices=SCORES)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=1, help="seed of the random baseline")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    args = parser.parse_args()

    module = load_module(args.checkpoint, args.device).to(args.device)
    cfg = OmegaConf.merge(module.cfg, OmegaConf.from_dotlist(args.set))
    splits = load_splits(cfg, (args.split,))
    dataset = splits.test if args.split == "test" else splits.val
    evaluate(
        module.model,
        dataset,
        cfg,
        splits.num_classes,
        args.score,
        args.out,
        args.batch_size,
        args.seed,
    )


if __name__ == "__main__":
    main()

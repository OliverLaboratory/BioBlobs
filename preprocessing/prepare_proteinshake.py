"""Download a ProteinShake task and cache its parsed structures for every split.

Usage:
    python preprocessing/prepare_proteinshake.py --dataset ec --split sequence --data-dir data
    python preprocessing/prepare_proteinshake.py --dataset go_mf --split structure

Writes ``<data-dir>/<task>_proteinshake/`` (PDB files, labels, split lists and
``cache/<split>_{train,val,test}_<edge_types>_0.7_cache.pt``). Then build the embedding cache
with ``preprocessing/prepare_plm_cache.py`` from those three cache files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bioblobs.data import proteinshake  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, choices=sorted(proteinshake.TASKS))
    parser.add_argument("--split", default="sequence", choices=["sequence", "structure"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--edge-types", default="knn_30", help="names the structure cache files")
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    task = proteinshake.TASKS[args.dataset]
    cfg = OmegaConf.create(
        {
            "datasets": {
                "dataset_name": args.dataset,
                "data_dir": args.data_dir,
                "split": args.split,
                "split_similarity_threshold": 0.7,
                "ec_level": 3,
                "edge_types": args.edge_types,
                "num_workers": args.num_workers,
            },
            "tasks": {"problem_type": task.problem_type},
        }
    )
    for split in ("train", "val", "test"):
        proteinshake.load_split(cfg, split)


if __name__ == "__main__":
    main()

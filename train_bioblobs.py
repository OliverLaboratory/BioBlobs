"""Train BioBlobs on one task, select on validation and test once.

Usage:
    # A reported BioBlobs setting (see conf/experiment/)
    python train_bioblobs.py +experiment=proteinshake/ec_sequence_esm2

    # Three seeds
    python train_bioblobs.py -m +experiment=venusx/act_esm2_labelfree training.seed=1,2,3
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from bioblobs.training.trainer import train


@hydra.main(version_base="1.3", config_path="conf", config_name="train_bioblobs")
def main(cfg: DictConfig) -> None:
    train(cfg, cfg.output_dir)


if __name__ == "__main__":
    main()

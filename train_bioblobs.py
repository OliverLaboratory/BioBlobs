"""Unified BioBlobs training entrypoint.

Defaults to the BioBlobs partitioner + MIL decoder. Override on the CLI to
switch to the mean-pooling MLP baseline:

    # BioBlobs (default)
    python train_bioblobs.py

    # Mean-pooling MLP baseline (encoder -> mean pool -> MLP)
    python train_bioblobs.py partitioners=none decoders=mlp pooling=mean
"""

from __future__ import annotations

import hydra
from loguru import logger
from omegaconf import DictConfig

from bioblobs.training.experiments import build_single_run_output_dir
from bioblobs.training.baseline_runner import run_baseline_experiment


@hydra.main(version_base="1.1", config_path="conf", config_name="train_bioblobs")
def main(cfg: DictConfig) -> None:
    print("=" * 70)
    logger.info("BioBlobs Training")
    print("=" * 70)

    output_dir = build_single_run_output_dir("outputs", cfg)
    run_baseline_experiment(cfg, output_dir)


if __name__ == "__main__":
    main()

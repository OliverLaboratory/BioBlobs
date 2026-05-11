"""Download + stage the VenusX site-fragment dataset (PDBs + per-split CSVs).

Run once per (target, split_strategy) pair before training or PLM caching:

    python preprocessing/prepare_venusx_site_fragment.py \\
        datasets.target=Act \\
        datasets.split_strategy=MF50

Targets:           Act, BindI, Evo, Motif, Dom
Split strategies:  MF50, MF70, MF90
"""

from __future__ import annotations

import hydra
from loguru import logger
from omegaconf import DictConfig

from bioblobs.datasets.venusx_dataset import ensure_prepared_dataset


@hydra.main(
    version_base="1.1",
    config_path="../conf",
    config_name="prepare_venusx_site_fragment",
)
def main(cfg: DictConfig) -> None:
    prepared_root = ensure_prepared_dataset(cfg)
    logger.success("Prepared VenusX dataset at {}", prepared_root)


if __name__ == "__main__":
    main()

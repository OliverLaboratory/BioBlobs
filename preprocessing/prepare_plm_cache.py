"""Build a direct-to-mmap PLM residue embedding cache for a ProteinShake dataset.

Loads structures for the configured dataset (EC / GO / Pfam / SCOP), unions
train+val+test, dedupes by ``protein_uid``/``name``, and writes a single
``embeddings.bin`` + ``meta.pt`` next to the dataset's prepared directory.
No per-protein ``.pt`` intermediates.

Usage:
    python preprocessing/prepare_plm_cache.py \\
        datasets=ec \\
        encoders=esm2_static \\
        encoders.precompute_device=cuda
"""

from __future__ import annotations

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig

from bioblobs.datasets.plm_static_mmap_cache import (
    prepare_esm2_static_mmap_cache,
    prepare_saprot_static_mmap_cache,
)
from bioblobs.datasets.proteinshake_dataset import load_dataset_structures


@hydra.main(version_base="1.1", config_path="../conf", config_name="baseline_example")
def main(cfg: DictConfig) -> None:
    encoder_name = str(cfg.encoders.name)
    if encoder_name not in {"esm2_static", "saprot_static"}:
        raise ValueError(
            f"Only esm2_static and saprot_static are supported (got {encoder_name!r})."
        )

    structures, num_classes = load_dataset_structures(cfg)
    all_structures = (
        list(structures["train"])
        + list(structures["val"])
        + list(structures["test"])
    )
    logger.info(
        "Dataset={}, splits sizes={{train: {}, val: {}, test: {}}}, classes={}",
        cfg.datasets.dataset_name,
        len(structures["train"]),
        len(structures["val"]),
        len(structures["test"]),
        num_classes,
    )
    if not all_structures:
        raise RuntimeError(f"No structures loaded for {cfg.datasets.dataset_name}.")

    device_str = str(cfg.encoders.get("precompute_device", "cpu"))
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"precompute_device={device_str!r} but no CUDA device is available."
        )
    device = torch.device(device_str)

    if encoder_name == "esm2_static":
        out_dir = prepare_esm2_static_mmap_cache(all_structures, cfg, device=device)
    else:
        out_dir = prepare_saprot_static_mmap_cache(all_structures, cfg, device=device)

    logger.success(
        "Built {} mmap cache for {} at {}", encoder_name, cfg.datasets.dataset_name, out_dir,
    )


if __name__ == "__main__":
    main()

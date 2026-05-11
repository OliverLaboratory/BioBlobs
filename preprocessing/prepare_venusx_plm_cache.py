"""Build a direct-to-mmap PLM residue embedding cache for one VenusX target.

Loads the prepared VenusX target (Act/BindI/Evo/Motif/Dom), unions
train+val+test proteins, dedupes by ``protein_uid``, and writes a single
``embeddings.bin`` + ``meta.pt`` next to the target's data root. No
per-protein ``.pt`` intermediates.

Usage:
    python preprocessing/prepare_venusx_plm_cache.py \\
        datasets.target=Act \\
        encoders=esm2_static \\
        encoders.precompute_device=cuda

    python preprocessing/prepare_venusx_plm_cache.py \\
        datasets.target=BindI \\
        encoders=saprot_static \\
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
from bioblobs.datasets.venusx_dataset import load_prepared_structures


@hydra.main(
    version_base="1.1",
    config_path="../conf",
    config_name="prepare_venusx_plm_cache",
)
def main(cfg: DictConfig) -> None:
    encoder_name = str(cfg.encoders.name)
    if encoder_name not in {"esm2_static", "saprot_static"}:
        raise ValueError(
            f"Only esm2_static and saprot_static are supported by this script "
            f"(got {encoder_name!r})."
        )

    structures, num_classes = load_prepared_structures(cfg)
    all_structures = (
        list(structures["train"])
        + list(structures["val"])
        + list(structures["test"])
    )
    logger.info(
        "VenusX target={}, splits sizes={{train: {}, val: {}, test: {}}}, classes={}",
        cfg.datasets.target,
        len(structures["train"]),
        len(structures["val"]),
        len(structures["test"]),
        num_classes,
    )
    if not all_structures:
        raise RuntimeError(f"No structures loaded for target {cfg.datasets.target}.")

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
        "Built {} mmap cache for VenusX target {} at {}",
        encoder_name, cfg.datasets.target, out_dir,
    )


if __name__ == "__main__":
    main()

"""Helpers for resolving encoder config metadata."""

from __future__ import annotations

from omegaconf import OmegaConf


ENCODER_TARGETS = {
    "esm2_static": (
        "bioblobs.modules.encoders.precomputed_encoder.PrecomputedNodeFeatureEncoder"
    ),
    "saprot_static": (
        "bioblobs.modules.encoders.precomputed_encoder.PrecomputedNodeFeatureEncoder"
    ),
}

# Fields used by ESM2 / SaProt cache infrastructure, not by encoder
# constructors. Stripped before Hydra instantiate for non-precomputed encoders.
_ESM_CACHE_FIELDS = {
    "model_name", "repr_layer", "window_size", "window_overlap",
    "max_batch_tokens", "cache_dir", "cache_dtype", "precompute_missing",
    "overwrite_cache", "precompute_device", "output_dim", "foldseek_bin",
}


def resolve_encoder_name(enc_cfg) -> str:
    """Return the configured encoder family name and validate its target."""
    if enc_cfg is None:
        raise ValueError("Missing cfg.encoders configuration")

    name = enc_cfg.get("name")
    if name is None:
        raise ValueError(
            "Missing cfg.encoders.name. Set the encoder family via the encoders config."
        )

    if name not in ENCODER_TARGETS:
        raise ValueError(
            f"Unsupported cfg.encoders.name {name!r}. "
            f"Supported values: {sorted(ENCODER_TARGETS)}"
        )

    target = enc_cfg.get("_target_")
    if target is None:
        raise ValueError(
            f"Missing cfg.encoders._target_ for encoder {name!r}. "
            "Programmatic configs must include both encoders.name and encoders._target_."
        )

    expected_target = ENCODER_TARGETS[name]
    if target != expected_target:
        raise ValueError(
            f"Encoder config mismatch: cfg.encoders.name={name!r} expects "
            f"_target_={expected_target!r}, got {target!r}"
        )

    return name


def sanitize_encoder_cfg(enc_cfg):
    """Return a copy of the encoder config without metadata-only fields."""
    resolve_encoder_name(enc_cfg)
    clean_cfg = OmegaConf.create(OmegaConf.to_container(enc_cfg, resolve=False))
    clean_cfg.pop("name", None)
    return clean_cfg

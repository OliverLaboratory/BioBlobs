"""Helpers for experiment naming, metrics, and output layout."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


DEFAULT_GO_BRANCH = "molecular_function"


def sanitize_path_component(value: str) -> str:
    """Make a value safe to use as a single path component."""
    sanitized = value.strip().replace("\\", "__").replace("/", "__").replace(":", "_")
    return sanitized or "default"


_GO_ALIAS_TO_BRANCH: dict[str, str] = {
    "go_mf": "molecular_function",
    "go_bp": "biological_process",
    "go_cc": "cellular_component",
}


def dataset_slug_from_cfg(cfg) -> str:
    """Build a dataset slug that is stable across output paths and caches."""
    dataset_name = cfg.datasets.dataset_name
    if dataset_name in _GO_ALIAS_TO_BRANCH:
        return f"go_{_GO_ALIAS_TO_BRANCH[dataset_name]}"
    if dataset_name == "go":
        go_branch = cfg.datasets.get("go_branch", DEFAULT_GO_BRANCH)
        return f"go_{go_branch}"
    if dataset_name == "ec":
        ec_level = cfg.datasets.get("ec_level", 1)
        if ec_level != 1:
            return f"ec_l{ec_level}"
    return str(dataset_name)


def partitioner_slug_from_cfg(cfg) -> str:
    """Return a normalized partitioner slug."""
    partitioner_cfg = cfg.get("partitioners", {})
    if not partitioner_cfg or not partitioner_cfg.get("enabled", False):
        return "none"
    return str(partitioner_cfg.get("name", "partitioner"))


def experiment_family_from_cfg(cfg) -> str:
    """Classify the run into the baseline or MIL experiment family."""
    if partitioner_slug_from_cfg(cfg) == "none":
        return "baselines"
    return "mil"


def derive_experiment_id(cfg) -> str:
    """Derive a stable experiment slug from the resolved config."""
    dataset_slug = dataset_slug_from_cfg(cfg)
    split_name = str(cfg.datasets.split)
    encoder_name = str(cfg.encoders.name)

    if experiment_family_from_cfg(cfg) == "baselines":
        return f"{dataset_slug}_{split_name}_{encoder_name}"

    partitioner_name = partitioner_slug_from_cfg(cfg)
    return f"{dataset_slug}_{split_name}_{encoder_name}_{partitioner_name}_mil"


def monitor_summary_key_from_cfg(cfg) -> str:
    """Return the summary key used to rank best validation performance."""
    tasks_cfg = cfg.get("tasks", {})
    monitor_metric = str(tasks_cfg.get("monitor_metric", "val_acc"))
    return f"best_{monitor_metric}"


def monitor_mode_from_cfg(cfg) -> str:
    """Return the optimization direction for the monitored validation metric."""
    tasks_cfg = cfg.get("tasks", {})
    return str(tasks_cfg.get("monitor_mode", "max"))


def test_primary_metric_key_from_cfg(cfg) -> str:
    """Return the preferred summary key for test-time comparison."""
    tasks_cfg = cfg.get("tasks", {})
    primary_metric = str(tasks_cfg.get("primary_metric", "accuracy"))
    if primary_metric == "accuracy":
        return "test_acc"
    return f"test_{primary_metric}"


def selection_slug_from_sweep_id(sweep_id: str) -> str:
    """Return the local selection slug used for promoted best runs."""
    return f"best_of_{sanitize_path_component(sweep_id)}"


def _run_slug(job_name: str, *, timestamp: str | None = None) -> str:
    resolved_timestamp = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return sanitize_path_component(f"{job_name}_{resolved_timestamp}")


def build_single_run_output_dir(
    base_dir: str | Path,
    cfg,
    *,
    timestamp: str | None = None,
) -> Path:
    """Build the structured output directory for a single Hydra-driven run.

    ``base_dir`` is used as a literal prefix.
    """
    base_path = Path(base_dir)
    family = experiment_family_from_cfg(cfg)
    dataset_slug = dataset_slug_from_cfg(cfg)
    split_name = sanitize_path_component(str(cfg.datasets.split))
    encoder_name = sanitize_path_component(str(cfg.encoders.name))
    job_name = str(cfg.get("wandb", {}).get("job_name", "run"))
    run_slug = _run_slug(job_name, timestamp=timestamp)

    if family == "baselines":
        return base_path / dataset_slug / split_name / encoder_name / "single" / run_slug

    partitioner_name = sanitize_path_component(partitioner_slug_from_cfg(cfg))
    return (
        base_path
        / dataset_slug
        / split_name
        / encoder_name
        / partitioner_name
        / "single"
        / run_slug
    )


def build_trial_output_dir(
    base_dir: str | Path,
    cfg,
    *,
    sweep_id: str,
    run_id: str,
) -> Path:
    """Build the structured output directory for one W&B sweep/tune trial.

    ``base_dir`` is used as a literal prefix — the caller is expected to name
    the root (e.g. ``outputs/mil`` for legacy layout, ``outputs/bioblobs_hoyer``
    for the Hoyer sweep).
    """
    base_path = Path(base_dir)
    family = experiment_family_from_cfg(cfg)
    dataset_slug = dataset_slug_from_cfg(cfg)
    split_name = sanitize_path_component(str(cfg.datasets.split))
    encoder_name = sanitize_path_component(str(cfg.encoders.name))
    sweep_slug = sanitize_path_component(sweep_id)
    run_slug = sanitize_path_component(run_id)

    if family == "baselines":
        return (
            base_path
            / dataset_slug
            / split_name
            / encoder_name
            / "sweep"
            / sweep_slug
            / "trials"
            / run_slug
        )

    partitioner_name = sanitize_path_component(partitioner_slug_from_cfg(cfg))
    return (
        base_path
        / dataset_slug
        / split_name
        / encoder_name
        / partitioner_name
        / "tune"
        / sweep_slug
        / "trials"
        / run_slug
    )


def build_best_output_dir(
    base_dir: str | Path,
    cfg,
    *,
    sweep_id: str,
) -> Path:
    """Build the structured directory that stores the promoted best config.

    ``base_dir`` is used as a literal prefix.
    """
    base_path = Path(base_dir)
    family = experiment_family_from_cfg(cfg)
    dataset_slug = dataset_slug_from_cfg(cfg)
    split_name = sanitize_path_component(str(cfg.datasets.split))
    encoder_name = sanitize_path_component(str(cfg.encoders.name))
    sweep_slug = sanitize_path_component(sweep_id)

    if family == "baselines":
        return (
            base_path
            / dataset_slug
            / split_name
            / encoder_name
            / "sweep"
            / sweep_slug
            / "best"
        )

    partitioner_name = sanitize_path_component(partitioner_slug_from_cfg(cfg))
    return (
        base_path
        / dataset_slug
        / split_name
        / encoder_name
        / partitioner_name
        / "tune"
        / sweep_slug
        / "best"
    )


def build_seed_eval_output_dir(
    base_dir: str | Path,
    cfg,
    *,
    sweep_id: str,
) -> Path:
    """Build the structured directory for fixed-seed promoted runs.

    ``base_dir`` is used as a literal prefix.
    """
    base_path = Path(base_dir)
    family = experiment_family_from_cfg(cfg)
    dataset_slug = dataset_slug_from_cfg(cfg)
    split_name = sanitize_path_component(str(cfg.datasets.split))
    encoder_name = sanitize_path_component(str(cfg.encoders.name))
    selection_slug = selection_slug_from_sweep_id(sweep_id)

    if family == "baselines":
        return (
            base_path
            / dataset_slug
            / split_name
            / encoder_name
            / "seed_eval"
            / selection_slug
        )

    partitioner_name = sanitize_path_component(partitioner_slug_from_cfg(cfg))
    return (
        base_path
        / dataset_slug
        / split_name
        / encoder_name
        / partitioner_name
        / "seed_eval"
        / selection_slug
    )

"""Shared helpers for baseline training, sweeps, and multi-seed evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import pytorch_lightning as pl
import torch
from hydra.core.global_hydra import GlobalHydra
from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from bioblobs.datasets import create_dataloader, get_dataset
from .experiments import (
    derive_experiment_id as derive_experiment_id_from_cfg,
    sanitize_path_component as sanitize_path_component_for_path,
)
from .utils import set_seed
from .train import BioBlobsLightningModule


REPO_ROOT = Path(__file__).resolve().parents[2]
CONF_DIR = REPO_ROOT / "conf"


def config_to_container(cfg: DictConfig | dict, *, resolve: bool = True) -> dict[str, Any]:
    """Convert an OmegaConf config into a plain Python dictionary."""
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=resolve)  # type: ignore[return-value]
    if isinstance(cfg, dict):
        return cfg
    raise TypeError(f"Unsupported config type: {type(cfg)!r}")


def clone_config(cfg: DictConfig | dict, *, resolve: bool = False) -> DictConfig:
    """Create an OmegaConf copy of the provided config."""
    return OmegaConf.create(config_to_container(cfg, resolve=resolve))


def compose_config(
    config_name: str = "baseline_example",
    overrides: list[str] | None = None,
) -> DictConfig:
    """Compose a Hydra config outside the CLI entrypoints."""
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with hydra.initialize_config_dir(version_base="1.1", config_dir=str(CONF_DIR)):
        cfg = hydra.compose(config_name=config_name, overrides=overrides or [])
    return cfg


def merge_dotlist_overrides(cfg: DictConfig, overrides: list[str] | None) -> DictConfig:
    """Merge Hydra-style dotlist overrides into a config."""
    if not overrides:
        return cfg
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))


def _set_nested_value(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    current = target
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def apply_parameter_updates(cfg: DictConfig, updates: dict[str, Any] | None) -> DictConfig:
    """Apply flat dotted-key updates such as sweep parameters into a config."""
    if not updates:
        return cfg

    nested_updates: dict[str, Any] = {}
    for key, value in updates.items():
        if key.startswith("_"):
            continue
        _set_nested_value(nested_updates, key, value)

    return OmegaConf.merge(cfg, OmegaConf.create(nested_updates))


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def derive_experiment_id(cfg: DictConfig) -> str:
    """Derive a stable experiment slug from the resolved baseline config."""
    return derive_experiment_id_from_cfg(cfg)


def sanitize_path_component(value: str) -> str:
    """Make a value safe to use as a single path component."""
    return sanitize_path_component_for_path(value)


def save_resolved_config(cfg: DictConfig, output_dir: str | Path) -> Path:
    """Persist the resolved config for later inspection and reuse."""
    output_dir = ensure_output_dir(output_dir)
    config_path = output_dir / "resolved_config.yaml"
    OmegaConf.save(clone_config(cfg, resolve=True), config_path)
    return config_path


def save_json(payload: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def build_run_name(cfg: DictConfig) -> str:
    from .experiments import dataset_slug_from_cfg

    dataset_slug = dataset_slug_from_cfg(cfg)
    split_name = cfg.datasets.split
    encoder_name = cfg.encoders.name
    partitioner_name = cfg.get("partitioners", {}).get("name", "none")
    job_name = cfg.get("wandb", {}).get("job_name", "baseline")
    if cfg.get("partitioners", {}).get("enabled", False):
        return f"{job_name}_{dataset_slug}_{split_name}_{encoder_name}_{partitioner_name}"
    return f"{job_name}_{dataset_slug}_{split_name}_{encoder_name}"


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class BestMetricTracker(pl.Callback):
    """Track the best validation metrics seen during training."""

    def __init__(self, monitor_metric: str, monitor_mode: str) -> None:
        super().__init__()
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode
        self.best_val_metric: float | None = None
        self.best_val_metric_epoch: int | None = None
        self.best_val_acc: float | None = None
        self.best_val_acc_epoch: int | None = None
        self.best_val_loss: float | None = None
        self.best_val_loss_epoch: int | None = None

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = trainer.callback_metrics
        current_epoch = int(trainer.current_epoch)

        monitored_value = _to_float(metrics.get(self.monitor_metric))
        if monitored_value is not None:
            is_better = False
            if self.best_val_metric is None:
                is_better = True
            elif self.monitor_mode == "min":
                is_better = monitored_value < self.best_val_metric
            else:
                is_better = monitored_value > self.best_val_metric

            if is_better:
                self.best_val_metric = monitored_value
                self.best_val_metric_epoch = current_epoch

        val_acc = _to_float(metrics.get("val_acc"))
        if val_acc is not None and (self.best_val_acc is None or val_acc > self.best_val_acc):
            self.best_val_acc = val_acc
            self.best_val_acc_epoch = current_epoch

        val_loss = _to_float(metrics.get("val_loss"))
        if val_loss is not None and (self.best_val_loss is None or val_loss < self.best_val_loss):
            self.best_val_loss = val_loss
            self.best_val_loss_epoch = current_epoch


def _normalize_tags(tags: Any) -> list[str] | None:
    if tags is None:
        return None
    if isinstance(tags, ListConfig):
        return [str(tag) for tag in tags]
    if isinstance(tags, list):
        return [str(tag) for tag in tags]
    return [str(tags)]


def create_wandb_logger(
    cfg: DictConfig,
    output_dir: str | Path,
    *,
    experiment: Any | None = None,
) -> WandbLogger | None:
    use_wandb = cfg.get("wandb", {}).get("use_wandb", False)
    if not use_wandb:
        logger.info("W&B disabled")
        return None

    wandb_cfg = cfg.get("wandb", {})
    logger_kwargs: dict[str, Any] = {
        "project": wandb_cfg.get("project_name", "BioBlobs"),
        "name": build_run_name(cfg),
        "save_dir": str(output_dir),
        "config": config_to_container(cfg, resolve=True),
    }

    entity = wandb_cfg.get("entity")
    if entity:
        logger_kwargs["entity"] = entity

    group = wandb_cfg.get("group")
    if group:
        logger_kwargs["group"] = group

    job_type = wandb_cfg.get("job_type")
    if job_type:
        logger_kwargs["job_type"] = job_type

    tags = _normalize_tags(wandb_cfg.get("tags"))
    if tags:
        logger_kwargs["tags"] = tags

    if experiment is not None:
        logger_kwargs["experiment"] = experiment

    wandb_logger = WandbLogger(**logger_kwargs)
    logger.success(
        "W&B logger initialized (project: {}, name: {})",
        logger_kwargs["project"],
        logger_kwargs["name"],
    )
    return wandb_logger


def _update_wandb_run_metadata(logger_instance: WandbLogger | None, result: dict[str, Any]) -> None:
    if logger_instance is None:
        return

    experiment = logger_instance.experiment
    if hasattr(experiment, "config"):
        experiment.config.update(
            {
                "resolved_cfg": result["resolved_config"],
            },
            allow_val_change=True,
        )
    if hasattr(experiment, "summary"):
        for key, value in result.items():
            if key == "resolved_config":
                continue
            experiment.summary[key] = value


def _checkpoint_filename_for_metric(monitor_metric: str) -> str:
    return f"best-{{epoch:02d}}-{{{monitor_metric}:.4f}}"


def _scalar_test_metrics(test_metrics: dict[str, Any]) -> dict[str, float]:
    payload: dict[str, float] = {}
    for key, value in test_metrics.items():
        numeric_value = _to_float(value)
        if numeric_value is not None:
            payload[key] = numeric_value
    return payload


def run_baseline_experiment(
    cfg: DictConfig,
    output_dir: str | Path,
    *,
    wandb_experiment: Any | None = None,
    save_last_checkpoint: bool = True,
    extra_callbacks: list[pl.Callback] | None = None,
) -> dict[str, Any]:
    """Run a baseline training job and return the tracked metrics."""
    output_dir = ensure_output_dir(output_dir)
    save_resolved_config(cfg, output_dir)

    logger.info("Configuration:\n{}", OmegaConf.to_yaml(cfg))
    set_seed(int(cfg.training.seed))
    logger.info("Using device: {}", "CUDA" if torch.cuda.is_available() else "CPU")
    logger.info("Output directory: {}", output_dir)

    train_dataset, val_dataset, test_dataset, num_classes = get_dataset(cfg)
    logger.success("Train: {} samples", len(train_dataset))
    logger.success("Val: {} samples", len(val_dataset))
    logger.success("Test: {} samples", len(test_dataset))
    logger.success("Num classes: {}", num_classes)

    train_loader = create_dataloader(
        train_dataset,
        cfg.training.batch_size,
        cfg.training.num_workers,
        shuffle=True,
        loader_cfg=cfg.training,
    )
    val_loader = create_dataloader(
        val_dataset,
        cfg.training.batch_size,
        cfg.training.num_workers,
        shuffle=False,
        loader_cfg=cfg.training,
    )
    test_loader = create_dataloader(
        test_dataset,
        cfg.training.batch_size,
        cfg.training.num_workers,
        shuffle=False,
        loader_cfg=cfg.training,
    )

    lightning_module = BioBlobsLightningModule(cfg, num_classes)
    total_params = sum(parameter.numel() for parameter in lightning_module.parameters())
    trainable_params = sum(
        parameter.numel()
        for parameter in lightning_module.parameters()
        if parameter.requires_grad
    )
    logger.success("Total parameters: {:,}", total_params)
    logger.success("Trainable parameters: {:,}", trainable_params)

    wandb_logger = create_wandb_logger(cfg, output_dir, experiment=wandb_experiment)
    monitor_metric = cfg.tasks.get("monitor_metric", "val_acc")
    monitor_mode = cfg.tasks.get("monitor_mode", "max")
    primary_metric_name = cfg.tasks.get("primary_metric", "accuracy")

    best_metric_tracker = BestMetricTracker(monitor_metric, monitor_mode)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir),
        filename=_checkpoint_filename_for_metric(monitor_metric),
        monitor=monitor_metric,
        mode=monitor_mode,
        save_top_k=1,
        save_last=save_last_checkpoint,
    )
    early_stopping_patience = cfg.training.get("early_stopping_patience", 15)
    early_stopping_min_delta = float(cfg.training.get("early_stopping_min_delta", 0.0))
    early_stopping_callback = EarlyStopping(
        monitor=monitor_metric,
        mode=monitor_mode,
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        verbose=True,
    )
    callbacks: list[pl.Callback] = [
        checkpoint_callback,
        best_metric_tracker,
        early_stopping_callback,
        *(extra_callbacks or []),
    ]
    logger.success(
        "Checkpoint callback configured (monitor: {}, mode: {})",
        monitor_metric,
        monitor_mode,
    )
    logger.success("Early stopping configured (patience: {})", early_stopping_patience)

    if cfg.training.get("use_cosine_schedule", False):
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))
        logger.success("LR monitor callback added")

    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        logger=wandb_logger,
        callbacks=callbacks,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
        deterministic=True,
        enable_progress_bar=True,
        default_root_dir=str(output_dir),
    )
    logger.success("Trainer configured:")
    logger.success("  - Max epochs: {}", cfg.training.epochs)
    logger.success("  - Accelerator: auto")
    logger.success("  - Deterministic: True")
    logger.success("  - Logs directory: {}", output_dir)

    trainer.fit(lightning_module, train_loader, val_loader)

    best_checkpoint_path = checkpoint_callback.best_model_path
    if not best_checkpoint_path:
        raise RuntimeError("Training finished without a best checkpoint path.")

    # The checkpoint was created by this run and contains OmegaConf objects, so
    # we opt into the full trusted restore path for evaluation on the best model.
    test_results = trainer.test(
        dataloaders=test_loader,
        ckpt_path=best_checkpoint_path,
        weights_only=False,
    )
    if not test_results:
        raise RuntimeError("Trainer returned no test metrics.")

    test_metrics = test_results[0]
    scalar_test_metrics = _scalar_test_metrics(test_metrics)
    primary_test_metric_key = f"test_{primary_metric_name}"
    if primary_metric_name == "accuracy" and primary_test_metric_key not in scalar_test_metrics:
        primary_test_metric_key = "test_acc"

    result = {
        "experiment_id": derive_experiment_id(cfg),
        "run_name": build_run_name(cfg),
        "output_dir": str(output_dir),
        "resolved_config_path": str(output_dir / "resolved_config.yaml"),
        "best_checkpoint_path": best_checkpoint_path,
        "best_val_metric": best_metric_tracker.best_val_metric,
        "best_val_metric_name": monitor_metric,
        "best_val_metric_epoch": best_metric_tracker.best_val_metric_epoch,
        "best_val_acc": best_metric_tracker.best_val_acc,
        "best_val_acc_epoch": best_metric_tracker.best_val_acc_epoch,
        "best_val_loss": best_metric_tracker.best_val_loss,
        "best_val_loss_epoch": best_metric_tracker.best_val_loss_epoch,
        "test_primary_metric": _to_float(scalar_test_metrics.get(primary_test_metric_key)),
        "test_primary_metric_name": primary_metric_name,
        "test_acc": _to_float(test_metrics.get("test_acc")),
        "test_loss": _to_float(test_metrics.get("test_loss")),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "resolved_config": config_to_container(cfg, resolve=True),
    }
    result[f"best_{monitor_metric}"] = best_metric_tracker.best_val_metric
    result.update(scalar_test_metrics)

    _update_wandb_run_metadata(wandb_logger, result)

    metrics_path = output_dir / "metrics_summary.json"
    save_json({key: value for key, value in result.items() if key != "resolved_config"}, metrics_path)
    result["metrics_summary_path"] = str(metrics_path)

    logger.info("Test Results:")
    if result["test_loss"] is not None:
        logger.info("  Test Loss: {:.4f}", result["test_loss"])
    if result["test_primary_metric"] is not None:
        logger.info(
            "  Test {}: {:.4f}",
            result["test_primary_metric_name"],
            result["test_primary_metric"],
        )
    if result["best_val_metric"] is not None:
        logger.info(
            "Best {}: {:.4f}",
            result["best_val_metric_name"],
            result["best_val_metric"],
        )
    logger.info("Best checkpoint: {}", result["best_checkpoint_path"])

    return result

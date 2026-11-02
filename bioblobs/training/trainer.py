"""Fit a model on one task with checkpoint selection on validation, then test once."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from bioblobs.data import class_weights, load_splits, make_loader
from bioblobs.training.lightning_module import BioBlobsModule


class AnyMetricEarlyStopping(pl.Callback):
    """Stop when none of several validation metrics improved for ``patience`` epochs.

    Args:
        monitors: Metric names; losses (names ending in ``loss``) are minimized, others maximized.
        patience: Epochs without improvement in any monitored metric before stopping.
    """

    def __init__(self, monitors: list[str], patience: int) -> None:
        self.monitors = monitors
        self.patience = patience
        self.best = {m: -float("inf") for m in monitors}
        self.wait = 0

    def on_validation_epoch_end(self, trainer: pl.Trainer, module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        improved = False
        for name in self.monitors:
            value = trainer.callback_metrics.get(name)
            if value is None:
                continue
            signed = -float(value) if name.endswith("loss") else float(value)
            if signed > self.best[name]:
                self.best[name] = signed
                improved = True
        self.wait = 0 if improved else self.wait + 1
        if self.wait >= self.patience:
            trainer.should_stop = True


def train(cfg: DictConfig, output_dir: str | Path) -> dict[str, Any]:
    """Train, select the best validation checkpoint and evaluate it on the test split.

    Args:
        cfg: Composed config.
        output_dir: Run directory for checkpoints, logs and ``metrics.json``.

    Returns:
        Best validation value, checkpoint path and test metrics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml", resolve=True)
    pl.seed_everything(int(cfg.training.seed), workers=True)

    train_cfg, sel = cfg.training, cfg.selection
    splits = load_splits(cfg)
    weights = class_weights(splits.class_counts, cfg.objective.get("class_balanced_beta"))
    module = BioBlobsModule(cfg, splits.num_classes, weights)
    logger.info(
        "train={} val={} test={} classes={}",
        len(splits.train),
        len(splits.val),
        len(splits.test or []),
        splits.num_classes,
    )

    def loader(dataset, shuffle: bool):
        return make_loader(
            dataset,
            train_cfg.batch_size,
            shuffle,
            num_workers=int(train_cfg.get("num_workers", 0)),
            bucket_by_size=bool(train_cfg.get("bucket_by_size", False)),
            seed=int(train_cfg.seed),
        )

    monitor = f"val_{sel.metric}"
    checkpoint = ModelCheckpoint(
        dirpath=output_dir, filename="best", monitor=monitor, mode="max", save_last=True
    )
    early_monitors = [f"val_{m}" for m in (sel.get("early_stop_any_of") or [sel.metric])]
    if len(early_monitors) > 1:
        stopper: pl.Callback = AnyMetricEarlyStopping(early_monitors, int(sel.patience))
    else:
        stopper = EarlyStopping(
            monitor=early_monitors[0],
            mode="max",
            patience=int(sel.patience),
            min_delta=float(sel.get("min_delta", 0.0)),
        )
    trainer = pl.Trainer(
        max_epochs=int(train_cfg.epochs),
        callbacks=[checkpoint, stopper],
        logger=CSVLogger(output_dir, name="logs"),
        accelerator="auto",
        devices=1,
        deterministic=bool(train_cfg.get("deterministic", True)),
        enable_progress_bar=bool(train_cfg.get("progress_bar", False)),
        default_root_dir=output_dir,
    )
    trainer.fit(module, loader(splits.train, True), loader(splits.val, False))
    result: dict[str, Any] = {
        "best_checkpoint": checkpoint.best_model_path,
        f"best_{monitor}": float(checkpoint.best_model_score or float("nan")),
    }
    if splits.test is not None:
        tested = trainer.test(
            module, loader(splits.test, False), ckpt_path="best", weights_only=False
        )
        result["test"] = {k: float(v) for k, v in tested[0].items()}
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    logger.success("Done: {}", json.dumps(result))
    return result


def load_module(checkpoint: str | Path, device: str = "cpu") -> BioBlobsModule:
    """Load a trained module from a checkpoint written by ``train``."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    hp = ckpt["hyper_parameters"]
    module = BioBlobsModule(OmegaConf.create(hp["cfg"]), hp["num_classes"])
    module.load_state_dict(ckpt["state_dict"], strict=False)
    return module.eval()

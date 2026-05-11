from __future__ import annotations

from collections import defaultdict

from tqdm import tqdm
import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig
from loguru import logger

from bioblobs_framework import BioBlobsFramework

from .fmax_metric import FMaxMetric
from .lr_schedule import get_cosine_schedule_with_warmup
from .task_metrics import compute_multiclass_metrics, compute_multilabel_metrics


def _prepare_labels(
    raw_labels: torch.Tensor,
    logits: torch.Tensor,
    *,
    problem_type: str,
    num_classes: int,
) -> torch.Tensor:
    if problem_type == "multi_label":
        return raw_labels.view(logits.size(0), num_classes).float()
    return raw_labels.view(-1).long()


def train_epoch(framework, dataloader, optimizer, device):
    """Legacy non-Lightning train loop for multi-class tasks."""
    framework.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        batch = batch.to(device)
        logits, _ = framework(batch)
        loss = framework.compute_cross_entropy_loss(logits, batch.y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = batch.y.size(0)
        total_loss += loss.item() * batch_size

        pred = torch.argmax(logits, dim=1)
        total_correct += (pred == batch.y).sum().item()
        total_samples += batch_size

        current_acc = total_correct / total_samples
        progress_bar.set_postfix(
            {
                "loss": f"{total_loss / total_samples:.4f}",
                "acc": f"{current_acc:.4f}",
            }
        )

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    logger.debug("Training epoch complete: loss={:.4f}, acc={:.4f}", avg_loss, avg_acc)
    return avg_loss, avg_acc


def evaluate(framework, dataloader, device):
    """Legacy non-Lightning eval loop for multi-class tasks."""
    framework.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Evaluating")
    with torch.no_grad():
        for batch in progress_bar:
            batch = batch.to(device)
            logits, _ = framework(batch)
            loss = framework.compute_cross_entropy_loss(logits, batch.y)

            batch_size = batch.y.size(0)
            total_loss += loss.item() * batch_size

            pred = torch.argmax(logits, dim=1)
            total_correct += (pred == batch.y).sum().item()
            total_samples += batch_size

            current_acc = total_correct / total_samples
            progress_bar.set_postfix(
                {
                    "loss": f"{total_loss / total_samples:.4f}",
                    "acc": f"{current_acc:.4f}",
                }
            )

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    logger.debug("Evaluation complete: loss={:.4f}, acc={:.4f}", avg_loss, avg_acc)
    return avg_loss, avg_acc


class BioBlobsLightningModule(pl.LightningModule):
    """Unified Lightning module supporting both multiclass and multilabel tasks."""

    def __init__(self, cfg: DictConfig, num_classes: int):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        self.num_classes = num_classes
        self.model = BioBlobsFramework(cfg, num_classes)

        self.problem_type = cfg.tasks.get("problem_type", "multi_class")
        self.loss_name = cfg.tasks.get("loss", "cross_entropy")
        self.metrics = list(cfg.tasks.get("metrics", []))
        self.primary_metric = cfg.tasks.get("primary_metric", "accuracy")

        self.has_partitioner = cfg.partitioners.get("enabled", False)

        if self.problem_type == "multi_label":
            self.criterion = nn.BCEWithLogitsLoss()
            self.loss_metric_name = "bce_loss"
            self.fmax_metric = FMaxMetric()
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.loss_metric_name = "cross_entropy_loss"
            self.fmax_metric = None

        self._epoch_predictions: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._epoch_targets: dict[str, list[torch.Tensor]] = defaultdict(list)

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        logits, extra = self(batch)
        labels = _prepare_labels(
            batch.y,
            logits,
            problem_type=self.problem_type,
            num_classes=self.num_classes,
        )

        task_loss = self._compute_task_loss(logits, labels)
        loss = task_loss
        if self.has_partitioner:
            partitioner_loss = self._compute_partitioner_loss(extra)
            if partitioner_loss is not None:
                loss = loss + partitioner_loss

        batch_size = logits.size(0)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
            prog_bar=True,
        )
        self.log(
            f"train_{self.loss_metric_name}",
            task_loss,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )

        self._accumulate_epoch_outputs("train", logits, labels)
        self._log_partitioner_metrics("train", extra, batch_size=batch_size)
        return loss

    def on_train_epoch_start(self) -> None:
        if not self.has_partitioner:
            return

        partitioner = getattr(self.model, "partitioner", None)
        if partitioner is None:
            return

        if hasattr(partitioner, "set_epoch"):
            partitioner.set_epoch(self.current_epoch)

    def validation_step(self, batch, batch_idx):
        logits, extra = self(batch)
        labels = _prepare_labels(
            batch.y,
            logits,
            problem_type=self.problem_type,
            num_classes=self.num_classes,
        )

        task_loss = self._compute_task_loss(logits, labels)
        loss = task_loss
        if self.has_partitioner:
            partitioner_loss = self._compute_partitioner_loss(extra)
            if partitioner_loss is not None:
                loss = loss + partitioner_loss

        batch_size = logits.size(0)
        self.log("val_loss", loss, on_step=False, on_epoch=True, batch_size=batch_size, prog_bar=True)
        self.log(
            f"val_{self.loss_metric_name}",
            task_loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        self._accumulate_epoch_outputs("val", logits, labels)
        self._log_partitioner_metrics("val", extra, batch_size=batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        logits, extra = self(batch)
        labels = _prepare_labels(
            batch.y,
            logits,
            problem_type=self.problem_type,
            num_classes=self.num_classes,
        )

        task_loss = self._compute_task_loss(logits, labels)
        loss = task_loss
        if self.has_partitioner:
            partitioner_loss = self._compute_partitioner_loss(extra)
            if partitioner_loss is not None:
                loss = loss + partitioner_loss

        batch_size = logits.size(0)
        self.log("test_loss", loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(
            f"test_{self.loss_metric_name}",
            task_loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        self._accumulate_epoch_outputs("test", logits, labels)
        self._log_partitioner_metrics("test", extra, batch_size=batch_size)
        return loss

    def on_train_epoch_end(self) -> None:
        self._log_epoch_metrics("train")

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics("val")

    def on_test_epoch_end(self) -> None:
        self._log_epoch_metrics("test")

    def configure_optimizers(self):
        lr = self.cfg.training.get("lr", 1e-3)
        weight_decay = self.cfg.training.get("weight_decay", 0.0)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        if self.cfg.training.get("use_cosine_schedule", False):
            warmup_epochs = self.cfg.training.get("warmup_epochs", 0)
            max_epochs = self.cfg.training.get("epochs", 100)

            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                warmup_epochs=warmup_epochs,
                max_epochs=max_epochs,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return optimizer

    def _compute_task_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.problem_type == "multi_label":
            return self.criterion(logits, labels)
        return self.criterion(logits, labels.long())

    def _accumulate_epoch_outputs(
        self,
        split: str,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        self._epoch_predictions[split].append(logits.detach().cpu())
        self._epoch_targets[split].append(labels.detach().cpu())

    def _compute_epoch_metrics(self, split: str) -> dict[str, float]:
        logits = torch.cat(self._epoch_predictions[split], dim=0).numpy()
        labels = torch.cat(self._epoch_targets[split], dim=0).numpy()
        if self.problem_type == "multi_label":
            return compute_multilabel_metrics(labels, logits, metric=self.fmax_metric)
        return compute_multiclass_metrics(labels, logits)

    def _log_epoch_metrics(self, split: str) -> None:
        if not self._epoch_predictions[split]:
            return

        metrics = self._compute_epoch_metrics(split)
        for metric_name, metric_value in metrics.items():
            log_name = f"{split}_{metric_name}"
            self.log(
                log_name,
                metric_value,
                on_step=False,
                on_epoch=True,
                prog_bar=(split != "test" and metric_name == self.primary_metric),
            )

            if metric_name == "accuracy":
                self.log(
                    f"{split}_acc",
                    metric_value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=(split != "test"),
                )

        self._epoch_predictions[split].clear()
        self._epoch_targets[split].clear()

    def _compute_partitioner_loss(self, extra: dict) -> torch.Tensor | None:
        return extra.get("partitioner_loss")

    def _compute_partitioner_metrics(self, extra: dict) -> dict | None:
        return extra.get("partitioner_metrics")

    def _log_partitioner_metrics(self, split: str, extra: dict, *, batch_size: int) -> None:
        if not self.has_partitioner:
            return

        partitioner_metrics = self._compute_partitioner_metrics(extra)
        if partitioner_metrics is None:
            return

        for metric_name, metric_value in partitioner_metrics.items():
            self.log(
                f"{split}_{metric_name}",
                metric_value,
                on_step=(split == "train"),
                on_epoch=True,
                batch_size=batch_size,
            )

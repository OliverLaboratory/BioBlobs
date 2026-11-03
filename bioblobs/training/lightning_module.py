"""Lightning module: training objective, epoch-level metrics and optimization.

One module covers every reported setting; the config selects the loss terms (``objective``),
the optimizer and learning-rate schedule (``training``) and the validation metric used for
checkpoint selection and early stopping (``selection``).
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

from bioblobs.model import ModelOutput, build_model
from bioblobs.objectives import blob_alignment, hoyer_sparsity, seed_alignment
from bioblobs.training.metrics import fmax, multiclass_metrics


class BioBlobsModule(pl.LightningModule):
    """Trains a ``BioBlobs`` model (or a comparator) for protein classification.

    Args:
        cfg: Composed config with model groups plus ``tasks``, ``objective`` and ``training``.
        num_classes: Number of output classes.
        class_weights: Optional per-class weights of the cross-entropy (class-balanced loss).
    """

    def __init__(
        self, cfg: DictConfig, num_classes: int, class_weights: torch.Tensor | None = None
    ) -> None:
        super().__init__()
        self.save_hyperparameters({"cfg": cfg, "num_classes": num_classes})
        self.cfg = cfg
        self.num_classes = num_classes
        self.model = build_model(cfg, num_classes)
        self.multi_label = cfg.tasks.problem_type == "multi_label"
        self.register_buffer("class_weights", class_weights, persistent=False)
        self.objective = cfg.get("objective", {})
        self._outputs: dict[str, list] = defaultdict(list)

    def forward(self, batch: Batch) -> ModelOutput:
        """Run the model."""
        return self.model(batch)

    def loss(self, batch: Batch, out: ModelOutput) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Total loss and its weighted terms for one batch."""
        if self.multi_label:
            task = F.binary_cross_entropy_with_logits(
                out.logits, batch.y.view_as(out.logits).float()
            )
        else:
            task = F.cross_entropy(out.logits, batch.y.view(-1).long(), weight=self.class_weights)
        terms = {"task": task}
        part = out.partition
        weight = self.objective.get
        if part is not None and part.candidate_mask is not None and weight("hoyer_weight", 0):
            terms["hoyer"] = weight("hoyer_weight") * hoyer_sparsity(
                part.memberships, part.candidate_mask, part.blob_mask
            )
        if out.quantized is not None and weight("rvq_weight", 0):
            terms["rvq"] = weight("rvq_weight") * out.quantized.commitment
        if weight("seed_align_weight", 0) or weight("blob_align_weight", 0):
            site = _site_mask(batch)
            if weight("seed_align_weight", 0):
                terms["seed_align"] = weight("seed_align_weight") * seed_alignment(
                    part.seed_logits, site, part.residue_mask
                )
            if weight("blob_align_weight", 0):
                terms["blob_align"] = weight("blob_align_weight") * blob_alignment(
                    part.memberships, site, part.residue_mask
                )
        return sum(terms.values()), terms

    def on_train_epoch_start(self) -> None:
        self.model.set_epoch(self.current_epoch)

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        out = self(batch)
        loss, terms = self.loss(batch, out)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at epoch {self.current_epoch}")
        size = out.logits.shape[0]
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=size)
        for name, value in terms.items():
            self.log(f"train_{name}", value, on_step=False, on_epoch=True, batch_size=size)
        self._collect("train", batch, out)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        out = self(batch)
        loss, _ = self.loss(batch, out)
        self.log("val_loss", loss, on_epoch=True, batch_size=out.logits.shape[0])
        self._collect("val", batch, out)

    def test_step(self, batch: Batch, batch_idx: int) -> None:
        out = self(batch)
        self._collect("test", batch, out)

    def on_train_epoch_end(self) -> None:
        self._log_epoch("train")

    def on_validation_epoch_end(self) -> None:
        self._log_epoch("val")

    def on_test_epoch_end(self) -> None:
        self._log_epoch("test")

    def _collect(self, split: str, batch: Batch, out: ModelOutput) -> None:
        self._outputs[f"{split}_logits"].append(out.logits.detach().float().cpu())
        self._outputs[f"{split}_labels"].append(batch.y.detach().cpu())
        if getattr(batch, "residue_target", None) is not None and out.partition is not None:
            site = _site_mask(batch).float()
            covered = out.partition.memberships.max(dim=1).values * out.partition.residue_mask
            annotated = site.sum(-1) > 0
            recall = (covered * site).sum(-1) / site.sum(-1).clamp_min(1.0)
            self._outputs[f"{split}_site_recall"].append(recall[annotated].detach().cpu())

    def _log_epoch(self, split: str) -> None:
        logits = self._outputs.pop(f"{split}_logits", [])
        if not logits:
            return
        logits_np = torch.cat(logits).numpy()
        labels = torch.cat(self._outputs.pop(f"{split}_labels"))
        if self.multi_label:
            metrics = fmax(labels.view(logits_np.shape).numpy(), logits_np)
        else:
            metrics = multiclass_metrics(labels.view(-1).numpy(), logits_np, self.num_classes)
        recalls = self._outputs.pop(f"{split}_site_recall", [])
        if recalls:
            metrics["site_soft_recall"] = float(torch.cat(recalls).mean())
            metrics["composite"] = 0.5 * metrics["macro_f1"] + 0.5 * metrics["site_soft_recall"]
        for name, value in metrics.items():
            self.log(f"{split}_{name}", value, prog_bar=name in {"macro_f1", "fmax"})

    def configure_optimizers(self) -> dict:
        train = self.cfg.training
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer_cls = (
            torch.optim.AdamW if train.get("optimizer", "adam") == "adamw" else torch.optim.Adam
        )
        optimizer = optimizer_cls(params, lr=train.lr, weight_decay=train.get("weight_decay", 0.0))
        schedule = train.get("schedule", "warmup_cosine")
        if schedule == "none":
            return {"optimizer": optimizer}
        if schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=train.epochs, eta_min=train.lr * train.get("eta_min_ratio", 0.01)
            )
        elif schedule == "warmup_cosine":
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, warmup_cosine(int(train.get("warmup_epochs", 0)), int(train.epochs))
            )
        else:
            raise ValueError(f"Unknown schedule {schedule!r}")
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


def warmup_cosine(warmup_epochs: int, max_epochs: int):
    """Epoch-level LR factor: linear warmup (floored at 1e-6), then cosine decay to zero."""

    def factor(epoch: int) -> float:
        if epoch < warmup_epochs:
            return max(1e-6, epoch / max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return factor


def _site_mask(batch: Batch) -> torch.Tensor:
    return to_dense_batch(batch.residue_target.float(), batch.batch)[0] > 0.5


def _as_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

"""Multi-label variant of MultiStageBioBlobsLightning for Gene Ontology dataset."""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from utils.train_w_codebook import MultiStageBioBlobsLightning
from utils.fmax_metric import FMaxMetric


class MultiStageBioBlobsMultiLabelLightning(MultiStageBioBlobsLightning):
    """Multi-stage BioBlobs Lightning module for multi-label classification (GO dataset)."""

    def __init__(self, model_cfg, train_cfg, num_classes):
        super().__init__(model_cfg, train_cfg, num_classes)
        
        # Override criterion for multi-label
        self.criterion = nn.BCEWithLogitsLoss()
        
        # FMax metric for multi-label evaluation
        self.fmax_metric = FMaxMetric()
        self.num_classes = num_classes
        
        # Track accumulated predictions and targets for epoch-level metrics
        self.train_predictions = []
        self.train_targets = []
        self.val_predictions = []
        self.val_targets = []
        self.test_predictions = []
        self.test_targets = []

    def training_step(self, batch, batch_idx):
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        seq = (
            batch.seq
            if hasattr(batch, "seq") and hasattr(self.model, "sequence_embedding")
            else None
        )

        # Forward pass
        if self.bypass_codebook:
            logits, assignment_matrix, extra = self._forward_bypass_codebook(
                h_V, batch.edge_index, h_E, seq, batch.batch
            )
        else:
            logits, assignment_matrix, extra = self.model(
                h_V, batch.edge_index, h_E, seq, batch.batch
            )

        # Fix label shape for multi-label: PyTorch Geometric concatenates but we need to stack
        batch_size = logits.size(0)
        num_classes = logits.size(1)
        labels = batch.y.view(batch_size, num_classes).float()

        # Multi-label classification loss
        bce_loss = self.criterion(logits, labels)
        vq_loss = extra.get("vq_loss", 0.0)
        total_loss = bce_loss + vq_loss

        # Store predictions and targets for epoch-level metrics
        probs = torch.sigmoid(logits)
        self.train_predictions.append(probs.detach().cpu())
        self.train_targets.append(labels.cpu())
        
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train_bce_loss", bce_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train_vq_loss", vq_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("stage", float(self.current_stage), on_step=True, on_epoch=True, batch_size=batch_size)

        return total_loss

    def validation_step(self, batch, batch_idx):
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        seq = (
            batch.seq
            if hasattr(batch, "seq") and hasattr(self.model, "sequence_embedding")
            else None
        )

        if self.bypass_codebook:
            logits, assignment_matrix, extra = self._forward_bypass_codebook(
                h_V, batch.edge_index, h_E, seq, batch.batch
            )
        else:
            logits, assignment_matrix, extra = self.model(
                h_V, batch.edge_index, h_E, seq, batch.batch
            )

        # Fix label shape for multi-label: PyTorch Geometric concatenates but we need to stack
        batch_size = logits.size(0)
        num_classes = logits.size(1)
        labels = batch.y.view(batch_size, num_classes).float()

        bce_loss = self.criterion(logits, labels)
        vq_loss = extra.get("vq_loss", 0.0)
        total_loss = bce_loss + vq_loss

        # Store predictions and targets for epoch-level metrics
        probs = torch.sigmoid(logits)
        self.val_predictions.append(probs.detach().cpu())
        self.val_targets.append(labels.cpu())
        
        self.log("val_loss", total_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("val_bce_loss", bce_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("val_vq_loss", vq_loss, on_step=False, on_epoch=True, batch_size=batch_size)

        return total_loss

    def test_step(self, batch, batch_idx):
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        seq = (
            batch.seq
            if hasattr(batch, "seq") and hasattr(self.model, "sequence_embedding")
            else None
        )

        if self.bypass_codebook:
            logits, assignment_matrix, extra = self._forward_bypass_codebook(
                h_V, batch.edge_index, h_E, seq, batch.batch
            )
            # For bypass mode, we can't get attention importance scores
            cluster_importance = None
        else:
            logits, assignment_matrix, extra, cluster_importance = self.model(
                h_V, batch.edge_index, h_E, seq, batch.batch, return_importance=True
            )

        # Fix label shape for multi-label: PyTorch Geometric concatenates but we need to stack
        batch_size = logits.size(0)
        num_classes = logits.size(1)
        labels = batch.y.view(batch_size, num_classes).float()

        bce_loss = self.criterion(logits, labels)
        vq_loss = extra.get("vq_loss", 0.0)
        total_loss = bce_loss + vq_loss

        # Store predictions and targets for epoch-level metrics
        probs = torch.sigmoid(logits)
        self.test_predictions.append(probs.detach().cpu())
        self.test_targets.append(labels.cpu())
        
        self.log("test_loss", total_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("test_bce_loss", bce_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("test_vq_loss", vq_loss, on_step=False, on_epoch=True, batch_size=batch_size)

        # Log importance statistics if available
        if cluster_importance is not None:
            importance_max = cluster_importance.max(dim=1)[0].mean()
            importance_std = cluster_importance.std(dim=1).mean()
            importance_entropy = -(
                cluster_importance * torch.log(cluster_importance + 1e-10)
            ).sum(dim=1).mean()

            self.log("test_importance_max", importance_max, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log("test_importance_std", importance_std, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log("test_importance_entropy", importance_entropy, on_step=False, on_epoch=True, batch_size=batch_size)

        return total_loss

    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics.get("train_loss_epoch", 0.0)
        
        if self.train_predictions and self.train_targets:
            # Concatenate all predictions and targets
            all_predictions = torch.cat(self.train_predictions, dim=0).numpy()
            all_targets = torch.cat(self.train_targets, dim=0).numpy()
            
            # Compute epoch-level FMax metrics
            try:
                fmax = self.fmax_metric.fmax(all_targets, all_predictions)
                precision = self.fmax_metric.precision(all_targets, all_predictions)
                recall = self.fmax_metric.recall(all_targets, all_predictions)
                
                self.log("train_fmax", fmax, on_epoch=True)
                self.log("train_precision", precision, on_epoch=True)
                self.log("train_recall", recall, on_epoch=True)
                
                stage_name = ["BASELINE", "JOINT-FINE-TUNING"][self.current_stage]
                print(
                    f"[{stage_name}] Epoch {self.stage_epoch - 1:3d} | Train Loss: {train_loss:.4f} | Train FMax: {fmax:.4f}"
                )
            except Exception as e:
                print(f"Warning: Could not compute training FMax metrics: {e}")
                stage_name = ["BASELINE", "JOINT-FINE-TUNING"][self.current_stage]
                print(
                    f"[{stage_name}] Epoch {self.stage_epoch - 1:3d} | Train Loss: {train_loss:.4f} | Train FMax: N/A"
                )
            
            # Clear accumulated data
            self.train_predictions.clear()
            self.train_targets.clear()
        else:
            stage_name = ["BASELINE", "JOINT-FINE-TUNING"][self.current_stage]
            print(
                f"[{stage_name}] Epoch {self.stage_epoch - 1:3d} | Train Loss: {train_loss:.4f}"
            )

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get("val_loss", 0.0)
        
        if self.val_predictions and self.val_targets:
            # Concatenate all predictions and targets
            all_predictions = torch.cat(self.val_predictions, dim=0).numpy()
            all_targets = torch.cat(self.val_targets, dim=0).numpy()
            
            # Compute epoch-level FMax metrics
            try:
                fmax = self.fmax_metric.fmax(all_targets, all_predictions)
                precision = self.fmax_metric.precision(all_targets, all_predictions)
                recall = self.fmax_metric.recall(all_targets, all_predictions)
                
                self.log("val_fmax", fmax, on_epoch=True, prog_bar=True)
                self.log("val_precision", precision, on_epoch=True)
                self.log("val_recall", recall, on_epoch=True)
                
                print(f"{'':15} | Val Loss:   {val_loss:.4f} | Val FMax:  {fmax:.4f}")
            except Exception as e:
                print(f"Warning: Could not compute validation FMax metrics: {e}")
                print(f"{'':15} | Val Loss:   {val_loss:.4f} | Val FMax:  N/A")
            
            # Clear accumulated data
            self.val_predictions.clear()
            self.val_targets.clear()
        else:
            print(f"{'':15} | Val Loss:   {val_loss:.4f}")
        
        print("-" * 65)

    def on_test_epoch_end(self):
        if self.test_predictions and self.test_targets:
            # Concatenate all predictions and targets
            all_predictions = torch.cat(self.test_predictions, dim=0).numpy()
            all_targets = torch.cat(self.test_targets, dim=0).numpy()
            
            # Compute epoch-level FMax metrics
            try:
                fmax = self.fmax_metric.fmax(all_targets, all_predictions)
                precision = self.fmax_metric.precision(all_targets, all_predictions)
                recall = self.fmax_metric.recall(all_targets, all_predictions)
                
                self.log("test_fmax", fmax, on_epoch=True)
                self.log("test_precision", precision, on_epoch=True)
                self.log("test_recall", recall, on_epoch=True)
                
                print("🏆 Test Results:")
                print(f"   FMax: {fmax:.4f}")
                print(f"   Best Precision: {precision:.4f}")
                print(f"   Best Recall: {recall:.4f}")
            except Exception as e:
                print(f"Warning: Could not compute test FMax metrics: {e}")
                self.log("test_fmax", 0.0, on_epoch=True)
                self.log("test_precision", 0.0, on_epoch=True)
                self.log("test_recall", 0.0, on_epoch=True)
            
            # Clear accumulated data
            self.test_predictions.clear()
            self.test_targets.clear()

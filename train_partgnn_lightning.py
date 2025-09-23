"""
PyTorch Lightning module for training PartGNN (GNN-based Partoken) models.

This module supports three types of tasks:
1. Binary classification (Enzyme Commission)
2. Multi-class classification (Structural Class/SCOP)
3. Multi-label classification (Gene Ontology)
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig

from gnn_partoken import PartTokenGNNModel
from utils.fmax_metric import FMaxMetric


"""
PyTorch Lightning module for training PartGNN (GNN-based Partoken) models.

This module supports three types of tasks:
1. Binary classification (Enzyme Commission)
2. Multi-class classification (Structural Class/SCOP)
3. Multi-label classification (Gene Ontology)
"""


class PartGNNLightning(pl.LightningModule):
    """
    PartGNN Lightning module that trains GNN + partitioner + global-cluster attention fusion.
    
    This module handles binary and multi-class classification tasks.
    """
    
    def __init__(self, model_cfg: DictConfig, train_cfg: DictConfig, num_classes: int):
        super().__init__()
        self.save_hyperparameters()
        
        # Create PartGNN model
        self.model = PartTokenGNNModel(
            embed_dim=model_cfg.get('embed_dim', 128),
            num_classes=num_classes,
            num_layers=model_cfg.get('num_layers', 3),
            drop_rate=model_cfg.get('drop_rate', 0.1),
            pooling=model_cfg.get('pooling', 'mean'),
            use_edge_attr=model_cfg.get('use_edge_attr', True),
            edge_attr_dim=model_cfg.get('edge_attr_dim', 1),
            pe=model_cfg.get('pe', 'learned'),
            # Partitioner hyperparameters
            max_clusters=model_cfg.get('max_clusters', 5),
            nhid=model_cfg.get('nhid', 50),
            k_hop=model_cfg.get('k_hop', 2),
            cluster_size_max=model_cfg.get('cluster_size_max', 15),
            termination_threshold=model_cfg.get('termination_threshold', 0.95),
            tau_init=model_cfg.get('tau_init', 1.0),
            tau_min=model_cfg.get('tau_min', 0.1),
            tau_decay=model_cfg.get('tau_decay', 0.95),
            # Codebook hyperparameters
            codebook_size=model_cfg.get('codebook_size', 512),
            codebook_dim=model_cfg.get('codebook_dim', None),
            codebook_beta=model_cfg.get('codebook_beta', 0.25),
            codebook_decay=model_cfg.get('codebook_decay', 0.99),
            codebook_eps=model_cfg.get('codebook_eps', 1e-5),
            codebook_distance=model_cfg.get('codebook_distance', 'l2'),
            codebook_cosine_normalize=model_cfg.get('codebook_cosine_normalize', False),
            # Loss weights
            lambda_vq=model_cfg.get('lambda_vq', 1.0),
            lambda_ent=model_cfg.get('lambda_ent', 0.0),
            lambda_psc=model_cfg.get('lambda_psc', 1e-2),
            lambda_card=model_cfg.get('lambda_card', 0.005),
            psc_temp=model_cfg.get('psc_temp', 0.3)
        )
        
        self.train_cfg = train_cfg
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, batch):
        """Forward pass through PartGNN model."""
        return self.model(batch)
    
    def training_step(self, batch, batch_idx):
        """Training step for PartGNN."""
        # Forward pass
        logits, assignment_matrix, extra = self.forward(batch)
        
        # Compute total loss
        labels = batch.y
        total_loss, metrics = self.model.compute_total_loss(logits, labels, extra)
        
        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        # Get batch size
        batch_size = labels.size(0)
        
        # Log metrics
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_ce_loss', metrics['ce_loss'], on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_vq_loss', metrics['vq_loss'], on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_acc', acc, on_step=True, on_epoch=True, batch_size=batch_size)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step for PartGNN."""
        # Forward pass
        logits, assignment_matrix, extra = self.forward(batch)
        
        # Compute total loss
        labels = batch.y
        total_loss, metrics = self.model.compute_total_loss(logits, labels, extra)
        
        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        # Get batch size
        batch_size = labels.size(0)
        
        # Log metrics
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val_ce_loss', metrics['ce_loss'], on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val_vq_loss', metrics['vq_loss'], on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val_acc', acc, on_step=False, on_epoch=True, batch_size=batch_size)
        
        return total_loss
    
    def test_step(self, batch, batch_idx):
        """Test step for PartGNN."""
        # Forward pass
        logits, assignment_matrix, extra = self.forward(batch)
        
        # Compute total loss
        labels = batch.y
        total_loss, metrics = self.model.compute_total_loss(logits, labels, extra)
        
        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        # Get batch size
        batch_size = labels.size(0)
        
        # Log metrics
        self.log('test_loss', total_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_ce_loss', metrics['ce_loss'], on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_vq_loss', metrics['vq_loss'], on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_acc', acc, on_step=False, on_epoch=True, batch_size=batch_size)
        
        return total_loss
    
    def on_train_epoch_start(self):
        """Update partitioner temperature each epoch."""
        self.model.update_epoch()
    
    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics.get('train_loss_epoch', 0.0)
        train_acc = self.trainer.callback_metrics.get('train_acc_epoch', 0.0)
        print(f"[PARTGNN] Epoch {self.current_epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    
    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get('val_loss', 0.0)
        val_acc = self.trainer.callback_metrics.get('val_acc', 0.0)
        print(f"{'':15} | Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print("-" * 65)
    
    def configure_optimizers(self):
        """Configure optimizer - simple Adam without scheduling."""
        lr = float(self.train_cfg.get('learning_rate', 1e-3))
        return torch.optim.Adam(self.parameters(), lr=lr)


class PartGNNMultiLabelLightning(PartGNNLightning):
    """
    PartGNN Lightning module for multi-label classification (Gene Ontology dataset).
    
    This class extends PartGNNLightning to handle multi-label classification with:
    - BCEWithLogitsLoss instead of CrossEntropyLoss
    - FMax metric instead of accuracy
    - Updated logging for multi-label metrics
    """
    
    def __init__(self, model_cfg: DictConfig, train_cfg: DictConfig, num_classes: int):
        super().__init__(model_cfg, train_cfg, num_classes)
        
        # Replace criterion for multi-label classification
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Initialize FMax metric
        self.fmax_metric = FMaxMetric()
        
        # Track accumulated predictions and targets for epoch-level metrics
        self.val_predictions = []
        self.val_targets = []
        self.test_predictions = []
        self.test_targets = []
    
    def _compute_multilabel_loss(self, logits, labels, extra):
        """Compute multi-label loss with proper label formatting."""
        # Fix label shape for multi-label: PyTorch Geometric concatenates but we need to stack
        batch_size = logits.size(0)
        num_classes = logits.size(1)
        
        # Reshape labels to [batch_size, num_classes]
        if labels.dim() == 1:
            labels = labels.view(batch_size, num_classes)
        labels = labels.float()
        
        # Multi-label classification loss
        bce_loss = self.criterion(logits, labels)
        
        # Add VQ and regularization losses
        vq_loss = extra.get('vq_loss', torch.tensor(0.0, device=logits.device))
        
        total_loss = bce_loss + self.model.lambda_vq * vq_loss
        
        metrics = {
            'ce_loss': bce_loss,  # Using ce_loss name for consistency
            'vq_loss': vq_loss
        }
        
        return total_loss, metrics, labels
    
    def _compute_multilabel_metrics(self, logits, labels):
        """Compute multi-label metrics (FMax, precision, recall)."""
        # Compute metrics using sigmoid probabilities
        probs = torch.sigmoid(logits)
        
        # Convert to numpy for FMax computation
        y_true_np = labels.cpu().numpy()
        y_pred_np = probs.detach().cpu().numpy()
        
        # Compute FMax metrics
        try:
            fmax_score = self.fmax_metric.fmax(y_true_np, y_pred_np)
            precision_score = self.fmax_metric.precision(y_true_np, y_pred_np, 0.5)
            recall_score = self.fmax_metric.recall(y_true_np, y_pred_np, 0.5)
        except Exception:
            # Fallback in case of numerical issues
            fmax_score = 0.0
            precision_score = 0.0
            recall_score = 0.0
        
        return fmax_score, precision_score, recall_score
    
    def training_step(self, batch, batch_idx):
        """Training step for multi-label PartGNN."""
        # Forward pass
        logits, assignment_matrix, extra = self.forward(batch)
        
        # Compute multi-label loss
        total_loss, metrics, labels = self._compute_multilabel_loss(logits, batch.y, extra)
        
        # Compute multi-label metrics
        fmax_score, precision_score, recall_score = self._compute_multilabel_metrics(logits, labels)
        
        # Get batch size
        batch_size = labels.size(0)
        
        # Log metrics (using same naming as parent class for consistency)
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_ce_loss', metrics['ce_loss'], on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_vq_loss', metrics['vq_loss'], on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_acc', fmax_score, on_step=True, on_epoch=True, batch_size=batch_size)  # Using fmax as accuracy
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step for multi-label PartGNN."""
        # Forward pass
        logits, assignment_matrix, extra = self.forward(batch)
        
        # Compute multi-label loss
        total_loss, metrics, labels = self._compute_multilabel_loss(logits, batch.y, extra)
        
        # Compute multi-label metrics
        fmax_score, precision_score, recall_score = self._compute_multilabel_metrics(logits, labels)
        
        # Get batch size
        batch_size = labels.size(0)
        
        # Log metrics (using same naming as parent class for consistency)
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val_ce_loss', metrics['ce_loss'], on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val_vq_loss', metrics['vq_loss'], on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val_acc', fmax_score, on_step=False, on_epoch=True, batch_size=batch_size)  # Using fmax as accuracy
        
        # Store for epoch-level metrics
        probs = torch.sigmoid(logits)
        self.val_predictions.append(probs.cpu())
        self.val_targets.append(labels.cpu())
        
        return total_loss
    
    def test_step(self, batch, batch_idx):
        """Test step for multi-label PartGNN."""
        # Forward pass
        logits, assignment_matrix, extra = self.forward(batch)
        
        # Compute multi-label loss
        total_loss, metrics, labels = self._compute_multilabel_loss(logits, batch.y, extra)
        
        # Compute multi-label metrics
        fmax_score, precision_score, recall_score = self._compute_multilabel_metrics(logits, labels)
        
        # Get batch size
        batch_size = labels.size(0)
        
        # Log metrics (using same naming as parent class for consistency)
        self.log('test_loss', total_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_ce_loss', metrics['ce_loss'], on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_vq_loss', metrics['vq_loss'], on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_acc', fmax_score, on_step=False, on_epoch=True, batch_size=batch_size)  # Using fmax as accuracy
        
        # Store for epoch-level metrics
        probs = torch.sigmoid(logits)
        self.test_predictions.append(probs.cpu())
        self.test_targets.append(labels.cpu())
        
        return total_loss
    
    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics.get('train_loss_epoch', 0.0)
        train_fmax = self.trainer.callback_metrics.get('train_acc_epoch', 0.0)  # Using acc metric name for fmax
        print(f"[PARTGNN] Epoch {self.current_epoch:3d} | Train Loss: {train_loss:.4f} | Train FMax: {train_fmax:.4f}")
    
    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get('val_loss', 0.0)
        val_fmax = self.trainer.callback_metrics.get('val_acc', 0.0)  # Using acc metric name for fmax
        print(f"{'':15} | Val Loss:   {val_loss:.4f} | Val FMax:  {val_fmax:.4f}")
        
        # Compute epoch-level metrics if we have predictions
        if self.val_predictions and self.val_targets:
            try:
                # Concatenate all predictions and targets
                all_preds = torch.cat(self.val_predictions, dim=0).numpy()
                all_targets = torch.cat(self.val_targets, dim=0).numpy()
                
                # Compute epoch-level FMax
                epoch_fmax = self.fmax_metric.fmax(all_targets, all_preds)
                epoch_precision = self.fmax_metric.precision(all_targets, all_preds, 0.5)
                epoch_recall = self.fmax_metric.recall(all_targets, all_preds, 0.5)
                
                self.log('val_epoch_fmax', epoch_fmax, on_epoch=True)
                print(f"{'':15} | Val Epoch FMax: {epoch_fmax:.4f} | Precision: {epoch_precision:.4f} | Recall: {epoch_recall:.4f}")
                
            except Exception as e:
                print(f"Warning: Could not compute epoch-level validation metrics: {e}")
            
            # Clear stored predictions
            self.val_predictions.clear()
            self.val_targets.clear()
        
        print("-" * 85)
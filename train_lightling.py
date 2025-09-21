import pytorch_lightning as pl
from partoken_model import ParTokenModel
from utils.lr_schedule import get_cosine_schedule_with_warmup
from utils.fmax_metric import FMaxMetric
import torch
import torch.nn as nn
from typing import Dict, Optional
from utils.interpretability import (
    dataset_inter_results, 
)


class PartGVPLightning(pl.LightningModule):
    """PartGVP Lightning module that trains only GVP + partitioner + global-cluster attention fusion.
    
    This is a simplified version that bypasses the VQ codebook entirely and focuses on
    the core GVP architecture with hierarchical partitioning and attention mechanisms.
    """
    
    def __init__(self, model_cfg, train_cfg, num_classes):
        super().__init__()
        self.save_hyperparameters()
        
        # Create model - we'll always bypass the codebook
        self.model = ParTokenModel(
            node_in_dim=model_cfg.node_in_dim,
            node_h_dim=model_cfg.node_h_dim,
            edge_in_dim=model_cfg.edge_in_dim,
            edge_h_dim=model_cfg.edge_h_dim,
            num_classes=num_classes,
            seq_in=model_cfg.seq_in,
            num_layers=model_cfg.num_layers,
            drop_rate=model_cfg.drop_rate,
            pooling=model_cfg.pooling,
            # Partitioner hyperparameters
            max_clusters=model_cfg.max_clusters,
            nhid=model_cfg.nhid,
            k_hop=model_cfg.k_hop,
            cluster_size_max=model_cfg.cluster_size_max,
            termination_threshold=model_cfg.termination_threshold,
            tau_init=model_cfg.tau_init,
            tau_min=model_cfg.tau_min,
            tau_decay=model_cfg.tau_decay,
            # Codebook hyperparameters (not used but needed for model init)
            codebook_size=model_cfg.get('codebook_size', 512),
            codebook_dim=model_cfg.get('codebook_dim', None),
            codebook_beta=model_cfg.get('codebook_beta', 0.25),
            codebook_decay=model_cfg.get('codebook_decay', 0.99),
            codebook_eps=model_cfg.get('codebook_eps', 1e-5),
            codebook_distance=model_cfg.get('codebook_distance', 'l2'),
            codebook_cosine_normalize=model_cfg.get('codebook_cosine_normalize', False),
            # Loss weights (VQ losses will be 0)
            lambda_vq=0.0,
            lambda_ent=0.0,
            lambda_psc=0.0,
            lambda_card=0.0,
            psc_temp=model_cfg.get('psc_temp', 0.3)
        )
        
        self.train_cfg = train_cfg
        self.criterion = nn.CrossEntropyLoss()
        
        # Always bypass codebook for PartGVP
        self.bypass_codebook = True
        
    def forward(self, h_V, edge_index, h_E, seq=None, batch=None):
        return self._forward_bypass_codebook(h_V, edge_index, h_E, seq, batch)
    
    def _forward_bypass_codebook(self, h_V, edge_index, h_E, seq, batch):
        """Forward pass bypassing codebook (PartGVP mode)."""
        # Get node features
        if seq is not None and self.model.seq_in:
            seq_embedding = self.model.sequence_embedding(seq)
            h_V_with_seq = (
                torch.cat([h_V[0], seq_embedding], dim=-1),
                h_V[1]
            )
        else:
            h_V_with_seq = h_V
            
        h_V_enc = self.model.node_encoder(h_V_with_seq)
        h_E_enc = self.model.edge_encoder(h_E)
        for layer in self.model.gvp_layers:
            h_V_enc = layer(h_V_enc, edge_index, h_E_enc)
        node_features = self.model.output_projection(h_V_enc)
        
        # Handle batch indices
        if batch is None:
            batch = torch.zeros(node_features.size(0), dtype=torch.long, device=node_features.device)
        
        # Convert to dense format for partitioning
        from torch_geometric.utils import to_dense_batch
        dense_x, mask = to_dense_batch(node_features, batch)
        
        # Dense map of global node ids to line up flat ↔ padded layouts
        dense_index, _ = to_dense_batch(
            torch.arange(node_features.size(0), device=node_features.device), batch
        )  # [B, max_N]
        
        # Apply partitioner
        cluster_features, assignment_matrix = self.model.partitioner(
            dense_x, None, mask, edge_index=edge_index, batch_vec=batch, dense_index=dense_index
        )
        cluster_valid_mask = (assignment_matrix.sum(dim=1) > 0)
        
        # Global residue pooling for attention query
        residue_pooled = self.model._pool_nodes(node_features, batch)
        
        # Global-to-cluster attention
        c_star, cluster_importance, _ = self.model.global_cluster_attn(residue_pooled, cluster_features, cluster_valid_mask)
        
        # Feature-wise gated fusion
        fused_cluster, _beta = self.model.fw_gate(residue_pooled, c_star)
        
        # Classification using fused representation
        logits = self.model.classifier(fused_cluster)
        
        # Create dummy extra dict for compatibility  
        extra = {
            "vq_loss": torch.tensor(0.0, device=logits.device),
            "vq_info": {"perplexity": torch.tensor(1.0), "codebook_loss": torch.tensor(0.0), "commitment_loss": torch.tensor(0.0)},
            "code_indices": torch.zeros(cluster_features.shape[:2], dtype=torch.long, device=logits.device),
            "presence": torch.zeros(cluster_features.shape[:2], device=logits.device)
        }
        
        return logits, assignment_matrix, cluster_importance, extra
    
    def training_step(self, batch, batch_idx):
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        seq = batch.seq if hasattr(batch, 'seq') and hasattr(self.model, 'sequence_embedding') else None
        
        logits, assignment_matrix, cluster_importance, extra = self.forward(h_V, edge_index=batch.edge_index, h_E=h_E, seq=seq, batch=batch.batch)
        
        # Only classification loss for PartGVP
        ce_loss = self.criterion(logits, batch.y)
        total_loss = ce_loss
        
        # Compute accuracy 
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch.y).float().mean()
        batch_size = batch.y.size(0)
        
        # Log metrics
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_ce_loss', ce_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_acc', acc, on_step=True, on_epoch=True, batch_size=batch_size)
        
        # Log cluster statistics
        if cluster_importance is not None:
            max_importance = cluster_importance.max(dim=1)[0].mean()
            importance_entropy = -(cluster_importance * torch.log(cluster_importance + 1e-8)).sum(dim=1).mean()
            self.log('train_importance_max', max_importance, on_step=True, on_epoch=True, batch_size=batch_size)
            self.log('train_importance_entropy', importance_entropy, on_step=True, on_epoch=True, batch_size=batch_size)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        seq = batch.seq if hasattr(batch, 'seq') and hasattr(self.model, 'sequence_embedding') else None
        
        logits, assignment_matrix, cluster_importance, extra = self.forward(h_V, edge_index=batch.edge_index, h_E=h_E, seq=seq, batch=batch.batch)
        
        ce_loss = self.criterion(logits, batch.y)
        total_loss = ce_loss
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch.y).float().mean()
        batch_size = batch.y.size(0)
        
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val_ce_loss', ce_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val_acc', acc, on_step=False, on_epoch=True, batch_size=batch_size)
        
        # Log cluster statistics
        if cluster_importance is not None:
            max_importance = cluster_importance.max(dim=1)[0].mean()
            importance_entropy = -(cluster_importance * torch.log(cluster_importance + 1e-8)).sum(dim=1).mean()
            self.log('val_importance_max', max_importance, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log('val_importance_entropy', importance_entropy, on_step=False, on_epoch=True, batch_size=batch_size)
        
        return total_loss
    
    def test_step(self, batch, batch_idx):
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        seq = batch.seq if hasattr(batch, 'seq') and hasattr(self.model, 'sequence_embedding') else None
        
        logits, assignment_matrix, cluster_importance, extra = self.forward(h_V, edge_index=batch.edge_index, h_E=h_E, seq=seq, batch=batch.batch)
        
        ce_loss = self.criterion(logits, batch.y)
        total_loss = ce_loss
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch.y).float().mean()
        batch_size = batch.y.size(0)
        
        self.log('test_loss', total_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_ce_loss', ce_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_acc', acc, on_step=False, on_epoch=True, batch_size=batch_size)
        
        # Log importance statistics if available
        if cluster_importance is not None:
            max_importance = cluster_importance.max(dim=1)[0].mean()
            importance_entropy = -(cluster_importance * torch.log(cluster_importance + 1e-8)).sum(dim=1).mean()
            self.log('test_importance_max', max_importance, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log('test_importance_entropy', importance_entropy, on_step=False, on_epoch=True, batch_size=batch_size)
        
        return total_loss
    
    def on_train_epoch_start(self):
        """Update partitioner temperature each epoch."""
        self.model.update_epoch()
    
    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics.get('train_loss_epoch', 0.0)
        train_acc = self.trainer.callback_metrics.get('train_acc_epoch', 0.0)
        print(f"[PARTGVP] Epoch {self.current_epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    
    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get('val_loss', 0.0)
        val_acc = self.trainer.callback_metrics.get('val_acc', 0.0)
        print(f"{'':15} | Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print("-" * 65)
    
    def get_inter_info(
        self, 
        dataloader, 
        device: Optional[torch.device] = None,
        max_batches: Optional[int] = None
    ) -> Dict:
        """
        Run interpretability analysis on the current model state.
        
        Args:
            dataloader: DataLoader to analyze
            device: Device to run analysis on
            max_batches: Maximum number of batches to process
            
        Returns:
            Dictionary containing interpretability results
        """
        # PartGVP supports interpretability since it has cluster importance scores
        return dataset_inter_results(
            model=self,
            dataloader=dataloader,
            device=device,
            max_batches=max_batches
        )
    
    def configure_optimizers(self):
        lr = float(self.train_cfg.get('lr', 1e-4))
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        if self.train_cfg.get('use_cosine_schedule', False):
            # Use epoch-based scheduling with the correct parameters
            warmup_epochs = self.train_cfg.get('warmup_epochs', 5)
            max_epochs = self.trainer.max_epochs
            
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                warmup_epochs=warmup_epochs,
                max_epochs=max_epochs
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',  # Changed from 'step' to 'epoch'
                    'frequency': 1,
                }
            }
        else:
            return optimizer


class PartGVPMultiLabelLightning(PartGVPLightning):
    """PartGVP Lightning module for multi-label classification (Gene Ontology dataset).
    
    This class extends PartGVPLightning to handle multi-label classification with:
    - BCEWithLogitsLoss instead of CrossEntropyLoss
    - FMax metric instead of accuracy
    - Updated logging for multi-label metrics
    """
    
    def __init__(self, model_cfg, train_cfg, num_classes):
        super().__init__(model_cfg, train_cfg, num_classes)
        
        # Replace criterion for multi-label classification
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Initialize FMax metric
        self.fmax_metric = FMaxMetric()
        self.num_classes = num_classes
        
        # Track accumulated predictions and targets for epoch-level metrics
        self.val_predictions = []
        self.val_targets = []
        self.test_predictions = []
        self.test_targets = []
    
    def training_step(self, batch, batch_idx):
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        seq = batch.seq if hasattr(batch, 'seq') and hasattr(self.model, 'sequence_embedding') else None
        
        logits, assignment_matrix, cluster_importance, extra = self.forward(h_V, edge_index=batch.edge_index, h_E=h_E, seq=seq, batch=batch.batch)
        
        # Fix label shape for multi-label: PyTorch Geometric concatenates but we need to stack
        batch_size = logits.size(0)
        num_classes = logits.size(1)
        labels = batch.y.view(batch_size, num_classes).float()
        
        # Multi-label classification loss
        bce_loss = self.criterion(logits, labels)
        total_loss = bce_loss
        
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
        
        # Log metrics
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_bce_loss', bce_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_fmax', fmax_score, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_precision', precision_score, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_recall', recall_score, on_step=True, on_epoch=True, batch_size=batch_size)
        
        # Log cluster statistics
        if cluster_importance is not None:
            max_importance = cluster_importance.max(dim=1)[0].mean()
            importance_entropy = -(cluster_importance * torch.log(cluster_importance + 1e-8)).sum(dim=1).mean()
            self.log('train_importance_max', max_importance, on_step=True, on_epoch=True, batch_size=batch_size)
            self.log('train_importance_entropy', importance_entropy, on_step=True, on_epoch=True, batch_size=batch_size)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        seq = batch.seq if hasattr(batch, 'seq') and hasattr(self.model, 'sequence_embedding') else None
        
        logits, assignment_matrix, cluster_importance, extra = self.forward(h_V, edge_index=batch.edge_index, h_E=h_E, seq=seq, batch=batch.batch)
        
        # Fix label shape for multi-label: PyTorch Geometric concatenates but we need to stack
        batch_size = logits.size(0)
        num_classes = logits.size(1)
        labels = batch.y.view(batch_size, num_classes).float()
        
        bce_loss = self.criterion(logits, labels)
        total_loss = bce_loss
        
        # Store predictions and targets for epoch-level metrics
        probs = torch.sigmoid(logits)
        self.val_predictions.append(probs.detach().cpu())
        self.val_targets.append(labels.cpu())
        
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val_bce_loss', bce_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        
        # Log cluster statistics
        if cluster_importance is not None:
            max_importance = cluster_importance.max(dim=1)[0].mean()
            importance_entropy = -(cluster_importance * torch.log(cluster_importance + 1e-8)).sum(dim=1).mean()
            self.log('val_importance_max', max_importance, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log('val_importance_entropy', importance_entropy, on_step=False, on_epoch=True, batch_size=batch_size)
        
        return total_loss
    
    def test_step(self, batch, batch_idx):
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        seq = batch.seq if hasattr(batch, 'seq') and hasattr(self.model, 'sequence_embedding') else None
        
        logits, assignment_matrix, cluster_importance, extra = self.forward(h_V, edge_index=batch.edge_index, h_E=h_E, seq=seq, batch=batch.batch)
        
        # Fix label shape for multi-label: PyTorch Geometric concatenates but we need to stack
        batch_size = logits.size(0)
        num_classes = logits.size(1)
        labels = batch.y.view(batch_size, num_classes).float()
        
        bce_loss = self.criterion(logits, labels)
        total_loss = bce_loss
        
        # Store predictions and targets for epoch-level metrics
        probs = torch.sigmoid(logits)
        self.test_predictions.append(probs.detach().cpu())
        self.test_targets.append(labels.cpu())
        
        self.log('test_loss', total_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_bce_loss', bce_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        
        # Log importance statistics if available
        if cluster_importance is not None:
            max_importance = cluster_importance.max(dim=1)[0].mean()
            importance_entropy = -(cluster_importance * torch.log(cluster_importance + 1e-8)).sum(dim=1).mean()
            self.log('test_importance_max', max_importance, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log('test_importance_entropy', importance_entropy, on_step=False, on_epoch=True, batch_size=batch_size)
        
        return total_loss
    
    def on_validation_epoch_end(self):
        if self.val_predictions and self.val_targets:
            # Concatenate all predictions and targets
            all_predictions = torch.cat(self.val_predictions, dim=0).numpy()
            all_targets = torch.cat(self.val_targets, dim=0).numpy()
            
            # Compute epoch-level FMax metrics
            try:
                fmax = self.fmax_metric.fmax(all_targets, all_predictions)
                precision = self.fmax_metric.precision(all_targets, all_predictions)
                recall = self.fmax_metric.recall(all_targets, all_predictions)
                
                self.log('val_fmax', fmax, prog_bar=True)
                self.log('val_precision', precision)
                self.log('val_recall', recall)
                
                # Print validation results
                val_loss = self.trainer.callback_metrics.get('val_loss', 0.0)
                print(f"[PARTGVP-ML] Epoch {self.current_epoch:3d} | Val Loss: {val_loss:.4f} | Val FMax: {fmax:.4f}")
                print(f"{'':18} | Val Prec: {precision:.4f} | Val Rec:  {recall:.4f}")
                
            except Exception as e:
                print(f"Warning: Could not compute validation FMax metrics: {e}")
                self.log('val_fmax', 0.0)
                self.log('val_precision', 0.0)
                self.log('val_recall', 0.0)
            
            # Clear accumulated data
            self.val_predictions.clear()
            self.val_targets.clear()
        
        print("-" * 75)
    
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
                
                self.log('test_fmax', fmax)
                self.log('test_precision', precision)
                self.log('test_recall', recall)
                
                print("🏆 Test Results:")
                print(f"   FMax: {fmax:.4f}")
                print(f"   Best Precision: {precision:.4f}")
                print(f"   Best Recall: {recall:.4f}")
                
            except Exception as e:
                print(f"Warning: Could not compute test FMax metrics: {e}")
                self.log('test_fmax', 0.0)
                self.log('test_precision', 0.0)
                self.log('test_recall', 0.0)
            
            # Clear accumulated data
            self.test_predictions.clear()
            self.test_targets.clear()
    
    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics.get('train_loss_epoch', 0.0)
        train_fmax = self.trainer.callback_metrics.get('train_fmax_epoch', 0.0)
        print(f"[PARTGVP-ML] Epoch {self.current_epoch:3d} | Train Loss: {train_loss:.4f} | Train FMax: {train_fmax:.4f}")

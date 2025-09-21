import pytorch_lightning as pl
from partoken_model import ParTokenModel
from utils.lr_schedule import get_cosine_schedule_with_warmup
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

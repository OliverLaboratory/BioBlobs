import os
import pytorch_lightning as pl
from partoken_model import ParTokenModel
from utils.lr_schedule import get_cosine_schedule_with_warmup
import torch
import torch.nn as nn
from typing import Dict, Optional
from utils.interpretability import (
    dataset_inter_results, 
)
from omegaconf import DictConfig


class JointTrainLightning(pl.LightningModule):
    """Joint Training Lightning module for ParToken model.
    
    This module trains the full ParToken model (GVP + partitioner + codebook) from the beginning
    without bypassing the codebook. It includes optional codebook initialization via K-means.
    """
    
    def __init__(self, model_cfg, train_cfg, num_classes):
        super().__init__()
        self.save_hyperparameters()
        
        # Create full ParToken model with codebook enabled
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
            # Codebook hyperparameters (fully enabled)
            codebook_size=model_cfg.codebook_size,
            codebook_dim=model_cfg.get('codebook_dim', None),
            codebook_beta=model_cfg.codebook_beta,
            codebook_decay=model_cfg.codebook_decay,
            codebook_eps=model_cfg.codebook_eps,
            codebook_distance=model_cfg.codebook_distance,
            codebook_cosine_normalize=model_cfg.codebook_cosine_normalize,
            # Loss weights (VQ losses enabled from start)
            lambda_vq=model_cfg.lambda_vq,
            lambda_ent=model_cfg.lambda_ent,
            lambda_psc=model_cfg.lambda_psc,
            lambda_card=model_cfg.get('lambda_card', 0.005),
            psc_temp=model_cfg.psc_temp
        )
        
        self.train_cfg = train_cfg
        self.criterion = nn.CrossEntropyLoss()
        
        # No codebook bypass - always use full model
        self.bypass_codebook = False
        
        # Codebook initialization tracking
        self.codebook_initialized = False
        self.initialization_epochs = train_cfg.get('codebook_init_epochs', 0)
        self.initialization_max_batches = train_cfg.get('codebook_init_max_batches', 50)
        
    def forward(self, h_V, edge_index, h_E, seq=None, batch=None):
        """Forward pass through full ParToken model with codebook."""
        return self.model(h_V, edge_index, h_E, seq, batch)
    
    def training_step(self, batch, batch_idx):
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        seq = batch.seq if hasattr(batch, 'seq') and hasattr(self.model, 'sequence_embedding') else None
        
        # Forward pass through full model (with codebook)
        logits, extra = self.forward(h_V, edge_index=batch.edge_index, h_E=h_E, seq=seq, batch=batch.batch)
        
        # Main classification loss
        ce_loss = self.criterion(logits, batch.y)
        
        # VQ losses from the extra dict
        vq_loss = extra.get("vq_loss", 0.0)
        vq_commit_loss = extra.get("vq_info", {}).get("commitment_loss", 0.0)
        vq_codebook_loss = extra.get("vq_info", {}).get("codebook_loss", 0.0)
        perplexity = extra.get("vq_info", {}).get("perplexity", 0.0)
        
        # Additional losses
        ent_loss = extra.get("ent_loss", 0.0)
        psc_loss = extra.get("psc_loss", 0.0)
        card_loss = extra.get("card_loss", 0.0)
        
        # Compute total loss
        total_loss = ce_loss + vq_loss + ent_loss + psc_loss + card_loss
        
        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch.y).float().mean()
        batch_size = batch.y.size(0)
        
        # Log metrics
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_ce_loss', ce_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_vq_loss', vq_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_vq_commit_loss', vq_commit_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_vq_codebook_loss', vq_codebook_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_ent_loss', ent_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_psc_loss', psc_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_card_loss', card_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_acc', acc, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_perplexity', perplexity, on_step=True, on_epoch=True, batch_size=batch_size)
        
        # Log codebook utilization
        if 'code_indices' in extra:
            unique_codes = len(torch.unique(extra['code_indices']))
            utilization = unique_codes / self.model.codebook.K
            self.log('train_codebook_utilization', utilization, on_step=True, on_epoch=True, batch_size=batch_size)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        seq = batch.seq if hasattr(batch, 'seq') and hasattr(self.model, 'sequence_embedding') else None
        
        logits, extra = self.forward(h_V, edge_index=batch.edge_index, h_E=h_E, seq=seq, batch=batch.batch)
        
        # Compute losses
        ce_loss = self.criterion(logits, batch.y)
        vq_loss = extra.get("vq_loss", 0.0)
        ent_loss = extra.get("ent_loss", 0.0)
        psc_loss = extra.get("psc_loss", 0.0)
        card_loss = extra.get("card_loss", 0.0)
        total_loss = ce_loss + vq_loss + ent_loss + psc_loss + card_loss
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch.y).float().mean()
        batch_size = batch.y.size(0)
        
        # Log validation metrics
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val_ce_loss', ce_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val_vq_loss', vq_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val_acc', acc, on_step=False, on_epoch=True, batch_size=batch_size)
        
        # Log VQ metrics
        perplexity = extra.get("vq_info", {}).get("perplexity", 0.0)
        self.log('val_perplexity', perplexity, on_step=False, on_epoch=True, batch_size=batch_size)
        
        return total_loss
    
    def test_step(self, batch, batch_idx):
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        seq = batch.seq if hasattr(batch, 'seq') and hasattr(self.model, 'sequence_embedding') else None
        
        logits, extra = self.forward(h_V, edge_index=batch.edge_index, h_E=h_E, seq=seq, batch=batch.batch)
        
        # Compute losses
        ce_loss = self.criterion(logits, batch.y)
        vq_loss = extra.get("vq_loss", 0.0)
        ent_loss = extra.get("ent_loss", 0.0)
        psc_loss = extra.get("psc_loss", 0.0)
        card_loss = extra.get("card_loss", 0.0)
        total_loss = ce_loss + vq_loss + ent_loss + psc_loss + card_loss
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch.y).float().mean()
        batch_size = batch.y.size(0)
        
        # Log test metrics
        self.log('test_loss', total_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_ce_loss', ce_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_vq_loss', vq_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_acc', acc, on_step=False, on_epoch=True, batch_size=batch_size)
        
        # Log VQ metrics
        perplexity = extra.get("vq_info", {}).get("perplexity", 0.0)
        self.log('test_perplexity', perplexity, on_step=False, on_epoch=True, batch_size=batch_size)
        
        return total_loss
    
    def on_train_epoch_start(self):
        """Handle epoch-based updates and codebook initialization."""
        # Update partitioner temperature
        self.model.update_epoch()
        
        # Initialize codebook with K-means if configured and not done yet
        if (not self.codebook_initialized and 
            self.initialization_epochs > 0 and 
            self.current_epoch < self.initialization_epochs):
            
            print(f"\n🎲 Initializing codebook via K-means (epoch {self.current_epoch})")
            self._initialize_codebook_from_data()
            self.codebook_initialized = True
    
    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics.get('train_loss_epoch', 0.0)
        train_acc = self.trainer.callback_metrics.get('train_acc_epoch', 0.0)
        train_perplexity = self.trainer.callback_metrics.get('train_perplexity_epoch', 0.0)
        
        print(f"[JOINT] Epoch {self.current_epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Perplexity: {train_perplexity:.2f}")
    
    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get('val_loss', 0.0)
        val_acc = self.trainer.callback_metrics.get('val_acc', 0.0)
        val_perplexity = self.trainer.callback_metrics.get('val_perplexity', 0.0)
        
        print(f"{'':12} | Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Perplexity: {val_perplexity:.2f}")
        print("-" * 80)
    
    def _initialize_codebook_from_data(self):
        """Initialize codebook using K-means on training data."""
        if hasattr(self.trainer, 'train_dataloader') and self.trainer.train_dataloader is not None:
            train_loader = self.trainer.train_dataloader
            device = next(self.model.parameters()).device
            
            try:
                # Use the model's built-in K-means initialization
                self.model.kmeans_init_from_loader(
                    loader=train_loader,
                    max_batches=self.initialization_max_batches,
                    device=device
                )
                print(f"✓ Codebook initialized with K-means using {self.initialization_max_batches} batches")
            except Exception as e:
                print(f"⚠️  Warning: Codebook initialization failed: {e}")
        else:
            print("⚠️  Warning: Training dataloader not available for codebook initialization")
    
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
        return dataset_inter_results(
            model=self,
            dataloader=dataloader,
            device=device,
            max_batches=max_batches
        )
    
    def configure_optimizers(self):
        lr = float(self.train_cfg.get('lr', 1e-3))
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        if self.train_cfg.get('use_cosine_schedule', False):
            # Use epoch-based scheduling with the correct parameters
            warmup_epochs = self.train_cfg.get('warmup_epochs', 10)
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
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }
        else:
            return optimizer

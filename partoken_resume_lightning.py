import pytorch_lightning as pl
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from omegaconf import DictConfig
from partoken_model import ParTokenModel
from utils.interpretability import dataset_inter_results


class ParTokenResumeTrainingLightning(pl.LightningModule):
    """
    Dedicated Lightning module for ParToken resume training from PartGVP checkpoint.
    
    This module handles:
    - Direct joint training (no multi-stage needed for EMA-based VQ)
    - Codebook integration with proper VQ loss logging
    - Comprehensive metric tracking
    """
    
    def __init__(self, model_cfg: DictConfig, train_cfg: DictConfig, multistage_cfg: DictConfig, num_classes: int):
        super().__init__()
        self.save_hyperparameters()
        
        # Create ParToken model with codebook enabled
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
            max_clusters=model_cfg.max_clusters,
            nhid=model_cfg.nhid,
            k_hop=model_cfg.k_hop,
            cluster_size_max=model_cfg.cluster_size_max,
            termination_threshold=model_cfg.termination_threshold,
            tau_init=model_cfg.tau_init,
            tau_min=model_cfg.tau_min,
            tau_decay=model_cfg.tau_decay,
            codebook_size=model_cfg.codebook_size,
            codebook_dim=model_cfg.codebook_dim,
            codebook_beta=model_cfg.codebook_beta,
            codebook_decay=model_cfg.codebook_decay,
            codebook_eps=model_cfg.codebook_eps,
            codebook_distance=model_cfg.codebook_distance,
            codebook_cosine_normalize=model_cfg.codebook_cosine_normalize,
            lambda_vq=model_cfg.lambda_vq,
            lambda_ent=model_cfg.lambda_ent,
            lambda_psc=model_cfg.lambda_psc,
            lambda_card=model_cfg.lambda_card,
            psc_temp=model_cfg.psc_temp
        )
        
        self.train_cfg = train_cfg
        self.multistage_cfg = multistage_cfg
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup joint training from the start (no freezing needed)
        self.setup_joint_training()
        
    def setup_joint_training(self):
        """Setup model for joint training of all components."""
        print(f"\n{'='*60}")
        print(f"SETTING UP JOINT TRAINING (BACKBONE + CODEBOOK)")
        print(f"{'='*60}")
        
        # Extract loss weights from stage0 config (now joint training)
        stage_cfg = self.multistage_cfg.stage0
        loss_weights = stage_cfg.get('loss_weights', {})
        lambda_vq = loss_weights.get('lambda_vq', 1.0)
        lambda_ent = loss_weights.get('lambda_ent', 0.1)
        lambda_psc = loss_weights.get('lambda_psc', 0.01)
        
        # Update model loss weights
        self.model.lambda_vq = lambda_vq
        self.model.lambda_ent = lambda_ent
        self.model.lambda_psc = lambda_psc
        
        # Ensure all parameters are trainable
        self.model.unfreeze_all()
        
        print(f"✓ Loss weights: λ_vq={lambda_vq:.1e}, λ_ent={lambda_ent:.1e}, λ_psc={lambda_psc:.1e}")
        print(f"✓ Training mode: Joint training (backbone + EMA codebook)")
        print(f"✓ Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"{'='*60}\n")
    
    def forward(self, h_V, edge_index, h_E, seq=None, batch=None):
        """Forward pass - always use full model with codebook (no bypass)."""
        return self.model(h_V, edge_index, h_E, seq, batch)
    
    def training_step(self, batch, batch_idx):
        """Training step with comprehensive VQ loss tracking."""
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        seq = batch.seq if hasattr(batch, 'seq') and hasattr(self.model, 'sequence_embedding') else None
        
        # Forward pass through full model (no bypass)
        result = self.model.forward(h_V, batch.edge_index, h_E, seq, batch.batch)
        logits, assignment_matrix, extra = result[:3]  # Handle potential extra returns
        
        # Main classification loss
        ce_loss = self.criterion(logits, batch.y)
        
        # Extract VQ losses and metrics from extra dict
        vq_loss = extra.get("vq_loss", torch.tensor(0.0, device=logits.device))
        vq_info = extra.get("vq_info", {})
        
        # Individual VQ loss components
        codebook_loss = vq_info.get("codebook_loss", torch.tensor(0.0, device=logits.device))
        commitment_loss = vq_info.get("commitment_loss", torch.tensor(0.0, device=logits.device))
        perplexity = vq_info.get("perplexity", torch.tensor(1.0, device=logits.device))
        
        # Coverage and entropy losses
        coverage_loss = extra.get("coverage_loss", torch.tensor(0.0, device=logits.device))
        entropy_loss = extra.get("entropy_loss", torch.tensor(0.0, device=logits.device))
        
        # Total loss
        total_loss = ce_loss + vq_loss
        
        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch.y).float().mean()
        batch_size = batch.y.size(0)
        
        # Comprehensive logging
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_ce_loss', ce_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_acc', acc, on_step=True, on_epoch=True, batch_size=batch_size)
        
        # VQ-specific metrics
        self.log('train_vq_loss', vq_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_codebook_loss', codebook_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_commitment_loss', commitment_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_perplexity', perplexity, on_step=True, on_epoch=True, batch_size=batch_size)
        
        # Coverage and entropy metrics
        self.log('train_coverage_loss', coverage_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_entropy_loss', entropy_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        
        # Codebook utilization metrics
        if 'code_indices' in extra:
            code_indices = extra['code_indices']
            unique_codes = len(torch.unique(code_indices))
            utilization = unique_codes / self.model.codebook.K
            self.log('train_codebook_utilization', utilization, on_step=True, on_epoch=True, batch_size=batch_size)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step with comprehensive VQ loss tracking."""
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        seq = batch.seq if hasattr(batch, 'seq') and hasattr(self.model, 'sequence_embedding') else None
        
        # Forward pass
        result = self.model.forward(h_V, batch.edge_index, h_E, seq, batch.batch)
        logits, assignment_matrix, extra = result[:3]  # Handle potential extra returns
        
        # Losses
        ce_loss = self.criterion(logits, batch.y)
        vq_loss = extra.get("vq_loss", torch.tensor(0.0, device=logits.device))
        total_loss = ce_loss + vq_loss
        
        # Accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch.y).float().mean()
        batch_size = batch.y.size(0)
        
        # VQ metrics
        vq_info = extra.get("vq_info", {})
        codebook_loss = vq_info.get("codebook_loss", torch.tensor(0.0, device=logits.device))
        commitment_loss = vq_info.get("commitment_loss", torch.tensor(0.0, device=logits.device))
        perplexity = vq_info.get("perplexity", torch.tensor(1.0, device=logits.device))
        
        # Logging
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val_ce_loss', ce_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val_vq_loss', vq_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val_acc', acc, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val_codebook_loss', codebook_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val_commitment_loss', commitment_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val_perplexity', perplexity, on_step=False, on_epoch=True, batch_size=batch_size)
        
        return total_loss
    
    def test_step(self, batch, batch_idx):
        """Test step with comprehensive metrics."""
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        seq = batch.seq if hasattr(batch, 'seq') and hasattr(self.model, 'sequence_embedding') else None
        
        # Forward pass
        result = self.model.forward(h_V, batch.edge_index, h_E, seq, batch.batch)
        logits, assignment_matrix, extra = result[:3]  # Handle potential extra returns
        
        # Losses
        ce_loss = self.criterion(logits, batch.y)
        vq_loss = extra.get("vq_loss", torch.tensor(0.0, device=logits.device))
        total_loss = ce_loss + vq_loss
        
        # Accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch.y).float().mean()
        batch_size = batch.y.size(0)
        
        # VQ metrics
        vq_info = extra.get("vq_info", {})
        codebook_loss = vq_info.get("codebook_loss", torch.tensor(0.0, device=logits.device))
        commitment_loss = vq_info.get("commitment_loss", torch.tensor(0.0, device=logits.device))
        perplexity = vq_info.get("perplexity", torch.tensor(1.0, device=logits.device))
        
        # Logging
        self.log('test_loss', total_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_ce_loss', ce_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_vq_loss', vq_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_acc', acc, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_codebook_loss', codebook_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_commitment_loss', commitment_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_perplexity', perplexity, on_step=False, on_epoch=True, batch_size=batch_size)
        
        # Importance scores if available
        if 'cluster_importance' in extra:
            importance_scores = extra['cluster_importance']
            if importance_scores is not None:
                # Log importance statistics
                max_importance = importance_scores.max(dim=1)[0].mean()
                importance_entropy = -(importance_scores * torch.log(importance_scores + 1e-8)).sum(dim=1).mean()
                self.log('test_importance_max', max_importance, on_step=False, on_epoch=True, batch_size=batch_size)
                self.log('test_importance_entropy', importance_entropy, on_step=False, on_epoch=True, batch_size=batch_size)
        
        return total_loss
    
    def on_train_epoch_start(self):
        """Handle epoch-based updates."""
        # Update model epoch-based parameters (tau annealing, etc.)
        self.model.update_epoch()
    
    def on_train_epoch_end(self):
        """Log epoch summary."""
        train_loss = self.trainer.callback_metrics.get('train_loss_epoch', 0.0)
        train_acc = self.trainer.callback_metrics.get('train_acc_epoch', 0.0)
        train_vq_loss = self.trainer.callback_metrics.get('train_vq_loss_epoch', 0.0)
        
        epoch = self.trainer.current_epoch
        print(f"[JOINT-TRAINING] Epoch {epoch:3d} | "
              f"Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | VQ: {train_vq_loss:.4f}")
    
    def on_validation_epoch_end(self):
        """Log validation summary."""
        val_loss = self.trainer.callback_metrics.get('val_loss', 0.0)
        val_acc = self.trainer.callback_metrics.get('val_acc', 0.0)
        val_vq_loss = self.trainer.callback_metrics.get('val_vq_loss', 0.0)
        print(f"{'':18} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | VQ: {val_vq_loss:.4f}")
        print("-" * 75)
    
    def get_inter_info(
        self, 
        dataloader, 
        device: Optional[torch.device] = None,
        max_batches: Optional[int] = None
    ) -> Dict:
        """Run interpretability analysis (always available since codebook is never bypassed)."""
        return dataset_inter_results(
            model=self,
            dataloader=dataloader,
            device=device,
            max_batches=max_batches
        )
    
    def configure_optimizers(self):
        """Configure optimizer with learning rate from stage0 config."""
        # Get learning rate from stage0 (now joint training)
        stage_cfg = self.multistage_cfg.stage0
        lr = stage_cfg.lr
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        if self.train_cfg.get('use_cosine_schedule', False):
            from utils.lr_schedule import get_cosine_schedule_with_warmup
            warmup_epochs = self.train_cfg.get('warmup_epochs', 5)
            max_epochs = stage_cfg.epochs
            
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

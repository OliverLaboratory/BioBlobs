import pytorch_lightning as pl
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from omegaconf import DictConfig
from partoken_model import ParTokenModel
from utils.interpretability import dataset_inter_results
from utils.fmax_metric import FMaxMetric


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
        
        # Ensure model is in training mode
        self.model.train()
        
        print(f"✓ Loss weights: λ_vq={lambda_vq:.1e}, λ_ent={lambda_ent:.1e}, λ_psc={lambda_psc:.1e}")
        print(f"✓ Training mode: Joint training (backbone + EMA codebook)")
        print(f"✓ Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"✓ Model training mode: {self.model.training}")
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
    
    def on_train_start(self):
        """Ensure model is in training mode at the start of training."""
        self.model.train()
        print(f"\n🚀 Training started - Model training mode: {self.model.training}")
        
        # Double-check all modules are in training mode
        eval_count = 0
        for name, module in self.model.named_modules():
            if hasattr(module, 'training') and not module.training:
                eval_count += 1
        
        if eval_count > 0:
            print(f"⚠️  {eval_count} modules still in eval mode at training start")
            print("   This may cause PyTorch Lightning warnings but shouldn't affect training.")
        else:
            print("✅ All modules are properly in training mode")
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


class ParTokenResumeTrainingMultiLabelLightning(ParTokenResumeTrainingLightning):
    """
    Multi-label version of ParToken resume training for Gene Ontology dataset.
    
    This class extends ParTokenResumeTrainingLightning to handle multi-label classification with:
    - BCEWithLogitsLoss instead of CrossEntropyLoss
    - FMax metric instead of accuracy
    - Updated logging for multi-label metrics
    """
    
    def __init__(self, model_cfg: DictConfig, train_cfg: DictConfig, multistage_cfg: DictConfig, num_classes: int):
        super().__init__(model_cfg, train_cfg, multistage_cfg, num_classes)
        
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
        """Training step with multi-label classification and comprehensive VQ loss tracking."""
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        seq = batch.seq if hasattr(batch, 'seq') and hasattr(self.model, 'sequence_embedding') else None
        
        # Forward pass through full model (no bypass)
        result = self.model.forward(h_V, batch.edge_index, h_E, seq, batch.batch)
        logits, assignment_matrix, extra = result[:3]  # Handle potential extra returns
        
        # Fix label shape for multi-label: PyTorch Geometric concatenates but we need to stack
        batch_size = logits.size(0)
        num_classes = logits.size(1)
        labels = batch.y.view(batch_size, num_classes).float()
        
        # Multi-label classification loss
        bce_loss = self.criterion(logits, labels)
        
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
        total_loss = bce_loss + vq_loss
        
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
        
        # Comprehensive logging
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_bce_loss', bce_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_fmax', fmax_score, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_precision', precision_score, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_recall', recall_score, on_step=True, on_epoch=True, batch_size=batch_size)
        
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
            unique_codes = torch.unique(extra['code_indices']).numel()
            utilization = unique_codes / self.model.codebook.K
            self.log('train_codebook_utilization', utilization, on_step=True, on_epoch=True, batch_size=batch_size)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step with multi-label classification and comprehensive VQ loss tracking."""
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        seq = batch.seq if hasattr(batch, 'seq') and hasattr(self.model, 'sequence_embedding') else None
        
        # Forward pass
        result = self.model.forward(h_V, batch.edge_index, h_E, seq, batch.batch)
        logits, assignment_matrix, extra = result[:3]  # Handle potential extra returns
        
        # Fix label shape for multi-label
        batch_size = logits.size(0)
        num_classes = logits.size(1)
        labels = batch.y.view(batch_size, num_classes).float()
        
        # Losses
        bce_loss = self.criterion(logits, labels)
        vq_loss = extra.get("vq_loss", torch.tensor(0.0, device=logits.device))
        total_loss = bce_loss + vq_loss
        
        # Compute metrics using sigmoid probabilities
        probs = torch.sigmoid(logits)
        
        # Store predictions and targets for epoch-level metrics
        self.val_predictions.append(probs.detach().cpu())
        self.val_targets.append(labels.cpu())
        
        # VQ metrics
        vq_info = extra.get("vq_info", {})
        codebook_loss = vq_info.get("codebook_loss", torch.tensor(0.0, device=logits.device))
        commitment_loss = vq_info.get("commitment_loss", torch.tensor(0.0, device=logits.device))
        perplexity = vq_info.get("perplexity", torch.tensor(1.0, device=logits.device))
        
        # Logging
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val_bce_loss', bce_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val_vq_loss', vq_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val_codebook_loss', codebook_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val_commitment_loss', commitment_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val_perplexity', perplexity, on_step=False, on_epoch=True, batch_size=batch_size)
        
        return total_loss
    
    def test_step(self, batch, batch_idx):
        """Test step with multi-label classification and comprehensive metrics."""
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        seq = batch.seq if hasattr(batch, 'seq') and hasattr(self.model, 'sequence_embedding') else None
        
        # Forward pass
        result = self.model.forward(h_V, batch.edge_index, h_E, seq, batch.batch)
        logits, assignment_matrix, extra = result[:3]  # Handle potential extra returns
        
        # Fix label shape for multi-label
        batch_size = logits.size(0)
        num_classes = logits.size(1)
        labels = batch.y.view(batch_size, num_classes).float()
        
        # Losses
        bce_loss = self.criterion(logits, labels)
        vq_loss = extra.get("vq_loss", torch.tensor(0.0, device=logits.device))
        total_loss = bce_loss + vq_loss
        
        # Compute metrics using sigmoid probabilities
        probs = torch.sigmoid(logits)
        
        # Store predictions and targets for epoch-level metrics
        self.test_predictions.append(probs.detach().cpu())
        self.test_targets.append(labels.cpu())
        
        # VQ metrics
        vq_info = extra.get("vq_info", {})
        codebook_loss = vq_info.get("codebook_loss", torch.tensor(0.0, device=logits.device))
        commitment_loss = vq_info.get("commitment_loss", torch.tensor(0.0, device=logits.device))
        perplexity = vq_info.get("perplexity", torch.tensor(1.0, device=logits.device))
        
        # Logging
        self.log('test_loss', total_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_bce_loss', bce_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_vq_loss', vq_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_codebook_loss', codebook_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_commitment_loss', commitment_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_perplexity', perplexity, on_step=False, on_epoch=True, batch_size=batch_size)
        
        # Importance scores if available
        if 'cluster_importance' in extra:
            cluster_importance = extra['cluster_importance']
            max_importance = cluster_importance.max(dim=1)[0].mean()
            importance_entropy = -(cluster_importance * torch.log(cluster_importance + 1e-8)).sum(dim=1).mean()
            self.log('test_importance_max', max_importance, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log('test_importance_entropy', importance_entropy, on_step=False, on_epoch=True, batch_size=batch_size)
        
        return total_loss
    
    def on_validation_epoch_end(self):
        """Compute epoch-level metrics for validation."""
        super().on_validation_epoch_end()
        
        if len(self.val_predictions) > 0:
            # Concatenate all predictions and targets
            all_predictions = torch.cat(self.val_predictions, dim=0).numpy()
            all_targets = torch.cat(self.val_targets, dim=0).numpy()
            
            # Compute epoch-level FMax metrics
            try:
                epoch_fmax = self.fmax_metric.fmax(all_targets, all_predictions)
                epoch_precision = self.fmax_metric.precision(all_targets, all_predictions, 0.5)
                epoch_recall = self.fmax_metric.recall(all_targets, all_predictions, 0.5)
                
                self.log('val_fmax', epoch_fmax, on_step=False, on_epoch=True)
                self.log('val_precision', epoch_precision, on_step=False, on_epoch=True)
                self.log('val_recall', epoch_recall, on_step=False, on_epoch=True)
            except Exception as e:
                print(f"Warning: Could not compute validation FMax metrics: {e}")
            
            # Clear accumulated predictions and targets
            self.val_predictions.clear()
            self.val_targets.clear()
    
    def on_test_epoch_end(self):
        """Compute epoch-level metrics for testing."""
        if len(self.test_predictions) > 0:
            # Concatenate all predictions and targets
            all_predictions = torch.cat(self.test_predictions, dim=0).numpy()
            all_targets = torch.cat(self.test_targets, dim=0).numpy()
            
            # Compute epoch-level FMax metrics
            try:
                epoch_fmax = self.fmax_metric.fmax(all_targets, all_predictions)
                epoch_precision = self.fmax_metric.precision(all_targets, all_predictions, 0.5)
                epoch_recall = self.fmax_metric.recall(all_targets, all_predictions, 0.5)
                
                self.log('test_fmax', epoch_fmax, on_step=False, on_epoch=True)
                self.log('test_precision', epoch_precision, on_step=False, on_epoch=True)
                self.log('test_recall', epoch_recall, on_step=False, on_epoch=True)
            except Exception as e:
                print(f"Warning: Could not compute test FMax metrics: {e}")
            
            # Clear accumulated predictions and targets
            self.test_predictions.clear()
            self.test_targets.clear()

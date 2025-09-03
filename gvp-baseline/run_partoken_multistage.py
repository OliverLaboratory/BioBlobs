import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Only if you want to use specific GPU

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm
import json
import argparse
import math
import copy
from utils.proteinshake_dataset import get_dataset, create_dataloader
from partoken_model import ParTokenModel
from utils.utils import set_seed
from utils.lr_schedule import get_cosine_schedule_with_warmup
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint
from typing import Dict, Optional, Tuple


class LossWeightScheduler:
    """Scheduler for gradually ramping loss weights during training."""
    
    def __init__(self, initial_weights: Dict[str, float], final_weights: Dict[str, float], ramp_epochs: int):
        self.initial_weights = initial_weights
        self.final_weights = final_weights
        self.ramp_epochs = ramp_epochs
        
    def get_weights(self, epoch: int) -> Dict[str, float]:
        """Get interpolated weights for current epoch."""
        if epoch >= self.ramp_epochs:
            return self.final_weights.copy()
            
        alpha = epoch / self.ramp_epochs
        weights = {}
        for key in self.initial_weights:
            initial_val = self.initial_weights[key]
            final_val = self.final_weights.get(key, initial_val)
            weights[key] = (1 - alpha) * initial_val + alpha * final_val
        return weights


class MultiStageParTokenLightning(pl.LightningModule):
    """Multi-stage ParToken Lightning module with stage management."""
    
    def __init__(self, model_cfg, train_cfg, multistage_cfg, num_classes):
        super().__init__()
        self.save_hyperparameters()
        
        # Create model with codebook initially disabled for stage 0
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
            psc_temp=model_cfg.psc_temp
        )
        
        self.train_cfg = train_cfg
        self.multistage_cfg = multistage_cfg
        self.criterion = nn.CrossEntropyLoss()
        
        # Stage management
        self.current_stage = 0
        self.stage_epoch = 0
        self.total_epoch = 0
        self.bypass_codebook = True  # Start in stage 0 mode
        
        # Teacher model for knowledge distillation
        self.teacher_classifier = None
        self.use_knowledge_distillation = False
        
        # Loss weight scheduling
        self.loss_weight_scheduler = None
        self.current_loss_weights = {}
        
        # Codebook freezing
        self.freeze_codebook_remaining_epochs = 0
        
    def setup_stage(self, stage_idx: int, stage_cfg: DictConfig):
        """Setup model for specific training stage."""
        self.current_stage = stage_idx
        self.stage_epoch = 0
        
        print(f"\n{'='*60}")
        print(f"SETTING UP STAGE {stage_idx}: {stage_cfg.name.upper()}")
        print(f"{'='*60}")
        
        if stage_idx == 0:  # Baseline stage
            self.bypass_codebook = True
            self.model.unfreeze_all()
            self.current_loss_weights = stage_cfg.loss_weights
            print("✓ Codebook bypassed")
            print("✓ All parameters unfrozen")
            
        elif stage_idx == 1:  # Codebook warmup stage
            self.bypass_codebook = False
            if stage_cfg.freeze_backbone:
                self.model.freeze_backbone_for_codebook()
                print("✓ Backbone frozen, codebook trainable")
            self.current_loss_weights = stage_cfg.loss_weights
            print("✓ Codebook activated")
            
        elif stage_idx == 2:  # Joint fine-tuning stage
            self.bypass_codebook = False
            self.model.unfreeze_all()
            
            # Setup knowledge distillation
            if stage_cfg.knowledge_distillation.enabled and self.teacher_classifier is not None:
                self.use_knowledge_distillation = True
                self.kd_temperature = stage_cfg.knowledge_distillation.temperature
                self.kd_alpha = stage_cfg.knowledge_distillation.alpha
                print("✓ Knowledge distillation enabled")
            
            # Setup loss weight ramping
            if stage_cfg.loss_ramp.enabled:
                self.loss_weight_scheduler = LossWeightScheduler(
                    stage_cfg.loss_ramp.initial_weights,
                    stage_cfg.loss_ramp.final_weights,
                    stage_cfg.loss_ramp.ramp_epochs
                )
                self.current_loss_weights = self.loss_weight_scheduler.get_weights(0)
                print("✓ Loss weight ramping enabled")
            else:
                # Use model's default weights if no ramping
                self.current_loss_weights = {
                    'lambda_vq': self.model.lambda_vq,
                    'lambda_ent': self.model.lambda_ent,
                    'lambda_psc': self.model.lambda_psc
                }
            
            # Setup codebook freezing for final epochs
            if stage_cfg.freeze_codebook_final.enabled:
                total_epochs = stage_cfg.epochs
                freeze_epochs = stage_cfg.freeze_codebook_final.epochs
                self.freeze_codebook_remaining_epochs = freeze_epochs
                print(f"✓ Codebook will be frozen for last {freeze_epochs} epochs")
            
            print("✓ All parameters unfrozen")
        
        # Update model loss weights
        self.model.lambda_vq = self.current_loss_weights.get('lambda_vq', 0.0)
        self.model.lambda_ent = self.current_loss_weights.get('lambda_ent', 0.0)
        self.model.lambda_psc = self.current_loss_weights.get('lambda_psc', 0.0)
        
        print(f"✓ Loss weights: λ_vq={self.model.lambda_vq:.1e}, λ_ent={self.model.lambda_ent:.1e}, λ_psc={self.model.lambda_psc:.1e}")
        print(f"{'='*60}\n")
    
    def save_teacher_model(self):
        """Save current classifier as teacher for knowledge distillation."""
        self.teacher_classifier = copy.deepcopy(self.model.classifier)
        for param in self.teacher_classifier.parameters():
            param.requires_grad = False
        self.teacher_classifier.eval()
        print("✓ Teacher model saved for knowledge distillation")
    
    def forward(self, h_V, edge_index, h_E, seq=None, batch=None):
        return self.model(h_V, edge_index, h_E, seq, batch)
    
    def training_step(self, batch, batch_idx):
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        seq = batch.seq if hasattr(batch, 'seq') and hasattr(self.model, 'sequence_embedding') else None
        
        # Forward pass with optional codebook bypass
        if self.bypass_codebook:
            # Stage 0: Bypass codebook, direct cluster features to classification
            logits, assignment_matrix, extra = self._forward_bypass_codebook(h_V, batch.edge_index, h_E, seq, batch.batch)
        else:
            # Stages 1-2: Use full model with quantization
            logits, assignment_matrix, extra = self.model(h_V, batch.edge_index, h_E, seq, batch.batch)
        
        # Main classification loss
        ce_loss = self.criterion(logits, batch.y)
        
        # Additional losses from extra dict
        vq_loss = extra.get("vq_loss", 0.0)
        
        # Compute total loss
        total_loss = ce_loss + vq_loss
        
        # Knowledge distillation loss
        kd_loss = 0.0
        if self.use_knowledge_distillation and self.teacher_classifier is not None:
            # Get teacher predictions
            with torch.no_grad():
                # Extract combined features for teacher (same as in model forward)
                combined_features = self._extract_combined_features(h_V, batch.edge_index, h_E, seq, batch.batch)
                teacher_logits = self.teacher_classifier(combined_features)
            
            # KL divergence with temperature scaling
            T = self.kd_temperature
            student_probs = F.log_softmax(logits / T, dim=1)
            teacher_probs = F.softmax(teacher_logits / T, dim=1)
            kd_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (T * T)
            total_loss += self.kd_alpha * kd_loss
        
        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch.y).float().mean()
        batch_size = batch.y.size(0)
        
        # Log metrics
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_ce_loss', ce_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_vq_loss', vq_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_kd_loss', kd_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_acc', acc, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('stage', float(self.current_stage), on_step=True, on_epoch=True, batch_size=batch_size)
        
        return total_loss
    
    def _forward_bypass_codebook(self, h_V, edge_index, h_E, seq, batch):
        """Forward pass bypassing codebook for Stage 0."""
        # Add sequence features if provided
        if seq is not None and self.model.seq_in:
            seq_emb = self.model.sequence_embedding(seq)
            h_V = (torch.cat([h_V[0], seq_emb], dim=-1), h_V[1])
            
        # Encode initial features
        h_V_enc = self.model.node_encoder(h_V)
        h_E_enc = self.model.edge_encoder(h_E)
        
        # Process through GVP layers
        for layer in self.model.gvp_layers:
            h_V_enc = layer(h_V_enc, edge_index, h_E_enc)
            
        # Extract scalar features
        node_features = self.model.output_projection(h_V_enc)
        
        # Handle batch indices
        if batch is None:
            batch = torch.zeros(node_features.size(0), dtype=torch.long, device=node_features.device)
        
        # Convert to dense format for partitioning
        from torch_geometric.utils import to_dense_batch, to_dense_adj
        dense_x, mask = to_dense_batch(node_features, batch)
        dense_adj = to_dense_adj(edge_index, batch)
        
        # Apply partitioner
        cluster_features, cluster_adj, assignment_matrix = self.model.partitioner(dense_x, dense_adj, mask)
        
        # Skip quantization, use cluster features directly
        cluster_valid_mask = (assignment_matrix.sum(dim=1) > 0)
        
        # Inter-cluster message passing on original cluster features
        refined_clusters = self.model.cluster_gcn(cluster_features, cluster_adj)
        
        # Global pooling
        cluster_pooled = self.model._masked_mean(refined_clusters, cluster_valid_mask)
        residue_pooled = self.model._pool_nodes(node_features, batch)
        
        # Combine representations
        combined_features = torch.cat([residue_pooled, cluster_pooled], dim=-1)
        
        # Classification
        logits = self.model.classifier(combined_features)
        
        # Create dummy extra dict for compatibility
        extra = {
            "vq_loss": torch.tensor(0.0, device=logits.device),
            "vq_info": {"perplexity": torch.tensor(1.0), "codebook_loss": torch.tensor(0.0), "commitment_loss": torch.tensor(0.0)},
            "code_indices": torch.zeros(cluster_features.shape[:2], dtype=torch.long, device=logits.device),
            "presence": torch.zeros(cluster_features.shape[:2], device=logits.device)
        }
        
        return logits, assignment_matrix, extra
    
    def _extract_combined_features(self, h_V, edge_index, h_E, seq, batch):
        """Extract combined features for teacher model."""
        # This replicates the feature extraction part of the model
        if seq is not None and self.model.seq_in:
            seq_emb = self.model.sequence_embedding(seq)
            h_V = (torch.cat([h_V[0], seq_emb], dim=-1), h_V[1])
            
        h_V_enc = self.model.node_encoder(h_V)
        h_E_enc = self.model.edge_encoder(h_E)
        
        for layer in self.model.gvp_layers:
            h_V_enc = layer(h_V_enc, edge_index, h_E_enc)
            
        node_features = self.model.output_projection(h_V_enc)
        
        if batch is None:
            batch = torch.zeros(node_features.size(0), dtype=torch.long, device=node_features.device)
        
        from torch_geometric.utils import to_dense_batch, to_dense_adj
        dense_x, mask = to_dense_batch(node_features, batch)
        dense_adj = to_dense_adj(edge_index, batch)
        
        cluster_features, cluster_adj, assignment_matrix = self.model.partitioner(dense_x, dense_adj, mask)
        cluster_valid_mask = (assignment_matrix.sum(dim=1) > 0)
        
        if self.bypass_codebook:
            refined_clusters = self.model.cluster_gcn(cluster_features, cluster_adj)
        else:
            quant_clusters, _, _, _ = self.model.codebook(cluster_features, mask=cluster_valid_mask)
            refined_clusters = self.model.cluster_gcn(quant_clusters, cluster_adj)
        
        cluster_pooled = self.model._masked_mean(refined_clusters, cluster_valid_mask)
        residue_pooled = self.model._pool_nodes(node_features, batch)
        
        return torch.cat([residue_pooled, cluster_pooled], dim=-1)
    
    def on_train_epoch_start(self):
        """Handle epoch-based updates."""
        # Update loss weights if ramping is enabled
        if self.loss_weight_scheduler is not None:
            self.current_loss_weights = self.loss_weight_scheduler.get_weights(self.stage_epoch)
            self.model.lambda_vq = self.current_loss_weights.get('lambda_vq', 0.0)
            self.model.lambda_ent = self.current_loss_weights.get('lambda_ent', 0.0)
            self.model.lambda_psc = self.current_loss_weights.get('lambda_psc', 0.0)
        
        # Handle codebook freezing in final epochs
        if self.freeze_codebook_remaining_epochs > 0:
            epochs_in_stage = self.stage_epoch
            total_stage_epochs = self.trainer.max_epochs  # This will be stage-specific
            
            if epochs_in_stage >= total_stage_epochs - self.freeze_codebook_remaining_epochs:
                # Freeze codebook
                for param in self.model.codebook.parameters():
                    param.requires_grad = False
                if self.stage_epoch == total_stage_epochs - self.freeze_codebook_remaining_epochs:
                    print(f"🔒 Codebook frozen for final {self.freeze_codebook_remaining_epochs} epochs")
        
        self.stage_epoch += 1
        self.total_epoch += 1
    
    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics.get('train_loss_epoch', 0.0)
        train_acc = self.trainer.callback_metrics.get('train_acc_epoch', 0.0)
        stage_name = ["BASELINE", "CODEBOOK", "JOINT"][self.current_stage]
        print(f"[{stage_name}] Epoch {self.stage_epoch-1:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    
    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get('val_loss', 0.0)
        val_acc = self.trainer.callback_metrics.get('val_acc', 0.0)
        print(f"{'':15} | Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print("-" * 65)
    
    def validation_step(self, batch, batch_idx):
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        seq = batch.seq if hasattr(batch, 'seq') and hasattr(self.model, 'sequence_embedding') else None
        
        if self.bypass_codebook:
            logits, assignment_matrix, extra = self._forward_bypass_codebook(h_V, batch.edge_index, h_E, seq, batch.batch)
        else:
            logits, assignment_matrix, extra = self.model(h_V, batch.edge_index, h_E, seq, batch.batch)
        
        ce_loss = self.criterion(logits, batch.y)
        vq_loss = extra.get("vq_loss", 0.0)
        total_loss = ce_loss + vq_loss
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch.y).float().mean()
        batch_size = batch.y.size(0)
        
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val_ce_loss', ce_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val_vq_loss', vq_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val_acc', acc, on_step=False, on_epoch=True, batch_size=batch_size)
        return total_loss
    
    def test_step(self, batch, batch_idx):
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        seq = batch.seq if hasattr(batch, 'seq') and hasattr(self.model, 'sequence_embedding') else None
        
        if self.bypass_codebook:
            logits, assignment_matrix, extra = self._forward_bypass_codebook(h_V, batch.edge_index, h_E, seq, batch.batch)
        else:
            logits, assignment_matrix, extra = self.model(h_V, batch.edge_index, h_E, seq, batch.batch)
        
        ce_loss = self.criterion(logits, batch.y)
        vq_loss = extra.get("vq_loss", 0.0)
        total_loss = ce_loss + vq_loss
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch.y).float().mean()
        batch_size = batch.y.size(0)
        
        self.log('test_loss', total_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_ce_loss', ce_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_vq_loss', vq_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_acc', acc, on_step=False, on_epoch=True, batch_size=batch_size)
        return total_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.get_current_lr())
        
        if self.train_cfg.get('use_cosine_schedule', False):
            warmup_epochs = self.train_cfg.get('warmup_epochs', 5)
            # Use current stage epochs for scheduler
            stage_epochs = self.get_current_stage_epochs()
            scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs, stage_epochs)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        else:
            return optimizer
    
    def get_current_lr(self):
        """Get learning rate for current stage."""
        stage_cfg = getattr(self.multistage_cfg, f'stage{self.current_stage}')
        return stage_cfg.lr
    
    def get_current_stage_epochs(self):
        """Get number of epochs for current stage."""
        stage_cfg = getattr(self.multistage_cfg, f'stage{self.current_stage}')
        return stage_cfg.epochs


@hydra.main(version_base="1.1", config_path='conf', config_name='config_partoken_multistage')
def main(cfg: DictConfig):
    print("🧬 Multi-Stage ParToken Training")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    
    set_seed(cfg.train.seed)
    
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    # Custom output directory
    custom_output_dir = os.path.join(
        "./outputs",
        cfg.train.wandb_project,
        cfg.data.dataset_name,
        f"multistage_{timestamp}"
    )
    os.makedirs(custom_output_dir, exist_ok=True)
    
    print(f"Output directory: {custom_output_dir}")
    
    # Get datasets
    train_dataset, val_dataset, test_dataset, num_classes = get_dataset(
        dataset_name=cfg.data.dataset_name,
        split=cfg.data.split,
        split_similarity_threshold=cfg.data.split_similarity_threshold,
        data_dir=cfg.data.data_dir,
    )
    
    train_loader = create_dataloader(train_dataset, cfg.train.batch_size, cfg.train.num_workers, shuffle=True)
    val_loader = create_dataloader(val_dataset, cfg.train.batch_size, cfg.train.num_workers, shuffle=False)
    test_loader = create_dataloader(test_dataset, cfg.train.batch_size, cfg.train.num_workers, shuffle=False)
    
    # Create multi-stage model
    model = MultiStageParTokenLightning(cfg.model, cfg.train, cfg.multistage, num_classes)
    
    # Multi-stage training loop
    stages = ['stage0', 'stage1', 'stage2']
    stage_names = ['BASELINE', 'CODEBOOK_WARMUP', 'JOINT_FINETUNING']
    
    for stage_idx, (stage_key, stage_name) in enumerate(zip(stages, stage_names)):
        stage_cfg = getattr(cfg.multistage, stage_key)
        
        print(f"\n🚀 STARTING STAGE {stage_idx}: {stage_name}")
        print("=" * 60)
        
        # Setup stage
        model.setup_stage(stage_idx, stage_cfg)
        
        # K-means initialization for codebook in stage 1
        if stage_idx == 1 and stage_cfg.get('kmeans_init', False):
            print("🔧 Initializing codebook with K-means...")
            model.model.kmeans_init_from_loader(
                train_loader, 
                max_batches=stage_cfg.get('kmeans_batches', 50)
            )
            print("✓ K-means initialization completed")
        
        # Logger for this stage
        wandb_logger = None
        if cfg.train.use_wandb:
            wandb_logger = WandbLogger(
                project=cfg.train.wandb_project,
                name=f"{stage_name.lower()}_{timestamp}",
                tags=[stage_name.lower(), cfg.data.dataset_name]
            )
        
        # Checkpoint callback for this stage
        stage_output_dir = os.path.join(custom_output_dir, f"stage_{stage_idx}")
        os.makedirs(stage_output_dir, exist_ok=True)
        
        checkpoint_callback = ModelCheckpoint(
            monitor='val_acc',
            mode='max',
            save_top_k=1,
            dirpath=stage_output_dir,
            filename=f'best-stage{stage_idx}-{{epoch:02d}}-{{val_acc:.3f}}',
            save_last=True
        )
        
        # Trainer for this stage
        trainer = pl.Trainer(
            max_epochs=stage_cfg.epochs,
            logger=wandb_logger,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1 if torch.cuda.is_available() else None,
            log_every_n_steps=10,
            default_root_dir=stage_output_dir,
            callbacks=[checkpoint_callback],
            enable_checkpointing=True,
            enable_progress_bar=True
        )
        
        # Reset stage epoch counter
        model.stage_epoch = 0
        
        # Train this stage
        print(f"\n📚 Training Stage {stage_idx} for {stage_cfg.epochs} epochs...")
        trainer.fit(model, train_loader, val_loader)
        
        # Post-stage processing
        if stage_idx == 0:  # After baseline stage
            print("💾 Saving teacher model...")
            model.save_teacher_model()
            
        elif stage_idx == 1:  # After codebook warmup
            print("🔓 Enabling codebook for joint training...")
            
        # Save stage checkpoint
        stage_checkpoint_path = os.path.join(stage_output_dir, f'stage_{stage_idx}_final.ckpt')
        trainer.save_checkpoint(stage_checkpoint_path)
        print(f"✓ Stage {stage_idx} checkpoint saved to {stage_checkpoint_path}")
        
        # Close wandb run for this stage
        if wandb_logger is not None:
            wandb.finish()
        
        print(f"✅ STAGE {stage_idx} COMPLETED")
        print("=" * 60)
    
    # Final testing
    print("\n🧪 FINAL TESTING")
    print("=" * 60)
    
    # Create final test trainer
    test_trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else None,
        logger=False
    )
    
    test_results = test_trainer.test(model, test_loader)
    
    # Save final model
    final_model_path = os.path.join(custom_output_dir, 'final_multistage_model.ckpt')
    test_trainer.save_checkpoint(final_model_path)
    
    print(f"\n🎉 MULTI-STAGE TRAINING COMPLETED!")
    print(f"📁 Final model saved to: {final_model_path}")
    print(f"📊 Final test accuracy: {test_results[0]['test_acc']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

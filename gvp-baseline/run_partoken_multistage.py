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
from utils.interpretability import (
    batch_interpretability_analysis, 
    print_interpretability_summary,
    plot_importance_distribution,
    run_interpretability_analysis_example
)

from utils.save_checkpoints import (
    save_stage_specific_checkpoint,
    create_checkpoint_summary
)
from utils.loss_schedulers import LossWeightScheduler

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
            # partitioner parameters
            max_clusters=model_cfg.max_clusters,
            nhid=model_cfg.nhid,
            k_hop=model_cfg.k_hop,
            cluster_size_max=model_cfg.cluster_size_max,
            termination_threshold=model_cfg.termination_threshold,
            tau_init=model_cfg.tau_init,
            tau_min=model_cfg.tau_min,
            tau_decay=model_cfg.tau_decay,
            # codebook parameters
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
        
        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch.y).float().mean()
        batch_size = batch.y.size(0)
        
        # Log metrics
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_ce_loss', ce_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_vq_loss', vq_loss, on_step=True, on_epoch=True, batch_size=batch_size)
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
        
        # Use attention-weighted pooling if available, otherwise fall back to masked mean
        if hasattr(self.model, '_attention_weighted_pooling'):
            cluster_pooled, _ = self.model._attention_weighted_pooling(refined_clusters, cluster_valid_mask)
        else:
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
            # For bypass mode, we can't get attention importance scores
            cluster_importance = None
        else:
            logits, assignment_matrix, extra, cluster_importance = self.model(
                h_V, batch.edge_index, h_E, seq, batch.batch, return_importance=True
            )
        
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
        
        # Log importance statistics if available
        if cluster_importance is not None:
            # Compute importance statistics
            valid_mask = (assignment_matrix.sum(dim=1) > 0)  # [B, S]
            masked_importance = cluster_importance * valid_mask.float()
            
            # Average max importance per protein
            max_importance = masked_importance.max(dim=1)[0].mean()
            
            # Entropy of importance distribution per protein
            importance_entropy = []
            for b in range(cluster_importance.size(0)):
                valid_imp = masked_importance[b][valid_mask[b]]
                if len(valid_imp) > 0:
                    p = valid_imp + 1e-8
                    entropy = (-p * torch.log(p)).sum()
                    importance_entropy.append(entropy)
            
            if importance_entropy:
                avg_entropy = torch.tensor(importance_entropy).mean()
                self.log('test_importance_max', max_importance, on_step=False, on_epoch=True, batch_size=batch_size)
                self.log('test_importance_entropy', avg_entropy, on_step=False, on_epoch=True, batch_size=batch_size)
        
        return total_loss
    
    def get_interpretability_analysis(
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
        if self.bypass_codebook:
            print("⚠️ Warning: Interpretability analysis not available in bypass_codebook mode")
            return None
            
        return batch_interpretability_analysis(
            model=self,
            dataloader=dataloader,
            device=device,
            max_batches=max_batches
        )
    
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
    
    # Use Hydra's output directory to ensure consistency
    from hydra.core.hydra_config import HydraConfig
    hydra_cfg = HydraConfig.get()
    custom_output_dir = hydra_cfg.runtime.output_dir
    
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
            timestamp = datetime.now().strftime("%Y-%m-%d")
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
        if stage_idx == 1:  # After codebook warmup
            print("🔓 Codebook training completed...")
            
        # Save stage checkpoint
        stage_checkpoint_path = os.path.join(stage_output_dir, f'stage_{stage_idx}_final.ckpt')
        trainer.save_checkpoint(stage_checkpoint_path)
        print(f"✓ Stage {stage_idx} checkpoint saved to {stage_checkpoint_path}")
        
        # Save stage-specific components checkpoint
        stage_info = {
            'epoch': stage_cfg.epochs,
            'val_acc': trainer.callback_metrics.get('val_acc', 0.0).item() if 'val_acc' in trainer.callback_metrics else 0.0,
            'val_loss': trainer.callback_metrics.get('val_loss', 0.0).item() if 'val_loss' in trainer.callback_metrics else 0.0,
        }
        
        component_checkpoint_path = save_stage_specific_checkpoint(
            model, 
            stage_idx, 
            stage_output_dir, 
            stage_info
        )
        print(f"✓ Stage {stage_idx} component checkpoint saved to {component_checkpoint_path}")
        
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
    
    # Run interpretability analysis if not in bypass mode
    if not model.bypass_codebook and cfg.multistage.get('run_interpretability', True):
        print("\n🔍 INTERPRETABILITY ANALYSIS")
        print("=" * 60)
        
        # Create interpretability output directory
        interp_output_dir = os.path.join(custom_output_dir, "interpretability")
        os.makedirs(interp_output_dir, exist_ok=True)
        
        # Run analysis
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("📊 Running interpretability analysis on test set...")
        interp_results = model.get_interpretability_analysis(
            test_loader, 
            device=device,
            max_batches=cfg.multistage.get('interpretability_max_batches', 20)
        )
        
        if interp_results is not None:
            # Save results
            import json
            results_path = os.path.join(interp_output_dir, "test_interpretability.json")
            
            # Convert tensors to lists for JSON serialization
            def convert_for_json(obj):
                if hasattr(obj, 'tolist'):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                else:
                    return obj
            
            serializable_results = convert_for_json(interp_results)
            
            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            # Print summary
            print_interpretability_summary(interp_results)
            
            # Create visualization
            try:
                plot_path = os.path.join(interp_output_dir, "importance_analysis.png")
                plot_importance_distribution(interp_results, save_path=plot_path)
            except Exception as e:
                print(f"⚠️  Could not create visualization: {e}")
            
            print(f"✓ Interpretability results saved to {interp_output_dir}")
        else:
            print("⚠️  Interpretability analysis skipped (bypass_codebook mode)")
    
    # Save final model
    final_model_path = os.path.join(custom_output_dir, 'final_multistage_model.ckpt')
    test_trainer.save_checkpoint(final_model_path)
    
    # Create checkpoint summary
    summary_path = create_checkpoint_summary(custom_output_dir)
    
    print(f"\n🎉 MULTI-STAGE TRAINING COMPLETED!")
    print(f"📁 Final model saved to: {final_model_path}")
    print(f"📋 Checkpoint summary: {summary_path}")
    print(f"📊 Final test accuracy: {test_results[0]['test_acc']:.4f}")
    
    # Print additional metrics if available
    if 'test_importance_max' in test_results[0]:
        print(f"🎯 Average max cluster importance: {test_results[0]['test_importance_max']:.3f}")
        print(f"📈 Average importance entropy: {test_results[0]['test_importance_entropy']:.3f}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
    # pass
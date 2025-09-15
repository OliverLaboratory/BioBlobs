import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from utils.proteinshake_dataset import get_dataset, create_dataloader
from utils.utils import set_seed
import torch
import wandb
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.interpretability import (
    print_interpretability_summary,
    save_interpretability_results,
)
from utils.save_checkpoints import (
    save_stage_specific_checkpoint,
    create_checkpoint_summary
)
from train_lightling import MultiStageParTokenLightning

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
        test_mode=cfg.data.get('test_mode', False),  # Add test_mode parameter
    )
    
    train_loader = create_dataloader(train_dataset, cfg.train.batch_size, cfg.train.num_workers, shuffle=True)
    val_loader = create_dataloader(val_dataset, cfg.train.batch_size, cfg.train.num_workers, shuffle=False)
    test_loader = create_dataloader(test_dataset, cfg.train.batch_size, cfg.train.num_workers, shuffle=False)
    
    # Create multi-stage model
    model = MultiStageParTokenLightning(cfg.model, cfg.train, cfg.multistage, num_classes)
    
    # Multi-stage training loop (two stages: baseline + joint)
    stages = ['stage0', 'stage1']
    stage_names = ['BASELINE', 'JOINT_FINETUNING']
    
    for stage_idx, (stage_key, stage_name) in enumerate(zip(stages, stage_names)):
        stage_cfg = getattr(cfg.multistage, stage_key)
        
        print(f"\n🚀 STARTING STAGE {stage_idx}: {stage_name}")
        print("=" * 60)
        
        # Setup stage
        model.setup_stage(stage_idx, stage_cfg)
        
        # K-means initialization for codebook when transitioning to stage 1
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
            trainer,  # Add trainer parameter
            model, 
            stage_idx, 
            stage_output_dir, 
            stage_info
        )
        print(f"✓ Stage {stage_idx} component checkpoint saved to {component_checkpoint_path}")
        
        # Run interpretability analysis for this stage
        run_interp = cfg.multistage.get('run_interpretability', True)
        # Allow interpretability for stage 0 if explicitly enabled
        if stage_idx == 0:
            run_interp = cfg.multistage.get('run_stage0_interpretability', run_interp)
        
        if (not model.bypass_codebook or stage_idx == 0) and run_interp:
            print(f"\n🔍 INTERPRETABILITY ANALYSIS - STAGE {stage_idx}")
            print("=" * 60)
            
            # Create stage-specific interpretability output directory
            stage_interp_output_dir = os.path.join(stage_output_dir, "interpretability")
            os.makedirs(stage_interp_output_dir, exist_ok=True)
            
            # Run analysis on validation set for this stage
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            print(f"📊 Running interpretability analysis on validation set for stage {stage_idx}...")
            stage_interp_results = model.get_inter_info(
                val_loader, 
                device=device,
                max_batches=cfg.multistage.get('interpretability_max_batches', 10)  # Limit batches for stage analysis
            )
            
            if stage_interp_results is not None:
                # Save stage-specific results
                stage_results_path = os.path.join(stage_interp_output_dir, f"stage_{stage_idx}_val_interpretability.json")
                save_interpretability_results(stage_interp_results, stage_results_path)
                
                # Print stage summary
                print(f"📈 Stage {stage_idx} Interpretability Summary:")
                print_interpretability_summary(stage_interp_results)
            else:
                print("⚠️  Interpretability analysis skipped (bypass_codebook mode)")
        
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
        interp_results = model.get_inter_info(
            test_loader, 
            device=device,
            max_batches=cfg.multistage.get('interpretability_max_batches', None)
        )
        
        if interp_results is not None:
            # Save results using helper function
            results_path = os.path.join(interp_output_dir, "test_interpretability.json")
            save_interpretability_results(interp_results, results_path)
            
            # Print summary
            print_interpretability_summary(interp_results)
        else:
            print("⚠️  Interpretability analysis skipped (bypass_codebook mode)")
    
    # Save final model
    final_model_path = os.path.join(custom_output_dir, 'final_multistage_model.ckpt')
    test_trainer.save_checkpoint(final_model_path)
    
    # Create checkpoint summary
    summary_path = create_checkpoint_summary(custom_output_dir)
    
    print("\n🎉 MULTI-STAGE TRAINING COMPLETED!")
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
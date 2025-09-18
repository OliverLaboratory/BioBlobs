import os
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
    create_checkpoint_summary
)
from train_lightling import PartGVPLightning
import json


def test_checkpoint(checkpoint_path, model_class, model_cfg, train_cfg, num_classes, test_loader, checkpoint_type="best", wandb_logger=None):
    """Test a specific checkpoint and return results."""
    print(f"\n🧪 Testing {checkpoint_type} checkpoint: {checkpoint_path}")
    
    # Load model from checkpoint
    model = model_class.load_from_checkpoint(
        checkpoint_path,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        num_classes=num_classes
    )
    
    # Create test trainer with the same logger if provided
    test_trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else None,
        logger=wandb_logger if wandb_logger is not None else False
    )
    
    # Run test
    test_results = test_trainer.test(model, test_loader)
    
    result = {
        'checkpoint_type': checkpoint_type,
        'checkpoint_path': str(checkpoint_path),
        'test_accuracy': test_results[0]['test_acc'],
        'test_loss': test_results[0]['test_loss'],
        'test_ce_loss': test_results[0]['test_ce_loss']
    }
    
    # Add importance metrics if available
    if 'test_importance_max' in test_results[0]:
        result['test_importance_max'] = test_results[0]['test_importance_max']
        result['test_importance_entropy'] = test_results[0]['test_importance_entropy']
    
    print(f"✓ {checkpoint_type.title()} checkpoint test accuracy: {result['test_accuracy']:.4f}")
    
    return result, model


@hydra.main(version_base="1.1", config_path='conf', config_name='config_partgvp')
def main(cfg: DictConfig):
    print("🧬 PartGVP Training (GVP + Partitioner + Global-Cluster Attention)")
    print("=" * 70)
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
        test_mode=cfg.data.get('test_mode', False),
    )

    # return
    
    train_loader = create_dataloader(train_dataset, cfg.train.batch_size, cfg.train.num_workers, shuffle=True)
    val_loader = create_dataloader(val_dataset, cfg.train.batch_size, cfg.train.num_workers, shuffle=False)
    test_loader = create_dataloader(test_dataset, cfg.train.batch_size, cfg.train.num_workers, shuffle=False)
    
    # Create PartGVP model
    model = PartGVPLightning(cfg.model, cfg.train, num_classes)
    
    print("\n🏗️  Model Architecture:")
    print(f"  • Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  • Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("  • Mode: PartGVP (bypass codebook)")
    
    # Logger
    wandb_logger = None
    if cfg.train.use_wandb:
        timestamp = datetime.now().strftime("%Y-%m-%d")
        wandb_logger = WandbLogger(
            project=cfg.train.wandb_project,
            name=f"{cfg.data.dataset_name}_{cfg.data.split}_seq_{cfg.model.seq_in}_{cfg.model.max_clusters}_{cfg.model.cluster_size_max}_{cfg.train.lr}_{timestamp}",
            tags=["partgvp", cfg.data.dataset_name, cfg.data.split]
        )
    
    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        save_top_k=1,
        dirpath=custom_output_dir,
        filename='best-partgvp-{epoch:02d}-{val_acc:.3f}',
        save_last=True
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        logger=wandb_logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else None,
        log_every_n_steps=10,
        default_root_dir=custom_output_dir,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
        enable_progress_bar=True
    )
    
    # Training
    print(f"\n📚 Training PartGVP for {cfg.train.epochs} epochs...")
    trainer.fit(model, train_loader, val_loader)
    
    # Get checkpoint paths
    best_checkpoint_path = checkpoint_callback.best_model_path
    last_checkpoint_path = checkpoint_callback.last_model_path
    
    print("\n📁 Checkpoints:")
    print(f"  • Best: {best_checkpoint_path}")
    print(f"  • Last: {last_checkpoint_path}")
    
    # Test both checkpoints
    results_summary = {
        'training_config': OmegaConf.to_container(cfg, resolve=True),
        'model_info': {
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'mode': 'PartGVP'
        },
        'checkpoints': {}
    }
    
    # Test best checkpoint
    if best_checkpoint_path and os.path.exists(best_checkpoint_path):
        best_results, best_model = test_checkpoint(
            best_checkpoint_path, PartGVPLightning, cfg.model, cfg.train, num_classes, test_loader, "best", wandb_logger
        )
        results_summary['checkpoints']['best'] = best_results
        final_model = best_model  # Use best model for interpretability
    else:
        print("⚠️  Best checkpoint not found")
        final_model = model
    
    # Test last checkpoint
    if last_checkpoint_path and os.path.exists(last_checkpoint_path):
        last_results, _ = test_checkpoint(
            last_checkpoint_path, PartGVPLightning, cfg.model, cfg.train, num_classes, test_loader, "last", wandb_logger
        )
        results_summary['checkpoints']['last'] = last_results
    else:
        print("⚠️  Last checkpoint not found")
    
    # Save results summary
    results_summary_path = os.path.join(custom_output_dir, 'results_summary.json')
    with open(results_summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n✓ Results summary saved to: {results_summary_path}")
    
    # Run interpretability analysis
    if cfg.interpretability.get('enabled', True):
        print("\n🔍 INTERPRETABILITY ANALYSIS")
        print("=" * 70)
        
        # Create interpretability output directory
        interp_output_dir = os.path.join(custom_output_dir, "interpretability")
        os.makedirs(interp_output_dir, exist_ok=True)
        
        # Run analysis on test set
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("📊 Running interpretability analysis on test set...")
        interp_results = final_model.get_inter_info(
            test_loader, 
            device=device,
            max_batches=cfg.interpretability.get('max_batches', None)
        )
        
        if interp_results is not None:
            # Save results
            results_path = os.path.join(interp_output_dir, "test_interpretability.json")
            save_interpretability_results(interp_results, results_path)
            
            # Print summary
            print_interpretability_summary(interp_results)
            
            # Add interpretability summary to results
            results_summary['interpretability'] = {
                'enabled': True,
                'results_path': results_path,
                'summary': interp_results['aggregated_stats']
            }
        else:
            print("⚠️  Interpretability analysis failed")
            results_summary['interpretability'] = {'enabled': False, 'error': 'Analysis failed'}
    else:
        print("⚠️  Interpretability analysis disabled")
        results_summary['interpretability'] = {'enabled': False}
    
    # Update results summary with interpretability info
    with open(results_summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Save final model
    final_model_path = os.path.join(custom_output_dir, 'final_partgvp_model.ckpt')
    trainer.save_checkpoint(final_model_path)
    
    # Create checkpoint summary
    summary_path = create_checkpoint_summary(custom_output_dir)
    
    # Close wandb if used
    if wandb_logger is not None:
        wandb.finish()
    
    print("\n🎉 PARTGVP TRAINING COMPLETED!")
    print("=" * 70)
    print(f"📁 Final model saved to: {final_model_path}")
    print(f"📋 Checkpoint summary: {summary_path}")
    print(f"📊 Results summary: {results_summary_path}")
    
    # Print final results summary
    if 'best' in results_summary['checkpoints']:
        print(f"🏆 Best checkpoint test accuracy: {results_summary['checkpoints']['best']['test_accuracy']:.4f}")
    if 'last' in results_summary['checkpoints']:
        print(f"📈 Last checkpoint test accuracy: {results_summary['checkpoints']['last']['test_accuracy']:.4f}")
    
    if results_summary['interpretability']['enabled']:
        print("🔍 Interpretability analysis completed")
        if 'summary' in results_summary['interpretability']:
            print(f"📊 Overall test accuracy: {results_summary['interpretability']['summary']['accuracy']:.4f}")
            print(f"🎯 Average attention concentration: {results_summary['interpretability']['summary']['avg_importance_concentration']:.3f}")
    
    print("=" * 70)


if __name__ == "__main__":
    main()

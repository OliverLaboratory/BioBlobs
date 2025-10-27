"""
BioBlobs Multi-Stage Training Script
Stage 0: Train baseline model (bypass codebook)
Stage 1: Fine-tune with codebook enabled
"""

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
from utils.train_w_codebook import MultiStageBioBlobsLightning
from utils.train_w_codebook_multilabel import MultiStageBioBlobsMultiLabelLightning
from hydra.utils import to_absolute_path
from utils.interpretability import run_interpretability_analysis
import json


def initialize_codebook(model, train_loader, device, max_batches=50):
    """Initialize codebook from training data using K-means clustering.
    
    Args:
        model: MultiStageBioBlobsLightning model with codebook
        train_loader: Training data loader
        device: Device to run on
        max_batches: Maximum number of batches to use for initialization
        
    Returns:
        Dictionary with initialization statistics
    """
    print(f"Initializing codebook with K-means (max_batches={max_batches})")
    
    # Move model to device
    model.to(device)
    
    # Call the model's codebook initialization method
    model.model.kmeans_init_from_loader(
        loader=train_loader,
        max_batches=max_batches,
        device=device
    )
    
    # Return initialization stats
    stats = {
        "codebook_size": model.model.codebook.K,
        "embedding_dim": model.model.codebook.D,
        "initialization_method": "kmeans_from_clusters",
        "max_batches_used": max_batches,
    }
    
    print(f"✓ Codebook initialized: {stats['codebook_size']} codes, {stats['embedding_dim']} dims")
    
    return stats


def create_wandb_logger(cfg):
    """Create WandB logger for the entire multi-stage training run."""
    if not cfg.train.use_wandb:
        return None
    
    timestamp = datetime.now().strftime("%Y-%m-%d")
    
    return WandbLogger(
        project=cfg.train.wandb_project,
        name=f"{cfg.data.dataset_name}_multistage_{timestamp}",
        tags=["bioblobs-multistage", "two-stage-training", cfg.data.dataset_name],
    )


def create_checkpoint_callback(custom_output_dir, stage_idx, cfg):
    """Create checkpoint callback for a specific stage."""
    # Use appropriate metric based on dataset type
    if cfg.data.dataset_name == "go":
        monitor_metric = "val_fmax"
        filename_template = f"best-stage{stage_idx}-{{epoch:02d}}-{{val_fmax:.3f}}"
    else:
        monitor_metric = "val_acc"
        filename_template = f"best-stage{stage_idx}-{{epoch:02d}}-{{val_acc:.3f}}"
    
    return ModelCheckpoint(
        monitor=monitor_metric,
        mode="max",
        save_top_k=1,
        dirpath=os.path.join(custom_output_dir, f"stage{stage_idx}"),
        filename=filename_template,
        save_last=True,
    )


def create_trainer(cfg, stage_idx, custom_output_dir, wandb_logger, checkpoint_callback):
    """Create PyTorch Lightning trainer for a specific stage.
    
    Args:
        cfg: Hydra config
        stage_idx: Current stage index (0 or 1)
        custom_output_dir: Base output directory
        wandb_logger: WandB logger instance (shared across stages)
        checkpoint_callback: Checkpoint callback for this stage
    """
    stage_cfg = getattr(cfg.train, f"stage{stage_idx}")
    
    # Log stage transition to WandB
    if wandb_logger is not None:
        stage_name = ["baseline", "joint_fine_tuning"][stage_idx]
        wandb_logger.experiment.config.update({
            f"stage{stage_idx}_epochs": stage_cfg.epochs,
            f"stage{stage_idx}_lr": stage_cfg.lr,
            f"stage{stage_idx}_name": stage_name,
        }, allow_val_change=True)
    
    return pl.Trainer(
        max_epochs=stage_cfg.epochs,
        logger=wandb_logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        log_every_n_steps=20,
        default_root_dir=os.path.join(custom_output_dir, f"stage{stage_idx}"),
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
        enable_progress_bar=True,
    )


def train_and_evaluate_stage(
    model, trainer, train_loader, val_loader, test_loader, 
    stage_idx, checkpoint_callback, cfg, custom_output_dir
):
    """Train and evaluate a specific stage."""
    stage_name = ["baseline", "joint_fine_tuning"][stage_idx]
    stage_display = ["BASELINE TRAINING", "JOINT FINE-TUNING WITH CODEBOOK"][stage_idx]
    
    print("\n" + "=" * 70)
    print(f"STAGE {stage_idx}: {stage_display}")
    print("=" * 70)
    
    # Log stage transition to WandB
    if trainer.logger is not None:
        trainer.logger.experiment.log({
            "stage/current": stage_idx,
            "stage/name": stage_name,
        })
    
    # Setup stage
    model.setup_stage(stage_idx=stage_idx)
    
    # Train
    stage_cfg = getattr(cfg.train, f"stage{stage_idx}")
    print(f"\n📚 Training Stage {stage_idx} for {stage_cfg.epochs} epochs...")
    trainer.fit(model, train_loader, val_loader)
    
    # Test
    print("\n" + "=" * 70)
    print(f"TESTING STAGE {stage_idx}")
    print("=" * 70)
    
    test_results = trainer.test(model, test_loader)
    
    # Log test results to WandB with stage prefix
    if trainer.logger is not None and test_results:
        stage_test_metrics = {f"stage{stage_idx}/{k}": v for k, v in test_results[0].items()}
        trainer.logger.experiment.log(stage_test_metrics)
    
    # Collect results
    stage_results = {
        "stage": stage_idx,
        "name": stage_name,
        "test_metrics": test_results[0] if test_results else {},
        "best_checkpoint": checkpoint_callback.best_model_path,
        "last_checkpoint": checkpoint_callback.last_model_path,
    }
    
    # Run interpretability analysis
    print(f"\n🔍 Running interpretability analysis for Stage {stage_idx}...")
    stage_output_dir = os.path.join(custom_output_dir, f"stage{stage_idx}")
    
    if stage_idx == 0:
        # Stage 0: Interpretability not available in bypass mode
        stage_results["interpretability"] = {
            "note": "Interpretability analysis not available in bypass_codebook mode (Stage 0)",
            "cluster_statistics_available": True,
        }
    else:
        # Stage 1: Full interpretability available
        interpretability_summary = run_interpretability_analysis(
            model, test_loader, cfg, stage_output_dir
        )
        stage_results["interpretability"] = interpretability_summary
    
    # Save stage results
    stage_results_path = os.path.join(stage_output_dir, f"stage{stage_idx}_results.json")
    with open(stage_results_path, "w") as f:
        json.dump(stage_results, f, indent=2)
    
    # Print summary
    print(f"\n✓ Stage {stage_idx} results saved to: {stage_results_path}")
    
    # Print appropriate metric based on dataset type
    if 'test_acc' in stage_results['test_metrics']:
        print(f"✓ Stage {stage_idx} test accuracy: {stage_results['test_metrics'].get('test_acc', 'N/A'):.4f}")
    elif 'test_fmax' in stage_results['test_metrics']:
        print(f"✓ Stage {stage_idx} test F-max: {stage_results['test_metrics'].get('test_fmax', 'N/A'):.4f}")
    
    if stage_idx == 1:
        interpretability_summary = stage_results.get("interpretability", {})
        if interpretability_summary and interpretability_summary.get("enabled"):
            print("✓ Interpretability analysis completed")
            print(f"  - Results saved to: {interpretability_summary.get('results_path', 'N/A')}")
    
    return stage_results


def save_final_summary(cfg, custom_output_dir, stage0_results, stage1_results, init_stats, wandb_logger=None):
    """Create and save final summary comparing both stages."""
    print("\n" + "=" * 70)
    print("FINAL SUMMARY - MULTI-STAGE TRAINING")
    print("=" * 70)
    
    final_summary = {
        "dataset": cfg.data.dataset_name,
        "split": cfg.data.split,
        "total_epochs": cfg.train.stage0.epochs + cfg.train.stage1.epochs,
        "batch_size": cfg.train.batch_size,
        "codebook_initialization": init_stats,
        "stage0": stage0_results,
        "stage1": stage1_results,
        "comparison": {},
    }
    
    # Build comparison based on available metrics
    stage0_metrics = stage0_results['test_metrics']
    stage1_metrics = stage1_results['test_metrics']
    
    if 'test_acc' in stage0_metrics and 'test_acc' in stage1_metrics:
        # Single-label classification
        final_summary["comparison"] = {
            "test_acc_improvement": stage1_metrics['test_acc'] - stage0_metrics['test_acc'],
            "test_loss_stage0": stage0_metrics.get('test_loss', 'N/A'),
            "test_loss_stage1": stage1_metrics.get('test_loss', 'N/A'),
        }
    elif 'test_fmax' in stage0_metrics and 'test_fmax' in stage1_metrics:
        # Multi-label classification
        final_summary["comparison"] = {
            "test_fmax_improvement": stage1_metrics['test_fmax'] - stage0_metrics['test_fmax'],
            "test_loss_stage0": stage0_metrics.get('test_loss', 'N/A'),
            "test_loss_stage1": stage1_metrics.get('test_loss', 'N/A'),
        }
    
    # Log final comparison to WandB
    if wandb_logger is not None:
        if 'test_acc_improvement' in final_summary['comparison']:
            # Single-label metrics
            wandb_logger.experiment.log({
                "final/test_acc_improvement": final_summary['comparison']['test_acc_improvement'],
                "final/stage0_test_acc": stage0_metrics['test_acc'],
                "final/stage1_test_acc": stage1_metrics['test_acc'],
                "final/stage0_test_loss": stage0_metrics.get('test_loss', 0),
                "final/stage1_test_loss": stage1_metrics.get('test_loss', 0),
            })
        elif 'test_fmax_improvement' in final_summary['comparison']:
            # Multi-label metrics
            wandb_logger.experiment.log({
                "final/test_fmax_improvement": final_summary['comparison']['test_fmax_improvement'],
                "final/stage0_test_fmax": stage0_metrics['test_fmax'],
                "final/stage1_test_fmax": stage1_metrics['test_fmax'],
                "final/stage0_test_loss": stage0_metrics.get('test_loss', 0),
                "final/stage1_test_loss": stage1_metrics.get('test_loss', 0),
            })
    
    # Save final summary
    final_summary_path = os.path.join(custom_output_dir, "final_summary.json")
    with open(final_summary_path, "w") as f:
        json.dump(final_summary, f, indent=2)
    
    # Print comparison
    print("\n📊 STAGE COMPARISON:")
    print("  Stage 0 (Baseline):")
    
    if 'test_acc' in stage0_metrics:
        # Single-label classification
        print(f"    - Test Accuracy: {stage0_metrics['test_acc']:.4f}")
        print(f"    - Test Loss:     {stage0_metrics.get('test_loss', 'N/A'):.4f}")
        print("  Stage 1 (With Codebook):")
        print(f"    - Test Accuracy: {stage1_metrics['test_acc']:.4f}")
        print(f"    - Test Loss:     {stage1_metrics.get('test_loss', 'N/A'):.4f}")
        print("  Improvement:")
        print(f"    - Accuracy Δ:    {final_summary['comparison']['test_acc_improvement']:+.4f}")
    elif 'test_fmax' in stage0_metrics:
        # Multi-label classification
        print(f"    - Test F-max:    {stage0_metrics['test_fmax']:.4f}")
        print(f"    - Test Loss:     {stage0_metrics.get('test_loss', 'N/A'):.4f}")
        print("  Stage 1 (With Codebook):")
        print(f"    - Test F-max:    {stage1_metrics['test_fmax']:.4f}")
        print(f"    - Test Loss:     {stage1_metrics.get('test_loss', 'N/A'):.4f}")
        print("  Improvement:")
        print(f"    - F-max Δ:       {final_summary['comparison']['test_fmax_improvement']:+.4f}")
    
    return final_summary


@hydra.main(version_base="1.1", config_path="conf", config_name="config_bioblobs_multistage")
def main(cfg: DictConfig):
    print("BioBlobs Multi-Stage Training")
    print("=" * 70)
    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.train.seed)

    from hydra.core.hydra_config import HydraConfig

    hydra_cfg = HydraConfig.get()
    custom_output_dir = hydra_cfg.runtime.output_dir

    print(f"Output directory: {custom_output_dir}")

    # Get datasets
    train_dataset, val_dataset, test_dataset, num_classes = get_dataset(
        dataset_name=cfg.data.dataset_name,
        split=cfg.data.split,
        split_similarity_threshold=cfg.data.split_similarity_threshold,
        data_dir=to_absolute_path(cfg.data.data_dir),
        test_mode=cfg.data.get("test_mode", False),
    )

    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset, cfg.train.batch_size, cfg.train.num_workers, shuffle=True
    )
    val_loader = create_dataloader(
        val_dataset, cfg.train.batch_size, cfg.train.num_workers, shuffle=False
    )
    test_loader = create_dataloader(
        test_dataset, cfg.train.batch_size, cfg.train.num_workers, shuffle=False
    )

    # Create multi-stage model (choose based on dataset type)
    print("\nCreating Multi-Stage BioBlobs model...")
    if cfg.data.dataset_name == "go":
        print("Using MultiStageBioBlobsMultiLabelLightning for multi-label GO classification")
        model = MultiStageBioBlobsMultiLabelLightning(cfg.model, cfg.train, num_classes)
    else:
        print("Using MultiStageBioBlobsLightning for single-label classification")
        model = MultiStageBioBlobsLightning(cfg.model, cfg.train, num_classes)

    print("\nModel Architecture:")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # Create single WandB logger for entire multi-stage training
    wandb_logger = create_wandb_logger(cfg)
    
    # Log initial config to WandB
    if wandb_logger is not None:
        wandb_logger.experiment.config.update({
            "dataset": cfg.data.dataset_name,
            "total_epochs": cfg.train.stage0.epochs + cfg.train.stage1.epochs,
            "batch_size": cfg.train.batch_size,
            "num_classes": num_classes,
        })

    # ============================================================================
    # STAGE 0: Baseline Training (Bypass Codebook)
    # ============================================================================
    
    checkpoint_callback_stage0 = create_checkpoint_callback(custom_output_dir, stage_idx=0, cfg=cfg)
    trainer_stage0 = create_trainer(
        cfg, stage_idx=0, custom_output_dir=custom_output_dir, 
        wandb_logger=wandb_logger, checkpoint_callback=checkpoint_callback_stage0
    )
    
    stage0_results = train_and_evaluate_stage(
        model, trainer_stage0, train_loader, val_loader, test_loader,
        stage_idx=0, checkpoint_callback=checkpoint_callback_stage0,
        cfg=cfg, custom_output_dir=custom_output_dir
    )

    # ============================================================================
    # CODEBOOK INITIALIZATION (K-MEANS) - Before Stage 1
    # ============================================================================
    
    print("\n" + "=" * 70)
    print("CODEBOOK INITIALIZATION (K-MEANS)")
    print("=" * 70)
    print("Initializing VQ codebook from Stage 0 cluster features...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_stats = initialize_codebook(
        model=model,
        train_loader=train_loader,
        device=device,
        max_batches=cfg.train.get("kmeans_max_batches", 50),
    )
    
    # Save initialization stats
    init_dir = os.path.join(custom_output_dir, "codebook_initialization")
    os.makedirs(init_dir, exist_ok=True)
    init_stats_path = os.path.join(init_dir, "initialization_stats.json")
    with open(init_stats_path, "w") as f:
        json.dump(init_stats, f, indent=2)
    
    print(f"✓ Codebook initialized: {init_stats['codebook_size']} codes, {init_stats['embedding_dim']} dims")
    print(f"✓ Initialization method: {init_stats['initialization_method']}")
    print(f"✓ Stats saved to: {init_stats_path}")
    
    # Log initialization to WandB
    if wandb_logger is not None:
        wandb_logger.experiment.log({
            "codebook/size": init_stats['codebook_size'],
            "codebook/embedding_dim": init_stats['embedding_dim'],
            "codebook/init_method": init_stats['initialization_method'],
        })

    # ============================================================================
    # STAGE 1: Joint Fine-Tuning (With Codebook)
    # ============================================================================
    
    checkpoint_callback_stage1 = create_checkpoint_callback(custom_output_dir, stage_idx=1, cfg=cfg)
    trainer_stage1 = create_trainer(
        cfg, stage_idx=1, custom_output_dir=custom_output_dir,
        wandb_logger=wandb_logger, checkpoint_callback=checkpoint_callback_stage1
    )
    
    stage1_results = train_and_evaluate_stage(
        model, trainer_stage1, train_loader, val_loader, test_loader,
        stage_idx=1, checkpoint_callback=checkpoint_callback_stage1,
        cfg=cfg, custom_output_dir=custom_output_dir
    )

    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    
    save_final_summary(cfg, custom_output_dir, stage0_results, stage1_results, init_stats, wandb_logger)
    
    # Finish WandB run after both stages complete
    if wandb_logger is not None:
        wandb.finish()


if __name__ == "__main__":
    main()

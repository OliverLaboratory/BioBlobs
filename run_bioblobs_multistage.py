"""
Example script showing how to use MultiStageBioBlobsLightning for progressive training.

This demonstrates:
1. Stage 0: Train baseline model (bypass codebook)
2. Stage 1: Fine-tune with codebook enabled
3. Access to global hyperparameters (batch_size, num_workers, etc.)
4. Stage-specific hyperparameters (epochs, lr, loss_weights)
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
from hydra.utils import to_absolute_path
from utils.interpretability import run_interpretability_analysis
import json


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


def create_checkpoint_callback(custom_output_dir, stage_idx):
    """Create checkpoint callback for a specific stage."""
    return ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        dirpath=os.path.join(custom_output_dir, f"stage{stage_idx}"),
        filename=f"best-stage{stage_idx}-{{epoch:02d}}-{{val_acc:.3f}}",
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
    print(f"✓ Stage {stage_idx} test accuracy: {stage_results['test_metrics'].get('test_acc', 'N/A'):.4f}")
    
    if stage_idx == 1:
        interpretability_summary = stage_results.get("interpretability", {})
        if interpretability_summary and interpretability_summary.get("enabled"):
            print("✓ Interpretability analysis completed")
            print(f"  - Results saved to: {interpretability_summary.get('results_path', 'N/A')}")
    
    return stage_results


def save_final_summary(cfg, custom_output_dir, stage0_results, stage1_results, wandb_logger=None):
    """Create and save final summary comparing both stages."""
    print("\n" + "=" * 70)
    print("FINAL SUMMARY - MULTI-STAGE TRAINING")
    print("=" * 70)
    
    final_summary = {
        "dataset": cfg.data.dataset_name,
        "split": cfg.data.split,
        "total_epochs": cfg.train.stage0.epochs + cfg.train.stage1.epochs,
        "batch_size": cfg.train.batch_size,
        "stage0": stage0_results,
        "stage1": stage1_results,
        "comparison": {
            "test_acc_improvement": (
                stage1_results['test_metrics'].get('test_acc', 0) - 
                stage0_results['test_metrics'].get('test_acc', 0)
            ),
            "test_loss_stage0": stage0_results['test_metrics'].get('test_loss', 'N/A'),
            "test_loss_stage1": stage1_results['test_metrics'].get('test_loss', 'N/A'),
        }
    }
    
    # Log final comparison to WandB
    if wandb_logger is not None:
        wandb_logger.experiment.log({
            "final/test_acc_improvement": final_summary['comparison']['test_acc_improvement'],
            "final/stage0_test_acc": stage0_results['test_metrics'].get('test_acc', 0),
            "final/stage1_test_acc": stage1_results['test_metrics'].get('test_acc', 0),
            "final/stage0_test_loss": stage0_results['test_metrics'].get('test_loss', 0),
            "final/stage1_test_loss": stage1_results['test_metrics'].get('test_loss', 0),
        })
    
    # Save final summary
    final_summary_path = os.path.join(custom_output_dir, "final_summary.json")
    with open(final_summary_path, "w") as f:
        json.dump(final_summary, f, indent=2)
    
    # Print comparison
    print("\n📊 STAGE COMPARISON:")
    print("  Stage 0 (Baseline):")
    print(f"    - Test Accuracy: {stage0_results['test_metrics'].get('test_acc', 'N/A'):.4f}")
    print(f"    - Test Loss:     {stage0_results['test_metrics'].get('test_loss', 'N/A'):.4f}")
    print("  Stage 1 (With Codebook):")
    print(f"    - Test Accuracy: {stage1_results['test_metrics'].get('test_acc', 'N/A'):.4f}")
    print(f"    - Test Loss:     {stage1_results['test_metrics'].get('test_loss', 'N/A'):.4f}")
    print("  Improvement:")
    print(f"    - Accuracy Δ:    {final_summary['comparison']['test_acc_improvement']:+.4f}")
    
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

    # Create multi-stage model
    print("\nCreating Multi-Stage BioBlobs model...")
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
    
    checkpoint_callback_stage0 = create_checkpoint_callback(custom_output_dir, stage_idx=0)
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
    # STAGE 1: Joint Fine-Tuning (With Codebook)
    # ============================================================================
    
    checkpoint_callback_stage1 = create_checkpoint_callback(custom_output_dir, stage_idx=1)
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
    
    save_final_summary(cfg, custom_output_dir, stage0_results, stage1_results, wandb_logger)
    
    # Finish WandB run after both stages complete
    if wandb_logger is not None:
        wandb.finish()


if __name__ == "__main__":
    main()

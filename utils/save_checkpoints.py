"""
Checkpoint saving utilities for multi-stage bioblobs training.

This module provides functions to save specific model components at different
training stages, allowing for fine-grained checkpoint management.
"""

import os
import torch
from datetime import datetime
import pytorch_lightning as pl
from omegaconf import OmegaConf


def evaluate_checkpoints(
    checkpoint_callback, model_class, cfg, num_classes, test_loader, wandb_logger, model
):
    """
    Test both best and last checkpoints on the test set.

    Args:
        checkpoint_callback: PyTorch Lightning checkpoint callback
        model_class: Model class to instantiate (BioBlobsLightning or BioBlobsMultiLabelLightning)
        cfg: Configuration object
        num_classes: Number of output classes
        test_loader: Test data loader
        wandb_logger: Weights & Biases logger
        model: Currently trained model (fallback if checkpoints not found)

    Returns:
        tuple: (results_summary dict, final_model for interpretability)
    """
    best_checkpoint_path = checkpoint_callback.best_model_path
    last_checkpoint_path = checkpoint_callback.last_model_path

    print("\nCheckpoints:")
    print(f"  • Best: {best_checkpoint_path}")
    print(f"  • Last: {last_checkpoint_path}")

    # Initialize results summary
    results_summary = {
        "training_config": OmegaConf.to_container(cfg, resolve=True),
        "model_info": {
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
        },
        "checkpoints": {},
    }

    # Test best checkpoint
    if best_checkpoint_path and os.path.exists(best_checkpoint_path):
        best_results, best_model = test_checkpoint(
            best_checkpoint_path,
            model_class,
            cfg.model,
            cfg.train,
            num_classes,
            test_loader,
            "best",
            wandb_logger,
        )
        results_summary["checkpoints"]["best"] = best_results
        final_model = best_model
    else:
        print("Best checkpoint not found")
        final_model = model

    # Test last checkpoint
    if last_checkpoint_path and os.path.exists(last_checkpoint_path):
        last_results, _ = test_checkpoint(
            last_checkpoint_path,
            model_class,
            cfg.model,
            cfg.train,
            num_classes,
            test_loader,
            "last",
            wandb_logger,
        )
        results_summary["checkpoints"]["last"] = last_results
    else:
        print("Last checkpoint not found")

    return results_summary, final_model


def test_checkpoint(
    checkpoint_path,
    model_class,
    model_cfg,
    train_cfg,
    num_classes,
    test_loader,
    checkpoint_type="best",
    wandb_logger=None,
):
    """Test a specific checkpoint and return results."""
    print(f"\n🧪 Testing {checkpoint_type} checkpoint: {checkpoint_path}")

    # Load model from checkpoint
    model = model_class.load_from_checkpoint(
        checkpoint_path,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        num_classes=num_classes,
    )

    # Create test trainer with the same logger if provided
    test_trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        logger=wandb_logger if wandb_logger is not None else False,
    )

    # Run test
    test_results = test_trainer.test(model, test_loader)

    result = {
        "checkpoint_type": checkpoint_type,
        "checkpoint_path": str(checkpoint_path),
        "test_loss": test_results[0]["test_loss"],
    }

    # Add dataset-specific metrics
    if "test_acc" in test_results[0]:
        # Single-label classification
        result["test_accuracy"] = test_results[0]["test_acc"]
        result["test_ce_loss"] = test_results[0]["test_ce_loss"]
        print(
            f"✓ {checkpoint_type.title()} checkpoint test accuracy: {result['test_accuracy']:.4f}"
        )
    elif "test_fmax" in test_results[0]:
        # Multi-label classification
        result["test_fmax"] = test_results[0]["test_fmax"]
        result["test_precision"] = test_results[0]["test_precision"]
        result["test_recall"] = test_results[0]["test_recall"]
        result["test_bce_loss"] = test_results[0]["test_bce_loss"]
        print(
            f"✓ {checkpoint_type.title()} checkpoint test FMax: {result['test_fmax']:.4f}"
        )
    else:
        print("Unknown test metrics format")

    # Add importance metrics if available
    if "test_importance_max" in test_results[0]:
        result["test_importance_max"] = test_results[0]["test_importance_max"]
        result["test_importance_entropy"] = test_results[0]["test_importance_entropy"]

    return result, model


def create_checkpoint_summary(output_dir: str) -> str:
    """
    Create a summary file of all stage checkpoints.

    Args:
        output_dir: Base output directory containing stage subdirectories

    Returns:
        Path to summary file
    """
    summary_data = {
        "checkpoint_summary": {
            "created_at": datetime.now().isoformat(),
            "base_directory": output_dir,
            "stages": {},
        }
    }

    # Scan for stage checkpoints
    for stage_idx in range(2):  # Only 2 stages now: 0 and 1
        stage_dir = os.path.join(output_dir, f"stage_{stage_idx}")
        stage_files = []

        if os.path.exists(stage_dir):
            for file in os.listdir(stage_dir):
                if file.endswith(".ckpt"):
                    file_path = os.path.join(stage_dir, file)
                    file_size = os.path.getsize(file_path)
                    stage_files.append(
                        {
                            "filename": file,
                            "full_path": file_path,
                            "size_mb": round(file_size / (1024 * 1024), 2),
                        }
                    )

        summary_data["checkpoint_summary"]["stages"][f"stage_{stage_idx}"] = {
            "directory": stage_dir,
            "files": stage_files,
        }

    # Save summary
    summary_path = os.path.join(output_dir, "checkpoint_summary.json")
    import json

    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"📋 Checkpoint summary created: {summary_path}")
    return summary_path


### Functions to be deleted in future versions ###


# def save_stage0_checkpoint(
#     trainer, model, output_dir: str, stage_info: Optional[Dict] = None
# ) -> str:
#     """
#     Save Stage 0 checkpoint using PyTorch Lightning's built-in functionality.

#     Args:
#         trainer: PyTorch Lightning trainer instance
#         model: The bioblobs Lightning model
#         output_dir: Directory to save the checkpoint
#         stage_info: Optional dictionary with stage metadata

#     Returns:
#         Path to saved checkpoint file
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     # Add stage metadata to model for inclusion in checkpoint
#     if stage_info:
#         model.stage_metadata = {
#             "stage": 0,
#             "stage_name": "baseline",
#             "timestamp": datetime.now().isoformat(),
#             **stage_info,
#         }

#     checkpoint_path = os.path.join(output_dir, "stage0_checkpoint.ckpt")
#     trainer.save_checkpoint(checkpoint_path)

#     print(f"✓ Stage 0 checkpoint saved: {checkpoint_path}")
#     print("  Using PyTorch Lightning's built-in checkpointing")

#     return checkpoint_path


# def save_stage1_checkpoint(
#     trainer, model, output_dir: str, stage_info: Optional[Dict] = None
# ) -> str:
#     """
#     Save Stage 1 checkpoint using PyTorch Lightning's built-in functionality.

#     Args:
#         trainer: PyTorch Lightning trainer instance
#         model: The bioblobs Lightning model
#         output_dir: Directory to save the checkpoint
#         stage_info: Optional dictionary with stage metadata

#     Returns:
#         Path to saved checkpoint file
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     # Add stage metadata to model for inclusion in checkpoint
#     if stage_info:
#         model.stage_metadata = {
#             "stage": 1,
#             "stage_name": "joint_finetuning",
#             "timestamp": datetime.now().isoformat(),
#             **stage_info,
#         }

#     checkpoint_path = os.path.join(output_dir, "stage1_checkpoint.ckpt")
#     trainer.save_checkpoint(checkpoint_path)

#     print(f"✓ Stage 1 checkpoint saved: {checkpoint_path}")
#     print("  Using PyTorch Lightning's built-in checkpointing")

#     return checkpoint_path


# def save_stage_specific_checkpoint(
#     trainer, model, stage_idx: int, output_dir: str, stage_info: Optional[Dict] = None
# ) -> str:
#     """
#     Save checkpoint for specific stage using PyTorch Lightning's built-in functionality.

#     Args:
#         trainer: PyTorch Lightning trainer instance
#         model: The bioblobs Lightning model
#         stage_idx: Stage index (0 or 1)
#         output_dir: Directory to save the checkpoint
#         stage_info: Optional dictionary with stage metadata

#     Returns:
#         Path to saved checkpoint file
#     """
#     if stage_idx == 0:
#         return save_stage0_checkpoint(trainer, model, output_dir, stage_info)
#     elif stage_idx == 1:
#         return save_stage1_checkpoint(trainer, model, output_dir, stage_info)
#     else:
#         raise ValueError(f"Invalid stage index: {stage_idx}. Must be 0 or 1.")


# def load_stage_checkpoint(checkpoint_path: str):
#     """
#     Load a PyTorch Lightning checkpoint.

#     Note: With PyTorch Lightning checkpoints, you typically use:
#     - trainer.fit(model, ckpt_path=checkpoint_path) to resume training
#     - Model.load_from_checkpoint(checkpoint_path) to load a trained model

#     This function is kept for backward compatibility and inspection purposes.

#     Args:
#         checkpoint_path: Path to the checkpoint file

#     Returns:
#         Dictionary containing checkpoint data
#     """
#     if not os.path.exists(checkpoint_path):
#         raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

#     print(f"📂 Loading PyTorch Lightning checkpoint: {checkpoint_path}")
#     checkpoint = torch.load(checkpoint_path, map_location="cpu")

#     # Print basic checkpoint info
#     if "epoch" in checkpoint:
#         print(f"   Epoch: {checkpoint['epoch']}")
#     if "global_step" in checkpoint:
#         print(f"   Global Step: {checkpoint['global_step']}")
#     if "stage_metadata" in checkpoint.get("state_dict", {}):
#         stage_info = checkpoint["state_dict"]["stage_metadata"]
#         print(f"   Stage: {stage_info.get('stage', 'unknown')}")
#         print(f"   Stage Name: {stage_info.get('stage_name', 'unknown')}")

#     print("✓ Checkpoint loaded successfully")
#     print("💡 To use this checkpoint:")
#     print("   - Resume training: trainer.fit(model, ckpt_path=checkpoint_path)")
#     print("   - Load trained model: Model.load_from_checkpoint(checkpoint_path)")

#     return checkpoint

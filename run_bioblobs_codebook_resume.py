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
    dataset_inter_results,
)
from train_lightling import (
    create_bioblobs_resume_model_from_checkpoint,
    initialize_codebook_from_dataloader,
)
from bioblob_resume_lightning import BioBlobsTrainingCodebookModule
import json


def test_checkpoint(
    checkpoint_path,
    model_class,
    model_cfg,
    train_cfg,
    multistage_cfg,
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
        multistage_cfg=multistage_cfg,
        num_classes=num_classes,
    )

    # Create test trainer
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
        "test_vq_loss": test_results[0]["test_vq_loss"],
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
        print("⚠️ Unknown test metrics format")

    # Add importance metrics if available
    if "test_importance_max" in test_results[0]:
        result["test_importance_max"] = test_results[0]["test_importance_max"]
        result["test_importance_entropy"] = test_results[0]["test_importance_entropy"]

    return result, model


def compare_with_bioblobs_baseline(
    bioblobs_checkpoint_path, bioblobs_results, output_dir
):
    """Compare BioBlobs with codebook results with original BioBlobs baseline."""
    comparison = {
        "bioblobs_checkpoint": str(bioblobs_checkpoint_path),
        "bioblobs_results": bioblobs_results,
        "comparison_summary": {
            "methodology": "Resumed training from BioBlobs checkpoint with codebook",
            "stages": ["codebook_initialization", "joint_finetuning"],
        },
    }   

    # Save comparison
    comparison_path = os.path.join(output_dir, "bioblobs_codebook_comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"✓ Comparison saved to: {comparison_path}")
    return comparison


@hydra.main(
    version_base="1.1", config_path="conf", config_name="config_bioblobs_resume"
)
def main(cfg: DictConfig):
    print("RUNNING BioBlobs RESUME TRAINING WITH VQ CODEBOOK")
    print("=" * 70)

    # Validate required checkpoint path
    if cfg.resume.model_checkpoint_path is None:
        raise ValueError(
            "Must provide model_checkpoint_path via command line: resume.model_checkpoint_path=path/to/checkpoint.ckpt"
        )

    if not os.path.exists(cfg.resume.model_checkpoint_path):
        raise FileNotFoundError(
            f"BioBlobs model checkpoint not found: {cfg.resume.model_checkpoint_path}"
        )

    print(f"📁 BioBlobs model checkpoint: {cfg.resume.model_checkpoint_path}")
    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.train.seed)

    # Use Hydra's output directory
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
        test_mode=cfg.data.get("test_mode", False),
    )

    train_loader = create_dataloader(
        train_dataset, cfg.train.batch_size, cfg.train.num_workers, shuffle=True
    )
    val_loader = create_dataloader(
        val_dataset, cfg.train.batch_size, cfg.train.num_workers, shuffle=False
    )
    test_loader = create_dataloader(
        test_dataset, cfg.train.batch_size, cfg.train.num_workers, shuffle=False
    )

    # Create BioBlobs model from BioBlobs checkpoint - choose appropriate version based on dataset
    print("\n🔄 RESUMING TRAINING FROM BIOBLOBS CHECKPOINT")
    print("=" * 70)

    if cfg.data.dataset_name == "go":
        print(
            "🧬 Using multi-label BioBlobs resume model for Gene Ontology classification"
        )
        from train_lightling import (
            create_bioblobs_resume_multilabel_model_from_checkpoint,
        )

        model = create_bioblobs_resume_multilabel_model_from_checkpoint(
            model_checkpoint_path=cfg.resume.model_checkpoint_path,
            model_cfg=cfg.model,
            train_cfg=cfg.train,
            multistage_cfg=cfg.multistage,
            num_classes=num_classes,
            load_model_config_from_checkpoint=cfg.resume.get(
                "load_model_config_from_checkpoint", True
            ),
        )
        model_type = "BioBlobs Resume Multi-Label"
    else:
        print("🧬 Using single-label BioBlobs resume model")
        model = create_bioblobs_resume_model_from_checkpoint(
            model_checkpoint_path=cfg.resume.model_checkpoint_path,
            model_cfg=cfg.model,
            train_cfg=cfg.train,
            multistage_cfg=cfg.multistage,
            num_classes=num_classes,
            load_model_config_from_checkpoint=cfg.resume.get(
                "load_model_config_from_checkpoint", True
            ),
        )
        model_type = "BioBlobs Resume"

    print("\n🏗️  Model Architecture:")
    print(f"  • Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"  • Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    print(f"  • Mode: {model_type} with VQ codebook")

    # Ensure model is in training mode after loading from checkpoint
    model.train()
    print("  • Model set to training mode")

    # Force ALL modules to training mode to avoid PyTorch Lightning warning
    # This is safe for resume training as we want all parameters to be trainable
    modules_set_to_train = 0
    for name, module in model.named_modules():
        if hasattr(module, "training") and not module.training:
            module.train()
            modules_set_to_train += 1

    if modules_set_to_train > 0:
        print(f"  • Forced {modules_set_to_train} modules from eval to training mode")

    # Final check for any modules still in eval mode
    eval_modules = []
    for name, module in model.named_modules():
        if hasattr(module, "training") and not module.training:
            eval_modules.append(name)

    if eval_modules:
        print(f"  ⚠️  Warning: {len(eval_modules)} modules still in eval mode:")
        for mod in eval_modules[:3]:  # Show first 3
            print(f"     - {mod}")
        if len(eval_modules) > 3:
            print(f"     ... and {len(eval_modules) - 3} more")
    else:
        print("  • All modules confirmed in training mode")

    # Run initial interpretability analysis before training
    print("\n🔍 INITIAL INTERPRETABILITY ANALYSIS (BEFORE TRAINING)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    initial_interp_results = dataset_inter_results(
        model=model,
        dataloader=test_loader,
        device=device,
        max_batches=cfg.evaluation.get("max_batches_interp", 20),
    )

    # Save initial interpretability results in interpretability subdirectory
    interp_dir = os.path.join(custom_output_dir, "interpretability")
    os.makedirs(interp_dir, exist_ok=True)
    initial_interp_path = os.path.join(interp_dir, "initial.json")
    save_interpretability_results(initial_interp_results, initial_interp_path)

    print(f"✓ Initial interpretability results saved to: {initial_interp_path}")
    print_interpretability_summary(initial_interp_results)

    # Initialize codebook using K-means

    print("\n🎲 CODEBOOK INITIALIZATION")
    print("=" * 70)

    init_stats = initialize_codebook_from_dataloader(
        model=model,
        train_loader=train_loader,
        device=device,
        max_batches=cfg.resume.kmeans_max_batches,
    )

    # Save initialization stats
    init_dir = os.path.join(custom_output_dir, "codebook_initialization")
    os.makedirs(init_dir, exist_ok=True)
    with open(os.path.join(init_dir, "initialization_stats.json"), "w") as f:
        json.dump(init_stats, f, indent=2)

    # Logger
    wandb_logger = None
    if cfg.train.use_wandb:
        timestamp = datetime.now().strftime("%Y-%m-%d")
        wandb_logger = WandbLogger(
            project=cfg.train.wandb_project,
            name=f"resume_{cfg.data.dataset_name}_{cfg.data.split}_{timestamp}",
            tags=["bioblobs-resume", cfg.data.dataset_name, cfg.data.split],
        )

    # Checkpoint callback - use appropriate metric based on dataset type
    if cfg.data.dataset_name == "go":
        filename_template = "best-bioblobs-{epoch:02d}-{val_fmax:.3f}"
        monitor_metric = "val_fmax"
    else:
        filename_template = "best-bioblobs-{epoch:02d}-{val_acc:.3f}"
        monitor_metric = "val_acc"

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        mode="max",
        save_top_k=1,
        dirpath=custom_output_dir,
        filename=filename_template,
        save_last=True,
    )

    # Trainer
    max_epochs = cfg.multistage.stage0.epochs
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=wandb_logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        log_every_n_steps=10,
        default_root_dir=custom_output_dir,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
        enable_progress_bar=True,
    )

    # Joint training (no multi-stage needed for EMA-based VQ)
    print("\n🎯 JOINT TRAINING (BACKBONE + CODEBOOK)")
    print("=" * 70)

    print(f"� Starting joint training ({max_epochs} epochs)")
    trainer.fit(model, train_loader, val_loader)  # Get checkpoint paths
    best_checkpoint_path = checkpoint_callback.best_model_path
    last_checkpoint_path = checkpoint_callback.last_model_path

    print("\n📁 Final Checkpoints:")
    print(f"  • Best: {best_checkpoint_path}")
    print(f"  • Last: {last_checkpoint_path}")

    # Test both checkpoints
    results_summary = {
        "training_config": OmegaConf.to_container(cfg, resolve=True),
        "model_info": {
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "mode": model_type,
            "source_checkpoint": str(cfg.resume.model_checkpoint_path),
        },
        "initialization_stats": init_stats,
        "initial_interpretability": {
            "enabled": True,
            "results_path": initial_interp_path,
            "summary": initial_interp_results["aggregated_stats"],
        },
        "checkpoints": {},
    }

    # Test best checkpoint - use the correct model class based on dataset
    if cfg.data.dataset_name == "go":
        from bioblob_resume_lightning import BioBlobsTrainingCodebookMultiLabelModule

        test_model_class = BioBlobsTrainingCodebookMultiLabelModule
    else:
        test_model_class = BioBlobsTrainingCodebookModule

    if best_checkpoint_path and os.path.exists(best_checkpoint_path):
        best_results, best_model = test_checkpoint(
            best_checkpoint_path,
            test_model_class,
            cfg.model,
            cfg.train,
            cfg.multistage,
            num_classes,
            test_loader,
            "best",
            wandb_logger,
        )
        results_summary["checkpoints"]["best"] = best_results
        final_model = best_model
    else:
        print("⚠️  Best checkpoint not found")
        final_model = model

    # Test last checkpoint
    if last_checkpoint_path and os.path.exists(last_checkpoint_path):
        last_results, _ = test_checkpoint(
            last_checkpoint_path,
            test_model_class,
            cfg.model,
            cfg.train,
            cfg.multistage,
            num_classes,
            test_loader,
            "last",
            wandb_logger,
        )
        results_summary["checkpoints"]["last"] = last_results

    # Save results summary
    results_summary_path = os.path.join(custom_output_dir, "results_summary.json")
    with open(results_summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    # Run interpretability analysis
    if cfg.evaluation.run_interpretability:
        print("\n🔍 INTERPRETABILITY ANALYSIS")
        print("=" * 70)

        interp_output_dir = os.path.join(custom_output_dir, "interpretability")
        os.makedirs(interp_output_dir, exist_ok=True)

        print("📊 Running interpretability analysis on test set...")

        try:
            interp_results = final_model.get_inter_info(
                test_loader,
                device=device,
                max_batches=cfg.evaluation.max_batches_interp,
            )
        except Exception as e:
            print(f"❌ Interpretability analysis failed with error: {e}")
            import traceback

            traceback.print_exc()
            interp_results = None

        if interp_results is not None:
            # Save results
            results_path = os.path.join(interp_output_dir, "bioblobs_final.json")
            save_interpretability_results(interp_results, results_path)
            print_interpretability_summary(interp_results)

            results_summary["interpretability"] = {
                "enabled": True,
                "results_path": results_path,
                "summary": interp_results["aggregated_stats"],
            }
        else:
            print("❌ Interpretability analysis returned None")
            print("   Possible causes:")
            print("   • Model still in bypass_codebook mode")
            print("   • dataset_inter_results function failed")
            print("   • Device/memory issues")
            results_summary["interpretability"] = {
                "enabled": False,
                "error": "Analysis returned None",
            }

    # # Compare with BioBlobs baseline
    # if cfg.evaluation.compare_with_baseline:
    #     comparison = compare_with_bioblobs_baseline(
    #         cfg.resume.model_checkpoint_path, results_summary, custom_output_dir
    #     )

    # Update results summary
    with open(results_summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    # Close wandb if used
    if wandb_logger is not None:
        wandb.finish()

    print("\n🎉 BIOBLOBS RESUME TRAINING COMPLETED!")
    print("=" * 70)
    print(f"📊 Results summary: {results_summary_path}")
    print(f"🔍 Initial interpretability: {initial_interp_path}")

    # Print final results
    if "best" in results_summary["checkpoints"]:
        best_checkpoint = results_summary["checkpoints"]["best"]
        if "test_accuracy" in best_checkpoint:
            print(
                f"🏆 Best checkpoint test accuracy: {best_checkpoint['test_accuracy']:.4f}"
            )
        elif "test_fmax" in best_checkpoint:
            print(f"🏆 Best checkpoint test FMax: {best_checkpoint['test_fmax']:.4f}")

    if results_summary["interpretability"]["enabled"]:
        print("🔍 Final interpretability analysis completed")
        if "summary" in results_summary["interpretability"]:
            overall_acc = results_summary["interpretability"]["summary"]["accuracy"]
            avg_concentration = results_summary["interpretability"]["summary"][
                "avg_importance_concentration"
            ]
            print(f"📊 Final test accuracy: {overall_acc:.4f}")
            print(f"🎯 Final average attention concentration: {avg_concentration:.3f}")

    # Print initial interpretability results
    if results_summary["initial_interpretability"]["enabled"]:
        print("🔍 Initial interpretability analysis completed")
        if "summary" in results_summary["initial_interpretability"]:
            initial_acc = results_summary["initial_interpretability"]["summary"][
                "accuracy"
            ]
            initial_concentration = results_summary["initial_interpretability"][
                "summary"
            ]["avg_importance_concentration"]
            print(f"📊 Initial test accuracy: {initial_acc:.4f}")
            print(
                f"🎯 Initial average attention concentration: {initial_concentration:.3f}"
            )

    print("=" * 70)


if __name__ == "__main__":
    main()

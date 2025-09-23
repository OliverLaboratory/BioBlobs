import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from utils.gnn_dataset import get_gnn_task, get_transformed_graph_dataset, get_data_loaders
from utils.utils import set_seed
from train_partgnn_lightning import PartGNNLightning, PartGNNMultiLabelLightning
import wandb
import torch
from datetime import datetime
import json


def create_model(cfg: DictConfig, task, num_classes: int):
    """Create the appropriate PartGNN model based on task type."""
    
    print("🧬 Task Info:")
    print(f"  Dataset: {cfg.data.dataset_name}")
    print(f"  Task type: {task.task_type}")
    print(f"  Number of classes: {num_classes}")
    
    # Determine model type based on dataset and task
    if cfg.data.dataset_name.lower() == "gene_ontology":
        print("🔬 Using PartGNNMultiLabelLightning for multi-label Gene Ontology classification")
        model_class = PartGNNMultiLabelLightning
        model_type = "PartGNN Multi-Label"
    else:
        print("🔬 Using PartGNNLightning for classification")
        model_class = PartGNNLightning
        if num_classes == 2:
            model_type = "PartGNN Binary"
        else:
            model_type = "PartGNN Multi-Class"
    
    print(f"  Model type: {model_type}")
    
    # Create model
    model = model_class(
        model_cfg=cfg.model,
        train_cfg=cfg.train,
        num_classes=num_classes
    )
    
    return model, model_type


def setup_callbacks(cfg: DictConfig, custom_output_dir: str, dataset_name: str):
    """Setup PyTorch Lightning callbacks."""
    callbacks = []
    
    # Learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    
    # Checkpoint callback - use appropriate metric based on task type
    if dataset_name.lower() == "gene_ontology":
        filename_template = "best-partgnn-{epoch:02d}-{val_fmax:.3f}"
        monitor_metric = "val_fmax"
    else:
        filename_template = "best-partgnn-{epoch:02d}-{val_acc:.3f}"
        monitor_metric = "val_acc"
    
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        mode="max",
        save_top_k=1,
        dirpath=custom_output_dir,
        filename=filename_template,
        save_last=True,
    )
    callbacks.append(checkpoint_callback)
    
    return callbacks, checkpoint_callback


def setup_logger(cfg: DictConfig, custom_output_dir: str, model_type: str):
    """Setup logging for the experiment."""
    loggers = []
    
    # CSV Logger
    csv_logger = pl.loggers.CSVLogger(
        save_dir=custom_output_dir,
        name='csv_logs'
    )
    loggers.append(csv_logger)
    
    # Weights & Biases logger
    if cfg.get('use_wandb', False) and not cfg.train.debug:
        timestamp = datetime.now().strftime("%Y-%m-%d")
        wandb_logger = WandbLogger(
            project=cfg.get('wandb_project', 'partgnn-protein'),
            name=f"{cfg.data.dataset_name}_{cfg.data.split}_seq_False_{cfg.model.max_clusters}_{cfg.model.get('cluster_size_max', 'NA')}_{cfg.train.learning_rate}_{timestamp}",
            save_dir=custom_output_dir,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=[
                "partgnn",
                cfg.data.dataset_name,
                cfg.data.split,
                model_type
            ]
        )
        loggers.append(wandb_logger)
    
    return loggers


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
        print(f"✓ {checkpoint_type.title()} checkpoint test accuracy: {result['test_accuracy']:.4f}")
    elif "test_fmax" in test_results[0]:
        # Multi-label classification
        result["test_fmax"] = test_results[0]["test_fmax"]
        result["test_precision"] = test_results[0]["test_precision"]
        result["test_recall"] = test_results[0]["test_recall"]
        result["test_bce_loss"] = test_results[0]["test_bce_loss"]
        print(f"✓ {checkpoint_type.title()} checkpoint test FMax: {result['test_fmax']:.4f}")
    else:
        print("⚠️ Unknown test metrics format")

    # Add importance metrics if available
    if "test_importance_max" in test_results[0]:
        result["test_importance_max"] = test_results[0]["test_importance_max"]
        result["test_importance_entropy"] = test_results[0]["test_importance_entropy"]

    return result, model


@hydra.main(version_base="1.1", config_path="conf", config_name="config_partgnn")
def main(cfg: DictConfig):
    print("🧬 PartGNN Training (GNN + Partitioner + Global-Cluster Attention)")
    print("=" * 70)
    
    # Set random seed
    set_seed(cfg.train.seed)

    # Use Hydra's output directory to ensure consistency
    from hydra.core.hydra_config import HydraConfig
    hydra_cfg = HydraConfig.get()
    custom_output_dir = hydra_cfg.runtime.output_dir
    print(f"📁 Output directory: {custom_output_dir}")

    # Load dataset and create data loaders
    print("\n📊 Loading Dataset...")
    task = get_gnn_task(cfg.data.dataset_name, root=cfg.data.data_dir)
    dataset = task.dataset

    # Handle regression tasks with target normalization
    y_transform = None
    if task.task_type[1] == 'regression':
        from sklearn.preprocessing import StandardScaler
        print("🔄 Setting up target normalization for regression...")
        task.compute_targets()
        all_y = task.train_targets
        y_transform = StandardScaler().fit(all_y.reshape(-1, 1))

    # Transform dataset
    dataset = get_transformed_graph_dataset(cfg, dataset, task, y_transform)
    
    # Create data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset, task,
        cfg.train.batch_size, cfg.train.num_workers
    )
    
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    print(f"  Training samples: {len(train_loader.dataset) if hasattr(train_loader.dataset, '__len__') else 'N/A'}")
    print(f"  Validation samples: {len(val_loader.dataset) if hasattr(val_loader.dataset, '__len__') else 'N/A'}")
    print(f"  Test samples: {len(test_loader.dataset) if hasattr(test_loader.dataset, '__len__') else 'N/A'}")
    
    # Examine data sample
    sample_batch = next(iter(train_loader))
    print("\n🔍 Sample Data Info:")
    print(f"  Node features shape: {sample_batch.x.shape}")
    print(f"  Edge index shape: {sample_batch.edge_index.shape}")
    print(f"  Edge attr shape: {sample_batch.edge_attr.shape if sample_batch.edge_attr is not None else 'None'}")
    print(f"  Batch shape: {sample_batch.batch.shape}")
    print(f"  Labels shape: {sample_batch.y.shape}")
    
    # Determine number of classes
    if hasattr(task, 'num_classes'):
        num_classes = task.num_classes
    else:
        # Infer from data
        if cfg.data.dataset_name.lower() == "gene_ontology":
            # Multi-label: classes are in the last dimension
            num_classes = sample_batch.y.shape[-1] if sample_batch.y.dim() > 1 else len(sample_batch.y.unique())
        else:
            # Multi-class or binary: count unique labels
            num_classes = len(sample_batch.y.unique())
    
    print(f"  Detected {num_classes} classes")

    # Create model
    print("\n🏗️  Creating Model...")
    model, model_type = create_model(cfg, task, num_classes)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Setup callbacks and logging
    print("\n⚙️  Setting up Training...")
    callbacks, checkpoint_callback = setup_callbacks(cfg, custom_output_dir, cfg.data.dataset_name)
    loggers = setup_logger(cfg, custom_output_dir, model_type)
    
    # Configure trainer
    accelerator = 'cpu' if cfg.train.debug else 'auto'
    limit_train_batches = 5 if cfg.train.debug else None
    limit_val_batches = 5 if cfg.train.debug else None
    
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=accelerator,
        devices='auto',
        logger=loggers,
        callbacks=callbacks,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        gradient_clip_val=cfg.train.get('gradient_clip_val', 1.0),
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=cfg.train.get('log_every_n_steps', 50),
        check_val_every_n_epoch=cfg.train.get('check_val_every_n_epoch', 1)
    )
    
    # Set steps per epoch for scheduler
    model.steps_per_epoch = len(train_loader)
    
    # Training
    print("\n🚀 Starting Training...")
    print(f"  Max epochs: {cfg.train.max_epochs}")
    print(f"  Batch size: {cfg.train.batch_size}")
    print(f"  Learning rate: {cfg.train.learning_rate}")
    print(f"  Accelerator: {accelerator}")
    
    try:
        trainer.fit(model, train_loader, val_loader)
        
        # Get checkpoint paths
        best_checkpoint_path = checkpoint_callback.best_model_path
        last_checkpoint_path = checkpoint_callback.last_model_path

        print("\n📁 Checkpoints:")
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
            },
            "checkpoints": {},
        }

        # Get wandb logger for testing
        wandb_logger = None
        for logger in loggers:
            if isinstance(logger, WandbLogger):
                wandb_logger = logger
                break

        # Determine model class for checkpoint loading
        if cfg.data.dataset_name.lower() == "gene_ontology":
            model_class = PartGNNMultiLabelLightning
        else:
            model_class = PartGNNLightning

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
            final_model = best_model  # Use best model for interpretability
        else:
            print("⚠️  Best checkpoint not found")
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
            print("⚠️  Last checkpoint not found")

        # Save results summary
        results_summary_path = os.path.join(custom_output_dir, "results_summary.json")
        with open(results_summary_path, "w") as f:
            json.dump(results_summary, f, indent=2)
        print(f"\n✓ Results summary saved to: {results_summary_path}")

        # Run interpretability analysis
        if cfg.get('interpretability', {}).get("enabled", True):
            print("\n🔍 INTERPRETABILITY ANALYSIS")
            print("=" * 70)

            # Create interpretability output directory
            interp_output_dir = os.path.join(custom_output_dir, "interpretability")
            os.makedirs(interp_output_dir, exist_ok=True)

            # Run analysis on test set
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            print("📊 Running interpretability analysis on test set...")
            try:
                interp_results = final_model.get_inter_info(
                    test_loader,
                    device=device,
                    max_batches=cfg.get('interpretability', {}).get("max_batches", None),
                )

                if interp_results is not None:
                    # Save results
                    from utils.interpretability import save_interpretability_results, print_interpretability_summary
                    results_path = os.path.join(interp_output_dir, "test_interpretability.json")
                    save_interpretability_results(interp_results, results_path)

                    # Print summary
                    print_interpretability_summary(interp_results)

                    # Add interpretability summary to results
                    results_summary["interpretability"] = {
                        "enabled": True,
                        "results_path": results_path,
                        "summary": interp_results["aggregated_stats"],
                    }
                else:
                    print("⚠️  Interpretability analysis failed")
                    results_summary["interpretability"] = {
                        "enabled": False,
                        "error": "Analysis failed",
                    }
            except Exception as e:
                print(f"⚠️  Interpretability analysis failed: {e}")
                results_summary["interpretability"] = {
                    "enabled": False,
                    "error": str(e),
                }
        else:
            print("⚠️  Interpretability analysis disabled")
            results_summary["interpretability"] = {"enabled": False}

        # Update results summary with interpretability info
        with open(results_summary_path, "w") as f:
            json.dump(results_summary, f, indent=2)

        # Save final model
        final_model_path = os.path.join(custom_output_dir, "final_partgnn_model.ckpt")
        trainer.save_checkpoint(final_model_path)

        print("\n🎉 PARTGNN TRAINING COMPLETED!")
        print("=" * 70)
        print(f"📁 Final model saved to: {final_model_path}")
        print(f"📊 Results summary: {results_summary_path}")

        # Print final results summary
        if "best" in results_summary["checkpoints"]:
            best_checkpoint = results_summary['checkpoints']['best']
            if "test_accuracy" in best_checkpoint:
                print(f"🏆 Best checkpoint test accuracy: {best_checkpoint['test_accuracy']:.4f}")
            elif "test_fmax" in best_checkpoint:
                print(f"🏆 Best checkpoint test FMax: {best_checkpoint['test_fmax']:.4f}")
                print(f"   Best precision: {best_checkpoint['test_precision']:.4f}")
                print(f"   Best recall: {best_checkpoint['test_recall']:.4f}")
        
        if "last" in results_summary["checkpoints"]:
            last_checkpoint = results_summary['checkpoints']['last']
            if "test_accuracy" in last_checkpoint:
                print(f"📈 Last checkpoint test accuracy: {last_checkpoint['test_accuracy']:.4f}")
            elif "test_fmax" in last_checkpoint:
                print(f"📈 Last checkpoint test FMax: {last_checkpoint['test_fmax']:.4f}")

        if results_summary["interpretability"]["enabled"]:
            print("🔍 Interpretability analysis completed")
            if "summary" in results_summary["interpretability"]:
                summary_stats = results_summary['interpretability']['summary']
                if 'accuracy' in summary_stats:
                    print(f"📊 Overall test accuracy: {summary_stats['accuracy']:.4f}")
                if 'avg_importance_concentration' in summary_stats:
                    print(f"🎯 Average attention concentration: {summary_stats['avg_importance_concentration']:.3f}")

        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # Cleanup
        if cfg.get('use_wandb', False) and not cfg.train.debug:
            wandb.finish()


if __name__ == "__main__":
    main()
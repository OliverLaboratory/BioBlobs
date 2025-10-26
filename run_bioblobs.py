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
from utils.interpretability import run_interpretability_analysis
from utils.save_checkpoints import create_checkpoint_summary, evaluate_checkpoints
from utils.verbose import print_final_results
from train_lightling import BioBlobsLightning, BioBlobsMultiLabelLightning
import json
from hydra.utils import to_absolute_path


@hydra.main(version_base="1.1", config_path="conf", config_name="config_bioblobs")
def main(cfg: DictConfig):
    print("BioBlobs Training (GVP + Partitioner + Global-Cluster Attention)")
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

    train_loader = create_dataloader(
        train_dataset, cfg.train.batch_size, cfg.train.num_workers, shuffle=True
    )
    val_loader = create_dataloader(
        val_dataset, cfg.train.batch_size, cfg.train.num_workers, shuffle=False
    )
    test_loader = create_dataloader(
        test_dataset, cfg.train.batch_size, cfg.train.num_workers, shuffle=False
    )

    if cfg.data.dataset_name == "go":
        print("Using BioBlobsMultiLabelLightning for multi-label GO classification")
        model_class = BioBlobsMultiLabelLightning
        model = model_class(cfg.model, cfg.train, num_classes)
    else:
        print("Using BioBlobsLightning for single-label classification")
        model_class = BioBlobsLightning
        model = model_class(cfg.model, cfg.train, num_classes)

    print("\nModel Architecture:")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # Wandb Logger
    wandb_logger = None
    if cfg.train.use_wandb:
        timestamp = datetime.now().strftime("%Y-%m-%d")
        wandb_logger = WandbLogger(
            project=cfg.train.wandb_project,
            name=f"{cfg.data.dataset_name}_{cfg.data.split}_seq_{cfg.model.seq_in}_{cfg.model.max_clusters}_{cfg.model.cluster_size_max}_{cfg.train.lr}_{timestamp}",
            tags=["bioblobs-baseline", cfg.data.dataset_name, cfg.data.split],
        )

    # Checkpoint callback - use appropriate metric based on task type
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
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        logger=wandb_logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        log_every_n_steps=20,
        default_root_dir=custom_output_dir,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
        enable_progress_bar=True,
    )

    # Training
    print(f"\n📚 Training BioBlobs for {cfg.train.epochs} epochs...")
    trainer.fit(model, train_loader, val_loader)

    # Evaluate checkpoints on test set
    results_summary, final_model = evaluate_checkpoints(
        checkpoint_callback, model_class, cfg, num_classes, test_loader, wandb_logger, model
    )

    # Save initial results summary
    results_summary_path = os.path.join(custom_output_dir, "results_summary.json")
    with open(results_summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n✓ Results summary saved to: {results_summary_path}")

    # Run interpretability analysis
    interpretability_summary = run_interpretability_analysis(
        final_model, test_loader, cfg, custom_output_dir
    )
    results_summary["interpretability"] = interpretability_summary

    # Update results summary with interpretability info
    with open(results_summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    # Save final model
    final_model_path = os.path.join(custom_output_dir, "final_bioblobs_model.ckpt")
    trainer.save_checkpoint(final_model_path)

    # Create checkpoint summary
    summary_path = create_checkpoint_summary(custom_output_dir)

    # Close wandb if used
    if wandb_logger is not None:
        wandb.finish()

    # Print final results
    print_final_results(results_summary)
    print(f"Final model saved to: {final_model_path}")
    print(f"Checkpoint summary: {summary_path}")
    print(f"Results summary: {results_summary_path}")


if __name__ == "__main__":
    main()

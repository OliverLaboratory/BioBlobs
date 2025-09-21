import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from utils.gnn_dataset import get_gnn_task, get_transformed_graph_dataset, get_data_loaders
from utils.utils import set_seed
import torch
import wandb
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint
import json


@hydra.main(version_base="1.1", config_path="conf", config_name="config_partgnn")
def main(cfg: DictConfig):
    print("🧬 PartGNN Training (GNN + Partitioner + Global-Cluster Attention)")
    print("=" * 70)
    # print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.train.seed)

    # Use Hydra's output directory to ensure consistency
    from hydra.core.hydra_config import HydraConfig

    hydra_cfg = HydraConfig.get()
    custom_output_dir = hydra_cfg.runtime.output_dir

    print(f"Output directory: {custom_output_dir}")

    task = get_gnn_task(cfg.data.dataset_name, root=cfg.data.data_dir)
    dataset = task.dataset

    y_transform = None

    if task.task_type[1] == 'regression':
        from sklearn.preprocessing import StandardScaler
        task.compute_targets()
        all_y = task.train_targets
        y_transform = StandardScaler().fit(all_y.reshape(-1, 1))

    dataset = get_transformed_graph_dataset(cfg, dataset, task, y_transform)

    print(dataset[0].x)

    train_loader, val_loader, test_loader = get_data_loaders(
        dataset, task,
        cfg.train.batch_size, cfg.train.num_workers
    )


    limit_train_batches = 5 if cfg.train.debug else None
    limit_val_batches = 5 if cfg.train.debug else None
    logger = pl.loggers.CSVLogger(cfg.paths.output_dir, name='csv_logs')

    callbacks = [
        pl.callbacks.LearningRateMonitor(),
    ]

    accelerator = 'cpu' if cfg.train.debug else 'auto'
    trainer = pl.Trainer(
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        max_epochs=cfg.train.epochs,
        devices='auto',
        accelerator=accelerator,
        enable_checkpointing=False,
        logger=[logger],
        callbacks=callbacks
    )


if __name__ == "__main__":
    main()
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from utils.gnn_dataset import get_gnn_task, get_transformed_graph_dataset, get_data_loaders
from utils.utils import set_seed
from train_partgnn_lightning import PartGNNLightning, PartGNNMultiLabelLightning
import wandb
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


def setup_callbacks(cfg: DictConfig, custom_output_dir: str):
    """Setup PyTorch Lightning callbacks."""
    callbacks = []
    
    # Learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(custom_output_dir, 'checkpoints'),
        filename='best-{epoch:02d}-{val_loss:.3f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    if cfg.train.get('early_stopping', True):
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.001,
            patience=cfg.train.get('patience', 20),
            verbose=True,
            mode='min'
        )
        callbacks.append(early_stop_callback)
    
    return callbacks


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
        wandb_logger = WandbLogger(
            project=cfg.get('wandb_project', 'partgnn-protein'),
            name=f"{model_type}_{cfg.data.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            save_dir=custom_output_dir,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=[
                model_type,
                cfg.data.dataset_name,
                f"clusters_{cfg.model.max_clusters}",
                f"layers_{cfg.model.num_layers}"
            ]
        )
        loggers.append(wandb_logger)
    
    return loggers


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
    callbacks = setup_callbacks(cfg, custom_output_dir)
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
        
        # Testing
        print("\n🧪 Starting Testing...")
        test_results = trainer.test(model, test_loader, ckpt_path='best')
        
        # Save results
        results_file = os.path.join(custom_output_dir, 'test_results.json')
        with open(results_file, 'w') as f:
            json.dump(test_results[0], f, indent=2)
        
        print("\n✅ Training completed successfully!")
        print(f"📊 Results saved to: {results_file}")
        
        # Print final metrics
        if test_results:
            print("\n🎯 Final Test Results:")
            for key, value in test_results[0].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        
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
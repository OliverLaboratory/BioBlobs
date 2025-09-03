import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Only if you want to use specific GPU

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm
import json
import argparse
from utils.proteinshake_dataset import get_dataset, create_dataloader
from diffpool_part import GVPDiffPoolGraphSAGEModel  # Import the new model
from utils.utils import set_seed
import torch
import torch.nn as nn
import wandb
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint

# Set torch matmul precision to suppress warnings
torch.set_float32_matmul_precision('medium')

class GVPDiffPool(pl.LightningModule):
    def __init__(self, model_cfg, train_cfg, num_classes):
        super().__init__()
        self.save_hyperparameters()
        self.model = GVPDiffPoolGraphSAGEModel(
            node_in_dim=model_cfg.node_in_dim,
            node_h_dim=model_cfg.node_h_dim,
            edge_in_dim=model_cfg.edge_in_dim,
            edge_h_dim=model_cfg.edge_h_dim,
            num_classes=num_classes,
            seq_in=model_cfg.seq_in,
            num_layers=model_cfg.num_layers,
            drop_rate=model_cfg.drop_rate,
            pooling=model_cfg.pooling,
            max_clusters=model_cfg.max_clusters,
            entropy_weight=model_cfg.entropy_weight,
            link_pred_weight=model_cfg.link_pred_weight
        )
        self.lr = train_cfg.lr
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, h_V, edge_index, h_E, seq=None, batch=None):
        return self.model(h_V, edge_index, h_E, seq, batch)

    def training_step(self, batch, batch_idx):
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        seq = batch.seq if hasattr(batch, 'seq') and hasattr(self.model, 'W_s') else None
        logits, aux_losses = self.model(h_V, batch.edge_index, h_E, seq, batch.batch)
        
        # Compute total loss with auxiliary losses
        total_loss, loss_dict = self.model.compute_total_loss(logits, batch.y, aux_losses)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch.y).float().mean()
        batch_size = batch.y.size(0)
        
        # Log all loss components
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_acc', acc, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_classification_loss', loss_dict['classification'], on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_entropy_loss', loss_dict['entropy'], on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_link_pred_loss', loss_dict['link_pred'], on_step=True, on_epoch=True, batch_size=batch_size)
        
        return total_loss

    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics.get('train_loss_epoch', 0.0)
        train_acc = self.trainer.callback_metrics.get('train_acc_epoch', 0.0)
        train_entropy = self.trainer.callback_metrics.get('train_entropy_loss_epoch', 0.0)
        train_link_pred = self.trainer.callback_metrics.get('train_link_pred_loss_epoch', 0.0)
        self.print(f"Epoch {self.current_epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        self.print(f"         | Entropy: {train_entropy:.4f} | Link Pred: {train_link_pred:.4f}")

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get('val_loss', 0.0)
        val_acc = self.trainer.callback_metrics.get('val_acc', 0.0)
        val_entropy = self.trainer.callback_metrics.get('val_entropy_loss', 0.0)
        val_link_pred = self.trainer.callback_metrics.get('val_link_pred_loss', 0.0)
        self.print(f"         | Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        self.print(f"         | Entropy: {val_entropy:.4f} | Link Pred: {val_link_pred:.4f}")
        self.print("-" * 75)

    def validation_step(self, batch, batch_idx):
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        seq = batch.seq if hasattr(batch, 'seq') and hasattr(self.model, 'W_s') else None
        logits, aux_losses = self.model(h_V, batch.edge_index, h_E, seq, batch.batch)
        
        # Compute total loss with auxiliary losses
        total_loss, loss_dict = self.model.compute_total_loss(logits, batch.y, aux_losses)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch.y).float().mean()
        batch_size = batch.y.size(0)
        
        # Log all loss components
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val_acc', acc, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val_classification_loss', loss_dict['classification'], on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val_entropy_loss', loss_dict['entropy'], on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val_link_pred_loss', loss_dict['link_pred'], on_step=False, on_epoch=True, batch_size=batch_size)
        
        return total_loss

    def test_step(self, batch, batch_idx):
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        seq = batch.seq if hasattr(batch, 'seq') and hasattr(self.model, 'W_s') else None
        logits, aux_losses = self.model(h_V, batch.edge_index, h_E, seq, batch.batch)
        
        # Compute total loss with auxiliary losses
        total_loss, loss_dict = self.model.compute_total_loss(logits, batch.y, aux_losses)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch.y).float().mean()
        batch_size = batch.y.size(0)
        
        # Log all loss components
        self.log('test_loss', total_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_acc', acc, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_classification_loss', loss_dict['classification'], on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_entropy_loss', loss_dict['entropy'], on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_link_pred_loss', loss_dict['link_pred'], on_step=False, on_epoch=True, batch_size=batch_size)
        
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

@hydra.main(version_base="1.1", config_path='conf', config_name='config_diffpool')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.train.seed)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Custom output directory: ./outputs/wandb_project/dataset_name/timestamp
    custom_output_dir = os.path.join(
        "./outputs",
        cfg.train.wandb_project,
        cfg.data.dataset_name,
        timestamp
    )
    os.makedirs(custom_output_dir, exist_ok=True)
    
    print(f"Output directory: {custom_output_dir}")

    # Get datasets
    train_dataset, val_dataset, test_dataset, num_classes = get_dataset(
        dataset_name=cfg.data.dataset_name,
        split=cfg.data.split,
        split_similarity_threshold=cfg.data.split_similarity_threshold,
        data_dir=cfg.data.data_dir,
    )

    train_loader = create_dataloader(train_dataset, cfg.train.batch_size, cfg.train.num_workers, shuffle=True)
    val_loader = create_dataloader(val_dataset, cfg.train.batch_size, cfg.train.num_workers, shuffle=False)
    test_loader = create_dataloader(test_dataset, cfg.train.batch_size, cfg.train.num_workers, shuffle=False)

    # Model
    model = GVPDiffPool(cfg.model, cfg.train, num_classes)

    # Logger
    wandb_logger = WandbLogger(project=cfg.train.wandb_project) if cfg.train.use_wandb else None

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        save_top_k=1,
        dirpath=custom_output_dir,
        filename='best-{epoch:02d}-{val_acc:.3f}',
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
        callbacks=[checkpoint_callback]
    )

    # Print training header
    print("\n" + "=" * 75)
    print("DIFFPOOL TRAINING STARTED")
    print("=" * 75)
    print(f"Dataset: {cfg.data.dataset_name}")
    print(f"Split: {cfg.data.split}")
    print(f"Number of classes: {num_classes}")
    print(f"DiffPool max clusters: {cfg.model.max_clusters}")
    print(f"Entropy weight: {cfg.model.entropy_weight}")
    print(f"Link prediction weight: {cfg.model.link_pred_weight}")
    print("=" * 75)

    trainer.fit(model, train_loader, val_loader)
    
    print("\n" + "=" * 75)
    print("TESTING")
    print("=" * 75)
    
    trainer.test(model, test_loader)

    # Save best model and summary manually (original functionality)
    if wandb_logger is not None:
        # Save model checkpoint
        best_model_path = os.path.join(custom_output_dir, 'best_model.pt')
        trainer.save_checkpoint(best_model_path)
        wandb_logger.experiment.save(best_model_path)
        # Log summary
        wandb_logger.experiment.log({
            "best_model_path": best_model_path
        })
        wandb.finish()

if __name__ == "__main__":
    main()


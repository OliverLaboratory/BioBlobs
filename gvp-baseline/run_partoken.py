import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Only if you want to use specific GPU

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm
import json
import argparse
import math
from utils.proteinshake_dataset import get_dataset, create_dataloader
from partoken_model import ParTokenModel
from utils.utils import set_seed
from utils.lr_schedule import get_cosine_schedule_with_warmup
import torch
import torch.nn as nn
import wandb
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint

# # Set torch matmul precision to suppress warnings
# torch.set_float32_matmul_precision('highest')


class ParTokenLightning(pl.LightningModule):
    def __init__(self, model_cfg, train_cfg, num_classes):
        super().__init__()
        self.save_hyperparameters()
        self.model = ParTokenModel(
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
            nhid=model_cfg.nhid,
            k_hop=model_cfg.k_hop,
            cluster_size_max=model_cfg.cluster_size_max,
            termination_threshold=model_cfg.termination_threshold,
            tau_init=model_cfg.tau_init,
            tau_min=model_cfg.tau_min,
            tau_decay=model_cfg.tau_decay,
            codebook_size=model_cfg.codebook_size,
            codebook_dim=model_cfg.codebook_dim,
            codebook_beta=model_cfg.codebook_beta,
            codebook_decay=model_cfg.codebook_decay,
            codebook_eps=model_cfg.codebook_eps,
            codebook_distance=model_cfg.codebook_distance,
            codebook_cosine_normalize=model_cfg.codebook_cosine_normalize,
            lambda_vq=model_cfg.lambda_vq,
            lambda_ent=model_cfg.lambda_ent,
            lambda_psc=model_cfg.lambda_psc,
            psc_temp=model_cfg.psc_temp
        )
        self.lr = train_cfg.lr
        self.train_cfg = train_cfg  # Store train config for scheduler
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, h_V, edge_index, h_E, seq=None, batch=None):
        return self.model(h_V, edge_index, h_E, seq, batch)

    def training_step(self, batch, batch_idx):
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        seq = batch.seq if hasattr(batch, 'seq') and hasattr(self.model, 'sequence_embedding') else None
        logits, assignment_matrix, extra = self.model(h_V, batch.edge_index, h_E, seq, batch.batch)
        
        # Main classification loss
        ce_loss = self.criterion(logits, batch.y)
        
        # Additional losses from extra dict
        vq_loss = extra.get("vq_loss", 0.0)
        total_loss = ce_loss + vq_loss
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch.y).float().mean()
        batch_size = batch.y.size(0)
        
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_ce_loss', ce_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_vq_loss', vq_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_acc', acc, on_step=True, on_epoch=True, batch_size=batch_size)
        return total_loss

    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics.get('train_loss_epoch', 0.0)
        train_acc = self.trainer.callback_metrics.get('train_acc_epoch', 0.0)
        self.print(f"Epoch {self.current_epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get('val_loss', 0.0)
        val_acc = self.trainer.callback_metrics.get('val_acc', 0.0)
        self.print(f"         | Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        self.print("-" * 55)

    def validation_step(self, batch, batch_idx):
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        seq = batch.seq if hasattr(batch, 'seq') and hasattr(self.model, 'sequence_embedding') else None
        logits, assignment_matrix, extra = self.model(h_V, batch.edge_index, h_E, seq, batch.batch)
        
        # Main classification loss
        ce_loss = self.criterion(logits, batch.y)
        
        # Additional losses from extra dict
        vq_loss = extra.get("vq_loss", 0.0)
        total_loss = ce_loss + vq_loss
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch.y).float().mean()
        batch_size = batch.y.size(0)
        
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val_ce_loss', ce_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val_vq_loss', vq_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val_acc', acc, on_step=False, on_epoch=True, batch_size=batch_size)
        return total_loss

    def test_step(self, batch, batch_idx):
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        seq = batch.seq if hasattr(batch, 'seq') and hasattr(self.model, 'sequence_embedding') else None
        logits, assignment_matrix, extra = self.model(h_V, batch.edge_index, h_E, seq, batch.batch)
        
        # Main classification loss
        ce_loss = self.criterion(logits, batch.y)
        
        # Additional losses from extra dict
        vq_loss = extra.get("vq_loss", 0.0)
        total_loss = ce_loss + vq_loss
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch.y).float().mean()
        batch_size = batch.y.size(0)
        
        self.log('test_loss', total_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_ce_loss', ce_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_vq_loss', vq_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_acc', acc, on_step=False, on_epoch=True, batch_size=batch_size)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        if self.train_cfg.get('use_cosine_schedule', False):
            warmup_epochs = self.train_cfg.get('warmup_epochs', 5)
            max_epochs = self.trainer.max_epochs if self.trainer else self.train_cfg.epochs
            scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs, max_epochs)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        else:
            return optimizer

@hydra.main(version_base="1.1", config_path='conf', config_name='config_partoken')

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
    model = ParTokenLightning(cfg.model, cfg.train, num_classes)

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
    print("\n" + "=" * 55)
    print("TRAINING STARTED")
    print("=" * 55)

    trainer.fit(model, train_loader, val_loader)
    
    print("\n" + "=" * 55)
    print("TESTING")
    print("=" * 55)
    
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

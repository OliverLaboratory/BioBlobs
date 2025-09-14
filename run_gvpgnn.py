import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Only if you want to use specific GPU

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from utils.proteinshake_dataset import get_dataset, create_dataloader
from baseline_model import BaselineGVPModel
from utils.utils import set_seed
from utils.checkpoint_testing import test_checkpoints_and_save_results
import torch
import torch.nn as nn
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint


class GVPBaseline(pl.LightningModule):
    def __init__(self, model_cfg, train_cfg, num_classes):
        super().__init__()
        self.save_hyperparameters()
        self.model = BaselineGVPModel(
            node_in_dim=model_cfg.node_in_dim,
            node_h_dim=model_cfg.node_h_dim,
            edge_in_dim=model_cfg.edge_in_dim,
            edge_h_dim=model_cfg.edge_h_dim,
            num_classes=num_classes,
            seq_in=model_cfg.seq_in,
            num_layers=model_cfg.num_layers,
            drop_rate=model_cfg.drop_rate,
            pooling=model_cfg.pooling
        )
        self.lr = train_cfg.lr
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, h_V, edge_index, h_E, seq=None, batch=None):
        return self.model(h_V, edge_index, h_E, seq, batch)

    def training_step(self, batch, batch_idx):
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        seq = batch.seq if hasattr(batch, 'seq') and hasattr(self.model, 'W_s') else None
        logits = self.model(h_V, batch.edge_index, h_E, seq, batch.batch)
        loss = self.criterion(logits, batch.y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch.y).float().mean()
        batch_size = batch.y.size(0)
        self.log('train_loss', loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_acc', acc, on_step=True, on_epoch=True, batch_size=batch_size)
        return loss

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
        seq = batch.seq if hasattr(batch, 'seq') and hasattr(self.model, 'W_s') else None
        logits = self.model(h_V, batch.edge_index, h_E, seq, batch.batch)
        loss = self.criterion(logits, batch.y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch.y).float().mean()
        batch_size = batch.y.size(0)
        self.log('val_loss', loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val_acc', acc, on_step=False, on_epoch=True, batch_size=batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        seq = batch.seq if hasattr(batch, 'seq') and hasattr(self.model, 'W_s') else None
        logits = self.model(h_V, batch.edge_index, h_E, seq, batch.batch)
        loss = self.criterion(logits, batch.y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch.y).float().mean()
        batch_size = batch.y.size(0)
        self.log('test_loss', loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_acc', acc, on_step=False, on_epoch=True, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

@hydra.main(version_base="1.1", config_path='conf', config_name='gvp_gnn_a6000')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.train.seed)

    # Use Hydra's output directory to ensure consistency
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
    )

    train_loader = create_dataloader(train_dataset, cfg.train.batch_size, cfg.train.num_workers, shuffle=True)
    val_loader = create_dataloader(val_dataset, cfg.train.batch_size, cfg.train.num_workers, shuffle=False)
    test_loader = create_dataloader(test_dataset, cfg.train.batch_size, cfg.train.num_workers, shuffle=False)


    # Model
    model = GVPBaseline(cfg.model, cfg.train, num_classes)

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
    
    # Test both checkpoints and save results using utility function
    test_checkpoints_and_save_results(
        trainer=trainer,
        model=model,
        test_loader=test_loader,
        checkpoint_callback=checkpoint_callback,
        output_dir=custom_output_dir,
        cfg=cfg,
        model_class=GVPBaseline,
        wandb_logger=wandb_logger
    )

    if wandb_logger is not None:
        wandb.finish()

if __name__ == "__main__":
    main()

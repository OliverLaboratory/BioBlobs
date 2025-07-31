from tqdm import tqdm
import json
import os
import argparse
from proteinclass_dataset import ProteinClassificationDataset, print_example_data, create_dataloader, \
get_dataset, generator_to_structures
from part_model_diffpool_graphsage import GVPDiffPoolGraphSAGEModel  # Import the new model
import torch
import random
import torch.optim as optim
import numpy as np
import torch.nn as nn
import wandb
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_name",
    type=str,
    default="enzymecommission",
    choices=["enzymecommission", "proteinfamily", "scope"],
)

parser.add_argument(
    "--split",
    type=str,
    default="structure",
    choices=["random", "sequence", "structure"],
)

parser.add_argument(
    "--split_similarity_threshold",
    type=float,
    default=0.7,
)

parser.add_argument(
    "--seed", type=int, default=42, help="Random seed for reproducibility"
)

parser.add_argument(
    "--max_clusters", type=int, default=30, help="Maximum number of clusters for DiffPool"
)

parser.add_argument(
    "--entropy_weight", type=float, default=0.1, help="Weight for entropy loss"
)

parser.add_argument(
    "--link_pred_weight", type=float, default=0.5, help="Weight for link prediction loss"
)

args = parser.parse_args()
dataset_name = args.dataset_name
split = args.split
split_similarity_threshold = args.split_similarity_threshold
seed = args.seed
max_clusters = args.max_clusters
entropy_weight = args.entropy_weight
link_pred_weight = args.link_pred_weight


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")


def train_diffpool_model(model, train_dataset, val_dataset, test_dataset, 
                        epochs=150, lr=1e-3, batch_size=128, num_workers=4,
                        models_dir="./models", device="cuda", use_wandb=True):
    
    # Create timestamp-based model ID and subfolder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"diffpool_{dataset_name}_{split}_{split_similarity_threshold}_{timestamp}"
    run_models_dir = os.path.join(models_dir, model_id)
    
    # Initialize wandb
    if use_wandb:
        run_name = f"diffpool_graphsage_intercluster_{dataset_name}_{split}_{split_similarity_threshold}"
        wandb.init(
            project="gvp-protein-classification",
            name=run_name,
            config={
                "dataset": dataset_name,
                "split": split,
                "epochs": epochs,
                "lr": lr,
                "model_id": model_id,
                "timestamp": timestamp,
                "max_clusters": max_clusters,
                "entropy_weight": entropy_weight,
                "link_pred_weight": link_pred_weight,
                "model_type": "GVPDiffPool"
            }
        )
    
    train_loader = create_dataloader(train_dataset, batch_size, num_workers, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size, num_workers, shuffle=False)
    test_loader = create_dataloader(test_dataset, batch_size, num_workers, shuffle=False)
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Track top 3 models: [(val_acc, model_path, epoch), ...]
    top_models = []
    os.makedirs(run_models_dir, exist_ok=True)
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Model ID: {model_id}")
    print(f"Models will be saved to: {run_models_dir}")
    print(f"DiffPool parameters: max_clusters={max_clusters}, entropy_weight={entropy_weight}, link_pred_weight={link_pred_weight}")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss, train_acc, train_aux_losses = train_epoch_diffpool(model, train_loader, optimizer, device)
        
        print(f"EPOCH {epoch} TRAIN loss: {train_loss:.4f} acc: {train_acc:.4f}")
        print(f"  Aux losses - entropy: {train_aux_losses['entropy']:.4f}, link_pred: {train_aux_losses['link_pred']:.4f}")
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss, val_acc, val_aux_losses = evaluate_model_diffpool(model, val_loader, device)
        
        print(f"EPOCH {epoch} VAL loss: {val_loss:.4f} acc: {val_acc:.4f}")
        print(f"  Aux losses - entropy: {val_aux_losses['entropy']:.4f}, link_pred: {val_aux_losses['link_pred']:.4f}")
        
        # Check if this model should be saved (top 3)
        should_save = len(top_models) < 3 or val_acc > min(top_models, key=lambda x: x[0])[0]
        
        if should_save:
            # Save model in the run-specific subfolder
            model_path = os.path.join(run_models_dir, f"epoch_{epoch}.pt")
            torch.save(model.state_dict(), model_path)
            
            # Add to top models list
            top_models.append((val_acc, model_path, epoch))
            
            # Sort by validation accuracy (descending)
            top_models.sort(key=lambda x: x[0], reverse=True)
            
            # Keep only top 3
            if len(top_models) > 3:
                # Remove the worst model file and entry
                worst_acc, worst_path, worst_epoch = top_models.pop()
                if os.path.exists(worst_path):
                    os.remove(worst_path)
                    print(f"Removed model from epoch {worst_epoch} (val_acc: {worst_acc:.4f})")
        
        # Log to wandb
        if use_wandb:
            wandb.log({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_entropy_loss": train_aux_losses['entropy'],
                "train_link_pred_loss": train_aux_losses['link_pred'],
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_entropy_loss": val_aux_losses['entropy'],
                "val_link_pred_loss": val_aux_losses['link_pred'],
                "epoch": epoch
            })
        
        # Display current top models
        best_val_acc = top_models[0][0] if top_models else 0.0
        print(f"BEST VAL acc: {best_val_acc:.4f}")
        print("Top 3 models:")
        for i, (acc, path, ep) in enumerate(top_models):
            print(f"  {i+1}. Epoch {ep}: {acc:.4f} - {os.path.basename(path)}")
        print("-" * 60)
    
    # Test with best model
    if top_models:
        best_val_acc, best_model_path, best_epoch = top_models[0]
        print(f"Loading best model from epoch {best_epoch}: {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))
        
        # Save a copy of the best model with a clear name
        best_model_final_path = os.path.join(run_models_dir, "best_model.pt")
        torch.save(model.state_dict(), best_model_final_path)
        print(f"Best model also saved as: {best_model_final_path}")
    else:
        print("No models were saved!")
        return model
    
    model.eval()
    with torch.no_grad():
        test_loss, test_acc, test_aux_losses = evaluate_model_diffpool(model, test_loader, device)
        
    print(f"TEST loss: {test_loss:.4f} acc: {test_acc:.4f}")
    print(f"Test aux losses - entropy: {test_aux_losses['entropy']:.4f}, link_pred: {test_aux_losses['link_pred']:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    
    
    # Log final results
    if use_wandb:
        wandb.log({
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_entropy_loss": test_aux_losses['entropy'],
            "test_link_pred_loss": test_aux_losses['link_pred'],
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch
        })
        # Log the model path for reference
        wandb.config.update({"model_save_path": run_models_dir})
        wandb.finish()
    
    return model


def evaluate_model_diffpool(model, dataloader, device):
    """
    Evaluate DiffPool model on given dataloader.
    """
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # Track auxiliary losses
    total_entropy_loss = 0.0
    total_link_pred_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc="Evaluating")
    
    for batch in progress_bar:
        # Move batch to device
        batch = batch.to(device)
        
        # Prepare inputs
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        seq = batch.seq if hasattr(batch, 'seq') else None
        
        # Forward pass
        logits, aux_losses = model(h_V, batch.edge_index, h_E, seq=None, batch=batch.batch)
        
        # Compute total loss
        total_loss_tensor, loss_dict = model.compute_total_loss(logits, batch.y, aux_losses)
        
        # Statistics
        total_loss += total_loss_tensor.item() * len(batch.y)
        total_entropy_loss += aux_losses.get('entropy', 0.0).item() * len(batch.y) if torch.is_tensor(aux_losses.get('entropy', 0.0)) else 0.0
        total_link_pred_loss += aux_losses.get('link_pred', 0.0).item() * len(batch.y) if torch.is_tensor(aux_losses.get('link_pred', 0.0)) else 0.0
        
        pred = torch.argmax(logits, dim=1)
        total_correct += (pred == batch.y).sum().item()
        total_samples += len(batch.y)
        
        # Update progress bar
        current_acc = total_correct / total_samples
        progress_bar.set_postfix({
            'loss': f'{total_loss/total_samples:.4f}',
            'acc': f'{current_acc:.4f}'
        })
        
        # Clear GPU cache
        torch.cuda.empty_cache()
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    avg_entropy_loss = total_entropy_loss / total_samples
    avg_link_pred_loss = total_link_pred_loss / total_samples
    
    aux_losses_avg = {
        'entropy': avg_entropy_loss,
        'link_pred': avg_link_pred_loss
    }
    
    return avg_loss, avg_acc, aux_losses_avg


def train_epoch_diffpool(model, dataloader, optimizer, device):
    """
    Train DiffPool model for one epoch.
    """
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # Track auxiliary losses
    total_entropy_loss = 0.0
    total_link_pred_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        # Move batch to device
        batch = batch.to(device)
        
        # Prepare inputs
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        seq = batch.seq if hasattr(batch, 'seq') else None
        
        # Forward pass
        logits, aux_losses = model(h_V, batch.edge_index, h_E, seq=None, batch=batch.batch)
        
        # Compute total loss
        total_loss_tensor, loss_dict = model.compute_total_loss(logits, batch.y, aux_losses)
        
        # Backward pass
        total_loss_tensor.backward()
        optimizer.step()
        
        # Statistics
        total_loss += total_loss_tensor.item() * len(batch.y)
        total_entropy_loss += aux_losses.get('entropy', 0.0).item() * len(batch.y) if torch.is_tensor(aux_losses.get('entropy', 0.0)) else 0.0
        total_link_pred_loss += aux_losses.get('link_pred', 0.0).item() * len(batch.y) if torch.is_tensor(aux_losses.get('link_pred', 0.0)) else 0.0
        
        pred = torch.argmax(logits, dim=1)
        total_correct += (pred == batch.y).sum().item()
        total_samples += len(batch.y)
        
        # Update progress bar
        current_acc = total_correct / total_samples
        progress_bar.set_postfix({
            'loss': f'{total_loss/total_samples:.4f}',
            'acc': f'{current_acc:.4f}'
        })
        
        # Clear GPU cache
        torch.cuda.empty_cache()
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    avg_entropy_loss = total_entropy_loss / total_samples
    avg_link_pred_loss = total_link_pred_loss / total_samples
    
    aux_losses_avg = {
        'entropy': avg_entropy_loss,
        'link_pred': avg_link_pred_loss
    }
    
    return avg_loss, avg_acc, aux_losses_avg


def main():
    # set seed
    set_seed(seed)

    # Get datasets
    train_dataset, val_dataset, test_dataset, num_classes = get_dataset(
        dataset_name=dataset_name,
        split=split,
        split_similarity_threshold=split_similarity_threshold,
        data_dir="./data",
    )

    model = GVPDiffPoolGraphSAGEModel(
        node_in_dim=(6, 3),
        node_h_dim=(100, 16),   
        edge_in_dim=(32, 1),
        edge_h_dim=(32, 1),
        num_classes=num_classes,
        seq_in=False,
        num_layers=3,
        drop_rate=0.1,
        pooling="sum",
        max_clusters=max_clusters,
        entropy_weight=entropy_weight,
        link_pred_weight=link_pred_weight
    )

    print(f"Dataset: {dataset_name}")
    print(f"Split: {split}")
    print(f"Number of classes: {num_classes}")
    print(f"DiffPool max clusters: {max_clusters}")
    print(f"Entropy weight: {entropy_weight}")
    print(f"Link prediction weight: {link_pred_weight}")

    trained_model = train_diffpool_model(
        model,
        train_dataset,
        val_dataset,
        test_dataset,
        epochs=150,
        lr=1e-4,
        batch_size=64,
        num_workers=4,
        models_dir="./models",
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_wandb=True
    )


if __name__ == "__main__":
    main()
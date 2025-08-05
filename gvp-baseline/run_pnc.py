from tqdm import tqdm
import os
import argparse
from proteinclass_dataset import create_dataloader, get_dataset
from part_model_PnC import GVPHardGumbelPartitionerModel  # Import the PnC model
import torch
import random
import torch.optim as optim
import numpy as np
import wandb
import torch.nn as nn
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
    "--max_clusters", type=int, default=15, help="Maximum number of clusters for PnC partitioner"
)

parser.add_argument(
    "--tau_init", type=float, default=1.0, help="Initial temperature for Gumbel-Softmax"
)

parser.add_argument(
    "--tau_min", type=float, default=0.1, help="Minimum temperature for Gumbel-Softmax"
)

parser.add_argument(
    "--tau_decay", type=float, default=0.95, help="Temperature decay rate for Gumbel-Softmax"
)

# NEW ARGUMENTS for k-hop connectivity and GCN layers
parser.add_argument(
    "--k_hop", type=int, default=2, help="k-hop neighborhood constraint for connectivity"
)

parser.add_argument(
    "--enable_connectivity", action="store_true", default=True, 
    help="Enable k-hop connectivity constraint (default: True)"
)

parser.add_argument(
    "--disable_connectivity", dest="enable_connectivity", action="store_false",
    help="Disable k-hop connectivity constraint"
)

parser.add_argument(
    "--num_gcn_layers", type=int, default=2, help="Number of GCN layers for inter-cluster message passing"
)

parser.add_argument(
    "--use_wandb",
    action="store_true",
    help="Use Weights & Biases for experiment tracking"
)

parser.add_argument(
    "--cluster_size_max", type=int, default=3, help="Maximum nodes per cluster for PnC partitioner"
)

parser.add_argument(
    "--nhid", type=int, default=50, help="Hidden dimension for partitioner context network"
)

args = parser.parse_args()
dataset_name = args.dataset_name
split = args.split
split_similarity_threshold = args.split_similarity_threshold
seed = args.seed
max_clusters = args.max_clusters
tau_init = args.tau_init
tau_min = args.tau_min
tau_decay = args.tau_decay
k_hop = args.k_hop
enable_connectivity = args.enable_connectivity
num_gcn_layers = args.num_gcn_layers
cluster_size_max = args.cluster_size_max
nhid = args.nhid
use_wandb = args.use_wandb


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")


def train_pnc_model(model, train_dataset, val_dataset, test_dataset, 
                   epochs=150, lr=1e-3, batch_size=128, num_workers=4,
                   models_dir="./models", device="cuda", use_wandb=True):
    
    # Create timestamp-based model ID and subfolder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"pnc_{dataset_name}_{split}_{split_similarity_threshold}_{timestamp}"
    run_models_dir = os.path.join(models_dir, model_id)
    
    # Initialize wandb with NEW parameters
    if use_wandb:
        run_name = f"pnc_hard_gumbel_{dataset_name}_{split}_{split_similarity_threshold}"
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
                "tau_init": tau_init,
                "tau_min": tau_min,
                "tau_decay": tau_decay,
                "k_hop": k_hop,
                "enable_connectivity": enable_connectivity,
                "num_gcn_layers": num_gcn_layers,
                "cluster_size_max": cluster_size_max,  # NEW
                "nhid": nhid,                          # NEW
                "model_type": "GVPHardGumbelPnC"
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
    print(f"PnC parameters: max_clusters={max_clusters}, tau_init={tau_init}, tau_min={tau_min}, tau_decay={tau_decay}")
    print(f"Cluster parameters: cluster_size_max={cluster_size_max}, k_hop={k_hop}, enable_connectivity={enable_connectivity}")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss, train_acc, train_metrics = train_epoch_pnc(model, train_loader, optimizer, device)
        
        print(f"EPOCH {epoch} TRAIN loss: {train_loss:.4f} acc: {train_acc:.4f}")
        print(f"  Temperature: {train_metrics['temperature']:.4f}, Avg clusters: {train_metrics['avg_clusters']:.2f}, Avg cluster size: {train_metrics['avg_cluster_size']:.2f}")
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss, val_acc, val_metrics = evaluate_model_pnc(model, val_loader, device)
        
        print(f"EPOCH {epoch} VAL loss: {val_loss:.4f} acc: {val_acc:.4f}")
        print(f"  Temperature: {val_metrics['temperature']:.4f}, Avg clusters: {val_metrics['avg_clusters']:.2f}, Avg cluster size: {val_metrics['avg_cluster_size']:.2f}")
        
        # Update temperature schedule
        model.update_epoch()
        
        # Check if this model should be saved (top 3)
        should_save = len(top_models) < 3 or val_acc > min(top_models, key=lambda x: x[0])[0]
        
        if should_save:
            # Save model
            model_path = os.path.join(run_models_dir, f"model_epoch_{epoch}_val_acc_{val_acc:.4f}.pt")
            torch.save(model.state_dict(), model_path)
            
            # Add to top models
            top_models.append((val_acc, model_path, epoch))
            top_models.sort(key=lambda x: x[0], reverse=True)  # Sort by accuracy, descending
            
            # Keep only top 3
            if len(top_models) > 3:
                # Remove the worst model file
                _, worst_path, _ = top_models.pop()
                if os.path.exists(worst_path):
                    os.remove(worst_path)
        
        # Log to wandb
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "temperature": train_metrics['temperature'],
                "train_avg_clusters": train_metrics['avg_clusters'],
                "val_avg_clusters": val_metrics['avg_clusters'],
                "train_avg_cluster_size": train_metrics['avg_cluster_size'],  # NEW
                "val_avg_cluster_size": val_metrics['avg_cluster_size']       # NEW
            })
        
        # Display current top models
        best_val_acc = top_models[0][0] if top_models else 0.0
        print(f"BEST VAL acc: {best_val_acc:.4f}")
        print("Top 3 models:")
        for i, (acc, path, ep) in enumerate(top_models):
            print(f"  {i+1}. Epoch {ep}, Val Acc: {acc:.4f}, Path: {os.path.basename(path)}")
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
        test_loss, test_acc, test_metrics = evaluate_model_pnc(model, test_loader, device)
        
    print(f"TEST loss: {test_loss:.4f} acc: {test_acc:.4f}")
    print(f"Test temperature: {test_metrics['temperature']:.4f}, Avg clusters: {test_metrics['avg_clusters']:.2f}, Avg cluster size: {test_metrics['avg_cluster_size']:.2f}")
    print(f"Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    
    # Save run summary
    summary_path = os.path.join(run_models_dir, "run_summary.txt")
    with open(summary_path, "w") as f:
        f.write("PnC Hard Gumbel Partitioner Training Summary\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Split: {split} (threshold: {split_similarity_threshold})\n")
        f.write(f"Max clusters: {max_clusters}\n")
        f.write(f"Cluster size max: {cluster_size_max}\n")                    # NEW
        f.write(f"k-hop constraint: {k_hop} (enabled: {enable_connectivity})\n")
        f.write(f"GCN layers: {num_gcn_layers}\n")
        f.write(f"Hidden dimension: {nhid}\n")                               # NEW
        f.write(f"Temperature params: init={tau_init}, min={tau_min}, decay={tau_decay}\n")
        f.write(f"Training epochs: {epochs}\n")
        f.write(f"Learning rate: {lr}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})\n")
        f.write(f"Test accuracy: {test_acc:.4f}\n")
        f.write(f"Final temperature: {test_metrics['temperature']:.4f}\n")
        f.write(f"Average clusters used: {test_metrics['avg_clusters']:.2f}\n")
        f.write(f"Average cluster size: {test_metrics['avg_cluster_size']:.2f}\n")  # NEW
    
    print(f"Run summary saved to: {summary_path}")
    
    # Log final results
    if use_wandb:
        wandb.log({
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_temperature": test_metrics['temperature'],
            "test_avg_clusters": test_metrics['avg_clusters'],
            "test_avg_cluster_size": test_metrics['avg_cluster_size'],  # NEW
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch
        })
        # Log the model path for reference
        wandb.config.update({"model_save_path": run_models_dir})
        wandb.finish()
    
    return model


def evaluate_model_pnc(model, dataloader, device):
    """
    Evaluate PnC model on given dataloader.
    """
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # Track PnC-specific metrics
    total_clusters = 0
    total_cluster_nodes = 0  # NEW: Track total nodes in clusters
    total_graphs = 0
    current_temp = model.partitioner.get_temperature()
    
    progress_bar = tqdm(dataloader, desc="Evaluating")
    
    for batch in progress_bar:
        # Move batch to device
        batch = batch.to(device)
        
        # Prepare inputs
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        
        # Forward pass
        logits, assignment_matrix = model(h_V, batch.edge_index, h_E, seq=None, batch=batch.batch)
        
        # Compute loss
        loss = model.compute_total_loss(logits, batch.y)
        
        # Statistics
        total_loss += loss.item() * len(batch.y)
        
        pred = torch.argmax(logits, dim=1)
        total_correct += (pred == batch.y).sum().item()
        total_samples += len(batch.y)
        
        # Track clustering metrics
        batch_size = logits.size(0)
        
        # Count actual clusters used and cluster sizes
        for b in range(batch_size):
            cluster_sizes = assignment_matrix[b].sum(dim=0)  # [max_clusters] - nodes per cluster
            clusters_used = (cluster_sizes > 0).sum().item()
            nodes_in_clusters = cluster_sizes.sum().item()
            
            total_clusters += clusters_used
            total_cluster_nodes += nodes_in_clusters
            total_graphs += 1
        
        # Update progress bar
        current_acc = total_correct / total_samples
        progress_bar.set_postfix({
            'loss': f'{total_loss/total_samples:.4f}',
            'acc': f'{current_acc:.4f}',
            'temp': f'{current_temp:.3f}'
        })
        
        # Clear GPU cache
        torch.cuda.empty_cache()
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    avg_clusters = total_clusters / total_graphs if total_graphs > 0 else 0
    avg_cluster_size = total_cluster_nodes / total_clusters if total_clusters > 0 else 0  # NEW
    
    metrics = {
        'temperature': current_temp,
        'avg_clusters': avg_clusters,
        'avg_cluster_size': avg_cluster_size  # NEW
    }
    
    return avg_loss, avg_acc, metrics


def train_epoch_pnc(model, dataloader, optimizer, device):
    """
    Train PnC model for one epoch.
    """
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # Track PnC-specific metrics
    total_clusters = 0
    total_cluster_nodes = 0  # NEW: Track total nodes in clusters
    total_graphs = 0
    current_temp = model.partitioner.get_temperature()
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        # Move batch to device
        batch = batch.to(device)
        
        # Prepare inputs
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        
        # Forward pass
        logits, assignment_matrix = model(h_V, batch.edge_index, h_E, seq=None, batch=batch.batch)
        
        # Compute loss
        loss = model.compute_total_loss(logits, batch.y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item() * len(batch.y)
        
        pred = torch.argmax(logits, dim=1)
        total_correct += (pred == batch.y).sum().item()
        total_samples += len(batch.y)
        
        # Track clustering metrics
        batch_size = logits.size(0)
        
        # Count actual clusters used and cluster sizes
        for b in range(batch_size):
            cluster_sizes = assignment_matrix[b].sum(dim=0)  # [max_clusters] - nodes per cluster
            clusters_used = (cluster_sizes > 0).sum().item()
            nodes_in_clusters = cluster_sizes.sum().item()
            
            total_clusters += clusters_used
            total_cluster_nodes += nodes_in_clusters
            total_graphs += 1
        
        # Update progress bar
        current_acc = total_correct / total_samples
        progress_bar.set_postfix({
            'loss': f'{total_loss/total_samples:.4f}',
            'acc': f'{current_acc:.4f}',
            'temp': f'{current_temp:.3f}'
        })
        
        # Clear GPU cache
        torch.cuda.empty_cache()
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    avg_clusters = total_clusters / total_graphs if total_graphs > 0 else 0
    avg_cluster_size = total_cluster_nodes / total_clusters if total_clusters > 0 else 0  # NEW
    
    metrics = {
        'temperature': current_temp,
        'avg_clusters': avg_clusters,
        'avg_cluster_size': avg_cluster_size  # NEW
    }
    
    return avg_loss, avg_acc, metrics


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

    # Create PnC model with ALL parameters matching part_model_PnC.py
    model = GVPHardGumbelPartitionerModel(
        node_in_dim=(6, 3),
        node_h_dim=(100, 16),   
        edge_in_dim=(32, 1),
        edge_h_dim=(32, 1),
        num_classes=num_classes,
        seq_in=False,
        num_layers=3,
        drop_rate=0.1,
        pooling="sum",
        max_clusters=max_clusters
    )
    
    # Update partitioner parameters to match arguments
    if hasattr(model, 'partitioner'):
        model.partitioner.tau_init = tau_init
        model.partitioner.tau_min = tau_min
        model.partitioner.tau_decay = tau_decay
        model.partitioner.k_hop = k_hop
        model.partitioner.cluster_size_max = cluster_size_max
        model.partitioner.enable_connectivity = enable_connectivity
        
        # Update context network hidden dimension if nhid is provided
        if nhid != model.partitioner.context_gru.hidden_size:
            ns = model.partitioner.selection_mlp[0].in_features - nhid  # nfeat
            model.partitioner.context_gru = nn.GRU(ns, nhid, batch_first=True)
            model.partitioner.context_init = nn.Linear(ns, nhid)
            model.partitioner.selection_mlp = nn.Sequential(
                nn.Linear(ns + nhid, nhid),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(nhid, 1)
            )

    print(f"Dataset: {dataset_name}")
    print(f"Split: {split}")
    print(f"Number of classes: {num_classes}")
    print(f"PnC max clusters: {max_clusters}")
    print(f"Cluster size max: {cluster_size_max}")
    print(f"k-hop constraint: {k_hop} (enabled: {enable_connectivity})")
    print(f"GCN layers: {num_gcn_layers}")
    print(f"Hidden dimension: {nhid}")
    print(f"Temperature schedule: init={tau_init}, min={tau_min}, decay={tau_decay}")

    train_pnc_model(
        model,
        train_dataset,
        val_dataset,
        test_dataset,
        epochs=150,
        lr=1e-4,
        batch_size=128,
        num_workers=4,
        models_dir="./models",
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_wandb=use_wandb
    )


if __name__ == "__main__":
    main()

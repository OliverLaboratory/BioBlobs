import torch
from tqdm import tqdm
import numpy as np
import wandb
import torch.optim as optim
import os
from datetime import datetime
from utils.proteinshake_dataset import create_dataloader, get_dataset


def train_epoch_pnc(model, dataloader, optimizer, device):
    """
    Trains gvp-partition model for one epoch.
    Args:
        model: The gvp-partition model to be trained. 
        dataloader: DataLoader providing batches of graph data.
        optimizer: Optimizer for updating model parameters.
        device: Device (CPU or CUDA).
    Returns:
        avg_loss (float): Average (cross entropy) loss over the epoch.
        avg_acc (float): Average accuracy over the epoch.
        metrics (dict): Dictionary containing partitioning and clustering metrics:
            - 'temperature': Current temperature used by the partitioner.
            - 'avg_clusters': Average number of clusters per graph.
            - 'avg_cluster_size': Average number of nodes per cluster.
            - 'avg_assignment_coverage': Average assignment coverage percentage.
            - 'termination_efficiency': Percentage of proteins above termination threshold.
    """

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # Accumulated partition metrics
    total_assignment_percentage = 0.0
    total_effective_clusters = 0.0
    total_proteins = 0
    proteins_above_threshold = 0
    total_cluster_size_sum = 0.0  # Add this for proper cluster size calculation
    
    current_temp = model.partitioner.get_temperature()
    termination_threshold = getattr(model.partitioner, 'termination_threshold', 0.95)
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        batch = batch.to(device)
        
        # node/edge scalars and vectors features
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        
        # Forward pass
        logits, assignment_matrix = model(h_V, batch.edge_index, h_E, seq=None, batch=batch.batch)
        
        # Compute cross entropy loss
        loss = model.compute_total_loss(logits, batch.y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Statistics
        total_loss += loss.item() * len(batch.y)
        
        pred = torch.argmax(logits, dim=1)
        total_correct += (pred == batch.y).sum().item()
        total_samples += len(batch.y)
        
        # Accumulate partition statistics from current batch
        if hasattr(batch, 'node_s'):
            from torch_geometric.utils import to_dense_batch
            _, mask = to_dense_batch(batch.node_s, batch.batch)
            
            batch_size = batch.y.size(0)
            
            # Count total valid residues per protein
            total_residues = mask.sum(dim=-1).float()  # [B]
            
            # Count assigned residues per protein
            assigned_residues = assignment_matrix.sum(dim=(1, 2))  # [B]
            
            # Calculate assignment percentages
            assignment_percentages = assigned_residues / (total_residues + 1e-8)  # [B]
            
            # Count effective clusters per protein and calculate cluster sizes
            effective_clusters = (assignment_matrix.sum(dim=1) > 0).sum(dim=-1)  # [B]
            
            # Calculate average cluster size per protein
            for b in range(batch_size):
                if effective_clusters[b] > 0:
                    cluster_sizes = assignment_matrix[b].sum(dim=0)  # [max_clusters]
                    active_cluster_sizes = cluster_sizes[cluster_sizes > 0.1]  # Only non-empty clusters
                    if len(active_cluster_sizes) > 0:
                        total_cluster_size_sum += active_cluster_sizes.mean().item()
            
            # Accumulate statistics
            total_assignment_percentage += assignment_percentages.sum().item()
            total_effective_clusters += effective_clusters.float().sum().item()
            total_proteins += batch_size
            proteins_above_threshold += (assignment_percentages >= termination_threshold).sum().item()
        
        # Update progress bar with current estimates
        current_acc = total_correct / total_samples
        
        progress_bar.set_postfix({
            'loss': f'{total_loss/total_samples:.4f}',
            'acc': f'{current_acc:.4f}',
            'temp': f'{current_temp:.3f}',
        })
        
        # Clear GPU cache
        torch.cuda.empty_cache()
    
    # Compute final epoch-level statistics
    if total_proteins > 0:
        avg_assignment_coverage = total_assignment_percentage / total_proteins
        avg_effective_clusters = total_effective_clusters / total_proteins
        termination_efficiency = proteins_above_threshold / total_proteins
        
        # Calculate avg_cluster_size correctly
        avg_cluster_size = total_cluster_size_sum / total_proteins if total_proteins > 0 else 0.0
    else:
        avg_assignment_coverage = 0.0
        avg_effective_clusters = 0.0
        termination_efficiency = 0.0
        avg_cluster_size = 0.0
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    
    metrics = {
        'temperature': current_temp,
        'avg_clusters': avg_effective_clusters,
        'avg_cluster_size': avg_cluster_size,
        'avg_assignment_coverage': avg_assignment_coverage,
        'termination_efficiency': termination_efficiency
    }
    
    return avg_loss, avg_acc, metrics



def evaluate_model_pnc(model, dataloader, device):
    """
    Evaluate efficient PnC model on given dataloader.
    """
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # Accumulated partition metrics (same as training)
    total_assignment_percentage = 0.0
    total_effective_clusters = 0.0
    total_proteins = 0
    proteins_above_threshold = 0
    total_cluster_size_sum = 0.0  # Add this for proper cluster size calculation
    
    current_temp = model.partitioner.get_temperature()
    termination_threshold = getattr(model.partitioner, 'termination_threshold', 0.95)
    
    progress_bar = tqdm(dataloader, desc="Evaluating")
    
    for batch in progress_bar:
        batch = batch.to(device)
        
        # node/edge scalars and vectors features
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        
        # Forward pass
        logits, assignment_matrix = model(h_V, batch.edge_index, h_E, seq=None, batch=batch.batch)
        
        # Compute cross entropy loss
        loss = model.compute_total_loss(logits, batch.y)
        
        # Statistics
        total_loss += loss.item() * len(batch.y)
        
        pred = torch.argmax(logits, dim=1)
        total_correct += (pred == batch.y).sum().item()
        total_samples += len(batch.y)
        
        # Accumulate partition statistics from current batch (same as training)
        if hasattr(batch, 'node_s'):
            from torch_geometric.utils import to_dense_batch
            _, mask = to_dense_batch(batch.node_s, batch.batch)
            
            batch_size = batch.y.size(0)
            
            # Count total valid residues per protein
            total_residues = mask.sum(dim=-1).float()  # [B]
            
            # Count assigned residues per protein
            assigned_residues = assignment_matrix.sum(dim=(1, 2))  # [B]
            
            # Calculate assignment percentages
            assignment_percentages = assigned_residues / (total_residues + 1e-8)  # [B]
            
            # Count effective clusters per protein and calculate cluster sizes
            effective_clusters = (assignment_matrix.sum(dim=1) > 0).sum(dim=-1)  # [B]
            
            # Calculate average cluster size per protein
            for b in range(batch_size):
                if effective_clusters[b] > 0:
                    cluster_sizes = assignment_matrix[b].sum(dim=0)  # [max_clusters]
                    active_cluster_sizes = cluster_sizes[cluster_sizes > 0.1]  # Only non-empty clusters
                    if len(active_cluster_sizes) > 0:
                        total_cluster_size_sum += active_cluster_sizes.mean().item()
            
            # Accumulate statistics
            total_assignment_percentage += assignment_percentages.sum().item()
            total_effective_clusters += effective_clusters.float().sum().item()
            total_proteins += batch_size
            proteins_above_threshold += (assignment_percentages >= termination_threshold).sum().item()
        
        # Update progress bar with current estimates
        current_acc = total_correct / total_samples
        
        progress_bar.set_postfix({
            'loss': f'{total_loss/total_samples:.4f}',
            'acc': f'{current_acc:.4f}',
            'temp': f'{current_temp:.3f}',
        })
        
        # Clear GPU cache
        torch.cuda.empty_cache()
    
    # Compute final epoch-level statistics (same as training)
    if total_proteins > 0:
        avg_assignment_coverage = total_assignment_percentage / total_proteins
        avg_effective_clusters = total_effective_clusters / total_proteins
        termination_efficiency = proteins_above_threshold / total_proteins
        
        # Calculate avg_cluster_size correctly
        avg_cluster_size = total_cluster_size_sum / total_proteins if total_proteins > 0 else 0.0
    else:
        avg_assignment_coverage = 0.0
        avg_effective_clusters = 0.0
        termination_efficiency = 0.0
        avg_cluster_size = 0.0
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    
    metrics = {
        'temperature': current_temp,
        'avg_clusters': avg_effective_clusters,
        'avg_cluster_size': avg_cluster_size,
        'avg_assignment_coverage': avg_assignment_coverage,
        'termination_efficiency': termination_efficiency
    }
    
    return avg_loss, avg_acc, metrics


def train_pnc_model(model, train_dataset, val_dataset, test_dataset, args,
                   epochs=150, lr=1e-3, batch_size=128, num_workers=4,
                   models_dir="./models", device="cuda", use_wandb=True):
    
    # Create timestamp-based model ID and subfolder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"efficient_pnc_{args.dataset_name}_{args.split}_{args.split_similarity_threshold}_{timestamp}"
    run_models_dir = os.path.join(models_dir, model_id)
    
    # Initialize wandb with updated parameters
    if use_wandb:
        run_name = f"efficient_pnc_{args.dataset_name}_{args.split}_{args.split_similarity_threshold}"
        wandb.init(
            project="gvp-protein-classification",
            name=run_name,
            config={
                "dataset": args.dataset_name,
                "split": args.split,
                "epochs": epochs,
                "lr": lr,
                "model_id": model_id,
                "timestamp": timestamp,
                "max_clusters": args.max_clusters,   
                "tau_init": args.tau_init,
                "tau_min": args.tau_min,
                "tau_decay": args.tau_decay,
                "k_hop": args.k_hop,
                "enable_connectivity": args.enable_connectivity,
                "num_gcn_layers": args.num_gcn_layers,
                "cluster_size_max": args.cluster_size_max,
                "nhid": args.nhid,
                "termination_threshold": args.termination_threshold,
                "model_type": "GVPGradientSafeHardGumbel"
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
    print(f"PnC parameters: max_clusters={args.max_clusters}, tau_init={args.tau_init}, tau_min={args.tau_min}, tau_decay={args.tau_decay}")
    print(f"Cluster parameters: cluster_size_max={args.cluster_size_max}, k_hop={args.k_hop}, enable_connectivity={args.enable_connectivity}")
    print(f"Termination threshold: {args.termination_threshold}")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss, train_acc, train_metrics = train_epoch_pnc(model, train_loader, optimizer, device)
        
        print(f"EPOCH {epoch} TRAIN loss: {train_loss:.4f} acc: {train_acc:.4f}")
        print(f"  Temperature: {train_metrics['temperature']:.4f}, Avg clusters: {train_metrics['avg_clusters']:.2f}, Avg cluster size: {train_metrics['avg_cluster_size']:.2f}")
        print(f"  Assignment coverage: {train_metrics['avg_assignment_coverage']:.2f}, Termination efficiency: {train_metrics['termination_efficiency']:.2f}")
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss, val_acc, val_metrics = evaluate_model_pnc(model, val_loader, device)
        
        print(f"EPOCH {epoch} VAL loss: {val_loss:.4f} acc: {val_acc:.4f}")
        print(f"  Temperature: {val_metrics['temperature']:.4f}, Avg clusters: {val_metrics['avg_clusters']:.2f}, Avg cluster size: {val_metrics['avg_cluster_size']:.2f}")
        print(f"  Assignment coverage: {val_metrics['avg_assignment_coverage']:.2f}, Termination efficiency: {val_metrics['termination_efficiency']:.2f}")
        
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
            log_dict = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "temperature": train_metrics['temperature'],
                "partition_stats/train_avg_clusters": train_metrics['avg_clusters'],
                "partition_stats/val_avg_clusters": val_metrics['avg_clusters'],
                "partition_stats/train_avg_cluster_size": train_metrics['avg_cluster_size'],
                "partition_stats/val_avg_cluster_size": val_metrics['avg_cluster_size'],
                "partition_stats/train_avg_assignment_coverage": train_metrics['avg_assignment_coverage'],
                "partition_stats/val_avg_assignment_coverage": val_metrics['avg_assignment_coverage'],
                "partition_stats/train_termination_efficiency": train_metrics['termination_efficiency'],
                "partition_stats/val_termination_efficiency": val_metrics['termination_efficiency']
            }
            
            wandb.log(log_dict)
        
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
    print(f"Test assignment coverage: {test_metrics['avg_assignment_coverage']:.2f}, Termination efficiency: {test_metrics['termination_efficiency']:.2f}")
    print(f"Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    
    # Save run summary
    summary_path = os.path.join(run_models_dir, "run_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Efficient PnC Hard Gumbel Partitioner Training Summary\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Dataset: {args.dataset_name}\n")
        f.write(f"Split: {args.split} (threshold: {args.split_similarity_threshold})\n")
        f.write(f"Max clusters: {args.max_clusters}\n")
        f.write(f"Cluster size max: {args.cluster_size_max}\n")
        f.write(f"k-hop constraint: {args.k_hop} (enabled: {args.enable_connectivity})\n")
        f.write(f"GCN layers: {args.num_gcn_layers}\n")
        f.write(f"Hidden dimension: {args.nhid}\n")
        f.write(f"Termination threshold: {args.termination_threshold}\n")
        f.write(f"Temperature params: init={args.tau_init}, min={args.tau_min}, decay={args.tau_decay}\n")
        f.write(f"Training epochs: {epochs}\n")
        f.write(f"Learning rate: {lr}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})\n")
        f.write(f"Test accuracy: {test_acc:.4f}\n")
        f.write(f"Final temperature: {test_metrics['temperature']:.4f}\n")
        f.write(f"Average clusters used: {test_metrics['avg_clusters']:.2f}\n")
        f.write(f"Average cluster size: {test_metrics['avg_cluster_size']:.2f}\n")
        f.write(f"Assignment coverage: {test_metrics['avg_assignment_coverage']:.2f}\n")
        f.write(f"Termination efficiency: {test_metrics['termination_efficiency']:.2f}\n")
    
    print(f"Run summary saved to: {summary_path}")
    
    # Log final results
    if use_wandb:
        final_log = {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_temperature": test_metrics['temperature'],
            "partition_stats/test_avg_clusters": test_metrics['avg_clusters'],
            "partition_stats/test_avg_cluster_size": test_metrics['avg_cluster_size'],
            "partition_stats/test_avg_assignment_coverage": test_metrics['avg_assignment_coverage'],
            "partition_stats/test_termination_efficiency": test_metrics['termination_efficiency'],
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch
        }
        
        wandb.log(final_log)
        wandb.config.update({"model_save_path": run_models_dir})
        wandb.finish()
    
    return model



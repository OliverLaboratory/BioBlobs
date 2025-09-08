from tqdm import tqdm
import json
import os
import argparse
from utils.proteinshake_dataset import ProteinClassificationDataset, print_example_data, create_dataloader
from baseline_model import BaselineGVPModel
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
    "--use_wandb",
    action="store_true",
    help="Use Weights & Biases for experiment tracking"
)

args = parser.parse_args()
dataset_name = args.dataset_name
split = args.split
split_similarity_threshold = args.split_similarity_threshold
seed = args.seed
use_wandb = args.use_wandb


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")


def generator_to_structures(generator, dataset_name="enzymecommission", token_map=None):
    """
    Convert generator of proteins to list of structures with name, sequence, and coordinates.
    Missing backbone atoms get infinite coordinates and will be filtered by ProteinGraphDataset.

    Args:
        generator: Generator yielding protein data dictionaries
        dataset_name: Name of the dataset for label extraction
        token_map: Pre-computed mapping from labels to integers (optional)

    Returns:
        tuple: (structures_list, token_map) where structures_list contains dicts with 'name', 'seq', 'coords', 'label' keys
    """
    structures = []
    labels_set = set()
    temp_data = []

    # Track processing statistics
    filtering_stats = {
        "total_processed": 0,
        "successful": 0,
        "partial_residues": 0,
        "zero_length_coords": 0,
    }
    partial_proteins = []

    # First pass: collect all data and extract unique labels (only if token_map not provided)
    print("First pass: collecting data and labels...")
    for protein_data in tqdm(generator, desc="Collecting data"):
        temp_data.append(protein_data)

        if token_map is None:
            protein_info = protein_data["protein"]
            if dataset_name == "enzymecommission":
                label = protein_info["EC"].split(".")[0]
            elif dataset_name == "proteinfamily":
                label = protein_info["Pfam"][0]
            elif dataset_name == "scope":
                label = protein_info["SCOP_class"]

            labels_set.add(label)

    # Create label mapping if not provided
    if token_map is None:
        token_map = {label: i for i, label in enumerate(sorted(list(labels_set)))}
        print(f"Found {len(labels_set)} unique labels: {sorted(list(labels_set))}")
    else:
        print(f"Using provided token map with {len(token_map)} labels")

    # Second pass: process the collected data
    print("Second pass: processing structures...")
    for protein_data in tqdm(temp_data, desc="Converting proteins"):
        filtering_stats["total_processed"] += 1

        # Extract protein information
        protein_info = protein_data["protein"]
        atom_info = protein_data["atom"]

        # Get basic information
        name = protein_info["ID"]
        seq = protein_info["sequence"]

        if dataset_name == "enzymecommission":
            label_key = protein_info["EC"].split(".")[0]
        elif dataset_name == "proteinfamily":
            label_key = protein_info["Pfam"][0]
        elif dataset_name == "scope":
            label_key = protein_info["SCOP_class"]
        
        # Skip if label not in token_map (can happen with split data)
        if label_key not in token_map:
            print(f"Warning: Label {label_key} not found in token_map for protein {name}")
            continue
            
        label = token_map[label_key]

        # Extract atom data
        x_coords = atom_info["x"]
        y_coords = atom_info["y"]
        z_coords = atom_info["z"]
        atom_types = atom_info["atom_type"]
        residue_numbers = atom_info["residue_number"]

        # Group atoms by residue number
        residues = {}
        total_residues = len(set(residue_numbers))

        for i in range(len(x_coords)):
            res_num = residue_numbers[i]
            atom_type = atom_types[i]
            coord = [x_coords[i], y_coords[i], z_coords[i]]

            if res_num not in residues:
                residues[res_num] = {}

            # Take the first occurrence of each backbone atom type per residue
            if (
                atom_type in ["N", "CA", "C", "O"]
                and atom_type not in residues[res_num]
            ):
                residues[res_num][atom_type] = coord

        # Build coords array in residue order
        coords = []
        complete_residues = 0

        for res_num in sorted(residues.keys()):
            backbone_atoms = residues[res_num]
            missing_atoms = [
                atom for atom in ["N", "CA", "C", "O"] if atom not in backbone_atoms
            ]

            # Always create residue coordinates, using inf for missing atoms
            residue_coords = []
            is_complete = True

            for atom_type in ["N", "CA", "C", "O"]:
                if atom_type in backbone_atoms:
                    residue_coords.append(backbone_atoms[atom_type])
                else:
                    residue_coords.append([float("inf"), float("inf"), float("inf")])
                    is_complete = False

            coords.append(residue_coords)

            if is_complete:
                complete_residues += 1

        # Only filter if we have absolutely no coordinates
        if len(coords) == 0:
            filtering_stats["zero_length_coords"] += 1
            continue

        # Track proteins with partial residues for statistics
        completion_rate = (
            complete_residues / total_residues if total_residues > 0 else 0
        )
        if completion_rate < 1.0:
            filtering_stats["partial_residues"] += 1
            partial_proteins.append(
                {
                    "name": name,
                    "total_residues": total_residues,
                    "complete_residues": complete_residues,
                    "completion_rate": completion_rate,
                }
            )

            if completion_rate < 0.1:  # Less than 10% complete - might want to warn
                print(
                    f"WARNING: {name} - Very low completion rate: {complete_residues}/{total_residues} ({completion_rate:.2f})"
                )

        # All proteins are now processed (none filtered)
        filtering_stats["successful"] += 1

        # Truncate sequence to match coords if necessary
        adjusted_seq = seq[: len(coords)] if len(seq) > len(coords) else seq

        structure = {
            "name": name,
            "seq": adjusted_seq,
            "coords": coords,
            "label": label,
        }
        structures.append(structure)

    # Print detailed statistics
    print(f"\n{'=' * 50}")
    print("PROCESSING SUMMARY")
    print(f"{'=' * 50}")
    print(f"Total proteins processed: {filtering_stats['total_processed']}")
    print(f"Successfully converted: {filtering_stats['successful']}")
    print(f"With partial residues: {filtering_stats['partial_residues']}")
    print(
        f"Success rate: {filtering_stats['successful'] / filtering_stats['total_processed']:.3f}"
    )

    if partial_proteins:
        print("\nProteins with incomplete backbone atoms:")
        print(f"{'Protein Name':<15} {'Complete/Total':<15} {'Completion Rate':<15}")
        print("-" * 50)
        for pp in partial_proteins[:10]:  # Show first 10
            comp_total = f"{pp['complete_residues']}/{pp['total_residues']}"
            comp_rate = f"{pp['completion_rate']:.3f}"
            print(f"{pp['name']:<15} {comp_total:<15} {comp_rate:<15}")

        if len(partial_proteins) > 10:
            print(f"... and {len(partial_proteins) - 10} more")

    return structures, token_map


def get_dataset(
    dataset_name, split="structure", split_similarity_threshold=0.7, data_dir="./data"
):
    """
    Get train, validation, and test datasets for the specified protein classification task.
    
    This function splits the data BEFORE converting to structures to preserve correct indices
    and avoid issues with filtering during conversion.

    Args:
        dataset_name (str): Name of the dataset ('enzymecommission', 'proteinfamily', 'scope')
        split (str): Split method ('random', 'sequence', 'structure')
        split_similarity_threshold (float): Similarity threshold for splitting
        data_dir (str): Directory to store/load data files

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, num_classes)
    """
    # Load the appropriate task
    if dataset_name == "enzymecommission":
        from proteinshake.tasks import EnzymeClassTask

        task = EnzymeClassTask(
            split=split, split_similarity_threshold=split_similarity_threshold
        )
        dataset = task.dataset
        print("Number of proteins:", task.size)
        num_classes = task.num_classes
        train_index, val_index, test_index = (
            task.train_index,
            task.val_index,
            task.test_index,
        )

    elif dataset_name == "proteinfamily":
        from proteinshake.tasks import ProteinFamilyTask

        task = ProteinFamilyTask(
            split=split, split_similarity_threshold=split_similarity_threshold
        )
        dataset = task.dataset
        num_classes = task.num_classes
        train_index, val_index, test_index = (
            task.train_index,
            task.val_index,
            task.test_index,
        )

    elif dataset_name == "scope":
        from proteinshake.tasks import StructuralClassTask

        task = StructuralClassTask(
            split=split, split_similarity_threshold=split_similarity_threshold
        )
        dataset = task.dataset
        num_classes = task.num_classes
        train_index, val_index, test_index = (
            task.train_index,
            task.val_index,
            task.test_index,
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Check if JSON files already exist for all splits
    train_json_path = os.path.join(data_dir, f"{dataset_name}_train.json")
    val_json_path = os.path.join(data_dir, f"{dataset_name}_val.json")
    test_json_path = os.path.join(data_dir, f"{dataset_name}_test.json")
    token_map_path = os.path.join(data_dir, f"{dataset_name}_token_map.json")
    
    if (os.path.exists(train_json_path) and os.path.exists(val_json_path) and 
        os.path.exists(test_json_path) and os.path.exists(token_map_path)):
        
        print("JSON files for all splits already exist. Loading from files...")
        
        # Load token map
        with open(token_map_path, "r") as f:
            token_map = json.load(f)
            
        # Load structures for each split
        with open(train_json_path, "r") as f:
            train_structures = json.load(f)
        with open(val_json_path, "r") as f:
            val_structures = json.load(f)
        with open(test_json_path, "r") as f:
            test_structures = json.load(f)
            
        print(f"Loaded {len(train_structures)} train, {len(val_structures)} val, {len(test_structures)} test structures")
        
    else:
        print("Converting proteins to structures with proper splitting...")
        
        # Get the full protein generator
        protein_generator = dataset.proteins(resolution="atom")
        print("Number of atom level proteins:", len(protein_generator))
        
        # Convert generator to list to enable indexing
        all_proteins = list(protein_generator)
        print(f"Loaded {len(all_proteins)} proteins into memory")
        
        # Create generators for each split using indices
        def create_split_generator(protein_list, indices):
            for idx in indices:
                if idx < len(protein_list):
                    yield protein_list[idx]
        
        # First, get token mapping from training data only to ensure consistency
        print("Creating token mapping from training data...")
        train_generator = create_split_generator(all_proteins, train_index)
        _, token_map = generator_to_structures(train_generator, dataset_name=dataset_name, token_map=None)
        
        print(f"Token map created with {len(token_map)} classes: {token_map}")
        
        # Now convert each split using the same token mapping
        print("\nConverting training split...")
        train_generator = create_split_generator(all_proteins, train_index)
        train_structures, _ = generator_to_structures(train_generator, dataset_name=dataset_name, token_map=token_map)
        
        print("\nConverting validation split...")
        val_generator = create_split_generator(all_proteins, val_index)
        val_structures, _ = generator_to_structures(val_generator, dataset_name=dataset_name, token_map=token_map)
        
        print("\nConverting test split...")
        test_generator = create_split_generator(all_proteins, test_index)
        test_structures, _ = generator_to_structures(test_generator, dataset_name=dataset_name, token_map=token_map)
        
        # Save all data to separate JSON files
        print("Saving processed data to JSON files...")
        
        with open(token_map_path, "w") as f:
            json.dump(token_map, f, indent=2)
            
        with open(train_json_path, "w") as f:
            json.dump(train_structures, f, indent=2)
            
        with open(val_json_path, "w") as f:
            json.dump(val_structures, f, indent=2)
            
        with open(test_json_path, "w") as f:
            json.dump(test_structures, f, indent=2)
            
        print(f"Saved {len(train_structures)} train structures to {train_json_path}")
        print(f"Saved {len(val_structures)} val structures to {val_json_path}")
        print(f"Saved {len(test_structures)} test structures to {test_json_path}")
        print(f"Saved token map to {token_map_path}")

    # Verify that we have the correct number of classes
    all_labels = set()
    for structures in [train_structures, val_structures, test_structures]:
        all_labels.update(s["label"] for s in structures)
    
    print(f"Found {len(all_labels)} unique labels in processed data: {sorted(all_labels)}")
    assert len(all_labels) <= num_classes, f"More labels found ({len(all_labels)}) than expected ({num_classes})"

    # Create separate datasets for each split
    train_dataset = ProteinClassificationDataset(
        train_structures, num_classes=num_classes
    )
    val_dataset = ProteinClassificationDataset(val_structures, num_classes=num_classes)
    test_dataset = ProteinClassificationDataset(
        test_structures, num_classes=num_classes
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset, num_classes


def train_baseline_model(model, train_dataset, val_dataset, test_dataset, 
                        epochs=100, lr=1e-3, batch_size=128, num_workers=4,
                        models_dir="./models", device="cuda", use_wandb=True):
    
    # Create timestamp-based model ID and subfolder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"{dataset_name}_{split}_{split_similarity_threshold}_{timestamp}"
    run_models_dir = os.path.join(models_dir, model_id)
    
    # Initialize wandb
    if use_wandb:
        run_name = f"{dataset_name}_{split}_{split_similarity_threshold}"
        wandb.init(
            project="gvp-protein-classification",
            name=run_name,
            config={
                "dataset": dataset_name,
                "split": split,
                "epochs": epochs,
                "lr": lr,
                "model_id": model_id,
                "timestamp": timestamp
            }
        )
    
    train_loader = create_dataloader(train_dataset, batch_size, num_workers, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size, num_workers, shuffle=False)
    test_loader = create_dataloader(test_dataset, batch_size, num_workers, shuffle=False)
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Track top 3 models: [(val_acc, model_path, epoch), ...]
    top_models = []
    os.makedirs(run_models_dir, exist_ok=True)
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Model ID: {model_id}")
    print(f"Models will be saved to: {run_models_dir}")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        print(f"EPOCH {epoch} TRAIN loss: {train_loss:.4f} acc: {train_acc:.4f}")
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        
        print(f"EPOCH {epoch} VAL loss: {val_loss:.4f} acc: {val_acc:.4f}")
        
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
                "val_loss": val_loss,
                "val_acc": val_acc,
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
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        
    print(f"TEST loss: {test_loss:.4f} acc: {test_acc:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    
    # Save run summary
    summary_path = os.path.join(run_models_dir, "run_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Run Summary\n")
        f.write(f"===========\n")
        f.write(f"Model ID: {model_id}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Split: {split}\n")
        f.write(f"Split Threshold: {split_similarity_threshold}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Learning Rate: {lr}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"\nResults:\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.4f} (epoch {best_epoch})\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"\nTop 3 Models:\n")
        for i, (acc, path, ep) in enumerate(top_models):
            f.write(f"  {i+1}. Epoch {ep}: {acc:.4f} - {os.path.basename(path)}\n")
    
    print(f"Run summary saved to: {summary_path}")
    
    # Log final results
    if use_wandb:
        wandb.log({
            "test_loss": test_loss,
            "test_acc": test_acc,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch
        })
        # Log the model path for reference
        wandb.config.update({"model_save_path": run_models_dir})
        wandb.finish()
    
    return model


def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate model on given dataloader.
    """
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc="Evaluating")
    
    for batch in progress_bar:
        # Move batch to device
        batch = batch.to(device)
        
        # Prepare inputs
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        seq = batch.seq if hasattr(batch, 'seq') else None
        
        # Forward pass
        logits = model(h_V, batch.edge_index, h_E, seq=None, batch=batch.batch)
        
        # Compute loss
        loss = criterion(logits, batch.y)
        
        # Statistics
        total_loss += loss.item() * len(batch.y)
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
    
    return avg_loss, avg_acc


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train model for one epoch.
    """
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
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
        logits = model(h_V, batch.edge_index, h_E, seq=None, batch=batch.batch)
        
        # Compute loss
        loss = criterion(logits, batch.y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item() * len(batch.y)
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
    
    return avg_loss, avg_acc



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

    model = BaselineGVPModel(
        node_in_dim = (6,3),
        node_h_dim = (100, 16),   
        edge_in_dim = (32,1),
        edge_h_dim = (32, 1),
        num_classes = num_classes,
        seq_in = False,
        num_layers = 3,
        drop_rate= 0.1,
        pooling="sum"
    )

    print(f"Dataset: {dataset_name}")
    print(f"Split: {split}")
    print(f"Number of classes: {num_classes}")

    trained_model = train_baseline_model(
        model,
        train_dataset,
        val_dataset,
        test_dataset,
        epochs=150,  # Reduced for testing
        lr=1e-4,
        batch_size=128,
        num_workers=4,
        models_dir="./models",
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_wandb=use_wandb
    )


if __name__ == "__main__":
    main()
"""
Create PDB files from ProteinShake datasets.

Usage:
    python create_pdb.py --dataset ec --split structure --subset train
    python create_pdb.py --dataset ec --split structure --subset all
"""

from proteinshake.tasks import EnzymeClassTask, GeneOntologyTask, StructuralClassTask
from utils.visualization import protein_to_pdb
import os
import argparse
from tqdm import tqdm


TASK_MAP = {
    "ec": EnzymeClassTask,
    "go": GeneOntologyTask,
    "scop": StructuralClassTask,
}


def get_split_indices(task, subset):
    """Get protein indices for a specific subset (train/val/test)."""
    if subset == "train":
        return set(task.train_index)
    elif subset == "val":
        return set(task.val_index)
    elif subset == "test":
        return set(task.test_index)
    else:
        return None  # All proteins


def save_proteins(task, save_dir, subset="all"):
    """Save proteins to PDB files for specified subset."""
    dataset = task.dataset
    protein_generator = dataset.proteins(resolution='atom')
    
    # Get indices for subset filtering
    split_indices = get_split_indices(task, subset)
    
    # Create save directory
    if subset != "all":
        subset_save_dir = os.path.join(save_dir, subset)
    else:
        subset_save_dir = save_dir
    os.makedirs(subset_save_dir, exist_ok=True)
    
    print(f"Saving PDB files to: {subset_save_dir}")
    
    saved_count = 0
    for i, protein in enumerate(tqdm(protein_generator, desc=f"Saving PDB files ({subset})")):
        # Filter by subset if specified
        if split_indices is not None and i not in split_indices:
            continue
        
        protein_id = protein['protein']['ID']
        seq_len = len(protein['protein']['sequence'])
        # Use saved_count as local index (starts from 0 for each subset)
        pdb_path = os.path.join(subset_save_dir, f"{saved_count}_{protein_id}_{seq_len}.pdb")
        protein_to_pdb(protein, pdb_path)
        saved_count += 1
    
    print(f"✓ Saved {saved_count} PDB files to {subset_save_dir}")
    return saved_count


def main():
    parser = argparse.ArgumentParser(description="Create PDB files from ProteinShake datasets")
    parser.add_argument("--dataset", type=str, default="ec", choices=["ec", "go", "scop"],
                        help="Dataset name (default: ec)")
    parser.add_argument("--split", type=str, default="structure", choices=["random", "structure"],
                        help="Split type (default: structure)")
    parser.add_argument("--subset", type=str, default="all", choices=["train", "val", "test", "all"],
                        help="Which subset to save (default: all)")
    parser.add_argument("--split_similarity_threshold", type=float, default=0.7,
                        help="Split similarity threshold (default: 0.7)")
    parser.add_argument("--data_path", type=str, 
                        default="/data/oliver_lab/wangx86/bioblobs/proteinshake_data",
                        help="Root data path")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Save directory (default: data_path/pdb/dataset/split)")
    
    args = parser.parse_args()
    
    # Set default save_dir based on dataset and split
    if args.save_dir is None:
        args.save_dir = os.path.join(args.data_path, "pdb", args.dataset, args.split)
    
    print("=" * 60)
    print("CREATE PDB FILES")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Subset: {args.subset}")
    print(f"Data path: {args.data_path}")
    print(f"Save dir: {args.save_dir}")
    print("=" * 60)
    
    # Create task
    task_class = TASK_MAP[args.dataset]
    task = task_class(
        split=args.split,
        split_similarity_threshold=args.split_similarity_threshold,
        root=args.data_path
    )
    
    # Save proteins
    if args.subset == "all":
        # Save all subsets separately
        for subset in ["train", "val", "test"]:
            save_proteins(task, args.save_dir, subset)
    else:
        save_proteins(task, args.save_dir, args.subset)
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()


# # Save all subsets (train/val/test) for EC dataset with structure split
# python create_pdb.py --dataset ec --split structure --subset all

# # Save only training set
# python create_pdb.py --dataset ec --split structure --subset train

# # Save validation set
# python create_pdb.py --dataset ec --split structure --subset val

# # Use different dataset
# python create_pdb.py --dataset go --split random --subset all

# # Custom save directory
# python create_pdb.py --dataset ec --split structure --save_dir /custom/path
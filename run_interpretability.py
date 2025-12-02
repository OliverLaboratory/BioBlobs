"""
Run interpretability analysis on a trained model checkpoint.

Usage:
    python run_interpretability.py \
        checkpoint_path=/path/to/checkpoint.ckpt \
        data.dataset_name=ec \
        data.split=random \
        split_type=test \
        output_dir=./interpretability_results
"""

import os
import argparse
import torch
from pathlib import Path
import json

from utils.proteinshake_dataset import get_dataset, create_dataloader
from utils.interpretability import (
    dataset_inter_results,
    save_interpretability_results,
    print_interpretability_summary,
)


def load_model_from_checkpoint(checkpoint_path, device):
    """
    Load model from checkpoint automatically detecting the model type.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint to inspect hyperparameters
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Try to determine model type from hyperparameters
    if "hyper_parameters" in checkpoint:
        hparams = checkpoint["hyper_parameters"]
        
        # Check for model class hints
        if "multistage_cfg" in hparams or "bypass_codebook" in hparams:
            # Multi-stage or resume model
            if hparams.get("train_cfg", {}).get("dataset_name") == "go":
                from bioblob_resume_lightning import BioBlobsTrainingCodebookMultiLabelModule
                model = BioBlobsTrainingCodebookMultiLabelModule.load_from_checkpoint(checkpoint_path)
                print("Loaded BioBlobsTrainingCodebookMultiLabelModule")
            else:
                from bioblob_resume_lightning import BioBlobsTrainingCodebookModule
                model = BioBlobsTrainingCodebookModule.load_from_checkpoint(checkpoint_path)
                print("Loaded BioBlobsTrainingCodebookModule")
        else:
            # Try standard BioBlobs model
            from train_lightling import BioBlobsLightning
            model = BioBlobsLightning.load_from_checkpoint(checkpoint_path)
            print("Loaded BioBlobsLightning")
    else:
        # Fallback: try BioBlobsLightning
        from train_lightling import BioBlobsLightning
        model = BioBlobsLightning.load_from_checkpoint(checkpoint_path)
        print("Loaded BioBlobsLightning (fallback)")
    
    model = model.to(device)
    model.eval()
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Run interpretability analysis on a checkpoint")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset_name", type=str, default="ec", help="Dataset name (ec, go, scop)")
    parser.add_argument("--split", type=str, default="random", help="Split type (random, structure)")
    parser.add_argument("--split_type", type=str, default="test", choices=["train", "val", "test"], 
                        help="Which dataset split to analyze")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--split_similarity_threshold", type=float, default=30.0, 
                        help="Split similarity threshold")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--max_batches", type=int, default=None, help="Max batches to process (None for all)")
    parser.add_argument("--output_dir", type=str, default="./interpretability_results", 
                        help="Output directory for results")
    parser.add_argument("--edge_types", type=str, default="knn_30", 
                        help="Edge types for graph construction")
    
    args = parser.parse_args()
    
    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("INTERPRETABILITY ANALYSIS")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Split: {args.split}")
    print(f"Split type: {args.split_type}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print("\nLoading dataset...")
    train_dataset, val_dataset, test_dataset, num_classes = get_dataset(
        dataset_name=args.dataset_name,
        split=args.split,
        split_similarity_threshold=args.split_similarity_threshold,
        data_dir=args.data_dir,
        test_mode=False,
        edge_types=args.edge_types,
    )
    
    # Select requested split
    dataset_map = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }
    selected_dataset = dataset_map[args.split_type]
    
    print(f"Dataset size ({args.split_type}): {len(selected_dataset)}")
    print(f"Number of classes: {num_classes}")
    
    # Create dataloader
    dataloader = create_dataloader(
        selected_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    
    # Load model
    print("\nLoading model...")
    model = load_model_from_checkpoint(args.checkpoint_path, device)
    
    # Run interpretability analysis
    print(f"\nRunning interpretability analysis on {args.split_type} set...")
    print(f"Processing {len(dataloader)} batches (max_batches={args.max_batches})")
    
    results = dataset_inter_results(
        model=model,
        dataloader=dataloader,
        device=device,
        max_batches=args.max_batches,
    )
    
    # Save results
    output_filename = f"interpretability_{args.dataset_name}_{args.split}_{args.split_type}.json"
    output_path = output_dir / output_filename
    save_interpretability_results(results, str(output_path))
    
    print(f"\n✓ Results saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print_interpretability_summary(results)
    
    # Save metadata
    metadata = {
        "checkpoint_path": str(args.checkpoint_path),
        "dataset_name": args.dataset_name,
        "split": args.split,
        "split_type": args.split_type,
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "num_classes": num_classes,
        "dataset_size": len(selected_dataset),
        "results_file": output_filename,
    }
    
    metadata_path = output_dir / f"metadata_{args.dataset_name}_{args.split}_{args.split_type}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved to: {metadata_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Standalone script to test ParToken resume training checkpoints.
Usage: python test_partoken_checkpoints.py --checkpoint_path /path/to/checkpoint.ckpt
"""

import os
import argparse
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from utils.proteinshake_dataset import get_dataset, create_dataloader
from utils.interpretability import (
    print_interpretability_summary,
    save_interpretability_results,
    dataset_inter_results,
)
from partoken_resume_lightning import ParTokenResumeTrainingLightning, ParTokenResumeTrainingMultiLabelLightning
import json


def load_checkpoint_config(checkpoint_path):
    """Load configuration from checkpoint hyperparameters."""
    print(f"📋 Loading configuration from checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'hyper_parameters' not in checkpoint:
        raise ValueError("Checkpoint does not contain hyperparameters")
    
    hyper_params = checkpoint['hyper_parameters']
    
    # Extract key configurations
    model_cfg = hyper_params.get('model_cfg', {})
    train_cfg = hyper_params.get('train_cfg', {})
    multistage_cfg = hyper_params.get('multistage_cfg', {})
    
    # Convert to OmegaConf for consistency
    model_cfg = OmegaConf.create(model_cfg) if isinstance(model_cfg, dict) else model_cfg
    train_cfg = OmegaConf.create(train_cfg) if isinstance(train_cfg, dict) else train_cfg
    multistage_cfg = OmegaConf.create(multistage_cfg) if isinstance(multistage_cfg, dict) else multistage_cfg
    
    print(f"✓ Configuration loaded successfully")
    return model_cfg, train_cfg, multistage_cfg


def determine_dataset_from_checkpoint(checkpoint_path):
    """Determine dataset type from checkpoint path or hyperparameters."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Try to get from hyperparameters first
    if 'hyper_parameters' in checkpoint:
        hyper_params = checkpoint['hyper_parameters']
        if hasattr(hyper_params, 'data_cfg'):
            return hyper_params.data_cfg.get('dataset_name', 'unknown')
    
    # Fallback: guess from checkpoint path
    if 'go' in checkpoint_path.lower() or 'geneontology' in checkpoint_path.lower():
        return 'geneontology'
    elif 'ec' in checkpoint_path.lower() or 'enzymecommission' in checkpoint_path.lower():
        return 'enzymecommission'
    elif 'scop' in checkpoint_path.lower():
        return 'scop'
    
    # Default guess
    print("⚠️  Could not determine dataset from checkpoint, defaulting to enzymecommission")
    return 'enzymecommission'


def test_single_checkpoint(checkpoint_path, dataset_name=None, split=None, data_dir=None, batch_size=128):
    """Test a single ParToken checkpoint and return results."""
    
    print(f"🧪 Testing ParToken checkpoint: {os.path.basename(checkpoint_path)}")
    print("=" * 80)
    
    # Determine dataset if not provided
    if dataset_name is None:
        dataset_name = determine_dataset_from_checkpoint(checkpoint_path)
    
    print(f"📊 Dataset: {dataset_name}")
    
    # Set default parameters based on dataset
    if split is None:
        split = 'structure'  # Default split
    
    if data_dir is None:
        data_dir = '/home/wangx86/partoken/partoken-protein/data'
    
    print(f"📁 Data directory: {data_dir}")
    print(f"🔀 Split: {split}")
    
    # Load checkpoint configuration
    try:
        model_cfg, train_cfg, multistage_cfg = load_checkpoint_config(checkpoint_path)
    except Exception as e:
        print(f"❌ Could not load checkpoint config: {e}")
        print("📋 Using default configuration...")
        
        # Create default configurations
        model_cfg = OmegaConf.create({
            'node_in_dim': [6, 3],
            'node_h_dim': [100, 16],
            'edge_in_dim': [32, 1],
            'edge_h_dim': [32, 1],
            'seq_in': False,
            'num_layers': 2 if dataset_name == 'geneontology' else 3,
            'drop_rate': 0.1,
            'pooling': 'sum',
            'max_clusters': 3 if dataset_name == 'geneontology' else 5,
            'nhid': 50,
            'k_hop': 1,
            'cluster_size_max': 15,
            'termination_threshold': 0.95,
            'tau_init': 1.0,
            'tau_min': 0.1,
            'tau_decay': 0.95,
            'codebook_size': 512,
            'codebook_dim': None,
            'codebook_beta': 0.25,
            'codebook_decay': 0.99,
            'codebook_eps': 1e-5,
            'codebook_distance': 'l2',
            'codebook_cosine_normalize': False,
            'lambda_vq': 1.0,
            'lambda_ent': 0.0,
            'lambda_psc': 0.01,
            'lambda_card': 0.005,
            'psc_temp': 0.3
        })
        
        train_cfg = OmegaConf.create({
            'batch_size': batch_size,
            'num_workers': 8,
            'seed': 42
        })
        
        multistage_cfg = OmegaConf.create({
            'stage0': {
                'name': 'joint_training',
                'epochs': 30,
                'lr': 1e-4
            }
        })
    
    # Get dataset
    print(f"\n📚 Loading {dataset_name} dataset...")
    try:
        train_dataset, val_dataset, test_dataset, num_classes = get_dataset(
            dataset_name=dataset_name,
            split=split,
            split_similarity_threshold=0.7,
            data_dir=data_dir,
            test_mode=False,
        )
        print(f"✓ Dataset loaded: {num_classes} classes")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return None
    
    # Create test dataloader
    test_loader = create_dataloader(test_dataset, batch_size, 8, shuffle=False)
    print(f"✓ Test loader created: {len(test_loader)} batches")
    
    # Determine model class based on dataset
    if dataset_name == "geneontology":
        print("🧬 Using multi-label model class")
        model_class = ParTokenResumeTrainingMultiLabelLightning
        metric_type = "multi-label"
    else:
        print("🧬 Using single-label model class")
        model_class = ParTokenResumeTrainingLightning
        metric_type = "single-label"
    
    # Load model from checkpoint
    print(f"\n🔄 Loading model from checkpoint...")
    try:
        model = model_class.load_from_checkpoint(
            checkpoint_path,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            multistage_cfg=multistage_cfg,
            num_classes=num_classes
        )
        print(f"✓ Model loaded successfully")
        print(f"  • Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  • Metric type: {metric_type}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
    # Create trainer
    print(f"\n🏃 Creating trainer...")
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else None,
        logger=False,
        enable_progress_bar=True
    )
    
    # Run test
    print(f"\n🧪 Running test...")
    try:
        test_results = trainer.test(model, test_loader)
        test_result = test_results[0]
        print(f"✓ Test completed successfully")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None
    
    # Extract and display results
    results = {
        'checkpoint_path': checkpoint_path,
        'dataset_name': dataset_name,
        'split': split,
        'num_classes': num_classes,
        'metric_type': metric_type,
        'test_loss': test_result.get('test_loss', 0.0),
        'test_vq_loss': test_result.get('test_vq_loss', 0.0)
    }
    
    print(f"\n📊 TEST RESULTS")
    print("=" * 50)
    print(f"Checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"Dataset: {dataset_name} ({split} split)")
    print(f"Classes: {num_classes}")
    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"VQ Loss: {results['test_vq_loss']:.4f}")
    
    # Add dataset-specific metrics
    if metric_type == "single-label" and 'test_acc' in test_result:
        results['test_accuracy'] = test_result['test_acc']
        results['test_ce_loss'] = test_result.get('test_ce_loss', 0.0)
        print(f"Accuracy: {results['test_accuracy']:.4f}")
        print(f"CE Loss: {results['test_ce_loss']:.4f}")
    elif metric_type == "multi-label" and 'test_fmax' in test_result:
        results['test_fmax'] = test_result['test_fmax']
        results['test_precision'] = test_result.get('test_precision', 0.0)
        results['test_recall'] = test_result.get('test_recall', 0.0)
        results['test_bce_loss'] = test_result.get('test_bce_loss', 0.0)
        print(f"FMax: {results['test_fmax']:.4f}")
        print(f"Precision: {results['test_precision']:.4f}")
        print(f"Recall: {results['test_recall']:.4f}")
        print(f"BCE Loss: {results['test_bce_loss']:.4f}")
    
    # Add VQ-specific metrics if available
    vq_metrics = ['test_codebook_loss', 'test_commitment_loss', 'test_perplexity']
    for metric in vq_metrics:
        if metric in test_result:
            results[metric] = test_result[metric]
            print(f"{metric.replace('test_', '').replace('_', ' ').title()}: {test_result[metric]:.4f}")
    
    # Add importance metrics if available
    if 'test_importance_max' in test_result:
        results['test_importance_max'] = test_result['test_importance_max']
        results['test_importance_entropy'] = test_result.get('test_importance_entropy', 0.0)
        print(f"Max Importance: {results['test_importance_max']:.4f}")
        print(f"Importance Entropy: {results['test_importance_entropy']:.4f}")
    
    print("=" * 50)
    
    # Run interpretability analysis if available
    try:
        print(f"\n🔍 Running interpretability analysis...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        interp_results = dataset_inter_results(
            model=model,
            dataloader=test_loader,
            device=device,
            max_batches=10  # Limit for faster testing
        )
        
        if interp_results is not None:
            print("✓ Interpretability analysis completed")
            print_interpretability_summary(interp_results)
            results['interpretability'] = interp_results['aggregated_stats']
        else:
            print("⚠️  Interpretability analysis returned None")
    except Exception as e:
        print(f"⚠️  Interpretability analysis failed: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Test ParToken resume training checkpoints')
    parser.add_argument('--checkpoint_path', required=True, help='Path to the checkpoint file')
    parser.add_argument('--dataset_name', help='Dataset name (auto-detected if not provided)')
    parser.add_argument('--split', default='structure', help='Dataset split (default: structure)')
    parser.add_argument('--data_dir', default='/home/wangx86/partoken/partoken-protein/data', help='Data directory')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for testing')
    parser.add_argument('--output_file', help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Validate checkpoint path
    if not os.path.exists(args.checkpoint_path):
        print(f"❌ Checkpoint not found: {args.checkpoint_path}")
        return
    
    # Test checkpoint
    results = test_single_checkpoint(
        checkpoint_path=args.checkpoint_path,
        dataset_name=args.dataset_name,
        split=args.split,
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    
    if results is None:
        print("❌ Testing failed")
        return
    
    # Save results if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {args.output_file}")
    
    print(f"\n🎉 Testing completed successfully!")


if __name__ == "__main__":
    main()
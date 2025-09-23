#!/usr/bin/env python3
"""
Test script for PartGNN Lightning modules.

This script tests the PartGNN Lightning modules across all three dataset types:
1. Enzyme Commission (binary classification)
2. Structural Class/SCOP (multi-class classification) 
3. Gene Ontology (multi-label classification)
"""

import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
import tempfile
import warnings

from train_partgnn_lightning import PartGNNLightning, PartGNNMultiLabelLightning
from utils.gnn_dataset import get_gnn_task, get_transformed_graph_dataset, get_data_loaders

warnings.filterwarnings("ignore")


def create_test_config(dataset_name: str, embed_dim: int = 64, max_epochs: int = 2, batch_size: int = 4):
    """Create a test configuration for the given dataset."""
    return DictConfig({
        'data': {
            'dataset_name': dataset_name,
            'data_dir': '/root/ICLR2026/partoken-protein/data',
            'graph_eps': 10.0
        },
        'model': {
            'embed_dim': embed_dim,
            'num_layers': 2,  # Reduced for faster testing
            'drop_rate': 0.1,
            'pooling': 'mean',
            'use_edge_attr': True,
            'edge_attr_dim': 1,
            'pe': 'learned',
            # Partitioner hyperparameters
            'max_clusters': 3,  # Reduced for faster testing
            'nhid': 32,
            'k_hop': 1,
            'cluster_size_max': 10,
            'termination_threshold': 0.95,
            'tau_init': 1.0,
            'tau_min': 0.1,
            'tau_decay': 0.95,
            # Codebook hyperparameters
            'codebook_size': 64,  # Reduced for faster testing
            'codebook_dim': None,
            'codebook_beta': 0.25,
            'codebook_decay': 0.99,
            'codebook_eps': 1e-5,
            'codebook_distance': 'l2',
            'codebook_cosine_normalize': False,
            # Loss weights
            'lambda_vq': 1.0,
            'lambda_ent': 0.0,
            'lambda_psc': 1e-2,
            'lambda_card': 0.005,
            'psc_temp': 0.3
        },
        'train': {
            'max_epochs': max_epochs,
            'batch_size': batch_size,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'num_workers': 0,  # No multiprocessing for testing
            'seed': 42,
            'debug': True
        }
    })


def test_dataset_loading(dataset_name: str):
    """Test loading a specific dataset."""
    print(f"\n📊 Testing {dataset_name} dataset loading...")
    
    try:
        cfg = create_test_config(dataset_name)
        
        # Load dataset
        task = get_gnn_task(cfg.data.dataset_name, root=cfg.data.data_dir)
        dataset = task.dataset
        
        # Transform dataset
        y_transform = None
        if task.task_type[1] == 'regression':
            from sklearn.preprocessing import StandardScaler
            task.compute_targets()
            all_y = task.train_targets
            y_transform = StandardScaler().fit(all_y.reshape(-1, 1))
        
        dataset = get_transformed_graph_dataset(cfg, dataset, task, y_transform)
        
        # Create data loaders
        train_loader, val_loader, test_loader = get_data_loaders(
            dataset, task, cfg.train.batch_size, cfg.train.num_workers
        )
        
        # Get sample batch
        sample_batch = next(iter(train_loader))
        
        # Determine number of classes
        if hasattr(task, 'num_classes'):
            num_classes = task.num_classes
        else:
            if dataset_name.lower() == "gene_ontology":
                num_classes = sample_batch.y.shape[-1] if sample_batch.y.dim() > 1 else len(sample_batch.y.unique())
            else:
                num_classes = len(sample_batch.y.unique())
        
        print("✓ Dataset loaded successfully!")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Sample batch shape: nodes={sample_batch.x.shape[0]}, edges={sample_batch.edge_index.shape[1]}")
        print(f"  Labels shape: {sample_batch.y.shape}")
        print(f"  Number of classes: {num_classes}")
        print(f"  Task type: {task.task_type}")
        
        return train_loader, val_loader, test_loader, task, num_classes
        
    except Exception as e:
        print(f"❌ Failed to load {dataset_name} dataset: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None


def test_model_creation(dataset_name: str, num_classes: int):
    """Test creating the appropriate model for the dataset."""
    print(f"\n🏗️  Testing model creation for {dataset_name}...")
    
    try:
        cfg = create_test_config(dataset_name)
        
        # Create appropriate model
        if dataset_name.lower() == "gene_ontology":
            model = PartGNNMultiLabelLightning(
                model_cfg=cfg.model,
                train_cfg=cfg.train,
                num_classes=num_classes
            )
            model_type = "Multi-Label"
        else:
            model = PartGNNLightning(
                model_cfg=cfg.model,
                train_cfg=cfg.train,
                num_classes=num_classes
            )
            if num_classes == 2:
                model_type = "Binary"
            else:
                model_type = "Multi-Class"
        
        # Check model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print("✓ Model created successfully!")
        print(f"  Model type: {model_type}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Embedding dimension: {model.model.embed_dim}")
        print(f"  Number of layers: {model.model.num_layers}")
        print(f"  Max clusters: {model.model.partitioner.max_clusters}")
        
        return model
        
    except Exception as e:
        print(f"❌ Failed to create model for {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_forward_pass(model, train_loader, dataset_name: str):
    """Test forward pass through the model."""
    print(f"\n🚀 Testing forward pass for {dataset_name}...")
    
    try:
        model.eval()
        
        # Get a batch
        batch = next(iter(train_loader))
        
        # Forward pass
        with torch.no_grad():
            if isinstance(model, PartGNNMultiLabelLightning):
                # Multi-label model
                logits, assignment_matrix, extra = model(batch)
                batch_size = logits.size(0)
                num_classes = logits.size(1)
                
                # Test loss computation
                labels = batch.y
                if labels.dim() == 1:
                    labels = labels.view(batch_size, num_classes)
                labels = labels.float()
                
                # Test metrics
                probs = torch.sigmoid(logits)
                test_acc = ((probs > 0.5) == labels).float().mean()
                
                print("✓ Multi-label forward pass successful!")
                print(f"  Logits shape: {logits.shape}")
                print(f"  Assignment matrix shape: {assignment_matrix.shape}")
                print(f"  Sample accuracy: {test_acc:.3f}")
                
            else:
                # Binary/Multi-class model
                logits, assignment_matrix, extra = model(batch)
                
                # Test loss computation
                labels = batch.y
                
                # Test metrics
                preds = torch.argmax(logits, dim=1)
                test_acc = (preds == labels).float().mean()
                
                print("✓ Classification forward pass successful!")
                print(f"  Logits shape: {logits.shape}")
                print(f"  Assignment matrix shape: {assignment_matrix.shape}")
                print(f"  Sample accuracy: {test_acc:.3f}")
            
            print(f"  Number of clusters used: {extra.get('num_clusters', 0):.2f}")
            
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed for {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step(model, train_loader, dataset_name: str):
    """Test a single training step."""
    print(f"\n🎯 Testing training step for {dataset_name}...")
    
    try:
        model.train()
        
        # Get a batch
        batch = next(iter(train_loader))
        
        # Training step
        loss = model.training_step(batch, 0)
        
        # Check gradients
        loss.backward()
        
        # Check that some parameters have gradients
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        
        print("✓ Training step successful!")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Parameters have gradients: {has_grad}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training step failed for {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lightning_trainer(model, train_loader, val_loader, dataset_name: str):
    """Test the model with PyTorch Lightning trainer."""
    print(f"\n⚡ Testing Lightning trainer for {dataset_name}...")
    
    try:
        # Create a minimal trainer for testing
        with tempfile.TemporaryDirectory():
            trainer = pl.Trainer(
                max_epochs=1,
                limit_train_batches=2,
                limit_val_batches=1,
                enable_checkpointing=False,
                logger=False,
                enable_progress_bar=False,
                accelerator='cpu',
                devices=1
            )
            
            # Test fit
            trainer.fit(model, train_loader, val_loader)
            
            print("✓ Lightning trainer test successful!")
            
        return True
        
    except Exception as e:
        print(f"❌ Lightning trainer test failed for {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset(dataset_name: str):
    """Test a complete pipeline for a specific dataset."""
    print(f"\n{'='*60}")
    print(f"🧬 Testing PartGNN Pipeline for {dataset_name.upper()}")
    print(f"{'='*60}")
    
    results = {
        'dataset_loading': False,
        'model_creation': False,
        'forward_pass': False,
        'training_step': False,
        'lightning_trainer': False
    }
    
    # Test 1: Dataset loading
    train_loader, val_loader, test_loader, task, num_classes = test_dataset_loading(dataset_name)
    if all([train_loader, val_loader, test_loader, task, num_classes]):
        results['dataset_loading'] = True
    else:
        print(f"❌ Skipping remaining tests for {dataset_name} due to dataset loading failure")
        return results
    
    # Test 2: Model creation
    model = test_model_creation(dataset_name, num_classes)
    if model:
        results['model_creation'] = True
    else:
        print(f"❌ Skipping remaining tests for {dataset_name} due to model creation failure")
        return results
    
    # Test 3: Forward pass
    if test_forward_pass(model, train_loader, dataset_name):
        results['forward_pass'] = True
    
    # Test 4: Training step
    if test_training_step(model, train_loader, dataset_name):
        results['training_step'] = True
    
    # Test 5: Lightning trainer
    if test_lightning_trainer(model, train_loader, val_loader, dataset_name):
        results['lightning_trainer'] = True
    
    return results


def main():
    """Run all tests for PartGNN Lightning modules."""
    print("🧪 PartGNN Lightning Module Comprehensive Test Suite")
    print("=" * 80)
    
    # Test datasets
    datasets = [
        'enzyme_commission',    # Binary classification
        'structural_class',     # Multi-class classification  
        'gene_ontology'         # Multi-label classification
    ]
    
    all_results = {}
    
    for dataset in datasets:
        try:
            results = test_dataset(dataset)
            all_results[dataset] = results
        except Exception as e:
            print(f"❌ Unexpected error testing {dataset}: {e}")
            all_results[dataset] = {
                'dataset_loading': False,
                'model_creation': False,
                'forward_pass': False,
                'training_step': False,
                'lightning_trainer': False
            }
    
    # Print summary
    print(f"\n{'='*80}")
    print("📊 TEST SUMMARY")
    print(f"{'='*80}")
    
    for dataset, results in all_results.items():
        print(f"\n🧬 {dataset.upper()}:")
        for test_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    # Overall summary
    total_tests = sum(len(results) for results in all_results.values())
    passed_tests = sum(sum(results.values()) for results in all_results.values())
    
    print("\n🎯 OVERALL RESULTS:")
    print(f"  Total tests: {total_tests}")
    print(f"  Passed: {passed_tests}")
    print(f"  Failed: {total_tests - passed_tests}")
    print(f"  Success rate: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! PartGNN Lightning modules are ready for use.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    main()
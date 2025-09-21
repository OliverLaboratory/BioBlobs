#!/usr/bin/env python3
"""
Test script for GNN Partoken model with graph dataset.

This script loads a graph dataset using the existing pipeline and tests
the new GNN-based Partoken model to ensure compatibility and functionality.
"""

import torch
from utils.gnn_dataset import get_gnn_task, get_transformed_graph_dataset, get_data_loaders
from gnn_partoken import create_optimized_gnn_model
from omegaconf import DictConfig
import warnings
warnings.filterwarnings("ignore")


def test_data_loading():
    """Test loading graph dataset with the existing pipeline."""
    print("🔬 Testing Graph Dataset Loading")
    print("=" * 50)
    
    # Create a simple config for testing
    cfg = DictConfig({
        'data': {
            'dataset_name': 'enzyme_commission',
            'data_dir': '/root/ICLR2026/partoken-protein/data',
            'graph_eps': 10.0
        },
        'train': {
            'batch_size': 4,
            'num_workers': 0
        }
    })
    
    try:
        # Load task and dataset
        print("📁 Loading dataset...")
        task = get_gnn_task(cfg.data.dataset_name, root=cfg.data.data_dir)
        dataset = task.dataset
        
        print(f"✓ Dataset loaded: {len(task.proteins)} samples")
        print(f"✓ Task type: {task.task_type}")
        
        # Transform dataset
        print("🔄 Transforming dataset...")
        y_transform = None
        if task.task_type[1] == 'regression':
            from sklearn.preprocessing import StandardScaler
            task.compute_targets()
            all_y = task.train_targets
            y_transform = StandardScaler().fit(all_y.reshape(-1, 1))
        
        dataset = get_transformed_graph_dataset(cfg, dataset, task, y_transform)
        print("✓ Dataset transformed")
        
        # Create data loaders with small batch size for testing
        train_loader, val_loader, test_loader = get_data_loaders(
            dataset, task,
            cfg.train.batch_size, cfg.train.num_workers
        )
        
        
        # Examine first batch
        print("\n📊 Examining first batch:")
        batch = next(iter(train_loader))
        print(f"  Batch type: {type(batch)}")
        print(f"  Node features shape: {batch.x.shape}")
        print(f"  Edge index shape: {batch.edge_index.shape}")
        print(f"  Edge attr shape: {batch.edge_attr.shape if batch.edge_attr is not None else 'None'}")
        print(f"  Batch shape: {batch.batch.shape}")
        print(f"  Labels shape: {batch.y.shape}")
        print(f"  Residue idx shape: {batch.residue_idx.shape if hasattr(batch, 'residue_idx') else 'None'}")
        
        # Check data ranges
        print(f"  Node feature range: [{batch.x.min().item():.2f}, {batch.x.max().item():.2f}]")
        print(f"  Unique node features: {batch.x.unique().numel()}")
        print(f"  Unique labels: {batch.y.unique().tolist()}")
        
        return train_loader, val_loader, test_loader, task
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def test_model_creation():
    """Test creating the GNN Partoken model."""
    print("\n🏗️  Testing Model Creation")
    print("=" * 50)
    
    try:
        # Test default model creation
        print("Creating default model...")
        model = create_optimized_gnn_model(num_classes=2)
        
        print("✓ Model created successfully")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Test model architecture
        print(f"  Embedding dim: {model.embed_dim}")
        print(f"  Number of layers: {model.num_layers}")
        print(f"  Dropout rate: {model.dropout}")
        print(f"  Pooling method: {model.pooling}")
        print(f"  Use edge attr: {model.use_edge_attr}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_model_forward_pass(model, data_loader):
    """Test forward pass with real data."""
    print("\n🚀 Testing Model Forward Pass")
    print("=" * 50)
    
    if model is None or data_loader is None:
        print("❌ Skipping forward pass test - model or data loader is None")
        return False
    
    try:
        # Get a batch
        batch = next(iter(data_loader))
        
        print("📥 Input batch:")
        print(f"  Batch size: {batch.batch.max().item() + 1}")
        print(f"  Total nodes: {batch.x.size(0)}")
        print(f"  Total edges: {batch.edge_index.size(1)}")
        
        # Test training mode
        print("\n🔥 Testing training mode...")
        model.train()
        
        # Forward pass
        logits, assignment_matrix, extra = model(batch, return_importance=True)
        
        print("✓ Forward pass successful!")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Assignment matrix shape: {assignment_matrix.shape}")
        print(f"  Number of clusters used: {extra.get('num_clusters', 0):.2f}")
        
        # Test loss computation
        labels = batch.y.squeeze() if batch.y.dim() > 1 else batch.y
        total_loss, metrics = model.compute_total_loss(logits, labels, extra)
        
        print("✓ Loss computation successful!")
        print(f"  Total loss: {total_loss.item():.4f}")
        
        print("\n📈 Loss components:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"    {key}: {value:.6f}")
        
        # Test evaluation mode
        print("\n🧪 Testing evaluation mode...")
        model.eval()
        
        with torch.no_grad():
            logits_eval, _, _ = model(batch)
            pred_labels = model.predict(batch)
            pred_probs = model.predict_proba(batch)
            
            print("✓ Evaluation mode successful!")
            print(f"  Predicted labels: {pred_labels.tolist()}")
            print(f"  True labels: {labels.tolist()}")
            print(f"  Max prediction confidence: {pred_probs.max(dim=1)[0].mean():.3f}")
        
        # Test interpretability features
        print("\n🔍 Testing interpretability features...")
        with torch.no_grad():
            importance_scores, cluster_assignments = model.get_cluster_importance(batch)
            node_importance = model.get_node_importance(batch)
            
            print("✓ Interpretability features working!")
            print(f"  Cluster importance shape: {importance_scores.shape}")
            print(f"  Node importance shape: {node_importance.shape}")
            print(f"  Max node importance: {node_importance.max():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step(model, data_loader):
    """Test a simple training step."""
    print("\n🎯 Testing Training Step")
    print("=" * 50)
    
    if model is None or data_loader is None:
        print("❌ Skipping training step test - model or data loader is None")
        return False
    
    try:
        # Set up optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Get a batch
        batch = next(iter(data_loader))
        labels = batch.y.squeeze() if batch.y.dim() > 1 else batch.y
        
        print("📦 Training batch:")
        print(f"  Batch size: {batch.batch.max().item() + 1}")
        print(f"  Labels: {labels.tolist()}")
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        logits, assignment_matrix, extra = model(batch)
        total_loss, metrics = model.compute_total_loss(logits, labels, extra)
        
        # Backward pass
        total_loss.backward()
        
        # Check gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        print("✓ Backward pass successful!")
        print(f"  Gradient norm: {grad_norm:.4f}")
        
        # Optimizer step
        optimizer.step()
        
        # Calculate accuracy
        pred_labels = logits.argmax(dim=1)
        accuracy = (pred_labels == labels).float().mean()
        
        print("✓ Training step completed!")
        print(f"  Loss: {total_loss.item():.4f}")
        print(f"  Accuracy: {accuracy.item():.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in training step: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_epoch_update(model):
    """Test epoch update functionality."""
    print("\n📅 Testing Epoch Update")
    print("=" * 50)
    
    if model is None:
        print("❌ Skipping epoch update test - model is None")
        return False
    
    try:
        # Test epoch update
        print("Calling model.update_epoch()...")
        model.update_epoch()
        print("✓ Epoch update successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in epoch update: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🧬 GNN Partoken Model Integration Test")
    print("=" * 60)
    
    # Test 1: Data loading
    train_loader, val_loader, test_loader, task = test_data_loading()
    
    # Test 2: Model creation
    if task is not None:
        # Determine number of classes from task
        if hasattr(task, 'num_classes'):
            num_classes = task.num_classes
        else:
            # Infer from data
            sample_batch = next(iter(train_loader))
            num_classes = len(sample_batch.y.unique())
        
        print(f"\n🎯 Creating model for {num_classes} classes")
        model = create_optimized_gnn_model(
            num_classes=num_classes,
            embed_dim=128,
            num_layers=3,
            use_edge_attr=True,
            pe='learned'
        )
    else:
        model = test_model_creation()
    
    # Test 3: Forward pass
    success_forward = test_model_forward_pass(model, train_loader)
    
    # Test 4: Training step
    success_training = test_training_step(model, train_loader)
    
    # Test 5: Epoch update
    success_epoch = test_epoch_update(model)
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 60)
    print(f"✓ Data loading: {'PASS' if train_loader is not None else 'FAIL'}")
    print(f"✓ Model creation: {'PASS' if model is not None else 'FAIL'}")
    print(f"✓ Forward pass: {'PASS' if success_forward else 'FAIL'}")
    print(f"✓ Training step: {'PASS' if success_training else 'FAIL'}")
    print(f"✓ Epoch update: {'PASS' if success_epoch else 'FAIL'}")
    
    all_passed = all([
        train_loader is not None,
        model is not None,
        success_forward,
        success_training,
        success_epoch
    ])
    
    if all_passed:
        print("\n🎉 All tests PASSED! GNN Partoken model is ready for use.")
    else:
        print("\n⚠️  Some tests FAILED. Please check the errors above.")
    
    return model, train_loader, val_loader, test_loader


if __name__ == "__main__":
    main()
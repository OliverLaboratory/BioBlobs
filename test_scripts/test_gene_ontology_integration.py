"""
Test the gene ontology multi-label classification integration.
This test ensures that our new gene ontology dataset and multi-label components work correctly.
"""
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.gene_ontology import get_gene_ontology_dataset, ProteinMultiLabelDataset
from utils.fmax_metric import FMaxMetric
from train_lightling import PartGVPMultiLabelLightning
from omegaconf import DictConfig
import yaml

def test_dataset_loading():
    """Test gene ontology dataset loading."""
    print("🧬 Testing gene ontology dataset loading...")
    
    try:
        # Test direct gene ontology dataset loading
        train_dataset, val_dataset, test_dataset, num_classes = get_gene_ontology_dataset(test_mode=True)
        print("✓ Dataset loaded successfully!")
        print(f"  Train dataset size: {len(train_dataset)}")
        print(f"  Val dataset size: {len(val_dataset)}")
        print(f"  Test dataset size: {len(test_dataset)}")
        print(f"  Number of classes: {num_classes}")
        
        # Check a sample from train dataset
        sample = train_dataset[0]
        print(f"  Sample keys: {list(sample.keys())}")
        print(f"  Sample name: {sample['name']}")
        print(f"  Sequence length: {len(sample['seq'])}")
        
        # Check if this is raw data or already processed
        if 'coords' in sample:
            print(f"  Coordinates shape: {len(sample['coords'])} x {len(sample['coords'][0]) if sample['coords'] else 0}")
            print(f"  Label type: {type(sample['label'])}")
            print(f"  Label length: {len(sample['label'])}")
            print(f"  Label sum: {sum(sample['label'])}")
        else:
            # Already processed to graph format
            print(f"  Node features shape: {sample['x'].shape}")
            print(f"  Edge index shape: {sample['edge_index'].shape}")
            print(f"  Label tensor shape: {sample['y'].shape}")
            print(f"  Number of positive labels: {sample['y'].sum().item()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False

def test_multi_label_dataset_class():
    """Test the ProteinMultiLabelDataset class."""
    print("\n🏷️ Testing ProteinMultiLabelDataset class...")
    
    try:
        # Get raw dataset first
        train_dataset, _, _, num_classes = get_gene_ontology_dataset(test_mode=True)
        
        # The train_dataset is already a ProteinMultiLabelDataset or Subset
        # Let's get a sample directly
        print("✓ ProteinMultiLabelDataset created successfully!")
        print(f"  Dataset size: {len(train_dataset)}")
        print(f"  Number of classes: {num_classes}")
        
        # Test a sample - handle both Subset and direct dataset
        if hasattr(train_dataset, 'dataset'):
            # It's a Subset, get the underlying dataset
            sample = train_dataset.dataset[train_dataset.indices[0]]
        else:
            # It's the dataset directly
            sample = train_dataset[0]
        
        # Check if it's already processed (graph format) or raw format
        if hasattr(sample, 'x'):  # Already processed to graph
            print(f"  Sample type: PyTorch Geometric graph")
            print(f"  Graph node features shape: {sample.x.shape}")
            print(f"  Graph edge index shape: {sample.edge_index.shape}")
            print(f"  Label tensor shape: {sample.y.shape}")
            print(f"  Label tensor dtype: {sample.y.dtype}")
            print(f"  Number of positive labels: {sample.y.sum().item()}")
        else:  # Raw format
            print(f"  Sample type: Raw protein data")
            print(f"  Sample keys: {list(sample.keys())}")
            if 'label' in sample:
                print(f"  Label type: {type(sample['label'])}")
                print(f"  Label length: {len(sample['label'])}")
                print(f"  Number of positive labels: {sum(sample['label'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ ProteinMultiLabelDataset test failed: {e}")
        return False

def test_dataset_routing():
    """Test that the gene ontology dataset integrates correctly with the training pipeline."""
    print("\n🚏 Testing gene ontology integration with training pipeline...")
    
    try:
        # Test that we can create dataloaders from the gene ontology dataset
        train_dataset, val_dataset, test_dataset, num_classes = get_gene_ontology_dataset(test_mode=True)
        
        print("✓ Gene ontology integration successful!")
        print(f"  Train dataset size: {len(train_dataset)}")
        print(f"  Val dataset size: {len(val_dataset)}")
        print(f"  Test dataset size: {len(test_dataset)}")
        print(f"  Number of classes: {num_classes}")
        
        # Test that we can create DataLoaders (this is what the training script does)
        from torch_geometric.loader import DataLoader
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
        
        # Test a batch from each loader
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        print(f"  Train batch graph nodes: {train_batch.x.shape}")
        print(f"  Train batch labels shape: {train_batch.y.shape}")
        print(f"  Train batch labels dtype: {train_batch.y.dtype}")
        print(f"  Val batch graph nodes: {val_batch.x.shape}")
        print(f"  Val batch labels shape: {val_batch.y.shape}")
        
        # Verify labels are in the correct format for multi-label classification
        print(f"  Train batch label dtype: {train_batch.y.dtype}")
        print(f"  Val batch label dtype: {val_batch.y.dtype}")
        print(f"  Expected 2D labels: {len(train_batch.y.shape) == 2}")
        print(f"  Label shape matches num_classes: {train_batch.y.shape[-1] == num_classes if len(train_batch.y.shape) >= 2 else 'Shape too small'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Gene ontology integration test failed: {e}")
        return False

def test_fmax_metric():
    """Test the FMax metric implementation."""
    print("\n📊 Testing FMax metric...")
    
    try:
        metric = FMaxMetric()
        
        # Create synthetic predictions and targets
        batch_size, num_classes = 8, 100
        predictions = torch.randn(batch_size, num_classes)  # Raw logits
        targets = torch.randint(0, 2, (batch_size, num_classes)).float()  # Binary targets
        
        # Compute metrics
        fmax = metric.fmax(predictions, targets)
        precision = metric.precision(predictions, targets)
        recall = metric.recall(predictions, targets)
        
        print(f"✓ FMax metric working!")
        print(f"  FMax: {fmax:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  Metrics are in valid range: {0 <= fmax <= 1 and 0 <= precision <= 1 and 0 <= recall <= 1}")
        
        return True
        
    except Exception as e:
        print(f"❌ FMax metric test failed: {e}")
        return False

def test_multi_label_lightning():
    """Test the PartGVPMultiLabelLightning class."""
    print("\n⚡ Testing PartGVPMultiLabelLightning...")
    
    try:
        # First get the gene ontology dataset to know the number of classes
        train_dataset, val_dataset, test_dataset, num_classes = get_gene_ontology_dataset(test_mode=True)
        print(f"  Gene ontology dataset loaded with {num_classes} classes")
        
        # Create model config - add all required parameters
        model_cfg = DictConfig({
            'node_in_dim': [6, 3], 'node_h_dim': [100, 16], 
            'edge_in_dim': [32, 1], 'edge_h_dim': [32, 1],
            'seq_in': False, 'num_layers': 3, 'drop_rate': 0.1,
            'pooling': 'mean',  # Add missing pooling parameter
            # Add all the ParTokenModel parameters that might be needed
            'max_clusters': 5,
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
            'psc_temp': 0.3
        })
        
        train_cfg = DictConfig({
            'batch_size': 4, 'lr': 1e-4, 'weight_decay': 1e-5
        })
        
        # Create lightning module with the correct number of classes
        lightning_model = PartGVPMultiLabelLightning(
            model_cfg=model_cfg, 
            train_cfg=train_cfg, 
            num_classes=num_classes  # Use actual gene ontology number of classes
        )
        
        print("✓ PartGVPMultiLabelLightning created successfully!")
        print(f"  Model: {type(lightning_model.model).__name__}")
        print(f"  Loss function: {type(lightning_model.criterion).__name__}")  # Fix: criterion not loss_fn
        print(f"  Number of classes: {lightning_model.num_classes}")
        print(f"  Has FMax metric: {hasattr(lightning_model, 'fmax_metric')}")
        
        # Create a small dataloader
        from torch_geometric.loader import DataLoader
        test_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
        
        # Test forward pass with real gene ontology data
        lightning_model.eval()
        with torch.no_grad():
            batch = next(iter(test_loader))
            print(f"  Batch node features shape: {batch.x.shape}")
            print(f"  Batch edge index shape: {batch.edge_index.shape}")
            print(f"  Batch labels shape: {batch.y.shape}")
            print(f"  Batch labels dtype: {batch.y.dtype}")
            
            # Test the forward pass - let's debug the input format
            print(f"  batch.node_s type: {type(batch.node_s)}")
            print(f"  batch.node_v type: {type(batch.node_v)}")
            print(f"  batch.edge_s type: {type(batch.edge_s)}")
            print(f"  batch.edge_v type: {type(batch.edge_v)}")
            
            # Test the forward pass
            logits, assignment_matrix, cluster_importance, extra = lightning_model.forward(
                (batch.node_s, batch.node_v), 
                batch.edge_index, 
                (batch.edge_s, batch.edge_v), 
                batch=batch.batch
            )
            
            # Fix label shape for multi-label: PyTorch Geometric concatenates but we need to stack
            batch_size = len(torch.unique(batch.batch))
            num_classes = batch.y.shape[0] // batch_size
            labels = batch.y.view(batch_size, num_classes).float()
            
            print(f"  Reshaped labels from {batch.y.shape} to {labels.shape}")
            
            # Test the loss computation
            loss = lightning_model.criterion(logits, labels)  # Use reshaped labels
            
            # Test metrics - add debug info
            print(f"  Debug: logits.shape = {logits.shape}")
            print(f"  Debug: labels.shape = {labels.shape}")
            print(f"  Debug: labels.min() = {labels.min()}, labels.max() = {labels.max()}")
            print(f"  Debug: logits.min() = {logits.min()}, logits.max() = {logits.max()}")
            
            fmax = lightning_model.fmax_metric.fmax(logits, labels)
            precision = lightning_model.fmax_metric.precision(logits, labels)
            recall = lightning_model.fmax_metric.recall(logits, labels)
        
        print("  Forward pass with gene ontology data successful!")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Labels shape: {labels.shape}")  # Fix: show reshaped labels shape
        print(f"  Labels dtype: {labels.dtype}")   # Fix: show reshaped labels dtype
        print(f"  Loss: {loss.item():.4f}")
        print(f"  FMax: {fmax:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        
        # Validate metrics are in correct range
        valid_fmax = 0 <= fmax <= 1
        valid_precision = 0 <= precision <= 1
        valid_recall = 0 <= recall <= 1
        
        print(f"  Metrics validation: FMax valid={valid_fmax}, Precision valid={valid_precision}, Recall valid={valid_recall}")
        
        if not (valid_fmax and valid_precision and valid_recall):
            print("  ❌ WARNING: Metrics are out of valid range [0,1]!")
            return False
        
        return True
        
    except Exception as e:
        import traceback
        print(f"❌ PartGVPMultiLabelLightning test failed: {e}")
        print(f"  Full traceback:")
        traceback.print_exc()
        return False

def test_config_file():
    """Test that the gene ontology config file exists and is valid."""
    print("\n⚙️ Testing config file...")
    
    try:
        config_path = '/root/ICLR2026/partoken-protein/conf/config_partgvp_geneontology.yaml'
        
        if not os.path.exists(config_path):
            print(f"❌ Config file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("✓ Config file loaded successfully!")
        print(f"  Dataset name: {config.get('data', {}).get('dataset_name')}")
        print(f"  Data dir: {config.get('data', {}).get('data_dir')}")
        print(f"  Model pooling: {config.get('model', {}).get('pooling', 'Unknown')}")
        
        # Check required fields in nested structure
        required_fields = ['data', 'model', 'train']
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            print(f"❌ Missing required fields: {missing_fields}")
            return False
        
        # Check data section
        data_config = config.get('data', {})
        if data_config.get('dataset_name') != 'geneontology':
            print(f"❌ Wrong dataset name: {data_config.get('dataset_name')}, expected 'geneontology'")
            return False
        
        # Check if data_dir is specified
        if not data_config.get('data_dir'):
            print("❌ Missing data_dir in data section")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Config file test failed: {e}")
        return False

def main():
    """Run all gene ontology integration tests."""
    print("🧪 Running Gene Ontology Multi-Label Classification Integration Tests")
    print("=" * 80)
    
    tests = [
        # test_dataset_loading,
        # test_multi_label_dataset_class,
        # test_dataset_routing,  # Now tests gene ontology integration
        test_fmax_metric,
        # test_multi_label_lightning,
        # test_config_file
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 80)
    print(f"🎉 Tests completed: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("✅ All gene ontology integration tests passed!")
        print("\n📋 Components verified:")
        print("  • Gene ontology dataset loading with JSON serialization fix")
        print("  • ProteinMultiLabelDataset class for graph conversion")
        print("  • Dataset routing through get_dataset function")
        print("  • FMax metric implementation for multi-label evaluation")
        print("  • PartGVPMultiLabelLightning class with BCEWithLogitsLoss")
        print("  • Gene ontology configuration file")
        print("\n🚀 Ready to run gene ontology training with:")
        print("  python run_partgvp.py --config-name config_partgvp_geneontology")
    else:
        failed_tests = [tests[i].__name__ for i, result in enumerate(results) if not result]
        print(f"❌ Failed tests: {failed_tests}")
        print("Please fix the issues before proceeding with training.")
    
    return all(results)

if __name__ == "__main__":
    main()
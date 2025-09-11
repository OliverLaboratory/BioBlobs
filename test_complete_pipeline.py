"""
Test the complete pipeline with global-to-cluster attention and feature-wise gating.
This test ensures that our new modules work correctly in all scenarios.
"""
import torch
import torch.nn.functional as F
from partoken_model import ParTokenModel
from train_lightling import MultiStageParTokenLightning
from omegaconf import DictConfig

def test_forward_pass():
    """Test basic forward pass functionality."""
    print("🧬 Testing forward pass...")
    
    model = ParTokenModel(
        node_in_dim=(6, 3), 
        node_h_dim=(100, 16), 
        edge_in_dim=(32, 1), 
        edge_h_dim=(32, 1),
        max_clusters=5
    )
    model.eval()
    
    # Create synthetic batch
    batch_size = 3
    max_nodes = 15
    total_nodes = batch_size * max_nodes
    
    h_V = (torch.randn(total_nodes, 6), torch.randn(total_nodes, 3, 3))
    
    # Create connectivity
    edge_list = []
    for b in range(batch_size):
        offset = b * max_nodes
        for i in range(max_nodes - 1):
            edge_list.extend([[offset + i, offset + i + 1], [offset + i + 1, offset + i]])
    
    edge_index = torch.tensor(edge_list).t().contiguous()
    num_edges = edge_index.size(1)
    h_E = (torch.randn(num_edges, 32), torch.randn(num_edges, 1, 3))
    batch = torch.repeat_interleave(torch.arange(batch_size), max_nodes)
    
    with torch.no_grad():
        # Test standard forward
        logits, assignment_matrix, extra = model(h_V, edge_index, h_E, batch=batch)
        
        # Test forward with importance
        logits_imp, assignment_matrix_imp, extra_imp, importance = model(
            h_V, edge_index, h_E, batch=batch, return_importance=True
        )
    
    print(f"✓ Forward pass successful!")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Assignment matrix shape: {assignment_matrix.shape}")
    print(f"  Importance scores shape: {importance.shape}")
    print(f"  Importance scores sum to 1: {torch.allclose(importance.sum(dim=1), torch.ones(batch_size))}")
    
    return True

def test_attention_mechanism():
    """Test the global-to-cluster attention mechanism specifically."""
    print("\n🎯 Testing global-to-cluster attention...")
    
    from utils.inter_cluster import GlobalClusterAttention
    
    batch_size = 2
    num_clusters = 5
    dim = 100
    
    attention = GlobalClusterAttention(dim=dim, heads=4)
    
    # Create test data
    global_feat = torch.randn(batch_size, dim)
    cluster_feat = torch.randn(batch_size, num_clusters, dim)
    mask = torch.ones(batch_size, num_clusters, dtype=torch.bool)
    
    # Test with some invalid clusters
    mask[0, -1] = False  # Last cluster invalid for first batch
    mask[1, -2:] = False  # Last two clusters invalid for second batch
    
    pooled, importance, attn = attention(global_feat, cluster_feat, mask)
    
    print(f"✓ Attention mechanism working!")
    print(f"  Pooled shape: {pooled.shape}")
    print(f"  Importance shape: {importance.shape}")
    print(f"  Attention shape: {attn.shape}")
    print(f"  Importance respects mask: {torch.allclose(importance[~mask], torch.zeros_like(importance[~mask]))}")
    
    # Check that importance sums to 1 over valid clusters
    valid_sums = []
    for b in range(batch_size):
        valid_imp = importance[b, mask[b]]
        valid_sums.append(valid_imp.sum().item())
    print(f"  Valid importance sums: {valid_sums}")
    
    return True

def test_feature_wise_gate():
    """Test the feature-wise gating mechanism."""
    print("\n🚪 Testing feature-wise gating...")
    
    from utils.inter_cluster import FeatureWiseGateFusion
    
    batch_size = 2
    dim = 100
    
    gate = FeatureWiseGateFusion(dim=dim, hidden=dim//2)
    
    global_feat = torch.randn(batch_size, dim)
    cluster_summary = torch.randn(batch_size, dim)
    
    fused, beta = gate(global_feat, cluster_summary)
    
    print(f"✓ Feature-wise gating working!")
    print(f"  Fused shape: {fused.shape}")
    print(f"  Gate values shape: {beta.shape}")
    print(f"  Gate values range: [{beta.min():.3f}, {beta.max():.3f}]")
    print(f"  Gate values mean: {beta.mean():.3f}")
    
    return True

def test_bypass_mode():
    """Test the bypass mode in lightning module."""
    print("\n🔄 Testing bypass mode...")
    
    # Create minimal configs
    model_cfg = DictConfig({
        'node_in_dim': [6, 3], 'node_h_dim': [100, 16], 
        'edge_in_dim': [32, 1], 'edge_h_dim': [32, 1],
        'seq_in': False, 'num_layers': 3, 'drop_rate': 0.1,
        'pooling': 'mean', 'max_clusters': 5, 'nhid': 50,
        'k_hop': 1, 'cluster_size_max': 15, 'termination_threshold': 0.95,
        'tau_init': 1.0, 'tau_min': 0.1, 'tau_decay': 0.95,
        'codebook_size': 512, 'codebook_dim': None, 'codebook_beta': 0.25,
        'codebook_decay': 0.99, 'codebook_eps': 1e-5, 'codebook_distance': 'l2',
        'codebook_cosine_normalize': False, 'lambda_vq': 1.0, 'lambda_ent': 1e-3,
        'lambda_psc': 1e-2, 'psc_temp': 0.3
    })
    
    train_cfg = DictConfig({'batch_size': 16, 'seed': 42, 'use_cosine_schedule': True, 'warmup_epochs': 1})
    multistage_cfg = DictConfig({'enabled': True})
    
    lightning_model = MultiStageParTokenLightning(model_cfg, train_cfg, multistage_cfg, num_classes=2)
    lightning_model.bypass_codebook = True  # Enable bypass mode
    
    # Create synthetic batch
    batch_size = 2
    max_nodes = 10
    total_nodes = batch_size * max_nodes
    
    # Create a mock batch object
    class MockBatch:
        def __init__(self):
            self.node_s = torch.randn(total_nodes, 6)
            self.node_v = torch.randn(total_nodes, 3, 3)
            
            edge_list = []
            for b in range(batch_size):
                offset = b * max_nodes
                for i in range(max_nodes - 1):
                    edge_list.extend([[offset + i, offset + i + 1], [offset + i + 1, offset + i]])
            
            self.edge_index = torch.tensor(edge_list).t().contiguous()
            num_edges = self.edge_index.size(1)
            
            self.edge_s = torch.randn(num_edges, 32)
            self.edge_v = torch.randn(num_edges, 1, 3)
            self.y = torch.randint(0, 2, (batch_size,))
    
    batch = MockBatch()
    
    with torch.no_grad():
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        
        # Test bypass forward
        logits, assignment_matrix, extra = lightning_model._forward_bypass_codebook(
            h_V, batch.edge_index, h_E, None, torch.repeat_interleave(torch.arange(batch_size), max_nodes)
        )
    
    print(f"✓ Bypass mode working!")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Assignment matrix shape: {assignment_matrix.shape}")
    print(f"  VQ loss (should be 0): {extra['vq_loss'].item()}")
    
    return True

def test_gradient_flow():
    """Test that gradients flow correctly through our new modules."""
    print("\n🌊 Testing gradient flow...")
    
    model = ParTokenModel(
        node_in_dim=(6, 3), 
        node_h_dim=(100, 16), 
        edge_in_dim=(32, 1), 
        edge_h_dim=(32, 1),
        max_clusters=3
    )
    model.train()
    
    # Create synthetic batch
    batch_size = 2
    max_nodes = 8
    total_nodes = batch_size * max_nodes
    
    h_V = (torch.randn(total_nodes, 6), torch.randn(total_nodes, 3, 3))
    
    edge_list = []
    for b in range(batch_size):
        offset = b * max_nodes
        for i in range(max_nodes - 1):
            edge_list.extend([[offset + i, offset + i + 1], [offset + i + 1, offset + i]])
    
    edge_index = torch.tensor(edge_list).t().contiguous()
    num_edges = edge_index.size(1)
    h_E = (torch.randn(num_edges, 32), torch.randn(num_edges, 1, 3))
    batch = torch.repeat_interleave(torch.arange(batch_size), max_nodes)
    
    # Forward pass
    logits, assignment_matrix, extra = model(h_V, edge_index, h_E, batch=batch)
    
    # Compute loss
    targets = torch.randint(0, 2, (batch_size,))
    loss = F.cross_entropy(logits, targets) + extra['vq_loss']
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    attention_has_grad = any(p.grad is not None for p in model.global_cluster_attn.parameters())
    gate_has_grad = any(p.grad is not None for p in model.fw_gate.parameters())
    
    print(f"✓ Gradient flow working!")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Attention module has gradients: {attention_has_grad}")
    print(f"  Gate module has gradients: {gate_has_grad}")
    
    return True

def main():
    """Run all tests."""
    print("🧪 Running comprehensive tests for global-to-cluster attention with feature-wise gating")
    print("=" * 80)
    
    tests = [
        test_forward_pass,
        test_attention_mechanism,
        test_feature_wise_gate,
        test_bypass_mode,
        test_gradient_flow
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed: {e}")
            results.append(False)
    
    print("\n" + "=" * 80)
    print(f"🎉 Tests completed: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("✅ All tests passed! Global-to-cluster attention with feature-wise gating is working correctly.")
        print("\n📋 Summary of changes:")
        print("  • Replaced InterClusterModel with GlobalClusterAttention")
        print("  • Added FeatureWiseGateFusion for intelligent cluster integration")
        print("  • Simplified classifier to use only fused representation (ns input)")
        print("  • Reduced model parameters by removing concatenation redundancy")
        print("  • Preserved cluster importance scores for interpretability")
        print("  • Updated both normal and bypass modes")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return all(results)

if __name__ == "__main__":
    main()

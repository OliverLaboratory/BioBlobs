import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse
from part_model_PnC import GVPHardGumbelPartitionerModel

# Import your model (adjust import path as needed)
# from your_model_file import GVPHardGumbelPartitionerModel

def create_dummy_protein_data(num_proteins=2, nodes_per_protein=[20, 15], device='cpu'):
    """
    Create dummy protein data in GVP format for testing
    
    Args:
        num_proteins: Number of proteins in batch
        nodes_per_protein: List of number of residues per protein
        device: Device to create tensors on
        
    Returns:
        h_V: Node features (scalars, vectors)
        edge_index: Edge connectivity 
        h_E: Edge features (scalars, vectors)
        seq: Sequence features
        batch: Batch assignment
        labels: Dummy classification labels
    """
    
    total_nodes = sum(nodes_per_protein)
    
    # Create node features (h_V)
    # GVP format: (scalar_features, vector_features)
    node_scalars = torch.randn(total_nodes, 6, device=device)  # [N, 6] - typical protein node features
    node_vectors = torch.randn(total_nodes, 3, 3, device=device)  # [N, 3, 3] - 3D vector features
    h_V = (node_scalars, node_vectors)
    
    # Create batch assignment
    batch = torch.cat([torch.full((n,), i, dtype=torch.long, device=device) 
                      for i, n in enumerate(nodes_per_protein)])
    
    # Create edges (connect each node to its neighbors + some random long-range connections)
    edge_indices = []
    edge_features_scalars = []
    edge_features_vectors = []
    
    node_offset = 0
    for i, num_nodes in enumerate(nodes_per_protein):
        # Local connectivity (each residue connected to neighbors)
        for j in range(num_nodes - 1):
            # Forward edge
            edge_indices.append([node_offset + j, node_offset + j + 1])
            edge_features_scalars.append(torch.randn(32))  # Edge scalar features
            edge_features_vectors.append(torch.randn(1, 3))  # Edge vector features
            
            # Backward edge (undirected graph)
            edge_indices.append([node_offset + j + 1, node_offset + j])
            edge_features_scalars.append(torch.randn(32))
            edge_features_vectors.append(torch.randn(1, 3))
        
        # Add some random long-range connections
        num_long_range = min(5, num_nodes // 2)
        for _ in range(num_long_range):
            src = np.random.randint(0, num_nodes)
            dst = np.random.randint(0, num_nodes)
            if src != dst:
                edge_indices.append([node_offset + src, node_offset + dst])
                edge_features_scalars.append(torch.randn(32))
                edge_features_vectors.append(torch.randn(1, 3))
        
        node_offset += num_nodes
    
    # Convert to tensors
    edge_index = torch.tensor(edge_indices, dtype=torch.long, device=device).t()  # [2, E]
    edge_scalars = torch.stack(edge_features_scalars).to(device)  # [E, 32]
    edge_vectors = torch.stack(edge_features_vectors).to(device)  # [E, 1, 3]
    h_E = (edge_scalars, edge_vectors)
    
    # Create sequence features (amino acid indices)
    seq = torch.randint(0, 20, (total_nodes,), device=device)  # [N] - amino acid indices
    
    # Create dummy labels
    labels = torch.randint(0, 2, (num_proteins,), device=device)  # [B] - binary classification
    
    return h_V, edge_index, h_E, seq, batch, labels


def test_model_creation():
    """Test model instantiation"""
    print("Testing model creation...")
    
    model = GVPHardGumbelPartitionerModel(
        node_in_dim=(6, 3),      # Input: 6 scalars, 3 vectors
        node_h_dim=(100, 16),    # Hidden: 100 scalars, 16 vectors  
        edge_in_dim=(32, 1),     # Edge input: 32 scalars, 1 vector
        edge_h_dim=(32, 1),      # Edge hidden: 32 scalars, 1 vector
        num_classes=2,           # Binary classification
        seq_in=True,             # Use sequence features
        num_layers=3,            # 3 GVP layers
        drop_rate=0.1,           # 10% dropout
        pooling='mean',          # Mean pooling
        max_clusters=5           # Up to 5 clusters
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model created successfully")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    return model


def test_forward_pass(model, device='cpu'):
    """Test forward pass with dummy data"""
    print("Testing forward pass...")
    
    # Create dummy data
    h_V, edge_index, h_E, seq, batch, labels = create_dummy_protein_data(
        num_proteins=3, 
        nodes_per_protein=[25, 18, 30],
        device=device
    )
    
    print(f"  - Input shapes:")
    print(f"    Node scalars: {h_V[0].shape}")
    print(f"    Node vectors: {h_V[1].shape}")
    print(f"    Edge index: {edge_index.shape}")
    print(f"    Edge scalars: {h_E[0].shape}")
    print(f"    Edge vectors: {h_E[1].shape}")
    print(f"    Sequence: {seq.shape}")
    print(f"    Batch: {batch.shape}")
    print(f"    Labels: {labels.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits, assignment_matrix = model(h_V, edge_index, h_E, seq, batch)
    
    batch_size = len(torch.unique(batch))
    max_nodes = max((batch == i).sum().item() for i in range(batch_size))
    
    print(f"  - Output shapes:")
    print(f"    Logits: {logits.shape}")
    print(f"    Assignment matrix: {assignment_matrix.shape}")
    
    # Check output shapes
    expected_logits_shape = (batch_size, 2)  # [B, num_classes]
    expected_assignment_shape = (batch_size, max_nodes, model.partitioner.max_clusters)  # [B, max_N, max_clusters]
    
    assert logits.shape == expected_logits_shape, f"Expected logits {expected_logits_shape}, got {logits.shape}"
    assert assignment_matrix.shape == expected_assignment_shape, f"Expected assignment {expected_assignment_shape}, got {assignment_matrix.shape}"
    
    print("✓ Forward pass successful")
    return logits, assignment_matrix


def test_backward_pass(model, device='cpu'):
    """Test backward pass and gradient computation"""
    print("Testing backward pass...")
    
    # Create dummy data
    h_V, edge_index, h_E, seq, batch, labels = create_dummy_protein_data(
        num_proteins=2, 
        nodes_per_protein=[15, 20],
        device=device
    )
    
    # Forward pass
    model.train()
    logits, assignment_matrix = model(h_V, edge_index, h_E, seq, batch)
    
    # Compute loss
    loss = model.compute_total_loss(logits, labels)
    print(f"  - Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    grad_norm = 0.0
    num_params_with_grad = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm += param.grad.norm().item() ** 2
            num_params_with_grad += 1
    
    grad_norm = grad_norm ** 0.5
    print(f"  - Gradient norm: {grad_norm:.4f}")
    print(f"  - Parameters with gradients: {num_params_with_grad}")
    
    print("✓ Backward pass successful")
    return loss.item()


def test_interpretability(model, device='cpu'):
    """Test interpretability features"""
    print("Testing interpretability features...")
    
    # Create dummy data
    h_V, edge_index, h_E, seq, batch, labels = create_dummy_protein_data(
        num_proteins=1, 
        nodes_per_protein=[10],
        device=device
    )
    
    model.eval()
    with torch.no_grad():
        logits, assignment_matrix = model(h_V, edge_index, h_E, seq, batch)
    
    # Analyze assignment matrix
    assignment = assignment_matrix[0]  # [max_N, max_clusters]
    
    print(f"  - Assignment matrix shape: {assignment.shape}")
    print(f"  - Assignments per cluster:")
    
    for cluster_idx in range(assignment.shape[1]):
        assigned_nodes = torch.where(assignment[:, cluster_idx] > 0.5)[0]
        if len(assigned_nodes) > 0:
            print(f"    Cluster {cluster_idx}: nodes {assigned_nodes.tolist()}")
        else:
            print(f"    Cluster {cluster_idx}: empty")
    
    # Check that each node is assigned to at most one cluster
    total_assignments = assignment.sum(dim=1)
    max_assignments = total_assignments.max().item()
    print(f"  - Max assignments per node: {max_assignments}")
    
    assert max_assignments <= 1.1, f"Node assigned to multiple clusters: {max_assignments}"
    
    print("✓ Interpretability test successful")


def test_temperature_schedule(model):
    """Test temperature annealing"""
    print("Testing temperature schedule...")
    
    initial_temp = model.partitioner.get_temperature()
    print(f"  - Initial temperature: {initial_temp:.4f}")
    
    # Simulate several epochs
    temperatures = [initial_temp]
    for epoch in range(10):
        model.update_epoch()
        temp = model.partitioner.get_temperature()
        temperatures.append(temp)
    
    print(f"  - Temperature after 10 epochs: {temperatures[-1]:.4f}")
    print(f"  - Temperature schedule: {[f'{t:.3f}' for t in temperatures[::2]]}")
    
    # Check that temperature decreases
    assert temperatures[-1] < temperatures[0], "Temperature should decrease over time"
    assert temperatures[-1] >= model.partitioner.tau_min, "Temperature should not go below minimum"
    
    print("✓ Temperature schedule test successful")


def test_different_batch_sizes(model, device='cpu'):
    """Test model with different batch sizes"""
    print("Testing different batch sizes...")
    
    batch_configs = [
        (1, [10]),           # Single protein
        (2, [15, 20]),       # Two proteins
        (4, [8, 12, 25, 18]) # Four proteins
    ]
    
    for num_proteins, nodes_per_protein in batch_configs:
        h_V, edge_index, h_E, seq, batch, labels = create_dummy_protein_data(
            num_proteins=num_proteins,
            nodes_per_protein=nodes_per_protein,
            device=device
        )
        
        model.eval()
        with torch.no_grad():
            logits, assignment_matrix = model(h_V, edge_index, h_E, seq, batch)
        
        expected_batch_size = num_proteins
        assert logits.shape[0] == expected_batch_size, f"Expected batch size {expected_batch_size}, got {logits.shape[0]}"
        
        print(f"  ✓ Batch size {num_proteins}: logits {logits.shape}, assignments {assignment_matrix.shape}")
    
    print("✓ Different batch sizes test successful")


def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("TESTING GVP Hard Gumbel Partitioner Model")
    print("="*60)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print()
    
    try:
        # Test 1: Model creation
        model = test_model_creation()
        model = model.to(device)
        print()
        
        # Test 2: Forward pass
        logits, assignment_matrix = test_forward_pass(model, device)
        print()
        
        # Test 3: Backward pass
        loss = test_backward_pass(model, device)
        print()
        
        # Test 4: Interpretability
        test_interpretability(model, device)
        print()
        
        # Test 5: Temperature schedule
        test_temperature_schedule(model)
        print()
        
        # Test 6: Different batch sizes
        test_different_batch_sizes(model, device)
        print()
        
        print("="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        exit(1)
#!/usr/bin/env python3
"""
Comprehensive training timing test for ParToken model.
Demonstrates both forward pass timing and full training loop timing.
"""

import torch
import torch.optim as optim
import time
from torch_geometric.data import Data, Batch
from partoken_model import create_optimized_model
from utils.training_timer import TrainingTimer, TimedTrainingLoop


def create_synthetic_batch(batch_size: int = 4, max_nodes: int = 50):
    """Create synthetic protein data for testing."""
    protein_data = []
    
    for b in range(batch_size):
        num_nodes = torch.randint(30, max_nodes + 1, (1,)).item()
        
        # Node features
        node_s = torch.randn(num_nodes, 6)  # Scalar features
        node_v = torch.randn(num_nodes, 3, 3)  # Vector features
        
        # Create chain-like connectivity
        edge_list = []
        for i in range(num_nodes - 1):
            edge_list.extend([[i, i + 1], [i + 1, i]])
            
            # Add some long-range connections
            if i % 5 == 0 and i + 5 < num_nodes:
                edge_list.extend([[i, i + 5], [i + 5, i]])
        
        edge_index = torch.tensor(edge_list).t().contiguous()
        num_edges = edge_index.size(1)
        
        # Edge features
        edge_s = torch.randn(num_edges, 32)
        edge_v = torch.randn(num_edges, 1, 3)
        
        # Random binary label
        y = torch.randint(0, 2, (1,))
        
        # Create PyTorch Geometric Data object
        data = Data(
            node_s=node_s,
            node_v=node_v,
            edge_index=edge_index,
            edge_s=edge_s,
            edge_v=edge_v,
            y=y
        )
        protein_data.append(data)
    
    return Batch.from_data_list(protein_data)


def test_forward_timing():
    """Test forward pass timing only."""
    print("🧬 Forward Pass Timing Test")
    print("=" * 50)
    
    model = create_optimized_model()
    model.eval()
    
    # Create synthetic data
    batch_data = create_synthetic_batch(batch_size=4, max_nodes=50)
    
    h_V = (batch_data.node_s, batch_data.node_v)
    edge_index = batch_data.edge_index
    h_E = (batch_data.edge_s, batch_data.edge_v)
    batch = batch_data.batch
    
    print(f"Batch size: {len(torch.unique(batch))}")
    print(f"Total nodes: {batch_data.node_s.size(0)}")
    print(f"Total edges: {batch_data.edge_index.size(1)}")
    
    # Test forward pass with timing
    with torch.no_grad():
        logits, assignment_matrix, extra = model(
            h_V, edge_index, h_E, batch=batch, print_timing=True
        )
    
    print(f"Output shapes: logits={logits.shape}, assignment={assignment_matrix.shape}")
    return model


def test_training_timing():
    """Test full training loop timing."""
    print("\n🔥 Full Training Timing Test")
    print("=" * 50)
    
    # Create model and optimizer
    model = create_optimized_model()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Create timer and training loop
    timer = TrainingTimer()
    training_loop = TimedTrainingLoop(model, optimizer, timer)
    
    # Simulate training data (small dataset for demo)
    train_batches = [create_synthetic_batch(batch_size=2, max_nodes=30) for _ in range(5)]
    
    print(f"Training on {len(train_batches)} batches per epoch")
    
    # === SINGLE BATCH TIMING TEST ===
    print("\n📦 Single Batch Training Timing:")
    model.train()
    
    batch_data = train_batches[0]
    h_V = (batch_data.node_s, batch_data.node_v)
    edge_index = batch_data.edge_index
    h_E = (batch_data.edge_s, batch_data.edge_v)
    labels = batch_data.y
    batch = batch_data.batch
    
    # Train single batch with detailed timing
    loss, metrics = training_loop.train_batch(
        h_V, edge_index, h_E, labels, batch, print_timing=True
    )
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Forward time: {metrics['timing/forward_ms']:.2f} ms")
    print(f"Backward time: {metrics['timing/backward_ms']:.2f} ms")
    print(f"Optimizer time: {metrics['timing/optimizer_ms']:.2f} ms")
    print(f"Total batch time: {metrics['timing/batch_total_ms']:.2f} ms")
    
    # === FULL EPOCH TIMING TEST ===
    print("\n🏆 Full Epoch Training Timing:")
    
    # Create a simple dataloader-like structure
    class SimpleDataLoader:
        def __init__(self, batches):
            self.batches = batches
        
        def __iter__(self):
            return iter(self.batches)
        
        def __len__(self):
            return len(self.batches)
    
    dataloader = SimpleDataLoader(train_batches)
    
    # Train one epoch with timing
    epoch_loss, epoch_metrics = training_loop.train_epoch(
        dataloader, epoch=1, print_batch_timing=True, print_every=2
    )
    
    print(f"\nEpoch Results:")
    print(f"Average Loss: {epoch_loss:.4f}")
    print(f"Average Forward Time: {epoch_metrics['timing/forward_ms']:.2f} ms")
    print(f"Average Backward Time: {epoch_metrics['timing/backward_ms']:.2f} ms")
    print(f"Average Optimizer Time: {epoch_metrics['timing/optimizer_ms']:.2f} ms")
    
    # === MULTI-EPOCH TIMING TEST ===
    print("\n🎯 Multi-Epoch Training Timing:")
    
    # Reset timer for clean multi-epoch test
    timer.reset()
    
    num_epochs = 3
    for epoch in range(1, num_epochs + 1):
        print(f"\n--- Epoch {epoch}/{num_epochs} ---")
        
        epoch_loss, epoch_metrics = training_loop.train_epoch(
            dataloader, epoch=epoch, print_batch_timing=False, print_every=10
        )
    
    # Print final training summary
    timer.print_training_summary(num_epochs)
    
    # Save timing data
    timer.save_timings("training_timing_results.json")
    
    return model, timer


def compare_training_vs_inference():
    """Compare training vs inference timing."""
    print("\n⚖️  Training vs Inference Timing Comparison")
    print("=" * 50)
    
    model = create_optimized_model()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Create test data
    batch_data = create_synthetic_batch(batch_size=4, max_nodes=50)
    h_V = (batch_data.node_s, batch_data.node_v)
    edge_index = batch_data.edge_index
    h_E = (batch_data.edge_s, batch_data.edge_v)
    labels = batch_data.y
    batch = batch_data.batch
    
    # === INFERENCE TIMING ===
    print("\n🔍 Inference Timing (eval mode, no gradients):")
    model.eval()
    
    inference_times = []
    for i in range(5):
        with torch.no_grad():
            start_time = torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            logits, _, _ = model(h_V, edge_index, h_E, batch=batch)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
    
    avg_inference = sum(inference_times) / len(inference_times)
    print(f"Average inference time: {avg_inference*1000:.2f} ms")
    
    # === TRAINING TIMING ===
    print("\n🔥 Training Timing (train mode, with gradients):")
    model.train()
    
    training_times = []
    for i in range(5):
        start_time = torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        # Forward pass
        logits, _, extra = model(h_V, edge_index, h_E, batch=batch)
        loss, _ = model.compute_total_loss(logits, labels, extra)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        training_time = time.time() - start_time
        training_times.append(training_time)
    
    avg_training = sum(training_times) / len(training_times)
    print(f"Average training time: {avg_training*1000:.2f} ms")
    
    print(f"\n📊 Comparison:")
    print(f"Training is {avg_training/avg_inference:.1f}x slower than inference")
    print(f"Additional training overhead: {(avg_training-avg_inference)*1000:.2f} ms")
    
    return avg_inference, avg_training


def main():
    """Run comprehensive timing tests."""
    print("🧬 ParToken Model Comprehensive Timing Analysis")
    print("=" * 60)
    
    # Test 1: Forward pass timing
    test_forward_timing()
    
    # Test 2: Full training timing  
    trained_model, timer = test_training_timing()
    
    # Test 3: Training vs inference comparison
    inf_time, train_time = compare_training_vs_inference()
    
    print(f"\n🎯 Summary:")
    print(f"• Forward pass component timing: Available")
    print(f"• Full training loop timing: Available") 
    print(f"• Batch-level timing: Available")
    print(f"• Epoch-level timing: Available")
    print(f"• Training vs inference comparison: Available")
    print(f"• Timing data saved to: training_timing_results.json")
    
    print(f"\n💡 Usage Tips:")
    print(f"• Use print_timing=True for component-level analysis")
    print(f"• Use TrainingTimer for comprehensive training analysis")
    print(f"• TimedTrainingLoop provides automatic timing integration")
    print(f"• All timings account for CUDA synchronization")


if __name__ == "__main__":
    main()

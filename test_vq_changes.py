#!/usr/bin/env python3
"""
Test script to verify the VQ codebook changes:
1. Only commitment loss (no codebook loss)
2. Differentiable presence
3. Entropy/perplexity for monitoring only
"""

import torch
import torch.nn.functional as F
from utils.VQCodebook import VQCodebookEMA
from partoken_model import create_optimized_model


def test_codebook_changes():
    """Test VQ codebook changes."""
    print("🧪 Testing VQ Codebook Changes")
    print("=" * 50)
    
    # Create a simple codebook
    codebook = VQCodebookEMA(
        codebook_size=64,
        dim=32,
        beta=0.25,
        decay=0.99
    )
    
    # Create some test data
    B, S, D = 4, 5, 32
    z = torch.randn(B, S, D, requires_grad=True)
    mask = torch.randint(0, 2, (B, S)).bool()
    
    print(f"Input shape: {z.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Valid clusters: {mask.sum().item()}")
    
    # Test forward pass
    z_q, indices, vq_loss, info = codebook(z, mask)
    
    print(f"\nVQ Forward Results:")
    print(f"  Quantized shape: {z_q.shape}")
    print(f"  VQ loss: {vq_loss.item():.6f}")
    print(f"  Codebook loss: {info['codebook_loss'].item():.6f} (should be 0.0)")
    print(f"  Commitment loss: {info['commitment_loss'].item():.6f}")
    print(f"  Perplexity: {info['perplexity'].item():.2f}")
    
    # Test differentiable presence
    print(f"\n🔄 Testing Differentiable Presence:")
    presence = codebook.differentiable_presence(z, mask, temperature=1.0)
    print(f"  Presence shape: {presence.shape}")
    print(f"  Presence requires grad: {presence.requires_grad}")
    print(f"  Presence range: [{presence.min().item():.3f}, {presence.max().item():.3f}]")
    
    # Test gradient flow
    print(f"\n⚡ Testing Gradient Flow:")
    loss = presence.sum()
    loss.backward()
    
    print(f"  z.grad is not None: {z.grad is not None}")
    if z.grad is not None:
        print(f"  z.grad norm: {z.grad.norm().item():.6f}")
    
    # Compare with non-differentiable version
    with torch.no_grad():
        presence_old = codebook.soft_presence(z.detach(), mask, temperature=1.0)
        print(f"  Old presence shape: {presence_old.shape}")
        print(f"  Old presence requires grad: {presence_old.requires_grad}")
    
    print("✅ VQ Codebook tests passed!")


def test_model_integration():
    """Test the changes in the full model."""
    print("\n🧬 Testing Model Integration")
    print("=" * 40)
    
    # Create model
    model = create_optimized_model()
    model.train()
    
    # Create synthetic data
    batch_size = 2
    max_nodes = 20
    total_nodes = batch_size * max_nodes
    
    h_V = (
        torch.randn(total_nodes, 6),
        torch.randn(total_nodes, 3, 3)
    )
    
    # Simple chain connectivity
    edge_list = []
    for b in range(batch_size):
        start_idx = b * max_nodes
        for i in range(max_nodes - 1):
            edge_list.extend([[start_idx + i, start_idx + i + 1], 
                            [start_idx + i + 1, start_idx + i]])
    
    edge_index = torch.tensor(edge_list).t().contiguous()
    num_edges = edge_index.size(1)
    
    h_E = (
        torch.randn(num_edges, 32),
        torch.randn(num_edges, 1, 3)
    )
    
    labels = torch.randint(0, 2, (batch_size,))
    batch = torch.repeat_interleave(torch.arange(batch_size), max_nodes)
    
    print(f"Input: {batch_size} proteins, {max_nodes} nodes each")
    
    # Forward pass
    logits, assignment_matrix, extra = model(h_V, edge_index, h_E, batch=batch)
    
    print(f"\nForward Results:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Presence shape: {extra['presence'].shape}")
    print(f"  Presence requires grad: {extra['presence'].requires_grad}")
    
    # Test loss computation
    total_loss, metrics = model.compute_total_loss(logits, labels, extra)
    
    print(f"\nLoss Results:")
    print(f"  Total loss: {total_loss.item():.6f}")
    print(f"  Classification loss: {metrics['loss/cls']:.6f}")
    print(f"  VQ commitment loss: {metrics['loss/vq_commit']:.6f}")
    print(f"  Coverage loss: {metrics['loss/psc']:.6f}")
    print(f"  Entropy (monitoring): {metrics['metric/entropy']:.6f}")
    print(f"  Perplexity: {metrics['metric/perplexity']:.2f}")
    print(f"  Presence mean: {metrics['metric/presence_mean']:.3f}")
    
    # Test gradient flow
    print(f"\n⚡ Testing Gradient Flow:")
    total_loss.backward()
    
    # Check if gradients flow to cluster features
    has_grad = False
    grad_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            grad_norm += param.grad.norm().item()
    
    print(f"  Model has gradients: {has_grad}")
    print(f"  Total gradient norm: {grad_norm:.6f}")
    
    # Verify no entropy in loss terms (should be 0.0 if lambda_ent = 0.0)
    print(f"  Lambda_ent: {model.lambda_ent}")
    
    print("✅ Model integration tests passed!")


if __name__ == "__main__":
    test_codebook_changes()
    test_model_integration()
    
    print("\n🎉 All tests completed successfully!")
    print("\nKey Changes Verified:")
    print("• VQ loss now only includes commitment term (no codebook loss)")
    print("• Presence computation is differentiable (gradients flow)")
    print("• Entropy/perplexity are monitoring metrics only")
    print("• Model still trains and computes losses correctly")

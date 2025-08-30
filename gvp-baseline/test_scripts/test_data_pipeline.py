#!/usr/bin/env python3
"""
Test script to verify the N, CA, C optimization works correctly.
"""

import torch
import numpy as np
from utils.proteinclass_dataset import ProteinClassificationDataset, BACKBONE

def test_backbone_optimization():
    """Test that the optimization to use only N, CA, C atoms works correctly."""
    
    print("Testing N, CA, C optimization...")
    print(f"BACKBONE atoms: {BACKBONE}")
    assert len(BACKBONE) == 3, f"Expected 3 atoms, got {len(BACKBONE)}"
    assert BACKBONE == ("N", "CA", "C"), f"Expected ('N', 'CA', 'C'), got {BACKBONE}"
    
    # Create a simple test protein with complete N, CA, C coordinates
    test_protein = {
        "name": "test_protein",
        "seq": "ACDE",  # 4 residues
        "coords": [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],  # Residue 1: N, CA, C
            [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]], # Residue 2: N, CA, C
            [[3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]], # Residue 3: N, CA, C
            [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]] # Residue 4: N, CA, C
        ],
        "label": 0
    }
    
    # Create dataset with the test protein
    dataset = ProteinClassificationDataset([test_protein], num_classes=1)
    
    # Get the featurized graph
    data = dataset[0]
    
    print(f"✅ Successfully created graph with {data.x.shape[0]} nodes")
    print(f"✅ Node scalar features shape: {data.node_s.shape}")
    print(f"✅ Node vector features shape: {data.node_v.shape}")
    print(f"✅ Edge scalar features shape: {data.edge_s.shape}")
    print(f"✅ Edge vector features shape: {data.edge_v.shape}")
    print(f"✅ Sequence length: {len(data.seq)}")
    print(f"✅ Mask shape: {data.mask.shape}")
    
    # Verify all residues are valid (mask should be all True)
    assert data.mask.all(), "Expected all residues to be valid"
    
    # Verify coordinates shape is correct (4 residues, 3 atoms each, 3 coordinates)
    coords_tensor = torch.tensor(test_protein["coords"])
    assert coords_tensor.shape == (4, 3, 3), f"Expected (4, 3, 3), got {coords_tensor.shape}"
    
    # Verify CA coordinates extraction works
    ca_coords = coords_tensor[:, 1]  # CA is at index 1
    assert ca_coords.shape == (4, 3), f"Expected (4, 3), got {ca_coords.shape}"
    
    print("✅ All tests passed! N, CA, C optimization is working correctly.")
    return True

def test_missing_atoms():
    """Test that missing atoms are handled correctly."""
    
    print("\nTesting missing atom handling...")
    
    # Create a test protein with some missing atoms (using inf)
    test_protein = {
        "name": "test_protein_missing",
        "seq": "AC",  # 2 residues
        "coords": [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],  # Complete residue
            [[float('inf'), float('inf'), float('inf')], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]  # Missing N atom
        ],
        "label": 0
    }
    
    dataset = ProteinClassificationDataset([test_protein], num_classes=1)
    data = dataset[0]
    
    print(f"✅ Graph created with missing atoms: {data.x.shape[0]} nodes")
    print(f"✅ Mask: {data.mask}")
    
    # First residue should be valid, second should be invalid
    assert data.mask[0] == True, "First residue should be valid"
    assert data.mask[1] == False, "Second residue should be invalid (missing N atom)"
    
    print("✅ Missing atom handling works correctly.")
    return True

if __name__ == "__main__":
    test_backbone_optimization()
    test_missing_atoms()
    print("\n🎉 All optimization tests passed!")

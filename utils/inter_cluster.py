import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGCN(nn.Module):
    """
    Simple GCN layer for inter-cluster message passing.
    
    Args:
        in_dim: Input feature dimension
        out_dim: Output feature dimension
    """
    
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of simple GCN.
        
        Args:
            x: Node features [batch_size, num_clusters, features]
            adj: Adjacency matrix [batch_size, num_clusters, num_clusters]
            
        Returns:
            Updated node features
        """
        # Add self-loops and normalize adjacency matrix
        eye = torch.eye(adj.size(-1), device=adj.device, dtype=adj.dtype)
        adj_with_self_loops = adj + eye.unsqueeze(0)
        
        # Degree normalization
        degree = adj_with_self_loops.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        adj_norm = adj_with_self_loops / degree
        
        # Message passing: A * X * W
        h = torch.bmm(adj_norm, x)
        return F.relu(self.linear(h))


class InterClusterModel(nn.Module):
    """
    Multi-layer GCN for inter-cluster message passing.
    
    Args:
        in_dim: Input feature dimension
        hidden_dim: Hidden feature dimension
        drop_rate: Dropout rate
    """
    
    def __init__(self, in_dim: int, hidden_dim: int, drop_rate: float = 0.1):
        super().__init__()
        self.gcn1 = SimpleGCN(in_dim, hidden_dim)
        self.dropout1 = nn.Dropout(drop_rate)
        self.gcn2 = SimpleGCN(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(drop_rate)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through cluster GCN.
        
        Args:
            x: Cluster features [B, num_clusters, features]
            adj: Cluster adjacency [B, num_clusters, num_clusters]
            
        Returns:
            Refined cluster features
        """
        h = self.gcn1(x, adj)
        h = self.dropout1(h)
        h = self.gcn2(h, adj)
        h = self.dropout2(h)
        return h
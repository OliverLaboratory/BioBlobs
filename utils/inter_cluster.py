import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalClusterAttention(nn.Module):
    """
    Single-query (global) → multi-key/value (clusters) attention.
    Returns a pooled cluster summary and per-cluster importance weights.
    """
    def __init__(self, dim: int, heads: int = 4, drop_rate: float = 0.1, temperature: float = 1.0):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads = heads
        self.d_head = dim // heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(drop_rate)
        self.temperature = temperature

    @torch.no_grad()
    def _safe_mask(self, scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # scores: [B,H,S], mask: [B,S] (bool)
        return scores.masked_fill(~mask.unsqueeze(1), -1e9)

    def forward(self, g: torch.Tensor, C: torch.Tensor, mask: torch.Tensor):
        """
        g:    [B, D]      global residue-pooled feature
        C:    [B, S, D]   cluster features (raw or quantized)
        mask: [B, S]      bool mask for valid clusters
        Returns:
            pooled:     [B, D]
            importance: [B, S]  (mean over heads; sums to 1 over valid S)
            attn:       [B, H, S]
        """
        B, S, D = C.shape
        H, d = self.heads, self.d_head

        q = self.q(g).view(B, H, d)                                  # [B,H,d]
        k = self.k(C).view(B, S, H, d).transpose(1, 2)               # [B,H,S,d]
        v = self.v(C).view(B, S, H, d).transpose(1, 2)               # [B,H,S,d]

        scale = (d ** 0.5) * max(self.temperature, 1e-6)
        scores = (q.unsqueeze(2) * k).sum(-1) / scale                 # [B,H,S]
        scores = self._safe_mask(scores, mask)

        attn = torch.softmax(scores, dim=-1)                          # [B,H,S]
        attn = attn * mask.unsqueeze(1).float()
        attn = attn / attn.sum(-1, keepdim=True).clamp_min(1e-6)
        attn = self.dropout(attn)

        pooled = (attn.unsqueeze(-1) * v).sum(2).reshape(B, D)        # [B,D]
        pooled = self.out(pooled)

        importance = attn.mean(1)                                     # [B,S]
        return pooled, importance, attn


class FeatureWiseGateFusion(nn.Module):
    """
    Feature-wise gated residual fusion:
        z = LN( g + beta ⊙ W_f c_star )
    where beta = sigmoid( MLP([g; c_star]) ) ∈ R^D
    """
    def __init__(self, dim: int, hidden: int = 0, drop_rate: float = 0.1):
        super().__init__()
        if hidden <= 0:
            hidden = max(32, dim // 2)
        self.beta_mlp = nn.Sequential(
            nn.Linear(2 * dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(hidden, dim),
        )
        self.proj_c = nn.Linear(dim, dim, bias=False)
        self.ln = nn.LayerNorm(dim)

        # gate bias near 0 so sigmoid ~ 0.5 at start
        nn.init.zeros_(self.beta_mlp[-1].bias)

    def forward(self, g: torch.Tensor, c_star: torch.Tensor):
        """
        g:      [B, D]
        c_star: [B, D]
        Returns:
            fused: [B, D]
            beta:  [B, D]  (feature-wise gate, useful for logging)
        """
        beta = torch.sigmoid(self.beta_mlp(torch.cat([g, c_star], dim=-1)))
        fused = self.ln(g + beta * self.proj_c(c_star))
        return fused, beta


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
        return x+h
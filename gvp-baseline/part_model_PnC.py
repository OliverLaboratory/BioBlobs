import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_scatter import scatter_mean, scatter_sum, scatter_max
import gvp
from gvp.models import GVP, GVPConvLayer, LayerNorm

class SimpleGCN(nn.Module):
    """Simple GCN layer for cluster message passing (same as your example)"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        
    def forward(self, x, adj):
        # x: [batch_size, num_clusters, features]
        # adj: [batch_size, num_clusters, num_clusters]
        
        # Normalize adjacency matrix (add self-loops and degree normalization)
        adj = adj + torch.eye(adj.size(-1), device=adj.device).unsqueeze(0)
        degree = adj.sum(dim=-1, keepdim=True)
        adj_norm = adj / (degree + 1e-8)
        
        # Message passing: A * X * W
        h = torch.matmul(adj_norm, x)
        h = self.linear(h)
        return F.relu(h)


class HardGumbelPartitioner(nn.Module):
    """Hard Gumbel-Softmax Partitioner replacing DiffPool"""
    def __init__(self, nfeat, max_clusters, nhid):
        super().__init__()
        self.max_clusters = max_clusters
        
        # Selection network (replaces DiffPool assignment)
        self.selection_mlp = nn.Sequential(
            nn.Linear(nfeat + nhid, nhid),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(nhid, 1)
        )
        
        # Context network for maintaining selection history
        self.context_gru = nn.GRU(nfeat, nhid, batch_first=True)
        self.context_init = nn.Linear(nfeat, nhid)
        
        # Temperature parameters
        self.register_buffer('epoch', torch.tensor(0))
        self.tau_init = 1.0
        self.tau_min = 0.1
        self.tau_decay = 0.95
        
    def get_temperature(self):
        return max(self.tau_min, self.tau_init * (self.tau_decay ** self.epoch))
    
    def gumbel_softmax_hard(self, logits, tau):
        """Hard Gumbel-Softmax with straight-through estimator"""
        # Add Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        noisy_logits = (logits + gumbel_noise) / tau
        
        # Soft selection (for backward pass)
        soft_selection = F.softmax(noisy_logits, dim=-1)
        
        # Hard selection (for forward pass)
        hard_selection = torch.zeros_like(soft_selection)
        hard_selection.scatter_(-1, soft_selection.argmax(dim=-1, keepdim=True), 1.0)
        
        # Straight-through estimator
        return hard_selection + (soft_selection - soft_selection.detach())
    
    def forward(self, x, adj, mask):
        """
        Args:
            x: [B, max_N, D] - Dense node features
            adj: [B, max_N, max_N] - Dense adjacency matrix  
            mask: [B, max_N] - Node mask (True for real nodes)
        Returns:
            cluster_features: [B, S, D] - Cluster representations
            cluster_adj: [B, S, S] - Inter-cluster adjacency
            assignment_matrix: [B, max_N, S] - Node-to-cluster assignments
        """
        batch_size, max_nodes, feat_dim = x.shape
        device = x.device
        tau = self.get_temperature()
        
        # Initialize
        available_mask = mask.clone()  # [B, max_N] - Track available nodes
        global_context = self.context_init(x.mean(dim=1))  # [B, H] - Initial context
        context_hidden = torch.zeros(1, batch_size, global_context.size(-1), device=device)
        
        cluster_embeddings = []  # List of [B, D] tensors
        assignment_matrix = torch.zeros(batch_size, max_nodes, self.max_clusters, device=device)
        cluster_history = torch.zeros(batch_size, 0, feat_dim, device=device)
        
        # Iterative clustering (similar to your max_iterations)
        for cluster_idx in range(self.max_clusters):
            # Check if any nodes remain
            if not available_mask.any():
                break
                
            # Expand context to all available nodes
            expanded_context = global_context.unsqueeze(1).expand(-1, max_nodes, -1)  # [B, max_N, H]
            
            # Compute selection logits for available nodes only
            combined_features = torch.cat([x, expanded_context], dim=-1)  # [B, max_N, D+H]
            logits = self.selection_mlp(combined_features).squeeze(-1)  # [B, max_N]
            
            # Mask out unavailable nodes
            logits = logits.masked_fill(~available_mask, float('-inf'))
            
            # Select one node per protein using hard Gumbel-Softmax
            selections = torch.zeros_like(available_mask, dtype=torch.float)  # [B, max_N]
            
            for b in range(batch_size):
                if available_mask[b].any():
                    batch_logits = logits[b][available_mask[b]]  # [N_available_b]
                    if len(batch_logits) > 0:
                        # Hard Gumbel-Softmax selection
                        selection_probs = self.gumbel_softmax_hard(batch_logits, tau)
                        selected_local_idx = selection_probs.argmax()
                        
                        # Map back to global index
                        available_indices = torch.where(available_mask[b])[0]
                        selected_global_idx = available_indices[selected_local_idx]
                        selections[b, selected_global_idx] = 1.0
                        
                        # Update assignment matrix
                        assignment_matrix[b, selected_global_idx, cluster_idx] = 1.0
            
            # Extract cluster embeddings (selected nodes only)
            cluster_emb = (selections.unsqueeze(-1) * x).sum(dim=1)  # [B, D]
            cluster_embeddings.append(cluster_emb)
            
            # Update cluster history
            cluster_history = torch.cat([cluster_history, cluster_emb.unsqueeze(1)], dim=1)  # [B, T+1, D]
            
            # Update global context using GRU memory
            if cluster_history.size(1) > 0:
                _, context_hidden = self.context_gru(cluster_history, context_hidden)
                global_context = context_hidden.squeeze(0)  # [B, H]
            
            # Remove selected nodes from available mask
            available_mask = available_mask & (selections <= 0.5)
        
        if not cluster_embeddings:
            # Fallback: single cluster with mean pooling
            cluster_embeddings = [x.mean(dim=1)]  # [B, D]
            assignment_matrix[:, :, 0] = mask.float()
        
        # Stack cluster embeddings
        cluster_features = torch.stack(cluster_embeddings, dim=1)  # [B, S, D]
        num_clusters = cluster_features.size(1)
        
        # Create inter-cluster adjacency (fully connected)
        cluster_adj = torch.ones(batch_size, num_clusters, num_clusters, device=device)
        cluster_adj = cluster_adj - torch.eye(num_clusters, device=device).unsqueeze(0)  # Remove self-loops
        
        return cluster_features, cluster_adj, assignment_matrix
    
    def update_epoch(self):
        self.epoch += 1


class GVPHardGumbelPartitionerModel(nn.Module):
    """
    GVP-GNN with Hard Gumbel-Softmax Partitioner for protein classification
    (Following your GVPDiffPoolGraphSAGEModel structure)
    """
    def __init__(self, node_in_dim, node_h_dim, 
                 edge_in_dim, edge_h_dim,
                 num_classes=2, seq_in=False, num_layers=3, 
                 drop_rate=0.1, pooling='mean', max_clusters=5):
        
        super().__init__()
        
        self.num_classes = num_classes
        self.pooling = pooling
        
        if seq_in:
            self.W_s = nn.Embedding(20, 20)
            node_in_dim = (node_in_dim[0] + 20, node_in_dim[1])
        
        # GVP layers (same as your example)
        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )
        
        self.layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) 
            for _ in range(num_layers))
        
        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0)))  # Extract scalar features only
        
        # Hard Gumbel-Softmax Partitioner (replaces your BatchedDiffPool)
        self.partitioner = HardGumbelPartitioner(nfeat=ns, max_clusters=max_clusters, nhid=ns//2)
        
        # Cluster GCN for inter-cluster message passing (same as your example)
        self.cluster_gcn = nn.Sequential(
            SimpleGCN(ns, ns),
            nn.Dropout(p=drop_rate),
            SimpleGCN(ns, ns),
            nn.Dropout(p=drop_rate)
        )
        
        # Classification head (same structure as your example)
        self.classifier = nn.Sequential(
            nn.Linear(2 * ns, 4 * ns), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(4 * ns, 2 * ns),
            nn.ReLU(inplace=True), 
            nn.Dropout(p=drop_rate),
            nn.Linear(2 * ns, num_classes)
        )

    def forward(self, h_V, edge_index, h_E, seq=None, batch=None):
        """Forward pass following your GVP-DiffPool structure"""
        if seq is not None:
            seq = self.W_s(seq)
            h_V = (torch.cat([h_V[0], seq], dim=-1), h_V[1])
            
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        
        # Process through GVP layers (same as your example)
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
            
        # Extract scalar features for partitioning (same as your example)
        residue_features = self.W_out(h_V)  # [N, ns]
        
        # Convert to dense format (same as your example)
        if batch is None:
            batch = torch.zeros(residue_features.size(0), dtype=torch.long, device=residue_features.device)
        
        dense_x, mask = to_dense_batch(residue_features, batch)  # [B, max_N, ns]
        dense_adj = to_dense_adj(edge_index, batch)  # [B, max_N, max_N]
        
        # Apply Hard Gumbel-Softmax Partitioner (replaces your DiffPool)
        cluster_features, cluster_adj, assignment_matrix = self.partitioner(dense_x, dense_adj, mask)
        
        # Inter-cluster message passing (same as your example)
        refined_cluster_features = self.cluster_gcn[0](cluster_features, cluster_adj)
        refined_cluster_features = self.cluster_gcn[1](refined_cluster_features)  # Dropout
        refined_cluster_features = self.cluster_gcn[2](refined_cluster_features, cluster_adj)
        refined_cluster_features = self.cluster_gcn[3](refined_cluster_features)  # Dropout
        
        # Pool cluster features to graph level (same as your example)
        cluster_pooled = refined_cluster_features.mean(dim=1)  # [B, ns]
        
        # Pool residue features to graph level (same as your example)
        residue_pooled = self._pool_nodes(residue_features, batch)  # [B, ns]
        
        # Concatenate residue and cluster representations (same as your example)
        combined_features = torch.cat([residue_pooled, cluster_pooled], dim=-1)  # [B, 2*ns]
        
        # Classification (same as your example)
        logits = self.classifier(combined_features)
        
        return logits, assignment_matrix
    
    def _pool_nodes(self, node_features, batch):
        """Pool node features to get graph-level representation (same as your example)"""
        if self.pooling == 'mean':
            return scatter_mean(node_features, batch, dim=0)
        elif self.pooling == 'max':
            return scatter_max(node_features, batch, dim=0)[0]
        elif self.pooling == 'sum':
            return scatter_sum(node_features, batch, dim=0)
        
        return scatter_mean(node_features, batch, dim=0)  # default to mean
    
    def compute_total_loss(self, logits, labels):
        """Compute classification loss (simplified compared to your aux losses)"""
        classification_loss = F.cross_entropy(logits, labels)
        return classification_loss
    
    def predict(self, h_V, edge_index, h_E, seq=None, batch=None):
        """Get class predictions (same as your example)"""
        with torch.no_grad():
            logits, _ = self.forward(h_V, edge_index, h_E, seq, batch)
            return torch.argmax(logits, dim=-1)
    
    def predict_proba(self, h_V, edge_index, h_E, seq=None, batch=None):
        """Get class probabilities (same as your example)"""
        with torch.no_grad():
            logits, _ = self.forward(h_V, edge_index, h_E, seq, batch)
            return torch.softmax(logits, dim=-1)
    
    def update_epoch(self):
        """Update temperature schedule"""
        self.partitioner.update_epoch()


# Usage example (following your pattern)
"""
model = GVPHardGumbelPartitionerModel(
    node_in_dim=(6, 3),      # GVP node dimensions
    node_h_dim=(100, 16),    # GVP hidden dimensions  
    edge_in_dim=(32, 1),     # GVP edge dimensions
    edge_h_dim=(32, 1),      # GVP edge hidden dimensions
    num_classes=2,           # Binary classification
    seq_in=False,            # Whether to use sequence
    num_layers=3,            # Number of GVP layers
    drop_rate=0.1,           # Dropout rate
    pooling='mean',          # Pooling strategy
    max_clusters=5           # Maximum number of clusters
)

# Forward pass
logits, assignment_matrix = model(h_V, edge_index, h_E, seq=seq, batch=batch)
loss = model.compute_total_loss(logits, labels)
"""
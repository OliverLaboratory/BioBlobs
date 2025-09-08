import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_scatter import scatter_mean, scatter_sum, scatter_max

class SimpleGCN(nn.Module):
    """A single GCN layer with row-normalized dense adjacency.

    Args:
        in_dim (int): Input feature dimension.
        out_dim (int): Output feature dimension.

    Forward:
        x (Tensor): [B, N, F_in] node features.
        adj (Tensor): [B, N, N] dense adjacency, no assumption about self-loops.

    Returns:
        Tensor: [B, N, F_out] features after one propagation step and ReLU.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Add self-loops and row-normalize (avoid div-by-zero)
        bsz, n, _ = x.shape
        device = x.device
        adj = adj + torch.eye(n, device=device).unsqueeze(0)
        degree = adj.sum(dim=-1, keepdim=True).clamp_min_(1e-8)
        adj_norm = adj / degree

        # Message passing: A_norm X W
        h = adj_norm @ x
        return F.relu(self.linear(h))



class HardGumbelPartitioner(nn.Module):
    """Hard Gumbel-Softmax Partitioner with Neighbor Expansion + k-Hop Relaxation"""
    def __init__(self, nfeat, max_clusters, nhid, k_hop=2, cluster_size_max=30, enable_connectivity=True):
        super().__init__()
        self.max_clusters = max_clusters
        self.k_hop = k_hop
        self.cluster_size_max = cluster_size_max
        self.enable_connectivity = enable_connectivity
        
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
        
    def compute_k_hop_mask(self, adj, seed_node_idx, k):
        """
        Compute k-hop neighborhood mask for a single seed node
        Args:
            adj: [max_N, max_N] - Single protein adjacency matrix
            seed_node_idx: int - Seed node index  
            k: int - Number of hops
        Returns:
            mask: [max_N] - Boolean mask for k-hop neighborhood
        """
        if k == 0:
            # Only seed node is selectable
            mask = torch.zeros(adj.size(0), dtype=torch.bool, device=adj.device)
            mask[seed_node_idx] = True
            return mask
            
        # Use matrix multiplication to compute k-hop reachability
        # Start with direct neighbors (1-hop)
        reachable = torch.zeros(adj.size(0), dtype=torch.bool, device=adj.device)
        reachable[seed_node_idx] = True  # Include seed node
        
        current_nodes = torch.zeros(adj.size(0), dtype=torch.bool, device=adj.device)
        current_nodes[seed_node_idx] = True
        
        # Iteratively expand to k-hops
        for hop in range(k):
            # Find neighbors of current nodes
            neighbor_mask = torch.matmul(current_nodes.float(), adj) > 0
            new_nodes = neighbor_mask & (~reachable)  # Only new nodes
            
            if not new_nodes.any():
                break  # No more nodes to expand to
                
            reachable = reachable | neighbor_mask
            current_nodes = new_nodes
            
        return reachable
    
    def select_from_candidates(self, x_batch, context_batch, candidate_mask, tau):
        """
        Select one node from candidates using Hard Gumbel-Softmax with pre-filtering
        Args:
            x_batch: [max_N, D] - Node features for single protein
            context_batch: [max_N, H] - Context features for single protein  
            candidate_mask: [max_N] - Boolean mask for candidate nodes
            tau: float - Temperature for Gumbel-Softmax
        Returns:
            selected_idx: int - Global index of selected node (-1 if no candidates)
        """
        candidate_indices = torch.where(candidate_mask)[0]
        
        if len(candidate_indices) == 0:
            return -1
            
        # Pre-filtering: extract features only for candidates
        candidate_features = x_batch[candidate_indices]  # [N_candidates, D]
        candidate_context = context_batch[candidate_indices]  # [N_candidates, H]
        
        # Compute logits only for candidates (no information leakage)
        combined_features = torch.cat([candidate_features, candidate_context], dim=-1)
        candidate_logits = self.selection_mlp(combined_features).squeeze(-1)  # [N_candidates]
        
        if len(candidate_logits) == 1:
            # Only one candidate, select deterministically
            return candidate_indices[0].item()
            
        # Hard Gumbel-Softmax selection
        selection_probs = self.gumbel_softmax_hard(candidate_logits, tau)
        selected_local_idx = selection_probs.argmax()
        selected_global_idx = candidate_indices[selected_local_idx]
        
        return selected_global_idx.item()
    
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
        
        # Iterative clustering with multi-node cluster formation
        for cluster_idx in range(self.max_clusters):
            # Check if any nodes remain across all proteins
            if not available_mask.any():
                break
                
            cluster_nodes_batch = []  # List of selected nodes per protein in this cluster
            cluster_embeddings_batch = torch.zeros(batch_size, feat_dim, device=device)
            
            # Process each protein in batch
            for b in range(batch_size):
                if not available_mask[b].any():
                    cluster_nodes_batch.append([])
                    continue
                    
                cluster_nodes = []  # Nodes selected for this cluster in protein b
                
                # Step 1: Select seed node (free selection from available nodes)
                expanded_context = global_context[b].unsqueeze(0).expand(max_nodes, -1)  # [max_N, H]
                seed_idx = self.select_from_candidates(
                    x[b], expanded_context, available_mask[b], tau
                )
                
                if seed_idx == -1:
                    cluster_nodes_batch.append([])
                    continue
                    
                cluster_nodes.append(seed_idx)
                available_mask[b, seed_idx] = False
                
                # Step 2: Expand cluster using k-hop neighbors (if enabled and cluster_size_max > 1)
                if self.enable_connectivity and self.cluster_size_max > 1:
                    # Compute k-hop neighborhood around seed
                    k_hop_mask = self.compute_k_hop_mask(adj[b], seed_idx, self.k_hop)
                    
                    # Iteratively select additional nodes from k-hop neighborhood
                    for additional_node in range(self.cluster_size_max - 1):
                        # Restrict candidates to available nodes within k-hop neighborhood
                        cluster_candidates = available_mask[b] & k_hop_mask
                        
                        if not cluster_candidates.any():
                            break  # No more candidates in k-hop neighborhood
                            
                        # Select next node using pre-filtering
                        next_idx = self.select_from_candidates(
                            x[b], expanded_context, cluster_candidates, tau
                        )
                        
                        if next_idx == -1:
                            break
                            
                        cluster_nodes.append(next_idx)
                        available_mask[b, next_idx] = False
                
                cluster_nodes_batch.append(cluster_nodes)
                
                # Update assignment matrix for this protein
                for node_idx in cluster_nodes:
                    assignment_matrix[b, node_idx, cluster_idx] = 1.0
                    
                # Compute cluster embedding (mean of selected nodes)
                if cluster_nodes:
                    cluster_node_features = x[b, cluster_nodes]  # [cluster_size, D]
                    cluster_embeddings_batch[b] = cluster_node_features.mean(dim=0)
            
            # Store cluster embeddings for this iteration
            cluster_embeddings.append(cluster_embeddings_batch)
            
            # Update cluster history
            cluster_history = torch.cat([cluster_history, cluster_embeddings_batch.unsqueeze(1)], dim=1)
            
            # Update global context using GRU memory
            if cluster_history.size(1) > 0:
                _, context_hidden = self.context_gru(cluster_history, context_hidden)
                global_context = context_hidden.squeeze(0)  # [B, H]
        
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
        self.partitioner = HardGumbelPartitioner(
            nfeat=ns, 
            max_clusters=max_clusters, 
            nhid=ns//2,
            k_hop=2,                    # 2-hop spatial constraint
            cluster_size_max=3,         # Max 3 nodes per cluster
            enable_connectivity=True    # Enable spatial constraints
        )
        
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

# The partitioner automatically uses:
# - k_hop=2: 2-hop spatial constraint for cluster formation
# - cluster_size_max=3: Maximum 3 nodes per cluster
# - enable_connectivity=True: Enforce spatial connectivity within clusters

# Forward pass
logits, assignment_matrix = model(h_V, edge_index, h_E, seq=seq, batch=batch)
# assignment_matrix[b, n, c] = 1 if node n in protein b belongs to cluster c

loss = model.compute_total_loss(logits, labels)

# For interpretability: assignment_matrix shows which residues form each functional cluster
"""
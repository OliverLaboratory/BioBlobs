import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class Partitioner(nn.Module):
    """
    Optimized Hard Gumbel-Softmax Partitioner with efficient clustering.
    
    This class implements a streamlined version of the partitioner that focuses on
    efficiency while maintaining gradient flow and clustering quality.
    
    Args:
        nfeat: Number of input features
        max_clusters: Maximum number of clusters
        nhid: Hidden dimension size
        k_hop: Number of hops for spatial constraints
        cluster_size_max: Maximum cluster size
        termination_threshold: Threshold for early termination
    """
    
    def __init__(
        self, 
        nfeat: int, 
        max_clusters: int, 
        nhid: int, 
        k_hop: int = 2, 
        cluster_size_max: int = 3,
        termination_threshold: float = 0.95
    ):
        super().__init__()
        self.max_clusters = max_clusters
        self.k_hop = k_hop
        self.cluster_size_max = cluster_size_max
        self.cluster_size_min = 1
        self.termination_threshold = termination_threshold
        
        # Simplified selection network
        self.seed_selector = nn.Sequential(
            nn.Linear(nfeat + nhid, nhid),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(nhid, 1)
        )
        
        # Size prediction network
        self.size_predictor = nn.Sequential(
            nn.Linear(nfeat + nhid, nhid),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(nhid, self.cluster_size_max)
        )
        
        # Context encoder (simplified from GRU)
        self.context_encoder = nn.Linear(nfeat, nhid)
        
        # Temperature parameters
        self.register_buffer('epoch', torch.tensor(0))
        self.tau_init = 1.0
        self.tau_min = 0.1
        self.tau_decay = 0.95
        
    def get_temperature(self) -> float:
        """Get current temperature for Gumbel-Softmax annealing."""
        return max(self.tau_min, self.tau_init * (self.tau_decay ** self.epoch))
    
    def _compute_k_hop_neighbors(
        self, 
        adj: torch.Tensor, 
        seed_indices: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Efficiently compute k-hop neighborhoods for seed nodes.
        
        Args:
            adj: Adjacency matrix [B, N, N]
            seed_indices: Seed node indices [B]
            mask: Valid node mask [B, N]
            
        Returns:
            k_hop_mask: Boolean mask for k-hop neighborhoods [B, N]
        """
        B, N, _ = adj.shape
        device = adj.device
        
        # Initialize with seed nodes
        current_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        reachable_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        
        # Set seed nodes
        valid_seeds = seed_indices >= 0
        current_mask[valid_seeds, seed_indices[valid_seeds]] = True
        reachable_mask = current_mask.clone()
        
        # Iteratively expand k hops
        for _ in range(self.k_hop):
            # Find neighbors efficiently
            neighbors = torch.bmm(current_mask.float().unsqueeze(1), adj).squeeze(1) > 0
            neighbors = neighbors & mask & (~reachable_mask)
            
            if not neighbors.any():
                break
                
            current_mask = neighbors
            reachable_mask = reachable_mask | neighbors
            
        return reachable_mask
    
    def _gumbel_softmax_selection(
        self, 
        logits: torch.Tensor, 
        tau: float, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Efficient Hard Gumbel-Softmax selection.
        
        Args:
            logits: Selection logits [B, N]
            tau: Temperature parameter
            mask: Valid selection mask [B, N]
            
        Returns:
            Hard selection with straight-through gradients [B, N]
        """
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        
        # Gumbel noise
        gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        noisy_logits = (logits + gumbel) / tau
        
        # Soft selection for gradients
        soft = F.softmax(noisy_logits, dim=-1)
        
        # Hard selection for forward pass
        hard = torch.zeros_like(soft)
        indices = soft.argmax(dim=-1, keepdim=True)
        hard.scatter_(-1, indices, 1.0)
        
        # Straight-through estimator
        return hard + (soft - soft.detach())
    
    def _check_termination(
        self, 
        assignment_matrix: torch.Tensor, 
        mask: torch.Tensor, 
        cluster_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Check termination condition efficiently.
        
        Args:
            assignment_matrix: Current assignments [B, N, S]
            mask: Valid node mask [B, N]
            cluster_idx: Current cluster index
            
        Returns:
            should_terminate: Boolean mask [B]
            active_proteins: Boolean mask [B]
        """
        total_nodes = mask.sum(dim=-1).float()
        assigned_nodes = assignment_matrix[:, :, :cluster_idx+1].sum(dim=(1, 2))
        coverage = assigned_nodes / (total_nodes + 1e-8)
        
        should_terminate = coverage >= self.termination_threshold
        active_proteins = (~should_terminate) & (total_nodes > 0)
        
        return should_terminate, active_proteins
    
    def forward(
        self, 
        x: torch.Tensor, 
        adj: torch.Tensor, 
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Optimized forward pass for clustering.
        
        Args:
            x: Dense node features [B, N, D]
            adj: Dense adjacency matrix [B, N, N]
            mask: Node validity mask [B, N]
            
        Returns:
            cluster_features: Cluster representations [B, S, D]
            cluster_adj: Inter-cluster adjacency [B, S, S]
            assignment_matrix: Node-to-cluster assignments [B, N, S]
        """
        B, N, D = x.shape
        device = x.device
        tau = self.get_temperature()
        
        # Initialize efficiently
        available_mask = mask.clone()
        global_context = self.context_encoder(x.masked_fill(~mask.unsqueeze(-1), 0).mean(dim=1))
        
        cluster_embeddings = []
        assignment_matrix = torch.zeros(B, N, self.max_clusters, device=device)
        terminated = torch.zeros(B, dtype=torch.bool, device=device)
        
        for cluster_idx in range(self.max_clusters):
            # Early global termination
            if not available_mask.any() or terminated.all():
                break
                
            # Check per-protein termination
            if cluster_idx > 0:
                should_terminate, active = self._check_termination(
                    assignment_matrix, mask, cluster_idx - 1
                )
                terminated = terminated | should_terminate
                if not active.any():
                    break
            else:
                active = torch.ones(B, dtype=torch.bool, device=device)
            
            # Seed selection
            context_expanded = global_context.unsqueeze(1).expand(-1, N, -1)
            combined_features = torch.cat([x, context_expanded], dim=-1)
            seed_logits = self.seed_selector(combined_features).squeeze(-1)
            
            selection_mask = available_mask & active.unsqueeze(-1)
            seed_selection = self._gumbel_softmax_selection(seed_logits, tau, selection_mask)
            seed_indices = seed_selection.argmax(dim=-1)
            
            # Update availability
            has_selection = selection_mask.sum(dim=-1) > 0
            valid_seeds = has_selection & active
            
            if valid_seeds.any():
                # Assign seeds
                assignment_matrix[valid_seeds, seed_indices[valid_seeds], cluster_idx] = 1.0
                
                # Update availability efficiently
                for b in range(B):
                    if valid_seeds[b]:
                        available_mask[b, seed_indices[b]] = False
                
                # Expand clusters with k-hop neighbors
                if self.cluster_size_max > 1:
                    k_hop_mask = self._compute_k_hop_neighbors(adj, seed_indices, mask)
                    candidates = available_mask & k_hop_mask & active.unsqueeze(-1)
                    
                    if candidates.any():
                        # Predict additional cluster size
                        seed_features = x[valid_seeds, seed_indices[valid_seeds]]
                        context_features = global_context[valid_seeds]
                        
                        size_input = torch.cat([seed_features, context_features], dim=-1)
                        size_logits = self.size_predictor(size_input)
                        size_probs = F.softmax(size_logits / tau, dim=-1)
                        predicted_sizes = (size_probs * torch.arange(1, self.cluster_size_max + 1, 
                                                                    device=device).float()).sum(dim=-1)
                        
                        # Select additional nodes efficiently
                        additional_needed = (predicted_sizes - 1).clamp(min=0).long()
                        
                        # Map valid_seeds indices to additional_needed indices
                        valid_indices = valid_seeds.nonzero(as_tuple=True)[0]
                        
                        for i, b in enumerate(valid_indices):
                            if additional_needed[i] > 0:
                                cand_indices = candidates[b].nonzero(as_tuple=True)[0]
                                if len(cand_indices) > 0:
                                    n_select = min(additional_needed[i].item(), len(cand_indices))
                                    if n_select > 0:
                                        # Select top candidates based on features
                                        cand_logits = seed_logits[b, cand_indices]
                                        _, top_indices = torch.topk(cand_logits, n_select)
                                        selected_nodes = cand_indices[top_indices]
                                        
                                        assignment_matrix[b, selected_nodes, cluster_idx] = 1.0
                                        available_mask[b, selected_nodes] = False
            
            # Compute cluster embeddings efficiently
            cluster_mask = assignment_matrix[:, :, cluster_idx] > 0.5
            cluster_size = cluster_mask.sum(dim=-1, keepdim=True).float().clamp(min=1)
            cluster_sum = (cluster_mask.unsqueeze(-1) * x).sum(dim=1)
            cluster_embedding = cluster_sum / cluster_size
            
            # Apply soft masking for terminated proteins
            active_weight = active.float().unsqueeze(-1)
            cluster_embedding = cluster_embedding * active_weight
            
            cluster_embeddings.append(cluster_embedding)
            
            # Update context efficiently - project cluster embedding to context space
            if active.any():
                # Project cluster embedding to context space to match dimensions
                cluster_context = self.context_encoder(cluster_embedding.detach())
                global_context = global_context + 0.1 * cluster_context
        
        # Handle edge cases
        if not cluster_embeddings:
            # Fallback: single cluster with all nodes
            cluster_embedding = (mask.unsqueeze(-1) * x).sum(dim=1) / mask.sum(dim=-1, keepdim=True).float()
            cluster_embeddings = [cluster_embedding]
            assignment_matrix[:, :, 0] = mask.float()
        
        # Stack outputs
        cluster_features = torch.stack(cluster_embeddings, dim=1)
        S = cluster_features.size(1)
        
        # Create fully connected cluster adjacency
        cluster_adj = torch.ones(B, S, S, device=device) - torch.eye(S, device=device).unsqueeze(0)
        
        return cluster_features, cluster_adj, assignment_matrix
    
    def update_epoch(self) -> None:
        """Update epoch counter for temperature annealing."""
        self.epoch += 1

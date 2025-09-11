import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_scatter import scatter_mean, scatter_sum, scatter_max
from gvp.models import GVP, GVPConvLayer, LayerNorm
from typing import Tuple, Optional
import numpy as np
from utils.VQCodebook import VQCodebookEMA
from utils.inter_cluster import InterClusterModel
from utils.pnc_partition import Partitioner


class ParTokenModel(nn.Module):
    """
    Optimized GVP-GNN with efficient partitioning for protein classification.
    
    This model combines GVP layers for geometric deep learning on proteins
    with a partitioner for hierarchical clustering.

    Args:
        node_in_dim: Input node dimensions (scalar, vector)
        node_h_dim: Hidden node dimensions (scalar, vector)
        edge_in_dim: Input edge dimensions (scalar, vector)
        edge_h_dim: Hidden edge dimensions (scalar, vector)
        num_classes: Number of output classes
        seq_in: Whether to use sequence features
        num_layers: Number of GVP layers
        drop_rate: Dropout rate
        pooling: Pooling strategy ('mean', 'max', 'sum')
        max_clusters: Maximum number of clusters
        termination_threshold: Early termination threshold
    """
    
    def __init__(
        self,
        node_in_dim: Tuple[int, int],
        node_h_dim: Tuple[int, int],
        edge_in_dim: Tuple[int, int],
        edge_h_dim: Tuple[int, int],
        num_classes: int = 2,
        seq_in: bool = False,
        num_layers: int = 3,
        drop_rate: float = 0.1,
        pooling: str = 'mean',
        # Partitioner hyperparameters
        max_clusters: int = 5,
        nhid: int = 50,
        k_hop: int = 2,
        cluster_size_max: int = 3,
        termination_threshold: float = 0.95,
        tau_init: float = 1.0,
        tau_min: float = 0.1,
        tau_decay: float = 0.95,
        # Codebook hyperparameters
        codebook_size: int = 512,
        codebook_dim: Optional[int] = None,
        codebook_beta: float = 0.25,
        codebook_decay: float = 0.99,
        codebook_eps: float = 1e-5,
        codebook_distance: str = "l2",
        codebook_cosine_normalize: bool = False,
        # Loss weights
        lambda_vq: float = 1.0,
        lambda_ent: float = 0.0,    # Not used in loss (kept for backward compatibility)
        lambda_psc: float = 1e-2,
        psc_temp: float = 0.3
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.pooling = pooling
        self.seq_in = seq_in
        
        # Adjust input dimensions for sequence features
        if seq_in:
            self.sequence_embedding = nn.Embedding(20, 20)
            node_in_dim = (node_in_dim[0] + 20, node_in_dim[1])
        
        # GVP layers
        self.node_encoder = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.edge_encoder = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )
        
        # GVP convolution layers
        self.gvp_layers = nn.ModuleList([
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) 
            for _ in range(num_layers)
        ])
        
        # Output projection (extract scalar features only)
        ns, _ = node_h_dim
        self.output_projection = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0))
        )
        # Partitioner
        self.partitioner = Partitioner(
            nfeat=ns,
            max_clusters=max_clusters,
            nhid=nhid,
            k_hop=k_hop,
            cluster_size_max=cluster_size_max,
            termination_threshold=termination_threshold
        )
        self.partitioner.tau_init = tau_init
        self.partitioner.tau_min = tau_min
        self.partitioner.tau_decay = tau_decay
        
        # Inter-cluster message passing
        self.cluster_gcn = InterClusterModel(ns, ns, drop_rate)

        self.codebook = VQCodebookEMA(
            codebook_size=codebook_size,
            dim=codebook_dim if codebook_dim is not None else ns,
            beta=codebook_beta,
            decay=codebook_decay,
            eps=codebook_eps,
            distance=codebook_distance,
            cosine_normalize=codebook_cosine_normalize
        )
        # Loss weights and coverage temperature
        self.lambda_vq = lambda_vq
        self.lambda_ent = lambda_ent
        self.lambda_psc = lambda_psc
        self.psc_temp = psc_temp
        
        # Use original cluster embeddings for classification (not quantized)
        self.use_quantized_for_classification = False
        
        # Attention mechanism for cluster importance
        self.cluster_attention = nn.Sequential(
            nn.Linear(ns, ns // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(ns // 2, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(2 * ns, 4 * ns), 
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(4 * ns, 2 * ns),
            nn.ReLU(inplace=True), 
            nn.Dropout(drop_rate),
            nn.Linear(2 * ns, num_classes)
        )

    def _masked_mean(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Masked mean over cluster dimension.

        Args:
            x: Tensor [B, S, D]
            mask: Boolean mask [B, S] (True = include)

        Returns:
            mean: [B, D]
        """
        w = mask.float().unsqueeze(-1)
        denom = w.sum(dim=1).clamp_min(1.0)
        return (x * w).sum(dim=1) / denom
    
    def _attention_weighted_pooling(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Attention-weighted pooling over cluster dimension with importance scores.

        Args:
            x: Cluster features [B, S, D]
            mask: Boolean mask [B, S] (True = include valid clusters)

        Returns:
            pooled: Attention-weighted pooled features [B, D]
            importance_scores: Normalized attention weights [B, S]
        """
        # Compute attention scores
        attention_logits = self.cluster_attention(x).squeeze(-1)  # [B, S]
        
        # Mask out invalid clusters (set to large negative value)
        attention_logits = attention_logits.masked_fill(~mask, -1e9)
        
        # Softmax over valid clusters
        attention_weights = torch.softmax(attention_logits, dim=-1)  # [B, S]
        
        # Zero out attention weights for invalid clusters (for safety)
        attention_weights = attention_weights * mask.float()
        
        # Weighted pooling
        pooled = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)  # [B, D]
        
        return pooled, attention_weights
    
    def _pool_nodes(self, node_features: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Pool node features to graph level."""
        if self.pooling == 'mean':
            return scatter_mean(node_features, batch, dim=0)
        elif self.pooling == 'max':
            return scatter_max(node_features, batch, dim=0)[0]
        elif self.pooling == 'sum':
            return scatter_sum(node_features, batch, dim=0)
        else:
            return scatter_mean(node_features, batch, dim=0)
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute classification loss."""
        return F.cross_entropy(logits, labels)

    def compute_total_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        extra: dict
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute total loss combining classification, VQ, and coverage components.

        Args:
            logits: Classification logits [B, num_classes]
            labels: Ground-truth labels [B]
            extra: Extra outputs from forward() with keys:
                - 'vq_loss': scalar
                - 'presence': [B, K] soft presence per graph
                - 'vq_info': dict with 'perplexity', 'commitment_loss'

        Returns:
            total_loss: Aggregated loss tensor
            metrics: Dict of scalar metrics for logging
        """
        L_cls = F.cross_entropy(logits, labels)
        L_vq = extra["vq_loss"]

        # Probabilistic set cover (coverage) loss
        p_gk = extra["presence"].clamp(0, 1)               # [B, K]
        coverage = 1.0 - torch.prod(1.0 - p_gk, dim=-1)    # [B]
        L_psc = -coverage.mean()

        # Total loss without entropy term
        total = L_cls + self.lambda_vq * L_vq + self.lambda_psc * L_psc

        # Compute entropy for monitoring only (not in loss)
        L_ent = self.codebook.entropy_loss(weight=1.0)

        metrics = {
            "loss/total": float(total.detach().cpu()),
            "loss/cls": float(L_cls.detach().cpu()),
            "loss/vq_commit": float(L_vq.detach().cpu()),
            "loss/psc": float(L_psc.detach().cpu()),
            "metric/entropy": float(L_ent.detach().cpu()),  # monitoring only
            "metric/perplexity": float(extra["vq_info"]["perplexity"].detach().cpu()),
            "codebook/commitment_loss": float(extra["vq_info"]["commitment_loss"].detach().cpu()),
            "coverage/mean": float(coverage.mean().detach().cpu()),
            "metric/presence_mean": float(extra["presence"].mean(dim=0).mean().detach().cpu()),  # batch usage
        }
        return total, metrics
   
    def predict(
        self, 
        h_V: Tuple[torch.Tensor, torch.Tensor], 
        edge_index: torch.Tensor, 
        h_E: Tuple[torch.Tensor, torch.Tensor], 
        seq: Optional[torch.Tensor] = None, 
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get class predictions."""
        with torch.no_grad():
            logits, _, _ = self.forward(h_V, edge_index, h_E, seq, batch)
            return torch.argmax(logits, dim=-1)
    
    def predict_proba(
        self, 
        h_V: Tuple[torch.Tensor, torch.Tensor], 
        edge_index: torch.Tensor, 
        h_E: Tuple[torch.Tensor, torch.Tensor], 
        seq: Optional[torch.Tensor] = None, 
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get class probabilities."""
        with torch.no_grad():
            logits, _, _ = self.forward(h_V, edge_index, h_E, seq, batch)
            return torch.softmax(logits, dim=-1)
    
    def update_epoch(self) -> None:
        """Update epoch counter for temperature annealing."""
        self.partitioner.update_epoch()
    
    def get_cluster_analysis(
        self, 
        h_V: Tuple[torch.Tensor, torch.Tensor], 
        edge_index: torch.Tensor, 
        h_E: Tuple[torch.Tensor, torch.Tensor], 
        seq: Optional[torch.Tensor] = None, 
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Combined function to get cluster importance scores and clustering statistics in one pass.
        
        Args:
            h_V: Node features (scalar, vector)
            edge_index: Edge connectivity
            h_E: Edge features (scalar, vector)
            seq: Optional sequence tensor
            batch: Batch vector
            
        Returns:
            predictions: Class predictions [B]
            probabilities: Class probabilities [B, num_classes]
            importance_scores: Cluster importance weights [B, S]
            stats: Dict with detailed clustering statistics including assignment_matrix
        """
        with torch.no_grad():
            # Single forward pass to get all needed information
            logits, assignment_matrix, extra, cluster_importance = self.forward(
                h_V, edge_index, h_E, seq, batch, return_importance=True
            )
            
            # Get predictions and probabilities
            predictions = torch.argmax(logits, dim=-1)
            probabilities = torch.softmax(logits, dim=-1)
            
            # Handle batch indices
            if batch is None:
                batch = torch.zeros(
                    h_V[0].size(0), 
                    dtype=torch.long, 
                    device=h_V[0].device
                )
            
            # Get node features for statistics 
            if seq is not None and self.seq_in:
                seq_emb = self.sequence_embedding(seq)
                h_V_aug = (torch.cat([h_V[0], seq_emb], dim=-1), h_V[1])
            else:
                h_V_aug = h_V
                
            h_V_processed = self.node_encoder(h_V_aug)
            for layer in self.gvp_layers:
                h_V_processed = layer(h_V_processed, edge_index, self.edge_encoder(h_E))
            
            node_features = self.output_projection(h_V_processed)
            dense_x, mask = to_dense_batch(node_features, batch)
            
            # Compute clustering statistics
            total_nodes = mask.sum(dim=-1).float()
            assigned_nodes = assignment_matrix.sum(dim=(1, 2))
            coverage = assigned_nodes / (total_nodes + 1e-8)
            effective_clusters = (assignment_matrix.sum(dim=1) > 0).sum(dim=-1)
            
            # Compute cluster sizes
            cluster_sizes = assignment_matrix.sum(dim=1)  # [B, S] - size of each cluster
            non_empty_clusters = cluster_sizes > 0
            
            # Average cluster size per protein (only counting non-empty clusters)
            avg_cluster_size_per_protein = []
            for b in range(cluster_sizes.size(0)):
                non_empty = cluster_sizes[b][non_empty_clusters[b]]
                if len(non_empty) > 0:
                    avg_cluster_size_per_protein.append(non_empty.float().mean().item())
                else:
                    avg_cluster_size_per_protein.append(0.0)
            
            avg_cluster_size_per_protein = torch.tensor(avg_cluster_size_per_protein)
            
            # Base statistics
            stats = {
                'assignment_matrix': assignment_matrix,
                'avg_coverage': coverage.mean().item(),
                'min_coverage': coverage.min().item(),
                'max_coverage': coverage.max().item(),
                'avg_clusters': effective_clusters.float().mean().item(),
                'avg_cluster_size': avg_cluster_size_per_protein.mean().item(),
                'min_cluster_size': avg_cluster_size_per_protein.min().item(),
                'max_cluster_size': avg_cluster_size_per_protein.max().item(),
                'total_proteins': len(coverage)
            }
            
            # Add importance statistics
            if cluster_importance is not None:
                # Mask importance scores for valid clusters only
                valid_importance = cluster_importance * non_empty_clusters.float()
                
                # Statistics per protein
                max_importance_per_protein = []
                min_importance_per_protein = []
                entropy_per_protein = []
                
                for b in range(cluster_importance.size(0)):
                    valid_scores = valid_importance[b][non_empty_clusters[b]]
                    if len(valid_scores) > 0:
                        max_importance_per_protein.append(valid_scores.max().item())
                        min_importance_per_protein.append(valid_scores.min().item())
                        # Compute entropy of importance distribution
                        p = valid_scores + 1e-8  # avoid log(0)
                        entropy_per_protein.append((-p * torch.log(p)).sum().item())
                    else:
                        max_importance_per_protein.append(0.0)
                        min_importance_per_protein.append(0.0)
                        entropy_per_protein.append(0.0)
                
                stats.update({
                    'avg_max_importance': np.mean(max_importance_per_protein),
                    'avg_min_importance': np.mean(min_importance_per_protein),
                    'avg_importance_entropy': np.mean(entropy_per_protein),
                    'importance_concentration': 1.0 - np.mean(entropy_per_protein),  # Higher = more concentrated
                })
            
            return predictions, probabilities, cluster_importance, stats
        
    @torch.no_grad()
    def extract_pre_gcn_clusters(
        self,
        h_V: Tuple[torch.Tensor, torch.Tensor],
        edge_index: torch.Tensor,
        h_E: Tuple[torch.Tensor, torch.Tensor],
        seq: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract pre-GCN cluster embeddings and their validity mask (no quantization).

        Args:
            h_V: Node features (scalar, vector)
            edge_index: Edge connectivity
            h_E: Edge features (scalar, vector)
            seq: Optional sequence tensor
            batch: Batch vector

        Returns:
            cluster_features: [B, S, ns] pre-GCN cluster embeddings
            cluster_valid_mask: [B, S] boolean mask for non-empty clusters
        """
        if seq is not None and self.seq_in:
            seq_emb = self.sequence_embedding(seq)
            h_V = (torch.cat([h_V[0], seq_emb], dim=-1), h_V[1])

        h_V_enc = self.node_encoder(h_V)
        h_E_enc = self.edge_encoder(h_E)
        for layer in self.gvp_layers:
            h_V_enc = layer(h_V_enc, edge_index, h_E_enc)

        node_features = self.output_projection(h_V_enc)  # [N, ns]

        if batch is None:
            batch = torch.zeros(node_features.size(0), dtype=torch.long, device=node_features.device)

        dense_x, mask = to_dense_batch(node_features, batch)  # [B, max_N, ns], [B, max_N]
        dense_adj = to_dense_adj(edge_index, batch)           # [B, max_N, max_N]

        cluster_features, _, assignment_matrix = self.partitioner(dense_x, dense_adj, mask)
        cluster_valid_mask = (assignment_matrix.sum(dim=1) > 0)
        return cluster_features, cluster_valid_mask

    @torch.no_grad()
    def kmeans_init_from_loader(
        self,
        loader,
        max_batches: int = 50,
        device: Optional[torch.device] = None
    ) -> None:
        """
        Initialize the codebook with k-means on cached pre-GCN cluster embeddings.

        Args:
            loader: DataLoader yielding batches compatible with model.forward
            max_batches: Maximum number of batches to sample
            device: Optional device override
        """
        self.eval()
        samples = []
        n_seen = 0
        for i, batch_data in enumerate(loader):
            if i >= max_batches:
                break
            
            # Handle PyTorch Geometric Batch object
            if device is not None:
                batch_data = batch_data.to(device)
            
            # Extract features from the batch
            h_V = (batch_data.node_s, batch_data.node_v)
            edge_index = batch_data.edge_index
            h_E = (batch_data.edge_s, batch_data.edge_v)
            batch = batch_data.batch

            clusters, mask = self.extract_pre_gcn_clusters(h_V, edge_index, h_E, batch=batch)
            if mask.any():
                samples.append(clusters[mask].detach().cpu())
                n_seen += int(mask.sum().item())

        if n_seen == 0:
            print("Warning: No valid clusters found for K-means initialization")
            return  # nothing to initialize
        samples = torch.cat(samples, dim=0)  # [N, ns]
        print(f"Initializing codebook with {len(samples)} cluster samples")
        self.codebook.kmeans_init(samples.to(self.codebook.embeddings.device))

    def freeze_backbone_for_codebook(self) -> None:
        """
        Freeze encoder, partitioner, and classifier so only the codebook trains.
        """
        for m in [self.node_encoder, self.edge_encoder, *self.gvp_layers, self.output_projection,
                  self.partitioner, self.cluster_gcn, self.cluster_attention, self.classifier]:
            for p in m.parameters():
                p.requires_grad = False
        for p in self.codebook.parameters():
            p.requires_grad = True

    def unfreeze_all(self) -> None:
        """
        Unfreeze all model parameters (joint fine-tuning).
        """
        for m in [self.node_encoder, self.edge_encoder, *self.gvp_layers, self.output_projection,
                  self.partitioner, self.cluster_gcn, self.cluster_attention, self.classifier, self.codebook]:
            for p in m.parameters():
                p.requires_grad = True

    def forward(
        self, 
        h_V: Tuple[torch.Tensor, torch.Tensor], 
        edge_index: torch.Tensor, 
        h_E: Tuple[torch.Tensor, torch.Tensor], 
        seq: Optional[torch.Tensor] = None, 
        batch: Optional[torch.Tensor] = None,
        return_importance: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            h_V: Node features (scalar, vector)
            edge_index: Edge connectivity
            h_E: Edge features (scalar, vector)
            seq: Sequence features (optional)
            batch: Batch indices (optional)
            
        Returns:
            logits: Classification logits
            assignment_matrix: Node-to-cluster assignments
        """
        # Add sequence features if provided
        if seq is not None and self.seq_in:
            seq_emb = self.sequence_embedding(seq)
            h_V = (torch.cat([h_V[0], seq_emb], dim=-1), h_V[1])
            
        # Encode initial features
        h_V = self.node_encoder(h_V)
        h_E = self.edge_encoder(h_E)
        
        # Process through GVP layers
        for layer in self.gvp_layers:
            h_V = layer(h_V, edge_index, h_E)
            
        # Extract scalar features
        node_features = self.output_projection(h_V)  # [N, ns]
        
        # Handle batch indices
        if batch is None:
            batch = torch.zeros(
                node_features.size(0), 
                dtype=torch.long, 
                device=node_features.device
            )
        
        # Convert to dense format for partitioning
        dense_x, mask = to_dense_batch(node_features, batch)  # [B, max_N, ns]
        dense_adj = to_dense_adj(edge_index, batch)  # [B, max_N, max_N]
        
        # Apply partitioner
        cluster_features, cluster_adj, assignment_matrix = self.partitioner(
            dense_x, dense_adj, mask
        )

        # --- VALID CLUSTER MASK (non-empty clusters) ---
        cluster_valid_mask = (assignment_matrix.sum(dim=1) > 0)  # [B, S]

        # === PRE-GCN QUANTIZATION (discrete dictionary) ===
        quant_clusters, code_indices, vq_loss, vq_info = self.codebook(
            cluster_features, mask=cluster_valid_mask
        )

        # Inter-cluster message passing
        # Choose between quantized or original clusters for classification
        if getattr(self, 'use_quantized_for_classification', True):
            # Use quantized clusters (default behavior)
            refined_clusters = self.cluster_gcn(quant_clusters, cluster_adj)  # [B, S, ns]
        else:
            # Use original clusters for classification, quantized for regularization only
            refined_clusters = self.cluster_gcn(cluster_features, cluster_adj)  # [B, S, ns]
        
        # Attention-weighted pooling with importance scores
        cluster_pooled, cluster_importance = self._attention_weighted_pooling(
            refined_clusters, cluster_valid_mask
        )  # [B, ns], [B, S]
        residue_pooled = self._pool_nodes(node_features, batch)  # [B, ns]
        
        # Combine representations
        combined_features = torch.cat([residue_pooled, cluster_pooled], dim=-1)

        # Classification
        logits = self.classifier(combined_features)
        
        # Use differentiable presence (no torch.no_grad())
        p_gk = self.codebook.differentiable_presence(
            cluster_features, cluster_valid_mask, temperature=self.psc_temp
        )

        # Extra info for loss/metrics (no aux_losses from partitioner)
        extra = {
            "vq_loss": vq_loss,
            "vq_info": vq_info,
            "code_indices": code_indices,
            "presence": p_gk,
        }

        if return_importance:
            return logits, assignment_matrix, extra, cluster_importance
        return logits, assignment_matrix, extra


def create_optimized_model():
    """
    Create an optimized model instance with recommended settings.
    
    Returns:
        ParToken model ready for training
    """
    model = ParTokenModel(
        node_in_dim=(6, 3),         # GVP node dimensions (scalar, vector)
        node_h_dim=(100, 16),       # GVP hidden dimensions  
        edge_in_dim=(32, 1),        # GVP edge dimensions
        edge_h_dim=(32, 1),         # GVP edge hidden dimensions
        num_classes=2,              # Binary classification
        seq_in=False,               # Whether to use sequence features
        num_layers=3,               # Number of GVP layers
        drop_rate=0.1,              # Dropout rate
        pooling='mean',             # Pooling strategy
        max_clusters=5,             # Maximum number of clusters
        termination_threshold=0.95  # Stop when 95% of residues are assigned
    )
    
    print("Optimized model initialized")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model



def demonstrate_model():
    """Demonstrate comprehensive model usage with synthetic data."""
    print("🧬 ParTokenModel Demonstration")
    print("=" * 50)
    
    model = create_optimized_model()
    
    # Create synthetic protein data
    batch_size = 4
    max_nodes = 50
    total_nodes = batch_size * max_nodes
    
    # Node features: (scalar_features, vector_features)
    h_V = (
        torch.randn(total_nodes, 6),      # Scalar features
        torch.randn(total_nodes, 3, 3)    # Vector features (3D vectors)
    )
    
    # Create chain-like connectivity for each protein
    edge_list = []
    for b in range(batch_size):
        start_idx = b * max_nodes
        for i in range(max_nodes - 1):
            # Chain connections
            edge_list.extend([
                [start_idx + i, start_idx + i + 1],
                [start_idx + i + 1, start_idx + i]
            ])
            
            # Add some long-range connections
            if i % 5 == 0 and i + 5 < max_nodes:
                edge_list.extend([
                    [start_idx + i, start_idx + i + 5],
                    [start_idx + i + 5, start_idx + i]
                ])
    
    edge_index = torch.tensor(edge_list).t().contiguous()
    num_edges = edge_index.size(1)
    
    # Edge features: (scalar_features, vector_features)
    h_E = (
        torch.randn(num_edges, 32),       # Scalar edge features
        torch.randn(num_edges, 1, 3)      # Vector edge features
    )
    
    labels = torch.randint(0, 2, (batch_size,))
    batch = torch.repeat_interleave(torch.arange(batch_size), max_nodes)
    
    print("\n📊 Input Data:")
    print(f"  Batch size: {batch_size}")
    print(f"  Max nodes per protein: {max_nodes}")
    print(f"  Total nodes: {total_nodes}")
    print(f"  Node scalar features: {h_V[0].shape}")
    print(f"  Node vector features: {h_V[1].shape}")
    print(f"  Edge index: {edge_index.shape}")
    print(f"  Edge scalar features: {h_E[0].shape}")
    print(f"  Edge vector features: {h_E[1].shape}")
    
    # === 1. TRAINING MODE DEMONSTRATION ===
    print("\n🔥 Training Mode Demonstration:")
    model.train()
    
    # Forward pass with all outputs
    logits, assignment_matrix, extra = model(h_V, edge_index, h_E, batch=batch)
    
    # Compute comprehensive loss
    total_loss, metrics = model.compute_total_loss(logits, labels, extra)
    
    print(f"  Logits shape: {logits.shape}")
    print(f"  Assignment matrix shape: {assignment_matrix.shape}")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Classification loss: {metrics['loss/cls']:.4f}")
    print(f"  VQ loss: {metrics['loss/vq_commit']:.4f}")
    print(f"  Entropy loss: {metrics['metric/entropy']:.6f}")
    print(f"  Coverage loss: {metrics['loss/psc']:.4f}")
    print(f"  Codebook perplexity: {metrics['metric/perplexity']:.2f}")
    print(f"  Mean coverage: {metrics['coverage/mean']:.3f}")
    
    # === 2. INFERENCE MODE DEMONSTRATION ===
    print("\n🎯 Inference Mode Demonstration:")
    model.eval()
    
    with torch.no_grad():
        # Get predictions
        predictions = model.predict(h_V, edge_index, h_E, batch=batch)
        probabilities = model.predict_proba(h_V, edge_index, h_E, batch=batch)
        
        print(f"  Predictions: {predictions}")
        print(f"  Probabilities shape: {probabilities.shape}")
        print(f"  Class 0 probabilities: {probabilities[:, 0].numpy()}")
        print(f"  Class 1 probabilities: {probabilities[:, 1].numpy()}")
    
    # === 3. CLUSTERING STATISTICS ===
    print("\n📈 Clustering Statistics:")
    stats = model.get_clustering_stats(h_V, edge_index, h_E, batch=batch)
    
    print(f"  Average coverage: {stats['avg_coverage']:.3f}")
    print(f"  Coverage range: {stats['min_coverage']:.3f} - {stats['max_coverage']:.3f}")
    print(f"  Average clusters per protein: {stats['avg_clusters']:.1f}")
    print(f"  Average cluster size: {stats['avg_cluster_size']:.1f}")
    print(f"  Cluster size range: {stats['min_cluster_size']:.1f} - {stats['max_cluster_size']:.1f}")
    print(f"  Total proteins processed: {stats['total_proteins']}")
    
    # === 4. VQ CODEBOOK FEATURES ===
    print("\n📚 VQ Codebook Features:")
    print(f"  Codebook size: {model.codebook.K}")
    print(f"  Embedding dimension: {model.codebook.D}")
    print(f"  Code indices shape: {extra['code_indices'].shape}")
    print(f"  Unique codes used: {len(torch.unique(extra['code_indices']))}")
    print(f"  Presence tensor shape: {extra['presence'].shape}")
    
    # === 5. PRE-GCN CLUSTER EXTRACTION ===
    print("\n🔍 Pre-GCN Cluster Analysis:")
    with torch.no_grad():
        cluster_features, cluster_mask = model.extract_pre_gcn_clusters(
            h_V, edge_index, h_E, batch=batch
        )
        
        print(f"  Pre-GCN cluster features: {cluster_features.shape}")
        print(f"  Valid clusters mask: {cluster_mask.shape}")
        print(f"  Total valid clusters: {cluster_mask.sum().item()}")
        print(f"  Valid clusters per protein: {cluster_mask.sum(dim=1).float().tolist()}")
    
    # === 6. MODEL TRAINING UTILITIES ===
    print("\n🛠️  Training Utilities:")
    
    # Demonstrate freezing/unfreezing
    initial_grad_status = next(model.classifier.parameters()).requires_grad
    
    model.freeze_backbone_for_codebook()
    frozen_grad_status = next(model.classifier.parameters()).requires_grad
    
    model.unfreeze_all()
    unfrozen_grad_status = next(model.classifier.parameters()).requires_grad
    
    print(f"  Initial gradient status: {initial_grad_status}")
    print(f"  After freezing backbone: {frozen_grad_status}")
    print(f"  After unfreezing all: {unfrozen_grad_status}")
    
    # Demonstrate epoch update
    old_temp = model.partitioner.get_temperature()
    model.update_epoch()
    new_temp = model.partitioner.get_temperature()
    print(f"  Temperature annealing: {old_temp:.6f} → {new_temp:.6f}")
    
    print("\n✅ Comprehensive demonstration completed!")
    
    print("\n🌟 Key ParTokenModel Features:")
    print("• GVP-based geometric deep learning")
    print("• Hierarchical protein partitioning") 
    print("• Vector Quantization (VQ) codebook")
    print("• Inter-cluster message passing")
    print("• Multi-component loss (classification + VQ + entropy + coverage)")
    print("• Comprehensive clustering statistics")
    print("• Flexible training modes (frozen/joint)")
    print("• Temperature annealing for partitioner")
    print("• Probabilistic set cover regularization")
    print("• Pre-GCN cluster analysis")
    
    return model, stats, metrics


# Backward compatibility aliases
GVPGradientSafeHardGumbelModel = ParTokenModel
GradientSafeVectorizedPartitioner = Partitioner


def create_model_and_train_example():
    """Backward compatibility function."""
    return create_optimized_model()


if __name__ == "__main__":
    demonstrate_model()
    

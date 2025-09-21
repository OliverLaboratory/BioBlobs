import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_mean, scatter_sum, scatter_max
from typing import Tuple, Optional
from utils.VQCodebook import VQCodebookEMA
from utils.inter_cluster import GlobalClusterAttention, FeatureWiseGateFusion
from utils.pnc_partition import Partitioner
from utils.positional_encoder import build_position_encoding
from gnn import GINConv


class PartTokenGNNModel(nn.Module):
    """
    GNN-based Partoken model with efficient partitioning for protein classification.
    
    This model combines GIN (Graph Isomorphism Network) layers for graph neural networks
    with a partitioner for hierarchical clustering, replacing the GVP backbone with 
    standard GNN convolutions while maintaining all Partoken innovations.

    Args:
        embed_dim: Node embedding dimension
        num_classes: Number of output classes
        num_layers: Number of GIN layers
        drop_rate: Dropout rate
        pooling: Pooling strategy ('mean', 'max', 'sum')
        use_edge_attr: Whether to use edge attributes
        pe: Positional encoding type ('learned', 'sine', None)
        max_clusters: Maximum number of clusters
        termination_threshold: Early termination threshold for partitioner
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 2,
        num_layers: int = 3,
        drop_rate: float = 0.1,
        pooling: str = 'mean',
        use_edge_attr: bool = False,
        edge_attr_dim: int = 1,
        pe: Optional[str] = 'learned',
        # Partitioner hyperparameters
        max_clusters: int = 5,
        nhid: int = 50,
        k_hop: int = 2,
        cluster_size_max: int = 15,
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
        lambda_ent: float = 0.0,
        lambda_psc: float = 1e-2,
        lambda_card: float = 0.005,
        psc_temp: float = 0.3
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = drop_rate
        self.pooling = pooling
        self.use_edge_attr = use_edge_attr
        
        # Protein embedding (21 amino acids + mask token)
        NUM_PROTEINS = 20
        NUM_PROTEINS_MASK = NUM_PROTEINS + 1
        self.x_embedding = nn.Embedding(NUM_PROTEINS_MASK, embed_dim)
        
        # Positional embedding
        self.position_embedding = build_position_encoding(embed_dim, pe)
        
        # GIN layers
        self.gin_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gin_layers.append(GINConv(embed_dim, use_edge_attr=use_edge_attr, edge_attr_dim=edge_attr_dim))
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(embed_dim))
        
        # Partitioner
        self.partitioner = Partitioner(
            nfeat=embed_dim,
            max_clusters=max_clusters,
            nhid=nhid,
            k_hop=k_hop,
            cluster_size_max=cluster_size_max,
            termination_threshold=termination_threshold,
            lambda_card=lambda_card
        )
        self.partitioner.tau_init = tau_init
        self.partitioner.tau_min = tau_min
        self.partitioner.tau_decay = tau_decay
        
        # Global-to-cluster attention and feature-wise gating
        self.global_cluster_attn = GlobalClusterAttention(
            dim=embed_dim, heads=4, drop_rate=drop_rate, temperature=1.0
        )
        self.fw_gate = FeatureWiseGateFusion(
            dim=embed_dim, hidden=embed_dim // 2, drop_rate=drop_rate
        )

        # VQ-VAE Codebook
        self.codebook = VQCodebookEMA(
            codebook_size=codebook_size,
            dim=codebook_dim if codebook_dim is not None else embed_dim,
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
        self.lambda_card = lambda_card
        self.psc_temp = psc_temp
        
        # Use original cluster embeddings for classification (not quantized)
        self.use_quantized_for_classification = False
        
        # Attention mechanism for cluster importance
        self.cluster_attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim // 2, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim), 
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(4 * embed_dim, 2 * embed_dim),
            nn.ReLU(inplace=True), 
            nn.Dropout(drop_rate),
            nn.Linear(2 * embed_dim, num_classes)
        )

    def _masked_mean(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute masked mean across specified dimension.
        
        Args:
            x: Input tensor [B, N, D]
            mask: Boolean mask [B, N] (True for valid nodes)
        
        Returns:
            Masked mean tensor [B, D]
        """
        masked_x = x * mask.unsqueeze(-1).float()
        sum_x = masked_x.sum(dim=1)
        count = mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        return sum_x / count
    
    def _attention_weighted_pooling(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention-weighted pooling to cluster features.
        
        Args:
            x: Cluster features [B, max_clusters, D]
            mask: Cluster mask [B, max_clusters]
        
        Returns:
            pooled_features: [B, D]
            attention_weights: [B, max_clusters]
        """
        # Compute attention scores
        attn_scores = self.cluster_attention(x).squeeze(-1)  # [B, max_clusters]
        
        # Mask invalid clusters
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        
        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, max_clusters]
        
        # Weighted sum
        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # [B, D]
        
        return pooled, attn_weights
    
    def _pool_nodes(self, node_features: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Pool node features to graph level."""
        if self.pooling == 'mean':
            return scatter_mean(node_features, batch, dim=0)
        elif self.pooling == 'sum':
            return scatter_sum(node_features, batch, dim=0)
        elif self.pooling == 'max':
            return scatter_max(node_features, batch, dim=0)[0]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
    
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
        Compute total loss including classification, VQ, cardinality, and PSC losses.
        
        Args:
            logits: Classification logits [B, num_classes]
            labels: Ground truth labels [B]
            extra: Dictionary containing additional loss components
        
        Returns:
            total_loss: Combined loss
            metrics: Dictionary of individual loss components
        """
        # Classification loss
        ce_loss = self.compute_loss(logits, labels)
        
        # VQ loss
        vq_loss = extra.get('vq_loss', torch.tensor(0.0, device=logits.device))
        
        # Cardinality loss (from partitioner)
        card_loss = extra.get('cardinality_loss', torch.tensor(0.0, device=logits.device))
        
        # PSC (Probabilistic Slot Coverage) loss
        psc_loss = extra.get('psc_loss', torch.tensor(0.0, device=logits.device))
        
        # Combine losses
        total_loss = (
            ce_loss + 
            self.lambda_vq * vq_loss + 
            self.lambda_card * card_loss + 
            self.lambda_psc * psc_loss
        )
        
        metrics = {
            'total_loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'vq_loss': vq_loss.item(),
            'card_loss': card_loss.item(),
            'psc_loss': psc_loss.item(),
            'num_clusters': extra.get('num_clusters', 0),
        }
        
        return total_loss, metrics
   
    def predict(self, data) -> torch.Tensor:
        """Predict class labels."""
        logits, _, _ = self(data)
        return torch.argmax(logits, dim=-1)
    
    def predict_proba(self, data) -> torch.Tensor:
        """Predict class probabilities."""
        logits, _, _ = self(data)
        return F.softmax(logits, dim=-1)
    
    def update_epoch(self) -> None:
        """Update epoch-dependent parameters (e.g., temperature decay)."""
        self.partitioner.update_epoch()
    
    def get_cluster_analysis(self, data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Get detailed cluster analysis for interpretability.
        
        Returns:
            cluster_features: Cluster representations
            assignment_matrix: Node-to-cluster assignments
            importance_scores: Cluster importance scores
            analysis: Dictionary with clustering statistics
        """
        # Extract data
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Node embedding
        node_features = self.x_embedding(x)
        if self.position_embedding is not None:
            pos = self.position_embedding(data)
            node_features = node_features + pos
        
        # GIN layers
        for layer in range(self.num_layers):
            node_features = self.gin_layers[layer](node_features, edge_index, edge_attr)
            node_features = self.batch_norms[layer](node_features)
            
            if layer == self.num_layers - 1:
                node_features = F.dropout(node_features, self.dropout, training=self.training)
            else:
                node_features = F.dropout(F.relu(node_features), self.dropout, training=self.training)
        
        # Convert to dense format for partitioner (following original ParTokenModel exactly)
        dense_features, node_mask = to_dense_batch(node_features, batch)  # [B, max_N, D], [B, max_N]
        
        # Create dense index mapping for partitioner (like original ParTokenModel)
        dense_index, _ = to_dense_batch(
            torch.arange(node_features.size(0), device=x.device), batch
        )  # [B, max_N]
        
        # Apply partitioner with sparse edge information (adj=None, pass edge_index instead)
        cluster_features, assignment_matrix = self.partitioner(
            dense_features, None, node_mask, edge_index=edge_index, batch_vec=batch, dense_index=dense_index
        )
        
        # Global-cluster attention (needs global pooled features)
        cluster_mask = assignment_matrix.sum(dim=1) > 0  # [B, max_clusters]
        residue_pooled = self._pool_nodes(node_features, batch)  # [B, embed_dim]
        attended_features, importance_scores, _ = self.global_cluster_attn(residue_pooled, cluster_features, cluster_mask)
        
        # Feature-wise gating (like original model)
        fused_features, _beta = self.fw_gate(residue_pooled, attended_features)
        
        analysis = {
            'num_clusters': cluster_mask.sum(dim=1).float().mean().item(),
            'cluster_sizes': assignment_matrix.sum(dim=1).mean(dim=0).cpu().numpy(),
            'assignment_entropy': 0,  # Partitioner doesn't return this separately
        }
        
        return cluster_features, assignment_matrix, importance_scores, analysis
        
    @torch.no_grad()
    def get_cluster_importance(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cluster importance scores for interpretability.
        
        Returns:
            importance_scores: [B, max_clusters] - Attention weights for each cluster
            cluster_assignments: [B, max_nodes, max_clusters] - Node-to-cluster assignments
        """
        _, assignment_matrix, importance_scores, _ = self.get_cluster_analysis(data)
        return importance_scores, assignment_matrix

    @torch.no_grad()
    def get_node_importance(self, data) -> torch.Tensor:
        """
        Get node-level importance scores by aggregating cluster importance.
        
        Returns:
            node_importance: [total_nodes] - Importance score for each node
        """
        importance_scores, assignment_matrix = self.get_cluster_importance(data)
        
        # Weight assignment matrix by cluster importance
        weighted_assignments = assignment_matrix * importance_scores.unsqueeze(1)  # [B, max_nodes, max_clusters]
        
        # Sum across clusters to get node importance
        node_importance_dense = weighted_assignments.sum(dim=-1)  # [B, max_nodes]
        
        # Convert back to sparse format
        batch = data.batch
        batch_size = batch.max().item() + 1
        node_importance = []
        
        for i in range(batch_size):
            batch_mask = (batch == i)
            num_nodes = batch_mask.sum().item()
            node_importance.append(node_importance_dense[i, :num_nodes])
        
        return torch.cat(node_importance, dim=0)

    def freeze_backbone_for_codebook(self) -> None:
        """Freeze GIN backbone layers for codebook training."""
        for param in self.x_embedding.parameters():
            param.requires_grad = False
        for param in self.gin_layers.parameters():
            param.requires_grad = False
        for param in self.batch_norms.parameters():
            param.requires_grad = False
        if self.position_embedding is not None:
            for param in self.position_embedding.parameters():
                param.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(
        self, 
        data,
        return_importance: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Forward pass of the GNN Partoken model.
        
        Args:
            data: PyTorch Geometric data object with x, edge_index, edge_attr, batch
            return_importance: Whether to return importance scores
        
        Returns:
            logits: Classification logits [B, num_classes]
            assignment_matrix: Node-to-cluster assignments [B, max_nodes, max_clusters] 
            extra: Dictionary with additional outputs and loss components
        """
        # Extract data
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Node embedding
        node_features = self.x_embedding(x)
        if self.position_embedding is not None:
            pos = self.position_embedding(data)
            node_features = node_features + pos
        
        # GIN layers
        for layer in range(self.num_layers):
            node_features = self.gin_layers[layer](node_features, edge_index, edge_attr)
            node_features = self.batch_norms[layer](node_features)
            
            if layer == self.num_layers - 1:
                node_features = F.dropout(node_features, self.dropout, training=self.training)
            else:
                node_features = F.dropout(F.relu(node_features), self.dropout, training=self.training)
        
        # Convert to dense format for partitioner (following original ParTokenModel exactly)
        dense_features, node_mask = to_dense_batch(node_features, batch)  # [B, max_N, D], [B, max_N]
        
        # Create dense index mapping like original ParTokenModel
        dense_index, _ = to_dense_batch(
            torch.arange(node_features.size(0), device=node_features.device), batch
        )  # [B, max_N]

        # Apply partitioner - pass None for adjacency and let partitioner handle edge_index
        cluster_features, assignment_matrix = self.partitioner(
            dense_features, None, node_mask, edge_index=edge_index, batch_vec=batch, dense_index=dense_index
        )
        
        # Global-cluster attention (needs global pooled features)
        cluster_mask = assignment_matrix.sum(dim=1) > 0  # [B, max_clusters]
        residue_pooled = self._pool_nodes(node_features, batch)  # [B, embed_dim]
        attended_features, cluster_importance, _ = self.global_cluster_attn(residue_pooled, cluster_features, cluster_mask)
        
        # Feature-wise gating (like original model)
        fused_features, _beta = self.fw_gate(residue_pooled, attended_features)
        
        # VQ-VAE codebook (optional)
        vq_loss = torch.tensor(0.0, device=x.device)
        if self.use_quantized_for_classification:
            quantized_features, vq_loss, _ = self.codebook(fused_features)  # fused_features is [B, D]
            fused_features = quantized_features
        
        # Classification (fused_features is already pooled)
        logits = self.classifier(fused_features)
        
        # Prepare extra outputs
        extra = {
            'vq_loss': vq_loss,
            'cardinality_loss': torch.tensor(0.0, device=x.device),  # Partitioner doesn't return this separately
            'psc_loss': torch.tensor(0.0, device=x.device),         # Partitioner doesn't return this separately
            'num_clusters': cluster_mask.sum(dim=1).float().mean().item(),
            'cluster_features': cluster_features,
            'cluster_mask': cluster_mask,
            'node_mask': node_mask,
        }
        
        if return_importance:
            extra['importance_scores'] = cluster_importance  # From global_cluster_attn
            extra['node_importance'] = self.get_node_importance(data)
        
        return logits, assignment_matrix, extra


def create_optimized_gnn_model(num_classes: int = 2, **kwargs):
    """
    Create an optimized GNN Partoken model instance with recommended settings.
    
    Args:
        num_classes: Number of output classes
        **kwargs: Additional model parameters
    
    Returns:
        PartTokenGNNModel ready for training
    """
    default_config = {
        'embed_dim': 128,
        'num_layers': 3,
        'drop_rate': 0.1,
        'pooling': 'mean',
        'use_edge_attr': True,
        'edge_attr_dim': 1,  # From graph dataset: edge_attr has shape [num_edges, 1]
        'pe': 'learned',
        'max_clusters': 5,
        'cluster_size_max': 15,
        'termination_threshold': 0.95,
        'lambda_card': 0.005
    }
    
    # Update with user provided kwargs
    default_config.update(kwargs)
    
    model = PartTokenGNNModel(
        num_classes=num_classes,
        **default_config
    )
    
    print(f"Created GNN Partoken model with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def demonstrate_gnn_model():
    """Demonstrate comprehensive GNN model usage with synthetic data."""
    print("🧬 GNN Partoken Model Demonstration")
    print("=" * 50)
    
    # Create model
    model = create_optimized_gnn_model(num_classes=2)
    
    # Create synthetic graph data
    batch_size = 4
    max_nodes = 50
    
    # Node features (amino acid indices)
    x_list = []
    edge_indices = []
    edge_attrs = []
    batch_list = []
    residue_idx_list = []
    
    for i in range(batch_size):
        num_nodes = torch.randint(30, max_nodes + 1, (1,)).item()
        
        # Node features (amino acid tokens 0-20)
        x = torch.randint(0, 21, (num_nodes,))
        x_list.append(x)
        
        # Residue indices for positional embedding
        residue_idx = torch.arange(num_nodes)
        residue_idx_list.append(residue_idx)
        
        # Edge connectivity (k-NN graph)
        num_edges = min(num_nodes * 5, num_nodes * (num_nodes - 1) // 2)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_indices.append(edge_index + sum(len(x) for x in x_list[:-1]))  # Offset for batching
        
        # Edge attributes
        edge_attr = torch.randn(num_edges, 32)
        edge_attrs.append(edge_attr)
        
        # Batch assignment
        batch_list.append(torch.full((num_nodes,), i))
    
    # Combine into batch
    x = torch.cat(x_list)
    edge_index = torch.cat(edge_indices, dim=1)
    edge_attr = torch.cat(edge_attrs, dim=0)
    batch = torch.cat(batch_list)
    residue_idx = torch.cat(residue_idx_list)
    
    # Create PyTorch Geometric data object
    from torch_geometric.data import Data
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=batch,
        residue_idx=residue_idx
    )
    
    labels = torch.randint(0, 2, (batch_size,))
    
    print("\n📊 Input Data:")
    print(f"  Batch size: {batch_size}")
    print(f"  Total nodes: {x.size(0)}")
    print(f"  Total edges: {edge_index.size(1)}")
    print(f"  Node features shape: {x.shape}")
    print(f"  Edge features shape: {edge_attr.shape}")
    
    # === 1. TRAINING MODE DEMONSTRATION ===
    print("\n🔥 Training Mode Demonstration:")
    model.train()
    
    # Forward pass
    logits, assignment_matrix, extra = model(data, return_importance=True)
    
    # Compute total loss
    total_loss, metrics = model.compute_total_loss(logits, labels, extra)
    
    print(f"  Logits shape: {logits.shape}")
    print(f"  Assignment matrix shape: {assignment_matrix.shape}")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Classification accuracy: {(logits.argmax(dim=1) == labels).float().mean():.3f}")
    print(f"  Average clusters used: {extra['num_clusters']:.2f}")
    
    print("\n📈 Loss Components:")
    for key, value in metrics.items():
        if 'loss' in key:
            print(f"    {key}: {value:.6f}")
    
    # === 2. EVALUATION MODE DEMONSTRATION ===
    print("\n🧪 Evaluation Mode Demonstration:")
    model.eval()
    
    with torch.no_grad():
        # Predictions
        pred_labels = model.predict(data)
        pred_probs = model.predict_proba(data)
        
        print(f"  Predicted labels: {pred_labels}")
        print(f"  True labels: {labels}")
        print(f"  Prediction probabilities shape: {pred_probs.shape}")
        print(f"  Max probability: {pred_probs.max(dim=1)[0].mean():.3f}")
    
    # === 3. INTERPRETABILITY DEMONSTRATION ===
    print("\n🔍 Interpretability Analysis:")
    
    with torch.no_grad():
        # Cluster analysis
        cluster_features, assignment_matrix, importance_scores, analysis = model.get_cluster_analysis(data)
        
        print(f"  Cluster features shape: {cluster_features.shape}")
        print(f"  Average clusters per graph: {analysis['num_clusters']:.2f}")
        print(f"  Assignment entropy: {analysis.get('assignment_entropy', 'N/A')}")
        
        # Node importance
        node_importance = model.get_node_importance(data)
        print(f"  Node importance shape: {node_importance.shape}")
        print(f"  Max node importance: {node_importance.max():.4f}")
        print(f"  Min node importance: {node_importance.min():.4f}")
    
    print("\n✅ GNN Partoken Model demonstration completed successfully!")
    return model, data, labels


# Backward compatibility aliases
GNNPartTokenModel = PartTokenGNNModel


if __name__ == "__main__":
    demonstrate_gnn_model()

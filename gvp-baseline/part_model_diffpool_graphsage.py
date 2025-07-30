import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum, scatter_max
from torch_geometric.utils import to_dense_adj, to_dense_batch
import gvp
import gvp.data  
from gvp.models import GVP, GVPConvLayer, LayerNorm


class BatchedGraphSAGE(nn.Module):
    def __init__(self, infeat, outfeat, use_bn=True, mean=False, add_self=False):
        super().__init__()
        self.add_self = add_self
        self.use_bn = use_bn
        self.mean = mean
        self.W = nn.Linear(infeat, outfeat, bias=True)
        nn.init.xavier_uniform_(self.W.weight, gain=nn.init.calculate_gain("relu"))
        
        if self.use_bn:
            self.bn = nn.BatchNorm1d(outfeat)

    def forward(self, x, adj):
        if self.add_self:
            adj = adj + torch.eye(adj.size(1), device=adj.device).unsqueeze(0)

        if self.mean:
            adj = adj / (adj.sum(-1, keepdim=True) + 1e-8)

        h_k_N = torch.matmul(adj, x)
        h_k = self.W(h_k_N)
        h_k = F.normalize(h_k, dim=2, p=2)
        h_k = F.relu(h_k)
        
        if self.use_bn:
            # Reshape for batch norm: [batch_size * num_nodes, features]
            batch_size, num_nodes, features = h_k.shape
            h_k = h_k.view(-1, features)
            h_k = self.bn(h_k)
            h_k = h_k.view(batch_size, num_nodes, features)
            
        return h_k


class DiffPoolAssignment(nn.Module):
    def __init__(self, nfeat, nnext):
        super().__init__()
        self.assign_mat = BatchedGraphSAGE(nfeat, nnext, use_bn=True)

    def forward(self, x, adj):
        s_l_init = self.assign_mat(x, adj)
        s_l = F.softmax(s_l_init, dim=-1)
        return s_l


class BatchedDiffPool(nn.Module):
    def __init__(self, nfeat, nnext, nhid):
        super().__init__()
        self.embed = BatchedGraphSAGE(nfeat, nhid, use_bn=True)
        self.assign = DiffPoolAssignment(nfeat, nnext)

    def forward(self, x, adj):
        z_l = self.embed(x, adj)
        s_l = self.assign(x, adj)
        
        xnext = torch.matmul(s_l.transpose(-1, -2), z_l)
        anext = torch.matmul(torch.matmul(s_l.transpose(-1, -2), adj), s_l)
        
        # Compute auxiliary losses
        entropy_loss = -torch.sum(s_l * torch.log(s_l + 1e-8), dim=-1).mean()
        link_pred_loss = F.mse_loss(torch.matmul(s_l, s_l.transpose(-1, -2)), adj)
        
        aux_losses = {
            'entropy': entropy_loss,        # ← Changed from 'EntropyLoss'
            'link_pred': link_pred_loss     # ← Changed from 'LinkPredLoss'
        }
        
        return xnext, anext, s_l, aux_losses


class SimpleGCN(nn.Module):
    """Simple GCN layer for cluster message passing"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        
    def forward(self, x, adj):
        # x: [batch_size, num_nodes, features]
        # adj: [batch_size, num_nodes, num_nodes]
        
        # Normalize adjacency matrix (add self-loops and degree normalization)
        adj = adj + torch.eye(adj.size(-1), device=adj.device).unsqueeze(0)
        degree = adj.sum(dim=-1, keepdim=True)
        adj_norm = adj / (degree + 1e-8)
        
        # Message passing: A * X * W
        h = torch.matmul(adj_norm, x)
        h = self.linear(h)
        return F.relu(h)


class GVPDiffPoolGraphSAGEModel(nn.Module):
    """
    GVP-GNN with GraphSAGE-based DiffPool integration for protein classification
    """
    def __init__(self, node_in_dim, node_h_dim, 
                 edge_in_dim, edge_h_dim,
                 num_classes=2, seq_in=False, num_layers=3, 
                 drop_rate=0.1, pooling='mean', max_clusters=30,
                 entropy_weight=0.1, link_pred_weight=0.5):
        
        super().__init__()
        
        self.num_classes = num_classes
        self.pooling = pooling
        self.entropy_weight = entropy_weight
        self.link_pred_weight = link_pred_weight
        
        if seq_in:
            self.W_s = nn.Embedding(20, 20)
            node_in_dim = (node_in_dim[0] + 20, node_in_dim[1])
        
        # GVP layers
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
            GVP(node_h_dim, (ns, 0)))
        
        # GraphSAGE-based DiffPool layer
        self.diffpool = BatchedDiffPool(nfeat=ns, nnext=max_clusters, nhid=ns)
        
        # Cluster GCN for inter-cluster message passing
        self.cluster_gcn = nn.Sequential(
            SimpleGCN(ns, ns),
            nn.Dropout(p=drop_rate),
            SimpleGCN(ns, ns),
            nn.Dropout(p=drop_rate)
        )
        
        # Classification head - input is 2*ns (residue_pooled + cluster_pooled)
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
        if seq is not None:
            seq = self.W_s(seq)
            h_V = (torch.cat([h_V[0], seq], dim=-1), h_V[1])
            
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        
        # Process through GVP layers
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
            
        # Extract scalar features for DiffPool
        residue_features = self.W_out(h_V)  # [num_nodes, ns]
        
        # Convert to dense format using PyTorch Geometric utilities
        if batch is None:
            # Single graph case - create dummy batch
            batch = torch.zeros(residue_features.size(0), dtype=torch.long, device=residue_features.device)
        
        # Convert to dense batched format
        dense_x, mask = to_dense_batch(residue_features, batch)  # [batch_size, max_nodes, features]
        dense_adj = to_dense_adj(edge_index, batch)  # [batch_size, max_nodes, max_nodes]
        
        # Apply DiffPool
        cluster_features, pooled_adj, assignment_matrix, aux_losses = self.diffpool(dense_x, dense_adj)
        
        # Inter-cluster message passing
        refined_cluster_features = self.cluster_gcn[0](cluster_features, pooled_adj)
        refined_cluster_features = self.cluster_gcn[1](refined_cluster_features)  # Dropout
        refined_cluster_features = self.cluster_gcn[2](refined_cluster_features, pooled_adj)
        refined_cluster_features = self.cluster_gcn[3](refined_cluster_features)  # Dropout
        
        # Pool cluster features to graph level
        cluster_pooled = refined_cluster_features.mean(dim=1)  # [batch_size, ns]
        
        # Pool residue features to graph level
        residue_pooled = self._pool_nodes(residue_features, batch)  # [batch_size, ns]
        
        # Concatenate residue and cluster representations
        combined_features = torch.cat([residue_pooled, cluster_pooled], dim=-1)  # [batch_size, 2*ns]
        
        # Classification
        logits = self.classifier(combined_features)
        
        return logits, aux_losses
    
    def _pool_nodes(self, node_features, batch):
        """Pool node features to get graph-level representation"""
        if self.pooling == 'mean':
            return scatter_mean(node_features, batch, dim=0)
        elif self.pooling == 'max':
            return scatter_max(node_features, batch, dim=0)[0]
        elif self.pooling == 'sum':
            return scatter_sum(node_features, batch, dim=0)
        elif self.pooling == 'attention':
            # Note: This attention pooling is simplified since we removed the attention layer
            # from __init__ as it was based on the old combined_dim
            return scatter_mean(node_features, batch, dim=0)
        
        return scatter_mean(node_features, batch, dim=0)  # default to mean
    
    def compute_total_loss(self, logits, labels, aux_losses):
        """Compute total loss including classification and auxiliary losses"""
        classification_loss = F.cross_entropy(logits, labels)
        
        entropy_loss = aux_losses.get('entropy', torch.tensor(0.0, device=logits.device))
        link_pred_loss = aux_losses.get('link_pred', torch.tensor(0.0, device=logits.device))
        
        total_loss = (classification_loss + 
                     self.entropy_weight * entropy_loss + 
                     self.link_pred_weight * link_pred_loss)
        
        return total_loss, {
            'classification': classification_loss,
            'entropy': entropy_loss,
            'link_pred': link_pred_loss,
            'total': total_loss
        }

    def predict(self, h_V, edge_index, h_E, seq=None, batch=None):
        """Get class predictions"""
        with torch.no_grad():
            logits, _ = self.forward(h_V, edge_index, h_E, seq, batch)
            return torch.argmax(logits, dim=-1)
    
    def predict_proba(self, h_V, edge_index, h_E, seq=None, batch=None):
        """Get class probabilities"""
        with torch.no_grad():
            logits, _ = self.forward(h_V, edge_index, h_E, seq, batch)
            return torch.softmax(logits, dim=-1)


def test_model():
    """Simple test function"""
    print("Testing GVPDiffPoolGraphSAGEModel...")
    
    model = GVPDiffPoolGraphSAGEModel(
        node_in_dim=(6, 3),
        node_h_dim=(16, 8),
        edge_in_dim=(32, 1),
        edge_h_dim=(32, 4),
        num_classes=3,
        max_clusters=5
    )
    
    # Test data
    num_nodes = 8
    node_s = torch.randn(num_nodes, 6)
    node_v = torch.randn(num_nodes, 3, 3)
    edge_s = torch.randn(12, 32)
    edge_v = torch.randn(12, 1, 3)
    edge_index = torch.randint(0, num_nodes, (2, 12))
    
    h_V = (node_s, node_v)
    h_E = (edge_s, edge_v)
    
    # Single graph test
    logits, aux_losses = model(h_V, edge_index, h_E)
    print(f"Single graph - Logits: {logits.shape}, Aux losses: {list(aux_losses.keys())}")
    
    # Batched test
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1])
    batch_logits, batch_aux_losses = model(h_V, edge_index, h_E, batch=batch)
    print(f"Batched - Logits: {batch_logits.shape}, Aux losses: {list(batch_aux_losses.keys())}")
    
    print("Test passed!")


if __name__ == "__main__":
    test_model()
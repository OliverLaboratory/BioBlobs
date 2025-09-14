import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum, scatter_max
from gvp.models import GVP, GVPConvLayer, LayerNorm


class BaselineGVPModel(nn.Module):
    '''
    GVP-GNN for Multi-Class Classification.
    
    Takes in protein structure graphs of type `torch_geometric.data.Data` 
    or `torch_geometric.data.Batch` and returns class logits for
    each graph in the batch in a `torch.Tensor` of shape [batch_size, num_classes]
    
    Should be used with `gvp.data.ProteinGraphDataset`, or with generators
    of `torch_geometric.data.Batch` objects with the same attributes.
    
    :param node_in_dim: node dimensions in input graph, should be
                        (6, 3) if using original features
    :param node_h_dim: node dimensions to use in GVP-GNN layers
    :param edge_in_dim: edge dimensions in input graph, should be
                        (32, 1) if using original features
    :param edge_h_dim: edge dimensions to embed to before use
                       in GVP-GNN layers
    :param num_classes: number of classes for classification
    :seq_in: if `True`, sequences will also be passed in with
             the forward pass; otherwise, sequence information
             is assumed to be part of input node embeddings
    :param num_layers: number of GVP-GNN layers
    :param drop_rate: rate to use in all dropout layers
    :param pooling: pooling method for graph-level representation
                   ('mean', 'max', 'sum', 'attention')
    '''
    def __init__(self, node_in_dim, node_h_dim, 
                 edge_in_dim, edge_h_dim,
                 num_classes=2, seq_in=False, num_layers=3, 
                 drop_rate=0.1, pooling='mean'):
        
        super(BaselineGVPModel, self).__init__()
        
        self.num_classes = num_classes
        self.pooling = pooling
        
        if seq_in:
            self.W_s = nn.Embedding(20, 20)
            node_in_dim = (node_in_dim[0] + 20, node_in_dim[1])
        
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
        
        # Attention pooling layer (optional)
        if pooling == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(ns, ns // 2),
                nn.ReLU(inplace=True),
                nn.Linear(ns // 2, 1)
            )
            
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(ns, 2*ns), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(2*ns, ns),
            nn.ReLU(inplace=True), 
            nn.Dropout(p=drop_rate),
            nn.Linear(ns, num_classes)
        )

    def forward(self, h_V, edge_index, h_E, seq=None, batch=None):      
        '''
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param seq: if not `None`, int `torch.Tensor` of shape [num_nodes]
                    to be embedded and appended to `h_V`
        :param batch: batch indices for pooling, if `None` assumes single graph
        :return: logits of shape [batch_size, num_classes]
        '''
        if seq is not None:
            seq = self.W_s(seq)
            h_V = (torch.cat([h_V[0], seq], dim=-1), h_V[1])
            
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
            
        out = self.W_out(h_V)  # [num_nodes, ns]
        
        # Graph-level pooling
        if batch is None: 
            # Single graph case
            out = self._pool_nodes(out, None)
        else: 
            # Batch case
            out = self._pool_nodes(out, batch)
        
        # Classification
        logits = self.classifier(out)  # [batch_size, num_classes]
        
        return logits
    
    def _pool_nodes(self, node_features, batch):
        '''
        Pool node features to get graph-level representation
        '''
        if batch is None:
            # Single graph
            if self.pooling == 'mean':
                return node_features.mean(dim=0, keepdim=True)
            elif self.pooling == 'max':
                return node_features.max(dim=0, keepdim=True)[0]
            elif self.pooling == 'sum':
                return node_features.sum(dim=0, keepdim=True)
            elif self.pooling == 'attention':
                weights = torch.softmax(self.attention(node_features), dim=0)
                return (weights * node_features).sum(dim=0, keepdim=True)
        else:
            # Batched graphs
            if self.pooling == 'mean':
                return scatter_mean(node_features, batch, dim=0)
            elif self.pooling == 'max':
                return scatter_max(node_features, batch, dim=0)[0]
            elif self.pooling == 'sum':
                return scatter_sum(node_features, batch, dim=0)
            elif self.pooling == 'attention':
                weights = torch.softmax(self.attention(node_features), dim=0)
                weighted_features = weights * node_features
                return scatter_sum(weighted_features, batch, dim=0)
                
        return node_features

    def predict(self, h_V, edge_index, h_E, seq=None, batch=None):
        '''
        Get class predictions (argmax of logits)
        '''
        with torch.no_grad():
            logits = self.forward(h_V, edge_index, h_E, seq, batch)
            return torch.argmax(logits, dim=-1)
    
    def predict_proba(self, h_V, edge_index, h_E, seq=None, batch=None):
        '''
        Get class probabilities (softmax of logits)
        '''
        with torch.no_grad():
            logits = self.forward(h_V, edge_index, h_E, seq, batch)
            return torch.softmax(logits, dim=-1)

def test_baseline_gvp_model():
    """
    Test function for the BaselineGVPModel class.
    Tests basic functionality including classification, prediction methods, and different pooling strategies.
    """
    print("Testing BaselineGVPModel...")
    
    # Example dimensions (default if using ProteinGraphDataset)
    node_in_dim = (6, 3)
    node_h_dim = (16, 8)
    edge_in_dim = (32, 1)
    edge_h_dim = (32, 4)
    num_classes = 3
    num_layers = 3
    drop_rate = 0.1

    # Test different pooling methods
    pooling_methods = ['mean', 'max', 'sum', 'attention']
    
    for pooling in pooling_methods:
        print(f"\n=== Testing with {pooling} pooling ===")
        
        # Instantiate the BaselineGVPModel
        model = BaselineGVPModel(
            node_in_dim=node_in_dim,
            node_h_dim=node_h_dim,
            edge_in_dim=edge_in_dim,
            edge_h_dim=edge_h_dim,
            num_classes=num_classes,
            seq_in=False,
            num_layers=num_layers,
            drop_rate=drop_rate,
            pooling=pooling
        )
        print(f"Model created with {num_layers} layers, {pooling} pooling, {num_classes} classes")

        # Create random input data for testing (5 nodes, 10 edges)
        num_nodes = 5
        num_edges = 10
        
        # Node features: scalar and vector features
        node_s = torch.randn(num_nodes, node_in_dim[0])
        node_v = torch.randn(num_nodes, node_in_dim[1], 3)
        
        # Edge features: scalar and vector features
        edge_s = torch.randn(num_edges, edge_in_dim[0])
        edge_v = torch.randn(num_edges, edge_in_dim[1], 3)
        
        # Edge connectivity
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        h_V = (node_s, node_v)
        h_E = (edge_s, edge_v)

        print(f"Input shapes - Nodes: {num_nodes}, Edges: {num_edges}")
        print(f"Node features: scalar {node_s.shape}, vector {node_v.shape}")
        print(f"Edge features: scalar {edge_s.shape}, vector {edge_v.shape}")
        
        # Test 1: Basic forward pass (single graph)
        print("\n--- Test 1: Single Graph Classification ---")
        logits = model(h_V, edge_index, h_E)
        print(f"Classification logits shape: {logits.shape}")
        print(f"Expected shape: (1, {num_classes})")
        print(f"Logits values: {logits}")
        
        # Test 2: Prediction methods
        print("\n--- Test 2: Prediction Methods ---")
        predictions = model.predict(h_V, edge_index, h_E)
        probabilities = model.predict_proba(h_V, edge_index, h_E)
        
        print(f"Predictions shape: {predictions.shape}")
        print(f"Predicted class: {predictions.item()}")
        print(f"Probabilities shape: {probabilities.shape}")
        print(f"Class probabilities: {probabilities}")
        print(f"Sum of probabilities: {probabilities.sum().item():.4f}")
        
        # Test 3: Batched processing
        print("\n--- Test 3: Batched Processing ---")
        
        # Create batch data (2 graphs: first with 3 nodes, second with 2 nodes)
        batch_node_s = torch.randn(5, node_in_dim[0])  # 3 + 2 = 5 nodes total
        batch_node_v = torch.randn(5, node_in_dim[1], 3)
        batch_edge_s = torch.randn(8, edge_in_dim[0])  # Some edges
        batch_edge_v = torch.randn(8, edge_in_dim[1], 3)
        batch_edge_index = torch.tensor([[0, 1, 1, 2, 3, 3, 4, 4],
                                        [1, 0, 2, 1, 4, 4, 3, 3]])  # Example edges
        batch_indices = torch.tensor([0, 0, 0, 1, 1])  # Graph assignment
        
        batch_h_V = (batch_node_s, batch_node_v)
        batch_h_E = (batch_edge_s, batch_edge_v)
        
        batch_logits = model(batch_h_V, batch_edge_index, batch_h_E, batch=batch_indices)
        batch_predictions = model.predict(batch_h_V, batch_edge_index, batch_h_E, batch=batch_indices)
        batch_probabilities = model.predict_proba(batch_h_V, batch_edge_index, batch_h_E, batch=batch_indices)
        
        print(f"Batch logits shape: {batch_logits.shape}")
        print(f"Expected shape: (2, {num_classes})")  # 2 graphs
        print(f"Batch predictions: {batch_predictions}")
        print(f"Batch probabilities shape: {batch_probabilities.shape}")
        print(f"Graph 1 probabilities: {batch_probabilities[0]}")
        print(f"Graph 2 probabilities: {batch_probabilities[1]}")
        
    # Test 4: Model with sequence input
    print("\n=== Test 4: Model with Sequence Input ===")
    
    model_with_seq = BaselineGVPModel(
        node_in_dim=node_in_dim,
        node_h_dim=node_h_dim,
        edge_in_dim=edge_in_dim,
        edge_h_dim=edge_h_dim,
        num_classes=num_classes,
        seq_in=True,  # Enable sequence input
        num_layers=num_layers,
        drop_rate=drop_rate,
        pooling='mean'
    )
    
    # Create sequence data
    seq = torch.randint(0, 20, (num_nodes,))  # 20 amino acids
    
    seq_logits = model_with_seq(h_V, edge_index, h_E, seq=seq)
    seq_predictions = model_with_seq.predict(h_V, edge_index, h_E, seq=seq)
    
    print(f"Model with sequence - logits shape: {seq_logits.shape}")
    print(f"Model with sequence - predictions: {seq_predictions}")
    print(f"Sequence input: {seq}")
    
    # Test 5: Model parameter count
    print("\n=== Test 5: Model Information ===")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Test 6: Gradient flow
    print("\n=== Test 6: Gradient Flow Test ===")
    
    model.train()
    logits = model(h_V, edge_index, h_E)
    
    # Create dummy target and compute loss
    target = torch.randint(0, num_classes, (1,))
    loss = F.cross_entropy(logits, target)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0 
                       for p in model.parameters())
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Target class: {target.item()}")
    print(f"Gradients computed: {has_gradients}")
    
    print("\n=== All BaselineGVPModel Tests Completed Successfully! ===")


if __name__ == "__main__":
    test_baseline_gvp_model()
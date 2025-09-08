import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum, scatter_max
import gvp
import gvp.data
from gvp.models import GVP, GVPConvLayer, LayerNorm



class GVPPartModel(nn.Module):
    '''
    GVP-GNN for neural partitioning protein graph.
    
    Takes in protein structure graphs of type `torch_geometric.data.Data` 
    or `torch_geometric.data.Batch` and returns 
    
    The model applies GVP-GNN layers to generate node embeddings, which can
    optionally be returned along with graph-level embeddings.
    
    :param node_in_dim: node dimensions in input graph, should be
                        (6, 3) if using original features
    :param node_h_dim: node dimensions to use in GVP-GNN layers
    :param node_in_dim: edge dimensions in input graph, should be
                        (32, 1) if using original features
    :param edge_h_dim: edge dimensions to embed to before use
                       in GVP-GNN layers
    :param num_layers: number of GVP-GNN layers
    :param drop_rate: rate to use in all dropout layers
    '''
    def __init__(self, node_in_dim, node_h_dim, edge_in_dim, edge_h_dim,
                 num_layers=3, drop_rate=0.1):
        
        super(GVPPartModel, self).__init__()
        
        self.num_layers = num_layers
        self.drop_rate = drop_rate
        self.node_h_dim = node_h_dim
        
        self.W_v = nn.Sequential(
            gvp.LayerNorm(node_in_dim),
            gvp.GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            gvp.LayerNorm(edge_in_dim),
            gvp.GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )
        
        self.layers = nn.ModuleList(
                gvp.GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) 
            for _ in range(num_layers))
        
        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            gvp.LayerNorm(node_h_dim),
            gvp.GVP(node_h_dim, (ns, 0), activations=(None, None)))

        self.dense = nn.Sequential(
            nn.Linear(ns, 2*ns), nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(2*ns, 1))
    
    def forward(self, h_V, edge_index, h_E, seq, return_embeddings=None, 
                graph_agg_method='mean'):
        '''
        Forward pass of the CPD model.
        
        :param h_V: tuple (s, V) of node features
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge features  
        :param seq: unused, kept for compatibility
        :param return_embeddings: str, controls what to return
                                 - None: return contact logits only (default)
                                 - 'node': return node embeddings only
                                 - 'graph': return graph embedding only  
                                 - 'both': return both node and graph embeddings
                                 - 'all': return embeddings and logits
        :param graph_agg_method: str, aggregation method for graph embedding
                                ('mean', 'sum', 'max')
        
        :returns: depending on return_embeddings:
                 - None: contact prediction logits
                 - 'node': node embeddings (s, V)
                 - 'graph': graph embedding (s, V)
                 - 'both': dict with 'node' and 'graph' embeddings
                 - 'all': dict with 'node', 'graph', and 'logits'
        '''
        
        # Embed input features
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        
        # Apply GVP-GNN layers (encoder)
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        
        # Store node embeddings after encoder layers
        node_embeddings = h_V
        
        # Handle different return modes
        if return_embeddings == 'node':
            return node_embeddings
        elif return_embeddings == 'graph':
            return self._compute_graph_embedding(node_embeddings, None, 
                                                graph_agg_method)
        elif return_embeddings == 'both':
            graph_emb = self._compute_graph_embedding(node_embeddings, None, 
                                                     graph_agg_method)
            return {
                'node': node_embeddings,
                'graph': graph_emb
            }
        
        # Continue with normal forward pass for contact prediction
        out = self.W_out(h_V)
        
        # Contact prediction through outer product
        out = self.dense(out).squeeze(-1)
        out = out.unsqueeze(0) + out.unsqueeze(1)
        
        if return_embeddings == 'all':
            graph_emb = self._compute_graph_embedding(node_embeddings, None, 
                                                     graph_agg_method)
            return {
                'node': node_embeddings,
                'graph': graph_emb,
                'logits': out
            }
        
        return out
    
    def _compute_graph_embedding(self, node_embeddings, batch_indices, agg_method='mean'):
        '''
        Compute graph-level embedding by aggregating node embeddings.
        
        :param node_embeddings: tuple (s, V) of node embeddings
        :param batch_indices: batch assignment for each node (for batched processing)
                             Should be None for single graphs or torch.Tensor with dtype int64
        :param agg_method: aggregation method ('mean', 'sum', 'max')
        
        :returns: tuple (s_graph, V_graph) of graph embeddings
        '''
        s, V = node_embeddings
        
        # Handle single graph case (no batch dimension)
        if batch_indices is None:
            if agg_method == 'mean':
                s_graph = torch.mean(s, dim=0, keepdim=True)
                V_graph = torch.mean(V, dim=0, keepdim=True)
            elif agg_method == 'sum':
                s_graph = torch.sum(s, dim=0, keepdim=True)
                V_graph = torch.sum(V, dim=0, keepdim=True)
            elif agg_method == 'max':
                s_graph = torch.max(s, dim=0, keepdim=True)[0]
                V_graph = torch.max(V, dim=0, keepdim=True)[0]
            else:
                raise ValueError(f"Unsupported aggregation method: {agg_method}")
        else:
            # Handle batched case - ensure batch_indices is int64
            if batch_indices.dtype != torch.int64:
                batch_indices = batch_indices.long()
                
            if agg_method == 'mean':
                s_graph = scatter_mean(s, batch_indices, dim=0)
                V_graph = scatter_mean(V, batch_indices, dim=0)
            elif agg_method == 'sum':
                s_graph = scatter_sum(s, batch_indices, dim=0)
                V_graph = scatter_sum(V, batch_indices, dim=0)
            elif agg_method == 'max':
                s_graph = scatter_max(s, batch_indices, dim=0)[0]
                V_graph = scatter_max(V, batch_indices, dim=0)[0]
            else:
                raise ValueError(f"Unsupported aggregation method: {agg_method}")
        
        return (s_graph, V_graph)
    
    def get_node_embeddings(self, h_V, edge_index, h_E, seq=None):
        '''
        Convenience method to get only node embeddings.
        
        :param h_V: tuple (s, V) of node features
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge features
        :param seq: unused, kept for compatibility
        
        :returns: tuple (s, V) of node embeddings after encoder layers
        '''
        return self.forward(h_V, edge_index, h_E, seq, return_embeddings='node')
    
    def get_graph_embedding(self, h_V, edge_index, h_E, seq=None, 
                           batch_indices=None, agg_method='mean'):
        '''
        Convenience method to get only graph embedding.
        
        :param h_V: tuple (s, V) of node features
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge features
        :param seq: unused, kept for compatibility
        :param batch_indices: batch assignment for each node
        :param agg_method: aggregation method ('mean', 'sum', 'max')
        
        :returns: tuple (s, V) of graph-level embeddings
        '''
        return self.forward(h_V, edge_index, h_E, seq, 
                          return_embeddings='graph', 
                          graph_agg_method=agg_method)
    
    def get_all_outputs(self, h_V, edge_index, h_E, seq=None,
                       batch_indices=None, agg_method='mean'):
        '''
        Convenience method to get node embeddings, graph embedding, and logits.
        
        :param h_V: tuple (s, V) of node features
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge features
        :param seq: unused, kept for compatibility
        :param batch_indices: batch assignment for each node
        :param agg_method: aggregation method ('mean', 'sum', 'max')
        
        :returns: dict with keys 'node', 'graph', 'logits'
        '''
        return self.forward(h_V, edge_index, h_E, seq,
                          return_embeddings='all',
                          graph_agg_method=agg_method)
    
def main():
    """
    Test function for the PartModel class.
    Tests basic functionality and all embedding extraction methods.
    """
    print("Testing PartModel...")
    
    # Example dimensions (default if using ProteinGraphDataset)
    node_in_dim = (6, 3)
    node_h_dim = (16, 8)
    edge_in_dim = (32, 1)
    edge_h_dim = (32, 4)
    num_layers = 3
    drop_rate = 0.1

    # Instantiate the PartModel
    model = GVPPartModel(
        node_in_dim,
        node_h_dim,
        edge_in_dim,
        edge_h_dim,
        num_layers=num_layers,
        drop_rate=drop_rate
    )
    print(f"Model created with {num_layers} layers")

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
    
    # Sequence (for compatibility, not used in contact prediction)
    seq = torch.randint(0, 20, (num_nodes,))  # 20 amino acids

    h_V = (node_s, node_v)
    h_E = (edge_s, edge_v)

    print(f"Input shapes - Nodes: {num_nodes}, Edges: {num_edges}")
    print(f"Node features: scalar {node_s.shape}, vector {node_v.shape}")
    print(f"Edge features: scalar {edge_s.shape}, vector {edge_v.shape}")
    
    # Test 1: Basic forward pass (contact prediction)
    print("\n=== Test 1: Contact Prediction ===")
    logits = model(h_V, edge_index, h_E, seq)
    print(f"Contact prediction logits shape: {logits.shape}")
    print(f"Expected shape: ({num_nodes}, {num_nodes})")
    
    # Test 2: Node embeddings only
    print("\n=== Test 2: Node Embeddings ===")
    node_emb = model.get_node_embeddings(h_V, edge_index, h_E, seq)
    node_s_emb, node_v_emb = node_emb
    print(f"Node embeddings - scalar: {node_s_emb.shape}, vector: {node_v_emb.shape}")
    
    # Test 3: Graph embedding only
    print("\n=== Test 3: Graph Embeddings ===")
    for agg_method in ['mean', 'sum', 'max']:
        graph_emb = model.get_graph_embedding(h_V, edge_index, h_E, seq, 
                                            agg_method=agg_method)
        graph_s_emb, graph_v_emb = graph_emb
        print(f"Graph embedding ({agg_method}) - scalar: {graph_s_emb.shape}, vector: {graph_v_emb.shape}")
    
    # Test 4: All outputs together
    print("\n=== Test 4: All Outputs ===")
    all_outputs = model.get_all_outputs(h_V, edge_index, h_E, seq)
    node_emb = all_outputs['node']
    graph_emb = all_outputs['graph']
    logits = all_outputs['logits']
    
    print(f"All outputs keys: {list(all_outputs.keys())}")
    print(f"Node embeddings - scalar: {node_emb[0].shape}, vector: {node_emb[1].shape}")
    print(f"Graph embeddings - scalar: {graph_emb[0].shape}, vector: {graph_emb[1].shape}")
    print(f"Contact logits: {logits.shape}")
    
    # Test 5: Test different return_embeddings modes
    print("\n=== Test 5: Different Return Modes ===")
    
    # Test 'both' mode
    both_emb = model.forward(h_V, edge_index, h_E, seq, return_embeddings='both')
    print(f"'both' mode keys: {list(both_emb.keys())}")
    
    # Test direct calls with different modes
    node_only = model.forward(h_V, edge_index, h_E, seq, return_embeddings='node')
    graph_only = model.forward(h_V, edge_index, h_E, seq, return_embeddings='graph')
    
    print(f"Node-only embeddings - scalar: {node_only[0].shape}, vector: {node_only[1].shape}")
    print(f"Graph-only embeddings - scalar: {graph_only[0].shape}, vector: {graph_only[1].shape}")
    
    # Test 6: Batched processing simulation
    print("\n=== Test 6: Batched Processing Simulation ===")
    batch_indices = torch.tensor([0, 0, 1, 1, 1])  # Two graphs: 2 nodes + 3 nodes
    
    for agg_method in ['mean', 'sum', 'max']:
        graph_emb_batched = model._compute_graph_embedding(
            node_emb, batch_indices, agg_method=agg_method
        )
        print(f"Batched graph embedding ({agg_method}) - scalar: {graph_emb_batched[0].shape}, vector: {graph_emb_batched[1].shape}")
    
    print("\n=== All Tests Completed Successfully! ===")
    

if __name__ == "__main__":
    main()
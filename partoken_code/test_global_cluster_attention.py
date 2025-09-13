from partoken_model import ParTokenModel
import torch

print('Testing forward pass...')
model = ParTokenModel(node_in_dim=(6, 3), node_h_dim=(100, 16), edge_in_dim=(32, 1), edge_h_dim=(32, 1))
model.eval()

# Create synthetic data
batch_size = 2
max_nodes = 20
total_nodes = batch_size * max_nodes

# Node features
h_V = (torch.randn(total_nodes, 6), torch.randn(total_nodes, 3, 3))

# Edge connectivity
edge_list = []
for b in range(batch_size):
    offset = b * max_nodes
    for i in range(max_nodes - 1):
        edge_list.extend([[offset + i, offset + i + 1], [offset + i + 1, offset + i]])

edge_index = torch.tensor(edge_list).t().contiguous()
num_edges = edge_index.size(1)

# Edge features
h_E = (torch.randn(num_edges, 32), torch.randn(num_edges, 1, 3))

# Batch tensor
batch = torch.repeat_interleave(torch.arange(batch_size), max_nodes)

with torch.no_grad():
    logits, assignment_matrix, extra = model(h_V, edge_index, h_E, batch=batch)

print(f'Forward pass successful!')
print(f'Logits shape: {logits.shape}')
print(f'Assignment matrix shape: {assignment_matrix.shape}')
print(f'VQ loss: {extra["vq_loss"].item():.4f}')
print(f'Codebook perplexity: {extra["vq_info"]["perplexity"].item():.2f}')
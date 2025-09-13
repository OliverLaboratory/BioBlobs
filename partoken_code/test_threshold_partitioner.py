
from partoken_model import create_optimized_model
import torch

print('🧬 Testing threshold-based ParToken model...')
model = create_optimized_model()

# Test with synthetic data
batch_size = 2
max_nodes = 20
h_V = (torch.randn(batch_size * max_nodes, 6), torch.randn(batch_size * max_nodes, 3, 3))
edge_list = []
for b in range(batch_size):
    start = b * max_nodes
    for i in range(max_nodes - 1):
        edge_list.extend([[start + i, start + i + 1], [start + i + 1, start + i]])
edge_index = torch.tensor(edge_list).t().contiguous()
h_E = (torch.randn(edge_index.size(1), 32), torch.randn(edge_index.size(1), 1, 3))
batch = torch.repeat_interleave(torch.arange(batch_size), max_nodes)
labels = torch.randint(0, 2, (batch_size,))

print(f'Input shapes: nodes={h_V[0].shape}, edges={edge_index.shape}')

# Forward pass
model.eval()
with torch.no_grad():
    logits, assignment_matrix, extra = model(h_V, edge_index, h_E, batch=batch)

print(f'Output shapes: logits={logits.shape}, assignment={assignment_matrix.shape}')

# Test loss computation
total_loss, metrics = model.compute_total_loss(logits, labels, extra)
print(f'Total loss: {total_loss.item():.4f}')
print(f'Classification loss: {metrics["loss/cls"]:.4f}')
print(f'VQ loss: {metrics["loss/vq_commit"]:.4f}')
print(f'PSC loss: {metrics["loss/psc"]:.4f}')
print(f'Cardinality loss: {metrics["loss/cardinality"]:.4f}')

print('✅ Threshold-based ParToken model test passed!')

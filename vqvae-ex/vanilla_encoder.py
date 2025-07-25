import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from residual import ResidualStack


class TransformerEncoderLayer(nn.Module):
    """
    A standard Transformer encoder layer with multi-head attention and feed-forward network
    """
    def __init__(self, d_model=128, nhead=4, dim_feedforward=512, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed Forward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Activation
        self.activation = nn.ReLU()
        
    def forward(self, src):
        # Self-attention block
        src2, _ = self.self_attn(src, src, src)
        src = src + src2  # Residual connection
        src = self.norm1(src)
        
        # Feed-forward block
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + src2  # Residual connection
        src = self.norm2(src)
        
        return src


class ProteinTransformerEncoder(nn.Module):
    """
    Transformer-based encoder for protein structures.
    
    Inputs:
    - in_dim: dimension of the input (e.g., 12 for atom types and coordinates)
    - h_dim: hidden dimension of the transformer
    - num_layers: number of transformer layers
    - nhead: number of attention heads
    - dim_feedforward: dimension of the feedforward network
    - out_dim: dimension of the output
    """
    
    def __init__(self, in_dim=12, h_dim=128, num_layers=3, nhead=4, dim_feedforward=512, out_dim=64):
        super(ProteinTransformerEncoder, self).__init__()
        
        # Initial projection
        self.input_projection = nn.Linear(in_dim, h_dim)
        
        # Stack of transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(h_dim, nhead, dim_feedforward) 
            for _ in range(num_layers)
        ])
        
        # Final projection
        self.output_projection = nn.Linear(h_dim, out_dim)
        
    def forward(self, x):
        # x has shape [batch_size, L, 4, 3]
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Reshape to [batch_size, L, 12]
        x = x.reshape(batch_size, seq_len, -1)
        
        # Project to hidden dimension
        x = self.input_projection(x)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Project to output dimension
        x = self.output_projection(x)
        
        return x


if __name__ == "__main__":
    # random protein coordinate data: batch_size, L atoms, 4 atom types, 3 coordinates
    batch_size = 10
    L = 150  # number of atoms
    # x = torch.randn(batch_size, L, 4, 3).float()
    x = torch.load('2NXC_X.pt').unsqueeze(0)  # Add batch dimension

    # test encoder
    encoder = ProteinTransformerEncoder(in_dim=12, h_dim=128, num_layers=3, nhead=4, dim_feedforward=512, out_dim=64)
    encoder_out = encoder(x)
    print('Encoder out shape:', encoder_out.shape)  # Should be [batch_size, L, 64]
    # print('Encoder out:', encoder_out)



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TransformerDecoderLayer(nn.Module):
    """
    Custom transformer decoder layer with bidirectional self-attention
    Inputs:
    - d_model: dimension of the model
    - nhead: number of attention heads
    - dim_feedforward: dimension of the feedforward network
    - dropout: dropout rate
    """
    def __init__(self, 
                 d_model, 
                 nhead, 
                 dim_feedforward=2048, 
                 dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = F.relu

    def forward(self, x):
        # Self attention block
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Feed forward block
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        
        return x


class ProteinTransformerDecoder(nn.Module):
    """
    Transformer-based decoder for protein structures.
    
    Inputs:
    - latent_dim: dimension of the quantized latent vectors
    - hidden_dim: hidden dimension of the transformer
    - num_layers: number of transformer layers
    - nhead: number of attention heads
    - dim_feedforward: dimension of the feedforward network
    - output_dim: dimension of the output (e.g., 3D coordinates)
    """
    def __init__(self, 
                 latent_dim, 
                 hidden_dim, 
                 num_layers, 
                 nhead, 
                 dim_feedforward, 
                 output_dim):
        super(ProteinTransformerDecoder, self).__init__()
        
        # Linear projection from latent dimension to hidden dimension
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        
        # Stack of bidirectional transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward
            ) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, z_q):
        """
        Args:
            z_q: quantized latent vectors [batch_size, L, latent_dim]
        Returns:
            output: predicted 3D coordinates [batch_size, L, output_dim]
        """
        # Project latent vectors to hidden dimension
        x = self.latent_to_hidden(z_q)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Project to output dimension (e.g., 3D coordinates)
        output = self.output_projection(x)
        
        return output


if __name__ == "__main__":
    # Test the decoder
    batch_size = 16
    seq_length = 150
    latent_dim = 64
    
    # Create random latent vectors (as if they came from the quantizer)
    z_q = torch.randn(batch_size, seq_length, latent_dim)
    
    # Initialize decoder
    decoder = ProteinTransformerDecoder(
        latent_dim=latent_dim,
        hidden_dim=128,
        num_layers=3,
        nhead=4,
        dim_feedforward=512,
        output_dim=12  # Example: 4 residues x 3 coordinates
    )
    
    # Forward pass
    output = decoder(z_q).reshape(batch_size, seq_length, 4, 3)
    print(f"Input shape: {z_q.shape}")
    print(f"Output shape: {output.shape}")

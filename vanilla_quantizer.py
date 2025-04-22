import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.
    
    Parameters:
        n_e (int): Number of embeddings in the codebook
        e_dim (int): Dimension of each embedding vector
        beta (float): Commitment cost coefficient for the loss term beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch_size, L, D) where:
        - L is the number of residues in the protein
        - D is the embedding dimension

        quantization pipeline:
            1. get encoder input (batch_size, L, D)
            2. flatten input to (batch_size*L, D)
            3. compute distances from z to embeddings e_j
            4. find closest encodings
            5. get quantized latent vectors
            6. compute loss for embedding
            7. preserve gradients
            8. compute perplexity
            9. reshape indices for output
        """
        # Store original shape
        batch_size, L, D = z.shape
        
        # Flatten input to (batch_size*L, D)
        z_flattened = z.reshape(-1, self.e_dim)
        
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(batch_size, L, D)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # Reshape indices for output
        min_encoding_indices = min_encoding_indices.reshape(batch_size, L)

        return loss, z_q, perplexity, min_encodings, min_encoding_indices

if __name__ == "__main__":
    from vanilla_encoder import ProteinTransformerEncoder
    x = torch.randn(16, 150, 4, 3).float().to(device)
    encoder = ProteinTransformerEncoder(in_dim=12, h_dim=128, num_layers=3, nhead=4, dim_feedforward=512, out_dim=64).to(device)
    z = encoder(x)
    quantizer = VectorQuantizer(n_e=1028, e_dim=64, beta=0.25).to(device)
    loss, z_q, perplexity, min_encodings, min_encoding_indices = quantizer(z)
    print(loss, perplexity)
    print(z_q.shape)
    # print(z_q)
    print(min_encodings.shape)
    print(min_encoding_indices.shape)
    
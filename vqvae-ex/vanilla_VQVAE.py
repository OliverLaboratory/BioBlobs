import torch
import torch.nn as nn
import numpy as np
from vanilla_encoder import ProteinTransformerEncoder
from vanilla_quantizer import VectorQuantizer
from vanilla_decoder import ProteinTransformerDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VQVAE(nn.Module):
    def __init__(self, h_dim, num_layers, nhead, dim_feedforward,
                 n_embeddings, embedding_dim, beta):
        super(VQVAE, self).__init__()
        # encode protein into continuous latent space
        self.encoder = ProteinTransformerEncoder(in_dim=12, 
                                                 h_dim=h_dim, 
                                                 num_layers=num_layers, 
                                                 nhead=nhead, 
                                                 dim_feedforward=dim_feedforward, 
                                                 out_dim=embedding_dim)
        
        self.vector_quantization = VectorQuantizer(n_e=n_embeddings, 
                                                   e_dim=embedding_dim, beta=beta)

        self.decoder = ProteinTransformerDecoder(latent_dim=embedding_dim, 
                                                 hidden_dim=h_dim, 
                                                 num_layers=num_layers, 
                                                 nhead=nhead, 
                                                 dim_feedforward=dim_feedforward, 
                                                 output_dim=12)

    def forward(self, x, verbose=False):
        z_e = self.encoder(x)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)
        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('quantized data shape:', z_q.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity
    
if __name__ == "__main__":
    # random protein coordinate data: batch_size, L atoms, 4 atom types, 3 coordinates
    batch_size = 10
    L = 150  # number of atoms
    x = torch.randn(batch_size, L, 4, 3).float().to(device)
    
    # test VQVAE
    vqvae = VQVAE(h_dim=128, 
                  num_layers=3, 
                  nhead=4, 
                  dim_feedforward=512, 
                  n_embeddings=1024, 
                  embedding_dim=64, 
                  beta=0.25).to(device)
    
    embedding_loss, x_hat, perplexity = vqvae(x, verbose=False)
    print('embedding loss:', embedding_loss)
    print('perplexity:', perplexity)
    print('recon data shape:', x_hat.shape)

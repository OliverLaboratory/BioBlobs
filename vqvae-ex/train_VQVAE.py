import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from vanilla_VQVAE import VQVAE
from torch.utils.data import random_split


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


data = torch.load('protein_data_list.pt')
total_samples = len(data)
train_size = int(0.8 * total_samples)  # 80% for training
valid_size = total_samples - train_size  # 20% for validation

# Create a random split
train_data, valid_data = random_split(data, [train_size, valid_size], 
                                     generator=torch.Generator().manual_seed(42))

# Now train_data and valid_data are Dataset objects
# To get the actual data items:
train_proteins = [data[i] for i in train_data.indices]
valid_proteins = [data[i] for i in valid_data.indices]

print(f"Training set size: {len(train_proteins)}")
print(f"Validation set size: {len(valid_proteins)}")


print('number of proteins:', len(data))

model = VQVAE(h_dim=128, 
              num_layers=3, 
              nhead=4, 
              dim_feedforward=512, 
              n_embeddings=1024, 
              embedding_dim=64, 
              beta=0.25).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0001, amsgrad=True)

model.train()



def train():
    loss_list = []
    num_epochs = 1000  # Reduced from 1000 since we'll process all data in each epoch
    lowest_valid_loss = float('inf')
    
    for epoch in range(num_epochs):
        total_train_loss = 0
        total_valid_loss = 0
        for i, x in enumerate(train_proteins):
            # print(f"Processing protein {i+1}/{len(data)}, shape: {x.shape}")
            x = x.unsqueeze(0)  
            x = x.to(device)
            
            optimizer.zero_grad()
            embedding_loss, x_hat, perplexity = model(x)
            x_hat = x_hat.reshape(x.shape)
            recon_loss = torch.mean((x_hat - x)**2)
            loss = recon_loss + embedding_loss
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        train_avg_loss = total_train_loss / len(train_proteins)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Average Loss: {train_avg_loss:.6f}")

        if (epoch + 1) % 10 == 0:
            # test on validation set
            model.eval()
            with torch.no_grad():
                for x in valid_proteins:
                    x = x.unsqueeze(0)
                    x = x.to(device)
                    embedding_loss, x_hat, perplexity = model(x)
                    x_hat = x_hat.reshape(x.shape)
                    recon_loss = torch.mean((x_hat - x)**2)
                    loss = recon_loss + embedding_loss
                    total_valid_loss += loss.item()

                valid_avg_loss = total_valid_loss / len(valid_proteins)
                print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {valid_avg_loss:.4f}, 
                        Perplexity: {perplexity:.4f}, 
                        Recon Loss: {recon_loss:.4f}, 
                        Embedding Loss: {embedding_loss:.4f}")
                
                if valid_avg_loss < lowest_valid_loss:
                    lowest_valid_loss = valid_avg_loss
                    torch.save({'Epoch': epoch,
                               'Model': model.state_dict(),
                               'Optimizer': optimizer.state_dict(),
                               'Validation Loss': valid_avg_loss,
                               'Perplexity': perplexity,
                               'Recon Loss': recon_loss,
                               'Embedding Loss': embedding_loss}, 
                               'vqvae_model.pt')
        
        # # Optional: Save checkpoint every few epochs
        # if (epoch + 1) % 10 == 0:
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'loss': avg_loss,
        #     }, f'vqvae_checkpoint_epoch_{epoch+1}.pt')

if __name__ == "__main__":
    train()

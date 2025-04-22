## Self Attention

In self-attention, we want each position in the sequence to gather information from all other positions in the same sequence. This is why we use the same input for all three parameters:

- Query (Q): Represents the current position asking "what should I pay attention to?"
- Key (K): Represents all positions answering "how relevant am I to your query?"
- Value (V): Represents the information to be aggregated from the relevant positions
  When all three are the same input (src), each position in the sequence:
- Forms a query based on its own representation
  Compares this query against keys from all positions (including itself)
  Aggregates values from all positions weighted by the query-key similarity

## VectorQuantizer

### Loss

- "Codebook loss" - Encourages the embedding vectors to move closer to the encoder outputs
- "Commitment loss" - Encourages the encoder to commit to codebook vectors (scaled by $\beta$)

```
loss = torch.mean((z_q.detach()-z)**2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
```

### z_q

- The quantized representation after replacing each input vector with its nearest codebook vector
- (batch_size, L, D)
- This is what you would pass to your decoder.
- It contains the discrete representation of your protein sequence, where each residue embedding has been replaced by its closest codebook vector.

```
z_q = torch.matmul(min_encodings, self.embedding.weight).view(batch_size, L, D)
```

### Perplexity

A metric that measures how effectively the codebook is being used

```
e_mean = torch.mean(min_encodings, dim=0)
perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
```

### min_encodings

The one-hot encodings of which codebook vector was selected for each position

- (batch_size*L, n_e)

```
min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(device)
min_encodings.scatter_(1, min_encoding_indices, 1)
```

### min_encoding_indices

The indices of the selected codebook vectors for each position. These are the actual discrete codes that represent your protein sequence. Each value is an integer in the range [0, n\_e-1] indicating which codebook vector was selected for that position.

- (batch_size, L)

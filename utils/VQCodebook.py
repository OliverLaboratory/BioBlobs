import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class VQCodebookEMA(nn.Module):
    """
    Vector-quantization codebook with EMA updates (VQ-VAE-2 style) for cluster/subgraph embeddings.

    This module quantizes before inter-cluster embeddings to a discrete codebook and provides:
    - VQ losses (codebook + commitment) with straight-through gradients
    - EMA code updates for stability
    - Usage entropy (KL to uniform) as a dead-code regularizer
    - Soft presence probabilities for coverage/frequency losses
    - K-means initialization from cached embeddings

    Args:
        codebook_size: Number of codes in the dictionary
        dim: Embedding dimension
        beta: Commitment loss weight
        decay: EMA decay for code updates
        eps: Numerical stability constant
        distance: 'l2' (squared Euclidean) or 'cos' (cosine distance = 1 - cosine similarity)
        cosine_normalize: If True, normalize embeddings before distance computation
    """

    def __init__(
        self,
        codebook_size: int,
        dim: int,
        beta: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
        distance: str = "l2",
        cosine_normalize: bool = False,
    ):
        super().__init__()
        self.K = codebook_size
        self.D = dim
        self.beta = beta
        self.decay = decay
        self.eps = eps
        self.distance = distance
        self.cosine_normalize = cosine_normalize

        # Codebook embeddings and EMA stats as buffers for easy device/state handling
        self.register_buffer("embeddings", torch.randn(self.K, self.D) * 0.05)
        self.register_buffer("ema_counts", torch.zeros(self.K))
        self.register_buffer("ema_sums", torch.zeros(self.K, self.D))

    def _pairwise_dist(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise distances between x and codebook e.

        Args:
            x: Query embeddings [M, D]
            e: Codebook embeddings [K, D]

        Returns:
            distances: [M, K]
        """
        if self.distance == "cos" or self.cosine_normalize:
            x = F.normalize(x, dim=-1)
            e = F.normalize(e, dim=-1)
            # cosine distance = 1 - cosine similarity
            return 1 - (x @ e.T)
        # L2^2 distance
        x2 = (x * x).sum(dim=1, keepdim=True)          # [M,1]
        e2 = (e * e).sum(dim=1, keepdim=True).T        # [1,K]
        return x2 - 2 * (x @ e.T) + e2

    @torch.no_grad()
    def _ema_update(self, z_valid: torch.Tensor, idx: torch.Tensor) -> None:
        """
        Update codebook embeddings with EMA statistics.

        Args:
            z_valid: Valid embeddings [M, D]
            idx: Assigned code indices [M]
        """
        onehot = F.one_hot(idx, num_classes=self.K).type_as(z_valid)
        counts = onehot.sum(dim=0)                     # [K]
        sums = onehot.T @ z_valid                      # [K, D]

        self.ema_counts.mul_(self.decay).add_(counts * (1 - self.decay))
        self.ema_sums.mul_(self.decay).add_(sums * (1 - self.decay))

        denom = self.ema_counts.unsqueeze(1) + self.eps
        self.embeddings.copy_(self.ema_sums / denom)

    def forward(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Quantize embeddings with straight-through estimator and EMA code updates.

        Args:
            z: Cluster embeddings [B, S, D]
            mask: Valid cluster mask [B, S] (True for non-empty clusters)

        Returns:
            z_q: Quantized embeddings [B, S, D]
            indices: Code indices per cluster [B, S] (-1 where invalid)
            vq_loss: Scalar VQ loss (beta * commitment only, no codebook loss)
            info: Dict with 'commitment_loss', 'perplexity', 'usage_probs'
        """
        B, S, D = z.shape
        if mask is None:
            mask = torch.ones(B, S, dtype=torch.bool, device=z.device)

        z_flat = z.reshape(-1, D)
        mask_flat = mask.reshape(-1)
        if not mask_flat.any():
            # Pass-through if no valid clusters
            idx_full = torch.full((B * S,), -1, dtype=torch.long, device=z.device)
            info = {
                "commitment_loss": z.new_tensor(0.0),
                "perplexity": self.perplexity(),
                "usage_probs": self.usage_probs()
            }
            return z, idx_full.view(B, S), z.new_tensor(0.0), info

        z_valid = z_flat[mask_flat]                   # [M, D]
        dists = self._pairwise_dist(z_valid, self.embeddings)  # [M, K]
        idx = torch.argmin(dists, dim=1)              # [M]
        e_q = self.embeddings[idx]                    # [M, D]

        # Losses (only commitment loss, no codebook loss)
        commitment_loss = F.mse_loss(z_valid, e_q.detach(), reduction='mean')
        vq_loss = self.beta * commitment_loss

        # Straight-through
        z_q_valid = z_valid + (e_q - z_valid).detach()

        # Scatter back
        z_q_flat = z_flat.clone()
        z_q_flat[mask_flat] = z_q_valid
        z_q = z_q_flat.view(B, S, D)

        idx_full = torch.full((B * S,), -1, dtype=torch.long, device=z.device)
        idx_full[mask_flat] = idx

        if self.training:
            self._ema_update(z_valid, idx)

        info = {
            "commitment_loss": commitment_loss.detach(),
            "perplexity": self.perplexity().detach(),
            "usage_probs": self.usage_probs().detach(),
        }
        return z_q, idx_full.view(B, S), vq_loss, info

    @torch.no_grad()
    def usage_probs(self) -> torch.Tensor:
        """
        Get EMA usage probabilities over codes.

        Returns:
            probs: [K] usage probabilities
        """
        total = self.ema_counts.sum() + self.eps
        return (self.ema_counts / total).clamp_min(self.eps)

    @torch.no_grad()
    def perplexity(self) -> torch.Tensor:
        """
        Compute codebook perplexity from EMA usage.
        """
        p = self.usage_probs()
        H = -(p * (p + self.eps).log()).sum()
        return torch.exp(H)

    def entropy_loss(self, weight: float = 1.0) -> torch.Tensor:
        """
        KL(usage || uniform) as a small regularizer to avoid dead codes.

        Args:
            weight: Loss weight multiplier

        Returns:
            entropy_loss: Scalar KL divergence
        """
        p = self.usage_probs()
        u = p.new_full((self.K,), 1.0 / self.K)
        return weight * (p * (p / (u + self.eps)).log()).sum()

    @torch.no_grad()
    def soft_presence(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
        temperature: float = 0.3
    ) -> torch.Tensor:
        """
        Compute soft presence p_k(G) per graph via softmax over codes and max over clusters.

        Args:
            z: Cluster embeddings [B, S, D]
            mask: Valid cluster mask [B, S]
            temperature: Softmax temperature (lower = harder)

        Returns:
            p_gk: Soft presence probabilities per graph and code [B, K]
        """
        B, S, D = z.shape
        z_flat = z.reshape(-1, D)
        mask_flat = mask.reshape(-1)

        if not mask_flat.any():
            return z.new_zeros(B, self.K)

        dists = self._pairwise_dist(z_flat[mask_flat], self.embeddings)       # [M, K]
        p_soft = F.softmax(-dists / temperature, dim=-1)                      # [M, K]

        p_full = z.new_zeros(B * S, self.K)
        p_full[mask_flat] = p_soft
        p_full = p_full.view(B, S, self.K)

        # Graph-level presence via max over clusters
        return p_full.max(dim=1).values                                       # [B, K]

    def differentiable_presence(
        self, 
        z: torch.Tensor, 
        valid_mask: torch.Tensor, 
        temperature: float = 1.0, 
        eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Compute differentiable presence probabilities per graph and code.
        
        Args:
            z: Cluster embeddings [B, C, D]
            valid_mask: Valid cluster mask [B, C] with 1 for real clusters and 0 for paddings
            temperature: Softmax temperature for soft assignments
            eps: Numerical stability constant
            
        Returns:
            presence: [B, K] differentiable presence probabilities
        """
        # Get codebook embeddings
        e = self.embeddings  # [K, D] - buffer updated by EMA
        
        # Compute squared Euclidean distances
        # [B,C,1,D] - [1,1,K,D] -> [B,C,K,D] -> sum over D -> [B,C,K]
        dists = (z.unsqueeze(2) - e.unsqueeze(0).unsqueeze(0)).pow(2).sum(dim=-1)
        
        # Soft assignments via softmax
        logits = -dists / max(temperature, eps)
        P = logits.softmax(dim=-1)  # [B, C, K]
        
        if valid_mask is not None:
            P = P * valid_mask.unsqueeze(-1)
        
        # Clamp for numerical stability, then smooth "OR" across clusters
        P = P.clamp(eps, 1 - eps)
        presence = 1.0 - torch.exp(torch.sum(torch.log1p(-P), dim=1))  # [B, K]
        return presence

    @torch.no_grad()
    def kmeans_init(self, samples: torch.Tensor, n_iters: int = 25, tol: float = 1e-4) -> None:
        """
        Initialize the codebook with k-means on cached cluster embeddings.

        Args:
            samples: Cached embeddings [N, D]
            n_iters: Maximum k-means iterations
            tol: Convergence tolerance on center movement (L2)

        Notes:
            - Uses simple PyTorch operations; expects N >> K.
            - Empty clusters are reinitialized from random samples.
        """
        assert samples.dim() == 2 and samples.size(1) == self.D, "Invalid samples shape."

        N = samples.size(0)
        K = self.K
        device = samples.device

        # Handle case where we have fewer samples than codebook entries
        if N < K:
            print(f"Warning: Only {N} samples available for {K} codebook entries. "
                  f"Using repeated samples for initialization.")
            # Repeat samples to fill codebook
            repeat_factor = (K + N - 1) // N  # Ceiling division
            expanded_samples = samples.repeat(repeat_factor, 1)[:K]  # [K, D]
            self.embeddings.copy_(expanded_samples)
            self.ema_counts.copy_(torch.ones(K, device=device))
            self.ema_sums.copy_(expanded_samples.clone())
            return

        # Choose K random distinct initial centers
        perm = torch.randperm(N, device=device)
        centers = samples[perm[:K]].clone()                                    # [K, D]

        prev_shift = None
        for _ in range(n_iters):
            # Assign
            dists = self._pairwise_dist(samples, centers)                       # [N, K]
            assign = torch.argmin(dists, dim=1)                                 # [N]

            # Update
            new_centers = torch.zeros_like(centers)
            counts = torch.zeros(K, device=device, dtype=torch.long)
            new_centers.index_add_(0, assign, samples)
            counts.index_add_(0, assign, torch.ones_like(assign, dtype=torch.long))
            # Handle empty clusters
            empty = counts == 0
            if empty.any():
                num_empty = int(empty.sum().item())
                # Sample replacement centers for empty clusters
                repl_indices = torch.randint(0, N, (num_empty,), device=device)
                repl = samples[repl_indices]  # [num_empty, D]
                new_centers[empty] = repl
                counts[empty] = 1  # avoid div by zero

            new_centers = new_centers / counts.clamp_min(1).unsqueeze(1)

            # Check convergence
            shift = torch.norm(new_centers - centers, p=2, dim=1).mean()
            centers = new_centers
            if prev_shift is not None and torch.abs(shift - prev_shift) < tol:
                break
            prev_shift = shift

        # Copy into codebook and reset EMA state consistently
        self.embeddings.copy_(centers)
        self.ema_counts.copy_(torch.ones(K, device=device))
        self.ema_sums.copy_(centers.clone())

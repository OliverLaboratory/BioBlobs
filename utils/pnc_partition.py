# pnc_partition.py
# Seed-conditioned, trainable expansion with rank-gated size coupling
# Includes a gradient-flow test.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SeedCondExpansion(nn.Module):
    """
    Lightweight seed-conditioned scorer + masked straight-through (ST) Gumbel-TopK.

    Scores k-hop candidates using a dot-product between a seed query and candidate keys,
    plus a normalized local link term to favor tight substructures. Selection is ST
    so gradients can flow.
    """

    def __init__(self, in_dim: int, hid_dim: int = 64, alpha: float = 0.2, tau: float = 1.0):
        """
        Args:
            in_dim: node feature size
            hid_dim: projection size for query/key
            alpha: weight for the normalized edges-to-cluster term
            tau: temperature for the soft relaxation (can be updated by caller)
        """
        super().__init__()
        self.Wq = nn.Linear(in_dim, hid_dim, bias=False)
        self.Wk = nn.Linear(in_dim, hid_dim, bias=False)
        self.alpha = alpha
        self.tau = tau

    @staticmethod
    def _edges_to_cluster(adj: torch.Tensor, cluster_mask: torch.Tensor) -> torch.Tensor:
        """
        Count edges from each node to the current cluster.
        Supports dense [B, N, N] and sparse COO with batch dimension.
        Returns [B, N].
        """
        if adj.is_sparse:
            adj_dense = adj.to_dense()
        else:
            adj_dense = adj
        return (adj_dense.float() @ cluster_mask.float().unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def _degree(adj: torch.Tensor) -> torch.Tensor:
        """Compute node degrees [B, N] for dense or sparse adjacency."""
        if adj.is_sparse:
            adj_dense = adj.to_dense()
        else:
            adj_dense = adj
        return adj_dense.float().sum(dim=-1)

    @staticmethod
    def _gumbel_noise_like(t: torch.Tensor) -> torch.Tensor:
        eps = 1e-10
        u = torch.rand_like(t)
        return -torch.log(-torch.log(u + eps) + eps)

    def forward(
        self,
        x: torch.Tensor,                 # [B, N, F]
        adj: torch.Tensor,               # [B, N, N], dense or sparse
        seed_idx: torch.Tensor,          # [B]
        cluster_mask: torch.Tensor,      # [B, N] (True at current cluster members; at start only the seed)
        cand_mask: torch.Tensor,         # [B, N] (True only for k-hop available candidates)
        pred_K: torch.Tensor,            # [B] (how many additional nodes to add)
        seed_repr: Optional[torch.Tensor] = None,  # [B, F] optional ST seed features
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            topk_idx_padded: [B, K_max] indices (padded) of selected nodes
            y_st: [B, N] ST selection weights (hard forward, soft backward)
            scores: [B, N] raw scores before masking/noise (for logging / ranking)
        """
        B, N, _ = x.shape

        # 1) Seed-conditioned dot-product scores
        x_k = self.Wk(x)  # [B, N, d]
        if seed_repr is None:
            q = self.Wq(x[torch.arange(B), seed_idx])  # [B, d] (hard)
        else:
            q = self.Wq(seed_repr)                     # [B, d] (ST-soft)
        scores = (x_k @ q.unsqueeze(-1)).squeeze(-1) / (x_k.size(-1) ** 0.5)  # [B, N]

        # 2) Add normalized local link term to prefer tight substructures
        e2c = self._edges_to_cluster(adj, cluster_mask)    # [B, N]
        deg = self._degree(adj).clamp_min(1.0)             # [B, N]
        z = e2c / deg                                      # fraction of neighbors in current cluster
        scores = scores + self.alpha * z

        # 3) Mask to k-hop available candidates only
        very_neg = torch.finfo(scores.dtype).min
        masked_scores = torch.where(cand_mask, scores, torch.full_like(scores, very_neg))

        # 4) ST Gumbel-TopK
        g = self._gumbel_noise_like(masked_scores)
        noisy = masked_scores + g

        K_max = int(pred_K.max().item()) if pred_K.numel() > 0 else 0
        y_hard = torch.zeros_like(scores)
        topk_idx_padded = torch.zeros((B, max(K_max, 1)), dtype=torch.long, device=scores.device)

        if K_max > 0:
            topk_all = torch.topk(noisy, k=K_max, dim=-1).indices  # [B, K_max]
            y_hard.scatter_(-1, topk_all, 1.0)                     # mark K_max per graph
            for b in range(B):
                keep = int(pred_K[b].item())
                if keep < K_max:
                    drop_idx = topk_all[b, keep:]
                    y_hard[b, drop_idx] = 0.0
                topk_idx_padded[b, :K_max] = topk_all[b]

        # Soft weights for gradient (softmax over perturbed, masked scores)
        y_soft = F.softmax(noisy / max(self.tau, 1e-6), dim=-1)

        # Straight-through
        y_st = y_hard + (y_soft - y_soft.detach())

        return topk_idx_padded, y_st, scores


class Partitioner(nn.Module):
    """
    Hard Gumbel-Softmax partitioner with trainable, seed-conditioned expansion
    and rank-gated size coupling.
    """

    def __init__(
        self,
        nfeat: int,
        max_clusters: int,
        nhid: int,
        k_hop: int = 2,
        cluster_size_max: int = 3,
        termination_threshold: float = 0.95,
        exp_hid: int = 64,
        exp_alpha: float = 0.2
    ):
        super().__init__()
        self.max_clusters = max_clusters
        self.k_hop = k_hop
        self.cluster_size_max = cluster_size_max
        self.cluster_size_min = 1
        self.termination_threshold = termination_threshold

        # Seed selection head (node + global context)
        self.seed_selector = nn.Sequential(
            nn.Linear(nfeat + nhid, nhid),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(nhid, 1)
        )

        # Size prediction head (seed features + global context + local scalars)
        self.size_extra_dim = 3  # n_cand_log, seed_link_frac, cand_density
        self.size_predictor = nn.Sequential(
            nn.Linear(nfeat + nhid + self.size_extra_dim, nhid),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(nhid, cluster_size_max)
        )

        # Global context encoder (mean-pooled nodes)
        self.context_encoder = nn.Linear(nfeat, nhid)

        # Trainable expansion module
        self.expander = SeedCondExpansion(in_dim=nfeat, hid_dim=exp_hid, alpha=exp_alpha, tau=1.0)

        # Temperature schedule
        self.register_buffer('epoch', torch.tensor(0))
        self.tau_init = 1.0
        self.tau_min = 0.1
        self.tau_decay = 0.95

    def get_temperature(self) -> float:
        return max(self.tau_min, self.tau_init * (self.tau_decay ** self.epoch))

    def _local_size_features(
        self,
        adj: torch.Tensor,          # [B, N, N], dense or sparse
        cand_mask: torch.Tensor,    # [B, N] bool
        seed_idx: torch.Tensor      # [B] long
    ) -> torch.Tensor:
        """
        Compute small local stats to guide size prediction.
        Returns phi: [B, 3] with (n_cand_log, seed_link_frac, cand_density)
        """
        if adj.is_sparse:
            adj_dense = adj.to_dense().float()
        else:
            adj_dense = adj.float()

        B, N, _ = adj_dense.shape
        cand_f = cand_mask.float()

        # number of candidates (log-scaled)
        n_cand = cand_mask.sum(dim=1).float()           # [B]
        n_cand_log = (n_cand + 1.0).log()               # [B]

        # fraction of seed's edges that go into the candidate set
        seed_rows = adj_dense[torch.arange(B), seed_idx]             # [B, N]
        deg_seed_cand = (seed_rows * cand_f).sum(dim=1)              # [B]
        seed_link_frac = deg_seed_cand / n_cand.clamp_min(1.0)       # [B]

        # density of the candidate-induced subgraph
        cand_adj = adj_dense * cand_f.unsqueeze(1) * cand_f.unsqueeze(2)   # [B, N, N]
        diag_sum = torch.diagonal(cand_adj, dim1=1, dim2=2).sum(dim=1)     # [B]
        e_cand = (cand_adj.sum(dim=(1, 2)) - diag_sum) * 0.5               # [B]
        denom = n_cand * (n_cand - 1.0) * 0.5                              # [B]
        cand_density = e_cand / denom.clamp_min(1.0)                       # [B]

        phi = torch.stack([n_cand_log, seed_link_frac, cand_density], dim=-1)  # [B, 3]
        return phi

    def _compute_k_hop_neighbors(
        self,
        adj: torch.Tensor,
        seed_indices: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute k-hop neighborhoods for each seed as a boolean mask.
        Returns [B, N].
        """
        B, N, _ = adj.shape
        device = adj.device

        current_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        reachable_mask = torch.zeros(B, N, dtype=torch.bool, device=device)

        valid_seeds = seed_indices >= 0
        current_mask[valid_seeds, seed_indices[valid_seeds]] = True
        reachable_mask = current_mask.clone()

        for _ in range(self.k_hop):
            neighbors = torch.bmm(current_mask.float().unsqueeze(1), adj).squeeze(1) > 0
            neighbors = neighbors & mask & (~reachable_mask)
            if not neighbors.any():
                break
            current_mask = neighbors
            reachable_mask = reachable_mask | neighbors

        return reachable_mask

    def _gumbel_softmax_selection(
        self,
        logits: torch.Tensor,
        tau: float,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Hard Gumbel-Softmax over nodes. Returns ST one-hots [B, N].
        """
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)

        gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        noisy_logits = (logits + gumbel) / max(tau, 1e-6)

        soft = F.softmax(noisy_logits, dim=-1)
        hard = torch.zeros_like(soft)
        indices = soft.argmax(dim=-1, keepdim=True)
        hard.scatter_(-1, indices, 1.0)

        return hard + (soft - soft.detach())

    def _check_termination(
        self,
        assignment_matrix: torch.Tensor,
        mask: torch.Tensor,
        cluster_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stop for graphs that reached coverage. Returns (should_terminate, active).
        """
        total_nodes = mask.sum(dim=-1).float()
        assigned_nodes = assignment_matrix[:, :, :cluster_idx + 1].sum(dim=(1, 2))
        coverage = assigned_nodes / (total_nodes + 1e-8)

        should_terminate = coverage >= self.termination_threshold
        active_proteins = (~should_terminate) & (total_nodes > 0)

        return should_terminate, active_proteins

    def forward(
        self,
        x: torch.Tensor,    # [B, N, D]
        adj: torch.Tensor,  # [B, N, N] (dense; sparse is tolerated in the expander)
        mask: torch.Tensor  # [B, N] bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run partitioning and return cluster features, cluster adjacency, and assignments.

        Returns:
            cluster_features: [B, S, D]
            cluster_adj: [B, S, S]
            assignment_matrix: [B, N, S]
        """
        B, N, D = x.shape
        device = x.device
        tau = self.get_temperature()

        # Initial availability and context
        available_mask = mask.clone()
        # mean over valid nodes (simple; can swap with masked mean if needed)
        node_mean = x.masked_fill(~mask.unsqueeze(-1), 0).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1).float()
        global_context = self.context_encoder(node_mean)

        cluster_embeddings = []
        assignment_matrix = torch.zeros(B, N, self.max_clusters, device=device)
        terminated = torch.zeros(B, dtype=torch.bool, device=device)

        for cluster_idx in range(self.max_clusters):
            if not available_mask.any() or terminated.all():
                break

            if cluster_idx > 0:
                should_terminate, active = self._check_termination(
                    assignment_matrix, mask, cluster_idx - 1
                )
                terminated = terminated | should_terminate
                if not active.any():
                    break
            else:
                active = torch.ones(B, dtype=torch.bool, device=device)

            # --- Seed selection ---
            context_expanded = global_context.unsqueeze(1).expand(-1, N, -1)
            combined_features = torch.cat([x, context_expanded], dim=-1)  # [B, N, D+nhid]
            seed_logits = self.seed_selector(combined_features).squeeze(-1)  # [B, N]

            selection_mask = available_mask & active.unsqueeze(-1)
            w_seed = self._gumbel_softmax_selection(seed_logits, tau, selection_mask)  # [B, N] ST
            seed_indices = w_seed.argmax(dim=-1)  # [B] for hard bookkeeping

            # ST seed embedding (gives gradient to seed head)
            x_seed_st = torch.einsum("bn,bnd->bd", w_seed, x)  # [B, D]

            # Assign seeds and mark unavailable (hard path for assignment)
            has_selection = selection_mask.sum(dim=-1) > 0
            valid_seeds = has_selection & active
            if valid_seeds.any():
                assignment_matrix[valid_seeds, seed_indices[valid_seeds], cluster_idx] = 1.0
                for b in range(B):
                    if valid_seeds[b]:
                        available_mask[b, seed_indices[b]] = False

                # --- Expansion with trainable scorer ---
                if self.cluster_size_max > 1:
                    k_hop_mask = self._compute_k_hop_neighbors(adj, seed_indices, mask)          # [B, N]
                    cand_mask = available_mask & k_hop_mask & active.unsqueeze(-1)               # [B, N]

                    if cand_mask.any():
                        # Local scalars for size
                        phi_all = self._local_size_features(adj, cand_mask, seed_indices)        # [B, 3]
                        n_cand_all = cand_mask.sum(dim=1).long()                                  # [B]

                        # Size predictor input on valid graphs
                        seed_features = x_seed_st[valid_seeds]                                   # [Bv, D]
                        context_features = global_context[valid_seeds]                           # [Bv, nhid]
                        phi = phi_all[valid_seeds]                                               # [Bv, 3]
                        size_input = torch.cat([seed_features, context_features, phi], dim=-1)   # [Bv, D+nhid+3]
                        size_logits = self.size_predictor(size_input)                            # [Bv, S_max]

                        # Capacity mask
                        device = x.device
                        S_max = self.cluster_size_max
                        n_cand_valid = n_cand_all[valid_seeds]                                   # [Bv]
                        allowed_max = torch.clamp(n_cand_valid + 1, max=S_max)                   # [Bv]
                        sizes = torch.arange(1, S_max + 1, device=device).unsqueeze(0)           # [1, S_max]
                        size_valid = sizes <= allowed_max.unsqueeze(1)                            # [Bv, S_max]
                        masked_logits = size_logits.masked_fill(~size_valid, -1e9)

                        size_probs = F.softmax(masked_logits / max(tau, 1e-6), dim=-1)           # [Bv, S_max]
                        exp_sizes = (size_probs * torch.arange(
                            1, S_max + 1, device=device, dtype=size_probs.dtype
                        ).unsqueeze(0)).sum(dim=-1)                                              # [Bv]
                        add_K_valid = (exp_sizes - 1).clamp(min=0).long()                         # [Bv]

                        # Build full pred_K over batch (zeros for invalid)
                        pred_K = torch.zeros(B, dtype=torch.long, device=device)
                        pred_K[valid_seeds] = add_K_valid

                        # Current cluster mask (only seeds set so far for this cluster)
                        cluster_mask_cur = assignment_matrix[:, :, cluster_idx] > 0.5            # [B, N]

                        # Sync expander temperature
                        self.expander.tau = float(tau)

                        # Call the expander (pass ST seed features for gradient flow)
                        topk_idx_padded, y_st, raw_scores = self.expander(
                            x=x,
                            adj=adj,
                            seed_idx=seed_indices,
                            cluster_mask=cluster_mask_cur,
                            cand_mask=cand_mask,
                            pred_K=pred_K,
                            seed_repr=x_seed_st
                        )

                        # Hard membership from ST output (forward equals one-hots)
                        expansion_mask = (y_st > 0)                                              # [B, N]
                        expansion_mask[torch.arange(B), seed_indices] = False

                        # Update assignments and availability (hard bookkeeping)
                        selected_pairs = expansion_mask.nonzero(as_tuple=False)                  # [K, 2]
                        if selected_pairs.numel() > 0:
                            b_idx = selected_pairs[:, 0]
                            n_idx = selected_pairs[:, 1]
                            assignment_matrix[b_idx, n_idx, cluster_idx] = 1.0
                            available_mask[b_idx, n_idx] = False

                        # --- Rank-gated weighting to couple size to features ---
                        # Broadcast size_probs to full batch; zeros where no valid seed
                        size_probs_all = torch.zeros(B, S_max, device=device, dtype=size_probs.dtype)
                        size_probs_all[valid_seeds] = size_probs

                        # Tail sums: w_rank(j) = sum_{s >= j} p(s)
                        tail = size_probs_all.flip(-1).cumsum(-1).flip(-1)  # [B, S_max]

                        # Per-node ranks among candidates by score (desc)
                        very_neg = torch.finfo(raw_scores.dtype).min
                        cand_scores = torch.where(cand_mask, raw_scores, torch.full_like(raw_scores, very_neg))
                        order = torch.argsort(cand_scores, dim=-1, descending=True)  # [B, N]
                        M = cand_mask.sum(dim=1)                                     # [B]

                        rank_pos = torch.full_like(order, fill_value=S_max + 1, dtype=torch.long)
                        for b in range(B):
                            m = int(M[b].item())
                            if m > 0:
                                rank_pos[b, order[b, :m]] = torch.arange(1, m + 1, device=device)  # 1..m

                        rank_clipped = rank_pos.clamp(min=1, max=S_max)                # [B, N]
                        w_nodes = torch.gather(tail, 1, rank_clipped - 1)              # [B, N]
                        w_nodes = w_nodes * cand_mask.float()                           # zero outside candidates

                        # Apply to expansion ST weights
                        y_rank = y_st * w_nodes                                        # [B, N]
                    else:
                        y_rank = torch.zeros(B, N, device=device, dtype=x.dtype)
                else:
                    y_rank = torch.zeros(B, N, device=device, dtype=x.dtype)
            else:
                # No valid seeds (should be rare with masks)
                y_rank = torch.zeros(B, N, device=device, dtype=x.dtype)
                w_seed = torch.zeros(B, N, device=device, dtype=x.dtype)

            # --- Cluster embedding for this cluster ---
            # Hard cluster embedding (existing mask average)
            cluster_mask = assignment_matrix[:, :, cluster_idx] > 0.5                  # [B, N]
            cluster_size = cluster_mask.sum(dim=-1, keepdim=True).float().clamp_min(1.0)
            h_hard = ((cluster_mask.unsqueeze(-1) * x).sum(dim=1) / cluster_size)     # [B, D]

            # Soft cluster embedding (ST seed + rank-gated expansion)
            m_soft = w_seed + y_rank                                                  # [B, N]
            den = m_soft.sum(dim=1, keepdim=True).clamp_min(1.0)
            h_soft = (m_soft.unsqueeze(-1) * x).sum(dim=1) / den                      # [B, D]

            # Straight-through feature: forward = hard, backward = soft
            cluster_embedding = h_hard + (h_soft - h_soft.detach())

            # Softly zero out finished graphs
            active_weight = active.float().unsqueeze(-1)
            cluster_embedding = cluster_embedding * active_weight

            cluster_embeddings.append(cluster_embedding)

            # Update global context with a light residual from this cluster (no gradient loop)
            if active.any():
                cluster_context = self.context_encoder(cluster_embedding.detach())     # [B, nhid]
                global_context = global_context + 0.1 * cluster_context

        # Fallback: one cluster with all valid nodes
        if not cluster_embeddings:
            cluster_embedding = (mask.unsqueeze(-1) * x).sum(dim=1) / mask.sum(dim=-1, keepdim=True).float().clamp_min(1.0)
            cluster_embeddings = [cluster_embedding]
            assignment_matrix[:, :, 0] = mask.float()

        # Stack cluster features and build a simple fully-connected cluster graph
        cluster_features = torch.stack(cluster_embeddings, dim=1)                      # [B, S, D]
        S = cluster_features.size(1)
        cluster_adj = torch.ones(B, S, S, device=device) - torch.eye(S, device=device).unsqueeze(0)

        return cluster_features, cluster_adj, assignment_matrix

    def update_epoch(self) -> None:
        self.epoch += 1


# ------------------------- Gradient Flow Test -------------------------

def _grad_flow_test():
    """
    Simple gradient flow test:
    - Builds a small random batch
    - Runs the partitioner
    - Classifies pooled cluster features
    - Checks that gradients hit seed_selector, expander (Wq/Wk), and size_predictor
    """
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Small synthetic graph batch
    B, N, D = 3, 12, 16
    Smax = 4
    nhid = 32
    x = torch.randn(B, N, D, device=device, requires_grad=True)

    # Build ring + random edges
    adj = torch.zeros(B, N, N, device=device)
    for b in range(B):
        for i in range(N):
            adj[b, i, (i + 1) % N] = 1.0
            adj[b, (i + 1) % N, i] = 1.0
        for _ in range(4):
            i, j = torch.randint(0, N, (2,), device=device)
            if i != j:
                adj[b, i, j] = 1.0
                adj[b, j, i] = 1.0

    mask = torch.ones(B, N, dtype=torch.bool, device=device)

    # Model
    part = Partitioner(
        nfeat=D, max_clusters=Smax, nhid=nhid,
        k_hop=2, cluster_size_max=3, termination_threshold=0.8
    ).to(device)
    part.train()

    # Simple classifier on mean of cluster features
    num_classes = 3
    clf = nn.Linear(D, num_classes).to(device)
    clf.train()

    # Forward
    cluster_features, cluster_adj, assignment = part(x, adj, mask)  # [B, S, D]
    # Pool over clusters
    pooled = cluster_features.mean(dim=1)                            # [B, D]
    logits = clf(pooled)                                             # [B, C]

    # Dummy labels
    y = torch.randint(0, num_classes, (B,), device=device)
    loss = F.cross_entropy(logits, y)

    # Backward
    for m in [part, clf]:
        m.zero_grad(set_to_none=True)
    if x.grad is not None:
        x.grad.zero_()
    loss.backward()

    # Checks
    def has_grad(module) -> bool:
        got = False
        for p in module.parameters():
            if p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0:
                got = True
                break
        return got

    seed_ok = has_grad(part.seed_selector)
    size_ok = has_grad(part.size_predictor)
    exp_wq_ok = part.expander.Wq.weight.grad is not None and part.expander.Wq.weight.grad.norm() > 0
    exp_wk_ok = part.expander.Wk.weight.grad is not None and part.expander.Wk.weight.grad.norm() > 0
    x_ok = x.grad is not None and x.grad.norm() > 0

    print("\nGradient flow check:")
    print(f"  seed_selector grad:    {'OK' if seed_ok else 'MISS'}")
    print(f"  size_predictor grad:   {'OK' if size_ok else 'MISS'}")
    print(f"  expander Wq grad:      {'OK' if exp_wq_ok else 'MISS'}")
    print(f"  expander Wk grad:      {'OK' if exp_wk_ok else 'MISS'}")
    print(f"  input x grad:          {'OK' if x_ok else 'MISS'}")

    assert seed_ok, "No gradient reached seed_selector (check ST seed wiring)."
    assert size_ok, "No gradient reached size_predictor (check rank-gated weighting)."
    assert exp_wq_ok and exp_wk_ok, "No gradient reached expander projections (check seed_repr and y_st use)."
    assert x_ok, "No gradient reached input x (check ST path into h_soft)."

    print("✅ Gradient flow passed.")


# ------------------------------ Demo / Tests ------------------------------

if __name__ == "__main__":
    print("Testing Partitioner functionality...")

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test parameters
    batch_size = 2
    num_nodes = 10
    node_features = 16
    max_clusters = 4
    hidden_dim = 32

    # Data
    x = torch.randn(batch_size, num_nodes, node_features, device=device)

    adj = torch.zeros(batch_size, num_nodes, num_nodes, device=device)
    for b in range(batch_size):
        # ring
        for i in range(num_nodes):
            adj[b, i, (i + 1) % num_nodes] = 1.0
            adj[b, (i + 1) % num_nodes, i] = 1.0
        # random
        for _ in range(3):
            i, j = torch.randint(0, num_nodes, (2,), device=device)
            if i != j:
                adj[b, i, j] = 1.0
                adj[b, j, i] = 1.0

    mask = torch.ones(batch_size, num_nodes, dtype=torch.bool, device=device)

    # Init
    partitioner = Partitioner(
        nfeat=node_features,
        max_clusters=max_clusters,
        nhid=hidden_dim,
        k_hop=2,
        cluster_size_max=3,
        termination_threshold=0.8
    ).to(device)

    print(f"Input shapes:")
    print(f"  x:   {x.shape}")
    print(f"  adj: {adj.shape}")
    print(f"  mask:{mask.shape}")

    # Forward pass
    try:
        cluster_features, cluster_adj, assignment_matrix = partitioner(x, adj, mask)

        print(f"\nOutput shapes:")
        print(f"  cluster_features: {cluster_features.shape}")
        print(f"  cluster_adj:      {cluster_adj.shape}")
        print(f"  assignment_matrix:{assignment_matrix.shape}")

        # Stats
        print(f"\nAssignment stats:")
        for b in range(batch_size):
            assigned_nodes = assignment_matrix[b].sum(dim=0)   # nodes per cluster
            node_assignments = assignment_matrix[b].sum(dim=1) # clusters per node

            print(f"  Batch {b}:")
            print(f"    Nodes per cluster: {assigned_nodes.tolist()}")
            print(f"    Total assigned:    {node_assignments.sum().item()}")
            print(f"    Coverage:          {float(node_assignments.sum().item()) / num_nodes:.2%}")
            multi_assigned = (node_assignments > 1).sum().item()
            print(f"    Multi-assigned:    {multi_assigned}")

        # k-hop demo
        print(f"\nK-hop demo:")
        seed_idx = torch.tensor([0, 1], device=device)
        khop_mask = partitioner._compute_k_hop_neighbors(adj, seed_idx, mask)
        for b in range(batch_size):
            reachable = int(khop_mask[b].sum().item())
            print(f"  Batch {b}: {reachable}/{num_nodes} nodes reachable from seed {seed_idx[b].item()}")

        # Temperature demo
        print(f"\nTemperature annealing:")
        initial_temp = partitioner.get_temperature()
        print(f"  Initial tau: {initial_temp:.4f}")
        for epoch in range(3):
            partitioner.update_epoch()
            temp = partitioner.get_temperature()
            print(f"  Epoch {epoch + 1} tau: {temp:.4f}")

        print("\n✅ Functional tests passed.")
    except Exception as e:
        print(f"\n❌ Functional test failed: {e}")
        import traceback
        traceback.print_exc()

    # Gradient flow test
    _grad_flow_test()

    print("\nAll tests complete.")

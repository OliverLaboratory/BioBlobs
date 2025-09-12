# pnc_partition.py
# Seed-conditioned, trainable expansion with threshold-based selection
# Includes a gradient-flow test.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def st_hard_sigmoid_gate(p: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
    """
    Straight-through hard gate.
    Forward: hard = (p > thresh).float()
    Backward: identity through p.
    """
    hard = (p > thresh).float()
    return hard + (p - p.detach())


class ThresholdHead(nn.Module):
    """
    Predicts a per-cluster threshold theta from seed/context/local stats.
    Input shape: [Bv, D_in]  (concat of seed_ST, global_ctx, phi_local)
    Output: theta in R, broadcastable to candidate dim
    """
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)  # [Bv]


class SeedCondExpansion(nn.Module):
    """
    Lightweight seed-conditioned scorer with threshold-based selection.
    Uses straight-through probabilities instead of categorical size prediction.
    """

    def __init__(self, score_dim: int, theta_in_dim: int, cluster_size_max: int, alpha: float = 0.2, tau_init: float = 1.0):
        """
        Args:
            score_dim: node feature size for scoring
            theta_in_dim: input dimension for threshold head
            cluster_size_max: maximum cluster size (for optional soft capping)
            alpha: weight for the normalized edges-to-cluster term
            tau_init: initial temperature for sigmoid (will be updated by scheduler)
        """
        super().__init__()
        self.Wq = nn.Linear(score_dim, 64, bias=False)  # Fixed hidden dim for scoring
        self.Wk = nn.Linear(score_dim, 64, bias=False)
        self.alpha = alpha
        self.cluster_size_max = cluster_size_max
        self.tau = tau_init
        self.threshold_head = ThresholdHead(theta_in_dim)

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

    def forward(
        self,
        x: torch.Tensor,                     # [B, N, F] node features
        adj: torch.Tensor,                   # [B, N, N] adjacency 
        seed_ctx_feats: torch.Tensor,        # [B, D_seed] seed + context features
        local_stats: torch.Tensor,           # [B, D_phi] local statistics
        cluster_mask: torch.Tensor,          # [B, N] current cluster members
        cand_mask: torch.Tensor,             # [B, N] k-hop candidates
        seed_repr: Optional[torch.Tensor] = None,  # [B, F] optional ST seed features
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Returns:
            hard_sel: [B, N]  hard 0/1 for assignment (ST)
            soft_sel: [B, N]  probabilities p_i for gradients and logging
            theta:    [B]     scalar threshold per cluster
            stats:    dict    misc scalars for logs
        """
        B, N, _ = x.shape

        # 1) Seed-conditioned dot-product scores
        x_k = self.Wk(x)  # [B, N, d]
        if seed_repr is None:
            # Extract seed from cluster_mask (should have exactly one True per batch)
            seed_indices = cluster_mask.argmax(dim=-1)  # [B]
            q = self.Wq(x[torch.arange(B), seed_indices])  # [B, d] (hard)
        else:
            q = self.Wq(seed_repr)  # [B, d] (ST-soft)
        
        scores = (x_k @ q.unsqueeze(-1)).squeeze(-1) / (x_k.size(-1) ** 0.5)  # [B, N]

        # 2) Add normalized local link term to prefer tight substructures
        e2c = self._edges_to_cluster(adj, cluster_mask)    # [B, N]
        deg = self._degree(adj).clamp_min(1.0)             # [B, N]
        z = e2c / deg                                      # fraction of neighbors in current cluster
        cand_scores = scores + self.alpha * z

        # 3) Build the threshold input (concat seed+context with local stats)
        theta_input = torch.cat([seed_ctx_feats, local_stats], dim=-1)
        theta = self.threshold_head(theta_input)  # [B]

        # 4) Compute masked probabilities
        tau = max(self.tau, 1e-6)
        logits = (cand_scores - theta.unsqueeze(-1)) / tau
        p = torch.sigmoid(logits) * cand_mask.float()

        # 5) Optional: enforce cluster_size_max softly by trimming top excess
        if self.cluster_size_max is not None and self.cluster_size_max > 0:
            # Keep seed elsewhere; here cap non-seed adds if needed
            k_cap = self.cluster_size_max - 1
            # If more than k_cap have p>0.5, trim to top by p
            with torch.no_grad():
                over = (p > 0.5).float().sum(dim=-1) - k_cap
                need_trim = over.clamp_min(0) > 0
            if need_trim.any():
                # zero out everything except top k_cap by p for those rows
                topk_vals, topk_idx = torch.topk(p, k=max(k_cap, 1), dim=-1)
                keep = torch.zeros_like(p)
                keep.scatter_(dim=-1, index=topk_idx, src=torch.ones_like(topk_vals))
                p = torch.where(need_trim.unsqueeze(-1), p * keep, p)

        # 6) Straight-through hard bits for assignment bookkeeping
        hard = st_hard_sigmoid_gate(p, thresh=0.5)

        stats = {
            "expected_size": p.sum(dim=-1).detach(),
            "mean_p": p.mean(dim=-1).detach(),
        }
        return hard, p, theta, stats


class Partitioner(nn.Module):
    """
    Hard Gumbel-Softmax partitioner with trainable, seed-conditioned expansion
    using threshold-based selection.
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
        exp_alpha: float = 0.2,
        lambda_card: float = 0.005
    ):
        super().__init__()
        self.max_clusters = max_clusters
        self.k_hop = k_hop
        self.cluster_size_max = cluster_size_max
        self.cluster_size_min = 1
        self.termination_threshold = termination_threshold
        self.lambda_card = lambda_card

        # Seed selection head (node + global context)
        self.seed_selector = nn.Sequential(
            nn.Linear(nfeat + nhid, nhid),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(nhid, 1)
        )

        # Global context encoder (mean-pooled nodes)
        self.context_encoder = nn.Linear(nfeat, nhid)

        # Local size features dimension
        self.size_extra_dim = 3  # n_cand_log, seed_link_frac, cand_density
        
        # Trainable expansion module with threshold head
        theta_in_dim = nfeat + nhid + self.size_extra_dim  # seed_ST + global_ctx + phi_local
        self.expander = SeedCondExpansion(
            score_dim=nfeat, 
            theta_in_dim=theta_in_dim,
            cluster_size_max=cluster_size_max,
            alpha=exp_alpha, 
            tau_init=1.0
        )

        # Temperature schedule
        self.register_buffer('epoch', torch.tensor(0))
        self.tau_init = 1.0
        self.tau_min = 0.1
        self.tau_decay = 0.95

    def get_temperature(self) -> float:
        return max(self.tau_min, self.tau_init * (self.tau_decay ** self.epoch))

    def cardinality_prior(self, phi_local: torch.Tensor) -> torch.Tensor:
        """
        Cheap 1-layer prior μ(φ) for expected cluster size.
        Args:
            phi_local: [B, 3] with (n_cand_log, seed_link_frac, cand_density)
        Returns:
            mu: [B] expected size prior
        """
        # Simple affine transformation of log #candidates
        mu = 2.0 + 3.0 * phi_local[..., 0]  # base size + scaled by log candidates
        return torch.clamp(mu, min=1.0, max=float(self.cluster_size_max))

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

                # --- Expansion with threshold-based scorer ---
                if self.cluster_size_max > 1:
                    k_hop_mask = self._compute_k_hop_neighbors(adj, seed_indices, mask)          # [B, N]
                    cand_mask = available_mask & k_hop_mask & active.unsqueeze(-1)               # [B, N]

                    if cand_mask.any():
                        # Local scalars for threshold prediction
                        phi_all = self._local_size_features(adj, cand_mask, seed_indices)        # [B, 3]

                        # Threshold predictor input on valid graphs
                        seed_features = x_seed_st[valid_seeds]                                   # [Bv, D]
                        context_features = global_context[valid_seeds]                           # [Bv, nhid]
                        phi = phi_all[valid_seeds]                                               # [Bv, 3]
                        seed_ctx_feats = torch.cat([seed_features, context_features], dim=-1)    # [Bv, D+nhid]

                        # Current cluster mask (only seeds set so far for this cluster)
                        cluster_mask_cur = assignment_matrix[:, :, cluster_idx] > 0.5            # [B, N]

                        # Sync expander temperature
                        self.expander.tau = float(tau)

                        # Call the new threshold-based expander
                        # Prepare inputs for valid seed graphs only
                        if valid_seeds.any():
                            valid_indices = torch.where(valid_seeds)[0]
                            x_valid = x[valid_seeds]                                             # [Bv, N, F]
                            adj_valid = adj[valid_seeds]                                         # [Bv, N, N] 
                            cluster_mask_valid = cluster_mask_cur[valid_seeds]                   # [Bv, N]
                            cand_mask_valid = cand_mask[valid_seeds]                             # [Bv, N]
                            x_seed_st_valid = x_seed_st[valid_seeds]                             # [Bv, F]
                            
                            hard_sel_valid, soft_sel_valid, theta, exp_stats = self.expander(
                                x=x_valid,
                                adj=adj_valid,
                                seed_ctx_feats=seed_ctx_feats,
                                local_stats=phi,
                                cluster_mask=cluster_mask_valid,
                                cand_mask=cand_mask_valid,
                                seed_repr=x_seed_st_valid
                            )
                            
                            # Broadcast back to full batch
                            hard_sel = torch.zeros(B, N, device=device, dtype=hard_sel_valid.dtype)
                            soft_sel = torch.zeros(B, N, device=device, dtype=soft_sel_valid.dtype)
                            hard_sel[valid_seeds] = hard_sel_valid
                            soft_sel[valid_seeds] = soft_sel_valid
                        else:
                            hard_sel = torch.zeros(B, N, device=device)
                            soft_sel = torch.zeros(B, N, device=device)
                            exp_stats = {"expected_size": torch.zeros(0, device=device)}

                        # Update assignments and availability (hard bookkeeping)
                        if valid_seeds.any():
                            for i, b in enumerate(valid_indices):
                                expansion_nodes = (hard_sel[b] > 0.5).nonzero(as_tuple=True)[0]
                                for n_idx in expansion_nodes:
                                    if cand_mask[b, n_idx]:  # Double-check it's a valid candidate
                                        assignment_matrix[b, n_idx, cluster_idx] = 1.0
                                        available_mask[b, n_idx] = False

                        # Use soft probabilities for cluster embeddings (gradients)
                        y_soft = soft_sel                                                        # [B, N]
                        
                        # Compute cardinality loss
                        if self.lambda_card > 0 and valid_seeds.any():
                            mu = self.cardinality_prior(phi_all[valid_seeds]).detach()           # [Bv] no gradient through prior
                            expected_size = exp_stats["expected_size"] if "expected_size" in exp_stats else torch.zeros_like(mu)
                            card_loss = self.lambda_card * ((expected_size - mu) ** 2).mean()
                            # Store cardinality loss for logging (you might need to aggregate this at model level)
                            if not hasattr(self, '_card_loss'):
                                self._card_loss = card_loss
                            else:
                                self._card_loss = self._card_loss + card_loss
                    else:
                        y_soft = torch.zeros(B, N, device=device, dtype=x.dtype)
                else:
                    y_soft = torch.zeros(B, N, device=device, dtype=x.dtype)
            else:
                # No valid seeds (should be rare with masks)
                y_soft = torch.zeros(B, N, device=device, dtype=x.dtype)
                w_seed = torch.zeros(B, N, device=device, dtype=x.dtype)

            # --- Cluster embedding for this cluster ---
            # Hard cluster embedding (existing mask average)
            cluster_mask = assignment_matrix[:, :, cluster_idx] > 0.5                  # [B, N]
            cluster_size = cluster_mask.sum(dim=-1, keepdim=True).float().clamp_min(1.0)
            h_hard = ((cluster_mask.unsqueeze(-1) * x).sum(dim=1) / cluster_size)     # [B, D]

            # Soft cluster embedding (ST seed + threshold-based expansion)
            m_soft = w_seed + y_soft                                                  # [B, N]
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

    def get_cardinality_loss(self) -> torch.Tensor:
        """Get accumulated cardinality loss and reset."""
        if hasattr(self, '_card_loss'):
            loss = self._card_loss
            delattr(self, '_card_loss')
            return loss
        return torch.tensor(0.0)

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
    threshold_ok = has_grad(part.expander.threshold_head)
    exp_wq_ok = part.expander.Wq.weight.grad is not None and part.expander.Wq.weight.grad.norm() > 0
    exp_wk_ok = part.expander.Wk.weight.grad is not None and part.expander.Wk.weight.grad.norm() > 0
    x_ok = x.grad is not None and x.grad.norm() > 0

    print("\nGradient flow check:")
    print(f"  seed_selector grad:    {'OK' if seed_ok else 'MISS'}")
    print(f"  threshold_head grad:   {'OK' if threshold_ok else 'MISS'}")
    print(f"  expander Wq grad:      {'OK' if exp_wq_ok else 'MISS'}")
    print(f"  expander Wk grad:      {'OK' if exp_wk_ok else 'MISS'}")
    print(f"  input x grad:          {'OK' if x_ok else 'MISS'}")

    assert seed_ok, "No gradient reached seed_selector (check ST seed wiring)."
    assert threshold_ok, "No gradient reached threshold_head (check threshold-based weighting)."
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
    num_nodes = 50
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
        cluster_size_max=15,
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

"""
BioBlobs proposal-style partitioner.

This partitioner emits a fixed blob proposal budget per protein and relies on
soft memberships for training-time blob formation. Hard, non-overlapping views
are only used by downstream interpretability exporters.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from .base import BasePartitioner


class BioBlobsPartitioner(BasePartitioner):
    """Seed-centered blob proposal module for MIL-style downstream decoding."""

    def __init__(
        self,
        input_dim: int,
        num_blobs_per_protein: int = 8,
        seed_hidden_dim: int = 128,
        attn_dim: int = 128,
        seed_radius: float = 12.0,
        proximity_bias: float = 0.5,
        seed_tau_init: float = 1.0,
        seed_tau_min: float = 0.25,
        seed_tau_decay: float = 0.95,
        membership_tau_init: float = 1.0,
        membership_tau_min: float = 0.25,
        membership_tau_decay: float = 0.95,
        membership_hoyer: float = 0.0,
        pooling: str = "mean",
        emit_interpretability: bool = False,
        eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__(
            pooling=pooling, emit_interpretability=emit_interpretability
        )

        self.input_dim = input_dim
        self.num_blobs_per_protein = num_blobs_per_protein
        self.seed_hidden_dim = seed_hidden_dim
        self.attn_dim = attn_dim
        self.seed_radius = seed_radius
        self.proximity_bias = proximity_bias
        self.membership_hoyer = membership_hoyer
        self.seed_tau_init = seed_tau_init
        self.seed_tau_min = seed_tau_min
        self.seed_tau_decay = seed_tau_decay
        self.membership_tau_init = membership_tau_init
        self.membership_tau_min = membership_tau_min
        self.membership_tau_decay = membership_tau_decay
        self.eps = eps

        self.seed_selector = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, seed_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(seed_hidden_dim, 1),
        )
        self.seed_query = nn.Linear(input_dim, attn_dim, bias=False)
        self.node_key = nn.Linear(input_dim, attn_dim, bias=False)
        self.q_norm = nn.RMSNorm(attn_dim)
        self.k_norm = nn.RMSNorm(attn_dim)

        self.register_buffer("epoch", torch.tensor(0, dtype=torch.long))

    def get_required_features(self):
        return ["node_features", "batch", "x"]

    def set_epoch(self, epoch: int) -> None:
        self.epoch.fill_(max(int(epoch), 0))

    def get_seed_temperature(self) -> float:
        return max(
            self.seed_tau_min,
            self.seed_tau_init * (self.seed_tau_decay ** int(self.epoch.item())),
        )

    def get_membership_temperature(self) -> float:
        return max(
            self.membership_tau_min,
            self.membership_tau_init
            * (self.membership_tau_decay ** int(self.epoch.item())),
        )

    def _select_seed_slots(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        tau: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Repeated ST top-1 extraction without replacement on a single logit pass.

        Returns:
            seed_assign_st: [B, K, N] straight-through one-hot selections
            seed_indices: [B, K] selected local residue indices or -1
            seed_valid: [B, K] validity mask for populated seed slots
        """
        batch_size, num_nodes = logits.shape
        seed_assignments = []
        seed_indices = torch.full(
            (batch_size, self.num_blobs_per_protein),
            -1,
            dtype=torch.long,
            device=logits.device,
        )
        seed_valid = torch.zeros(
            (batch_size, self.num_blobs_per_protein),
            dtype=torch.bool,
            device=logits.device,
        )

        available = mask.clone()
        tau = max(float(tau), 1e-6)
        fill_value = torch.finfo(logits.dtype).min

        for slot_idx in range(self.num_blobs_per_protein):
            hard_slot = torch.zeros_like(logits)
            soft_slot = torch.zeros_like(logits)
            active_rows = available.any(dim=-1)
            if active_rows.any():
                slot_logits = logits.masked_fill(~available, fill_value)
                soft_slot[active_rows] = F.softmax(
                    slot_logits[active_rows] / tau, dim=-1
                )
                selected_idx = slot_logits[active_rows].argmax(dim=-1)
                active_batch_idx = active_rows.nonzero(as_tuple=True)[0]
                hard_slot[active_batch_idx, selected_idx] = 1.0
                seed_indices[active_batch_idx, slot_idx] = selected_idx
                seed_valid[active_batch_idx, slot_idx] = True
                available[active_batch_idx, selected_idx] = False

            seed_assignments.append(hard_slot + (soft_slot - soft_slot.detach()))

        seed_assign_st = torch.stack(seed_assignments, dim=1)
        return seed_assign_st, seed_indices, seed_valid

    def _compute_dense_inputs(
        self, batch_data
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        node_features = batch_data.node_features
        batch = batch_data.batch
        coords = batch_data.x
        dense_features, batch_mask = to_dense_batch(node_features, batch)
        dense_coords, _ = to_dense_batch(coords, batch)

        node_mask = getattr(batch_data, "mask", None)
        if node_mask is None:
            dense_valid_mask = batch_mask
        else:
            dense_node_mask, _ = to_dense_batch(node_mask.bool(), batch)
            dense_valid_mask = batch_mask & dense_node_mask

        return dense_features, dense_coords, dense_valid_mask, batch

    def _compute_interpretability_payload(
        self,
        batch_data,
        assignment_matrix: torch.Tensor,
        valid_mask: torch.Tensor,
        seed_indices: torch.Tensor,
        seed_scores: torch.Tensor,
    ) -> None:
        batch_size = assignment_matrix.size(0)
        num_blobs_per_graph = torch.full(
            (batch_size,),
            self.num_blobs_per_protein,
            device=assignment_matrix.device,
            dtype=torch.long,
        )

        valid_node_indices = []
        assignment_matrices = []
        for graph_idx in range(batch_size):
            node_idx = valid_mask[graph_idx].nonzero(as_tuple=True)[0]
            valid_node_indices.append(node_idx)
            assignment_matrices.append(
                assignment_matrix[graph_idx, :, node_idx].transpose(0, 1)
            )

        self._set_interpretability_payload(
            batch_data=batch_data,
            assignment_type="soft",
            num_blobs_per_graph=num_blobs_per_graph,
            valid_node_local_indices_per_graph=valid_node_indices,
            assignment_matrices_per_graph=assignment_matrices,
        )
        if not self.emit_interpretability:
            return

        batch_data.partitioner_interpretability.update(
            {
                "seed_indices_per_graph": seed_indices,
                "seed_scores_per_graph": seed_scores,
            }
        )

    def forward(self, batch_data):
        self._validate_inputs(batch_data)
        self._clear_interpretability_payload(batch_data)

        dense_features, dense_coords, valid_mask, batch = self._compute_dense_inputs(
            batch_data
        )
        device = dense_features.device
        batch_size, max_nodes, feature_dim = dense_features.shape

        seed_logits = self.seed_selector(dense_features).squeeze(-1)
        seed_assign_st, seed_indices, seed_valid = self._select_seed_slots(
            seed_logits,
            valid_mask,
            tau=self.get_seed_temperature(),
        )

        safe_seed_indices = seed_indices.clamp_min(0)
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(-1)
        seed_repr = torch.einsum("bkn,bnd->bkd", seed_assign_st, dense_features)
        hard_seed_coords = dense_coords[batch_idx, safe_seed_indices]
        hard_seed_coords = hard_seed_coords * seed_valid.unsqueeze(-1)

        seed_queries = self.q_norm(self.seed_query(seed_repr))
        node_keys = self.k_norm(self.node_key(dense_features))
        scores = torch.einsum(
            "bkh,bnh->bkn", seed_queries, node_keys
        ) / math.sqrt(float(self.attn_dim))

        # Keep invalid nodes inert under the validity mask while avoiding the
        # broadcasted [B, K, N, 3] subtraction intermediate.
        masked_dense_coords = torch.where(
            valid_mask.unsqueeze(-1),
            dense_coords,
            torch.zeros_like(dense_coords),
        )
        distances = torch.cdist(
            hard_seed_coords,
            masked_dense_coords,
            compute_mode="use_mm_for_euclid_dist",
        )
        candidate_mask = (
            valid_mask.unsqueeze(1)
            & seed_valid.unsqueeze(-1)
            & (distances <= self.seed_radius)
        )

        seed_one_hot = (
            F.one_hot(safe_seed_indices, num_classes=max_nodes).to(torch.bool)
            & seed_valid.unsqueeze(-1)
        )
        candidate_mask = candidate_mask | seed_one_hot
        proximity = (
            1.0 - distances / max(float(self.seed_radius), self.eps)
        ).clamp_min(0.0)
        score_logits = scores + self.proximity_bias * proximity

        membership_tau = max(self.get_membership_temperature(), 1e-6)
        membership_soft = torch.sigmoid(score_logits / membership_tau)
        membership_soft = membership_soft * candidate_mask.float()
        membership_soft = torch.where(
            seed_one_hot,
            torch.ones_like(membership_soft),
            membership_soft,
        )

        assignment_matrix = membership_soft * valid_mask.unsqueeze(1).float()

        blob_mass = assignment_matrix.sum(dim=-1)
        blob_embeddings = torch.einsum(
            "bkn,bnd->bkd", assignment_matrix, dense_features
        )
        blob_embeddings = blob_embeddings / blob_mass.unsqueeze(-1).clamp_min(self.eps)
        blob_embeddings = torch.where(
            blob_mass.unsqueeze(-1) > self.eps,
            blob_embeddings,
            torch.zeros_like(blob_embeddings),
        )

        blob_features = blob_embeddings.reshape(batch_size * self.num_blobs_per_protein, feature_dim)
        blob_batch = torch.arange(
            batch_size, device=device, dtype=torch.long
        ).repeat_interleave(self.num_blobs_per_protein)

        batch_data.blob_features = blob_features
        batch_data.blob_batch = blob_batch
        batch_data.blob_seed_valid = seed_valid.reshape(-1)
        batch_data.blob_seed_valid_per_graph = seed_valid
        batch_data.seed_indices_per_graph = seed_indices
        batch_data.seed_scores_per_graph = torch.gather(
            seed_logits, dim=1, index=safe_seed_indices
        ) * seed_valid.float()
        batch_data.blob_memberships_soft = assignment_matrix

        # Diagnostic L1 (no longer added to the loss; kept for logging).
        total_candidates = candidate_mask.float().sum().clamp_min(1.0)
        membership_l1 = assignment_matrix.sum() / total_candidates

        # Hoyer-Square sparsity, per blob, normalized by candidate-set size:
        #   HS_k = (||a_k||_1 / ||a_k||_2)^2 / n_k  ∈ [1/n_k, 1]
        # Invalid blobs (seed_valid == False) have a_k = 0 and are masked out.
        # See experiments/partition_loss_hoyer.md for the derivation.
        blob_l1 = assignment_matrix.sum(dim=-1)                                  # [B, K]
        blob_l2 = assignment_matrix.pow(2).sum(dim=-1).clamp_min(self.eps).sqrt()  # [B, K]
        blob_cand_count = candidate_mask.float().sum(dim=-1).clamp_min(1.0)      # [B, K]
        blob_hoyer = (blob_l1 / blob_l2).pow(2) / blob_cand_count                # [B, K]

        valid_blob_f = seed_valid.float()
        valid_blob_total = valid_blob_f.sum().clamp_min(1.0)
        hoyer_mean = (blob_hoyer * valid_blob_f).sum() / valid_blob_total
        partitioner_loss = self.membership_hoyer * hoyer_mean
        batch_data.partitioner_loss = partitioner_loss

        zero = dense_features.new_zeros(())
        avg_mass = blob_mass[seed_valid].mean() if seed_valid.any() else zero
        avg_seed_logit = (
            batch_data.seed_scores_per_graph[seed_valid].mean()
            if seed_valid.any()
            else zero
        )

        # Residue coverage (union of all blob candidate sets): fraction of
        # valid residues that lie within seed_radius of at least one valid
        # seed. Purely geometric, threshold-free. Per-protein, then mean.
        num_valid = valid_mask.float().sum(dim=-1).clamp_min(1.0)        # [B]
        any_blob = candidate_mask.any(dim=1)                             # [B, N]
        all_blobs_frac = (
            (any_blob & valid_mask).float().sum(dim=-1) / num_valid
        ).mean()

        # blob_avg_size: literal residue count per blob, averaged over valid
        # blobs. Threshold at 0.5 so this is an integer-valued "how many
        # residues does this blob contain". Complements blob_avg_mass (soft L1)
        # and blob_hoyer (shape concentration).
        hard_mask = (assignment_matrix >= 0.5) & valid_mask.unsqueeze(1)  # [B, K, N]
        blob_hard_size = hard_mask.float().sum(dim=-1)                   # [B, K]
        avg_hard_size = (blob_hard_size * valid_blob_f).sum() / valid_blob_total

        batch_data.partitioner_metrics = {
            "blob_membership_l1": membership_l1.detach(),
            "blob_hoyer": hoyer_mean.detach(),
            "blob_avg_mass": avg_mass.detach(),
            "blob_avg_size": avg_hard_size.detach(),
            "blob_avg_seed_logit": avg_seed_logit.detach(),
            "all_blobs_frac": all_blobs_frac.detach(),
        }

        self._compute_interpretability_payload(
            batch_data=batch_data,
            assignment_matrix=assignment_matrix,
            valid_mask=valid_mask,
            seed_indices=seed_indices,
            seed_scores=batch_data.seed_scores_per_graph,
        )
        return batch_data

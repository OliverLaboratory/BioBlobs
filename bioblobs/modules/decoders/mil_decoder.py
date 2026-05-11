"""
MIL Decoder Module

Implements Multiple Instance Learning (MIL) decoder for protein blob classification.
Wraps AttentionMILHead to handle variable-length blob features from partitioner.

This decoder combines pooling + classification in a single module:
- Pads blob features to num_blobs_per_protein
- Creates attention mask for valid blobs
- Applies attention-based MIL pooling
- Returns logits + attention weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

from bioblobs.modules.decoders.mlp_decoder import MLPDecoder


def rank_blob_importance(
    attention_weights: torch.Tensor,
    mask: torch.Tensor,
    top_k: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rank valid blobs by MIL attention scores.

    Args:
        attention_weights: [B, K] attention scores from the MIL head
        mask: [B, K] boolean mask (True for valid blobs)
        top_k: Optional number of blobs to keep per protein. ``None`` keeps all.

    Returns:
        importance_values: [B, S] tensor of sorted attention scores
        importance_indices: [B, S] tensor of sorted local blob indices

    Notes:
        - Invalid / padded blobs are excluded from ranking.
        - Ties preserve lower original blob indices via stable sorting.
        - Output is padded with ``-inf`` / ``-1`` where proteins have fewer than
          ``S`` valid blobs.
    """
    if attention_weights.ndim != 2 or mask.ndim != 2:
        raise ValueError("attention_weights and mask must both be 2D tensors")
    if attention_weights.shape != mask.shape:
        raise ValueError("attention_weights and mask must have identical shapes")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive when provided")

    batch_size = attention_weights.shape[0]
    valid_counts = mask.sum(dim=-1)
    max_valid = int(valid_counts.max().item()) if batch_size > 0 else 0
    if max_valid == 0:
        return (
            attention_weights.new_full((batch_size, 0), float("-inf")),
            torch.empty((batch_size, 0), dtype=torch.long, device=mask.device),
        )

    selected_width = max_valid if top_k is None else min(top_k, max_valid)
    importance_values = attention_weights.new_full(
        (batch_size, selected_width), float("-inf")
    )
    importance_indices = torch.full(
        (batch_size, selected_width),
        -1,
        dtype=torch.long,
        device=mask.device,
    )

    for batch_idx in range(batch_size):
        valid_idx = mask[batch_idx].nonzero(as_tuple=True)[0]
        if valid_idx.numel() == 0:
            continue

        valid_scores = attention_weights[batch_idx, valid_idx]
        ranking = torch.argsort(valid_scores, descending=True, stable=True)
        ranked_idx = valid_idx[ranking]
        ranked_scores = attention_weights[batch_idx, ranked_idx]

        keep = ranked_idx.numel() if top_k is None else min(top_k, ranked_idx.numel())
        importance_values[batch_idx, :keep] = ranked_scores[:keep]
        importance_indices[batch_idx, :keep] = ranked_idx[:keep]

    return importance_values, importance_indices


class BlobSelfAttention(nn.Module):
    """Single-head self-attention across K blob tokens.

    Enriches each blob's representation with context from all other valid blobs
    before the MIL attention gate fires. The pairwise attention matrix [B, K, K]
    is returned for downstream interpretability.
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.scale = dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:    [B, K, D]  blob feature vectors
            mask: [B, K]     True = valid blob

        Returns:
            out:  [B, K, D]  context-enriched blobs (residual + LayerNorm)
            corr: [B, K, K]  pairwise attention weights
        """
        q, k, v = self.q(x), self.k(x), self.v(x)
        corr = torch.bmm(q, k.transpose(1, 2)) * self.scale        # [B, K, K]
        corr = corr.masked_fill(~mask.unsqueeze(1), float("-inf"))  # mask key dim
        attn = F.softmax(corr, dim=-1)
        attn = attn * mask.unsqueeze(-1).float()                    # zero invalid query rows
        attn = self.dropout(attn)
        context = torch.bmm(attn, v)                                # [B, K, D]
        out = self.norm(x + self.proj(context))
        return out, attn


class AttentionMILHead(nn.Module):
    """
    Attention-based Multiple Instance Learning head for blob classification.

    Architecture:
        1. Instance scoring: u_k = φ(b_k) where φ is a linear projection
        2. (Optional) Blob interaction: self-attention across K blobs
        3. Attention computation: α_k = softmax(a(b_k)) over valid blobs
        4. Bag aggregation: z = Σ α_k * u_k
        5. Classification: ŷ = softmax(W * z)

    Args:
        embedding_dim: Dimension of blob embeddings (D)
        num_classes: Number of output classes
        dropout: Dropout rate for regularization
        use_blob_interaction: If True, apply BlobSelfAttention after instance scoring

    Shapes:
        Input:
            blob_embeddings: [B, K, D] where B=batch, K=max_blobs, D=embedding_dim
            mask: [B, K] boolean tensor (True for valid blobs, False for padding)
        Output:
            logits: [B, num_classes]
            attention_weights: [B, K] normalized importance scores
            blob_corr: [B, K, K] pairwise correlation, or None if use_blob_interaction=False
    """

    def __init__(
        self,
        embedding_dim: int = 100,
        num_classes: int = 2,
        dropout: float = 0.1,
        use_blob_interaction: bool = False,
        classifier_hidden_multipliers: List[int] = [4, 2],
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.use_blob_interaction = use_blob_interaction

        # Instance scoring network φ: [D] → [D]
        self.instance_scorer = nn.Linear(embedding_dim, embedding_dim)

        # Optional inter-blob self-attention
        if use_blob_interaction:
            self.blob_interaction = BlobSelfAttention(embedding_dim, dropout=dropout)

        # Attention gate a: [D] → [1]
        self.attention_gate = nn.Linear(embedding_dim, 1)

        # Bag-level classifier: mirrors the baseline MLP head so MIL and MLP
        # decoders share identical bag/protein-level classifier capacity.
        self.classifier = MLPDecoder(
            input_dim=embedding_dim,
            num_classes=num_classes,
            hidden_multipliers=classifier_hidden_multipliers,
            drop_rate=dropout,
        )

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        nn.init.xavier_uniform_(self.instance_scorer.weight)
        nn.init.zeros_(self.instance_scorer.bias)

        nn.init.xavier_uniform_(self.attention_gate.weight)
        nn.init.zeros_(self.attention_gate.bias)

    def _compute_instance_features(
        self,
        blob_embeddings: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Build per-blob features used by both bag prediction and post-hoc blob scoring."""
        instance_features = self.instance_scorer(blob_embeddings)  # [B, K, D]
        instance_features = F.relu(instance_features)
        instance_features = self.dropout(instance_features)

        blob_corr = None
        if self.use_blob_interaction:
            instance_features, blob_corr = self.blob_interaction(instance_features, mask)

        return instance_features, blob_corr

    def _classify_instance_features(
        self,
        instance_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project per-blob features into class logits/probabilities for analysis."""
        B, K, D = instance_features.shape
        flat_features = instance_features.reshape(B * K, D)
        flat_logits = self.classifier(flat_features)
        instance_logits = flat_logits.reshape(B, K, self.num_classes)
        instance_probabilities = F.softmax(instance_logits, dim=-1)

        valid_mask = mask.unsqueeze(-1).float()
        instance_logits = instance_logits * valid_mask
        instance_probabilities = instance_probabilities * valid_mask
        return instance_logits, instance_probabilities

    def forward(
        self,
        blob_embeddings: torch.Tensor,
        mask: torch.Tensor,
        *,
        return_instance_predictions: bool = False,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """
        Forward pass through MIL head.

        Args:
            blob_embeddings: [B, K, D] blob feature vectors
            mask: [B, K] boolean mask (True = valid blob, False = padding)

        Returns:
            logits: [B, num_classes] class predictions
            attention_weights: [B, K] normalized importance scores (sum to 1)
            blob_corr: [B, K, K] pairwise blob correlation, or None
            instance_logits: [B, K, C] per-blob logits, or None
            instance_probabilities: [B, K, C] per-blob probabilities, or None
        """
        # Step 1: Instance scoring φ(b_k) → [B, K, D]
        instance_features, blob_corr = self._compute_instance_features(
            blob_embeddings, mask
        )

        # Step 3: Compute attention logits a(b_k) → [B, K, 1]
        attention_logits = self.attention_gate(blob_embeddings)  # [B, K, 1]
        attention_logits = attention_logits.squeeze(-1)  # [B, K]

        # Step 4: Mask invalid blobs (padding)
        attention_logits = self._apply_mask(attention_logits, mask)

        # Step 5: Normalize to get attention weights α_k
        attention_weights = F.softmax(attention_logits, dim=-1)  # [B, K]

        # Ensure masked positions have exactly zero attention
        attention_weights = attention_weights * mask.float()

        # Renormalize to ensure sum = 1.0 over valid blobs
        attention_sum = attention_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        attention_weights = attention_weights / attention_sum

        # Step 6: Weighted aggregation z = Σ α_k * u_k
        bag_representation = torch.einsum(
            "bk,bkd->bd", attention_weights, instance_features
        )

        # Step 7: Classification
        logits = self.classifier(bag_representation)  # [B, num_classes]
        instance_logits = None
        instance_probabilities = None
        if return_instance_predictions:
            instance_logits, instance_probabilities = self._classify_instance_features(
                instance_features,
                mask,
            )

        return (
            logits,
            attention_weights,
            blob_corr,
            instance_logits,
            instance_probabilities,
        )

    def _apply_mask(self, scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply mask to attention scores by setting invalid positions to -inf."""
        return scores.masked_fill(~mask, float("-inf"))

    def get_blob_importance_ranking(
        self,
        attention_weights: torch.Tensor,
        mask: torch.Tensor,
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience wrapper for ranking MIL blobs by attention."""
        return rank_blob_importance(attention_weights, mask, top_k=top_k)

    def get_instance_predictions(
        self, blob_embeddings: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get per-blob class scores from the MIL instance scorer.

        Args:
            blob_embeddings: [B, K, D] blob embeddings
            mask: [B, K] boolean mask

        Returns:
            instance_logits: [B, K, C] per-blob class logits
            instance_probabilities: [B, K, C] per-blob class probabilities
        """
        instance_features, _ = self._compute_instance_features(blob_embeddings, mask)
        return self._classify_instance_features(instance_features, mask)


class MILDecoder(nn.Module):
    """
    MIL-based decoder that handles variable-length blob features.

    Combines pooling + classification for Multiple Instance Learning:
    - Accepts blob features from partitioner (variable K per protein)
    - Pads to num_blobs_per_protein and creates mask
    - Applies attention-based MIL pooling
    - Returns logits + attention weights for interpretability

    Args:
        input_dim: Dimension of blob embeddings (from encoder)
        num_classes: Number of output classes
        dropout: Dropout rate (default: 0.1)
        return_attention: Return attention weights in extra dict (default: True)
        num_blobs_per_protein: Manual override for blob count (default: None = auto-detect)

    Example:
        >>> decoder = MILDecoder(input_dim=100, num_classes=3)
        >>> # After partitioner: blob_features [total_blobs, 100]
        >>> logits, extra = decoder(batch_data)
        >>> attention = extra['attention_weights']  # [B, max_K]
    """

    # Pipeline marker: framework should pass the full batch_data to this
    # decoder (and skip the explicit pooling step).
    consumes_batch_data: bool = True

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        return_attention: bool = True,
        return_instance_predictions: bool = False,
        num_blobs_per_protein: Optional[int] = None,
        use_blob_interaction: bool = False,
        classifier_hidden_multipliers: List[int] = [4, 2],
        mil_dim: int = 512,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.mil_dim = mil_dim
        self.embedding_dim = mil_dim  # Alias for compatibility
        self.num_classes = num_classes
        self.return_attention = return_attention
        self.return_instance_predictions = return_instance_predictions
        self.num_blobs_per_protein = num_blobs_per_protein

        # Project encoder-native blob embeddings to a shared MIL-head width so
        # attention capacity and classifier hidden sizes are comparable across
        # encoders of different dimensionality (ESM2=640, SaProt=1280, etc.).
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, mil_dim),
            nn.LayerNorm(mil_dim),
        )

        # Core MIL head
        self.mil_head = AttentionMILHead(
            embedding_dim=mil_dim,
            num_classes=num_classes,
            dropout=dropout,
            use_blob_interaction=use_blob_interaction,
            classifier_hidden_multipliers=classifier_hidden_multipliers,
        )

    def forward(self, batch_data) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through MIL decoder.
        
        Args:
            batch_data: PyTorch Geometric Batch object with:
                - blob_features: [total_blobs, D] from partitioner
                - blob_batch: [total_blobs] blob → protein mapping
        
        Returns:
            logits: [B, num_classes] classification logits
            extra: Dict with optional attention weights and mask
        """
        # Extract blob features and batch assignment
        blob_features = batch_data.blob_features
        blob_batch = batch_data.blob_batch

        blob_features = self.input_proj(blob_features)

        # Prepare padded batch with mask
        padded_features, mask = self._prepare_mil_batch(
            blob_features, blob_batch, self.num_blobs_per_protein
        )

        # Override mask with structural seed validity if available
        seed_valid_per_graph = getattr(batch_data, "blob_seed_valid_per_graph", None)
        if seed_valid_per_graph is not None:
            mask = seed_valid_per_graph

        # Forward through MIL head
        (
            logits,
            attention_weights,
            blob_corr,
            instance_logits,
            instance_probabilities,
        ) = self.mil_head(
            padded_features,
            mask,
            return_instance_predictions=self.return_instance_predictions,
        )

        # Package auxiliary outputs
        extra = {}
        if self.return_attention or self.return_instance_predictions:
            extra['blob_mask'] = mask
            # Store blob counts per protein for analysis
            if seed_valid_per_graph is not None:
                num_blobs = seed_valid_per_graph.sum(dim=-1)
            else:
                batch_size = int(blob_batch.max().item()) + 1
                num_blobs = torch.bincount(
                    blob_batch, minlength=batch_size
                )
            extra['num_blobs_per_protein'] = num_blobs
        if self.return_attention:
            extra['attention_weights'] = attention_weights
            if blob_corr is not None:
                extra['blob_correlation'] = blob_corr  # [B, K, K]
        if self.return_instance_predictions:
            extra['instance_logits'] = instance_logits
            extra['instance_probabilities'] = instance_probabilities

        return logits, extra

    def _has_regular_blob_layout(
        self,
        blob_batch: torch.Tensor,
        batch_size: int,
        num_blobs_per_protein: int,
    ) -> bool:
        """Check whether blobs are already laid out as [B * K] in graph-major order."""
        expected_total_blobs = batch_size * num_blobs_per_protein
        if blob_batch.numel() != expected_total_blobs:
            return False

        expected_batch = torch.arange(
            batch_size, device=blob_batch.device, dtype=blob_batch.dtype
        ).repeat_interleave(num_blobs_per_protein)
        return torch.equal(blob_batch, expected_batch)

    def _prepare_mil_batch(
        self,
        blob_features: torch.Tensor,
        blob_batch: torch.Tensor,
        num_blobs_per_protein: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert partitioner output to MIL input format.

        Transforms:
            blob_features: [total_blobs, D] → [B, max_K, D]
            blob_batch: [total_blobs] → mask: [B, max_K]

        Args:
            blob_features: Concatenated blob features from all proteins
            blob_batch: Maps each blob to its protein index
            num_blobs_per_protein: Maximum blob count (None = auto-detect from batch)

        Returns:
            padded_features: [B, max_K, D] zero-padded blob features
            mask: [B, max_K] boolean mask (True = valid, False = padding)
        """
        device = blob_features.device
        batch_size = int(blob_batch.max().item()) + 1
        num_blobs = torch.bincount(blob_batch, minlength=batch_size)

        # Determine max blobs
        if num_blobs_per_protein is None:
            num_blobs_per_protein = int(num_blobs.max().item())

        D = blob_features.shape[-1]

        if self._has_regular_blob_layout(
            blob_batch, batch_size, num_blobs_per_protein
        ):
            padded = blob_features.reshape(batch_size, num_blobs_per_protein, D)
            mask = torch.ones(
                batch_size,
                num_blobs_per_protein,
                dtype=torch.bool,
                device=device,
            )
            return padded, mask

        sort_idx = torch.argsort(blob_batch, stable=True)
        sorted_batch = blob_batch[sort_idx]
        sorted_features = blob_features[sort_idx]
        start_offsets = num_blobs.cumsum(dim=0) - num_blobs
        local_positions = (
            torch.arange(sorted_batch.numel(), device=device)
            - start_offsets[sorted_batch]
        )

        # Initialize padded tensor and mask
        padded = torch.zeros(batch_size, num_blobs_per_protein, D, device=device)
        max_positions = torch.arange(
            num_blobs_per_protein, device=device
        ).unsqueeze(0)
        clipped_counts = num_blobs.clamp(max=num_blobs_per_protein).unsqueeze(1)
        mask = max_positions < clipped_counts

        valid_positions = local_positions < num_blobs_per_protein
        padded[
            sorted_batch[valid_positions],
            local_positions[valid_positions],
        ] = sorted_features[valid_positions]

        return padded, mask

    def get_output_dim(self) -> int:
        """Return output dimension (for compatibility)."""
        return self.num_classes


class LightAttentionMILHead(nn.Module):
    """Light-Attention pooling over K blob tokens.

    Stärk 2021's Light-Attention head, ported from the residue axis to the
    blob axis. Two parallel ``Conv1d(D, D, kernel_size)`` along the K-blob
    axis: the "attention" branch is masked-softmaxed across blobs per channel;
    the "value" branch's masked max-pool and the attention-weighted sum are
    concatenated to ``[B, 2D]`` and fed to ``MLPDecoder``.

    Drop-in replacement for ``AttentionMILHead`` inside ``LightAttentionMILDecoder``.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        num_classes: int = 2,
        dropout: float = 0.1,
        kernel_size: int = 3,
        classifier_hidden_multipliers: List[int] = [2, 1],
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be odd to keep K unchanged via symmetric "
                f"padding, got {kernel_size}"
            )
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.kernel_size = kernel_size

        pad = kernel_size // 2
        self.attn_conv = nn.Conv1d(
            embedding_dim, embedding_dim, kernel_size, padding=pad
        )
        self.value_conv = nn.Conv1d(
            embedding_dim, embedding_dim, kernel_size, padding=pad
        )

        self.classifier = MLPDecoder(
            input_dim=2 * embedding_dim,
            num_classes=num_classes,
            hidden_multipliers=list(classifier_hidden_multipliers),
            drop_rate=dropout,
        )

    def forward(
        self,
        blob_embeddings: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            blob_embeddings: [B, K, D]
            mask: [B, K] boolean, True for valid blob slots.
        Returns:
            logits: [B, num_classes]
            attention_weights: [B, K] mean over channels, restricted to mask.
        """
        # Conv1d expects [B, D, K].
        b_t = blob_embeddings.transpose(1, 2)                    # [B, D, K]
        attn_logits = self.attn_conv(b_t)                        # [B, D, K]
        values = self.value_conv(b_t)                            # [B, D, K]

        mask_b1k = mask.unsqueeze(1)                             # [B, 1, K]
        attn_logits = attn_logits.masked_fill(~mask_b1k, float("-inf"))
        attn = F.softmax(attn_logits, dim=-1)                    # [B, D, K]

        attended = (values * attn).sum(dim=-1)                   # [B, D]
        values_masked = values.masked_fill(~mask_b1k, float("-inf"))
        max_pooled = values_masked.amax(dim=-1)                  # [B, D]

        feat = torch.cat([attended, max_pooled], dim=-1)         # [B, 2D]
        logits = self.classifier(feat)                           # [B, num_classes]

        attention_weights = attn.mean(dim=1) * mask.float()      # [B, K]
        return logits, attention_weights


class LightAttentionMILDecoder(MILDecoder):
    """MIL decoder that swaps the gated-attention head for Light-Attention.

    Subclasses ``MILDecoder`` so the framework's ``isinstance(_, MILDecoder)``
    routing still applies. Reuses ``input_proj``, padding, and mask handling
    from the parent; only the head is different.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        return_attention: bool = True,
        num_blobs_per_protein: Optional[int] = None,
        kernel_size: int = 3,
        classifier_hidden_multipliers: List[int] = [2, 1],
        mil_dim: int = 512,
    ):
        # Initialise nn.Module via grandparent to avoid building parent's
        # AttentionMILHead (which we'd then immediately replace).
        nn.Module.__init__(self)

        self.input_dim = input_dim
        self.mil_dim = mil_dim
        self.embedding_dim = mil_dim
        self.num_classes = num_classes
        self.return_attention = return_attention
        self.return_instance_predictions = False
        self.num_blobs_per_protein = num_blobs_per_protein

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, mil_dim),
            nn.LayerNorm(mil_dim),
        )

        self.mil_head = LightAttentionMILHead(
            embedding_dim=mil_dim,
            num_classes=num_classes,
            dropout=dropout,
            kernel_size=kernel_size,
            classifier_hidden_multipliers=classifier_hidden_multipliers,
        )

    def forward(self, batch_data) -> Tuple[torch.Tensor, Dict]:
        blob_features = batch_data.blob_features
        blob_batch = batch_data.blob_batch

        blob_features = self.input_proj(blob_features)

        padded_features, mask = self._prepare_mil_batch(
            blob_features, blob_batch, self.num_blobs_per_protein
        )

        seed_valid_per_graph = getattr(batch_data, "blob_seed_valid_per_graph", None)
        if seed_valid_per_graph is not None:
            mask = seed_valid_per_graph

        logits, attention_weights = self.mil_head(padded_features, mask)

        extra: Dict = {}
        if self.return_attention:
            extra["blob_mask"] = mask
            extra["attention_weights"] = attention_weights
            if seed_valid_per_graph is not None:
                extra["num_blobs_per_protein"] = seed_valid_per_graph.sum(dim=-1)
        return logits, extra


def create_mil_decoder(
    input_dim: int,
    num_classes: int,
    dropout: float = 0.1,
    return_attention: bool = True,
    return_instance_predictions: bool = False,
    mil_dim: int = 512,
) -> MILDecoder:
    """
    Factory function to create MIL decoder with standard settings.
    
    Args:
        input_dim: Blob embedding dimension
        num_classes: Number of output classes
        dropout: Dropout rate
        return_attention: Return attention weights for analysis
        return_instance_predictions: Return per-blob predictions for analysis
    
    Returns:
        Initialized MILDecoder instance
    """
    return MILDecoder(
        input_dim=input_dim,
        num_classes=num_classes,
        dropout=dropout,
        return_attention=return_attention,
        return_instance_predictions=return_instance_predictions,
        mil_dim=mil_dim,
    )

"""BioBlobs model: protein encoder -> neural blob partitioner -> (RVQ) -> MIL predictor.

The same class also assembles the comparators: residue-level heads (mean or attention
pooling, optionally after a residue GNN) and learned-pooling partitioners (DiffPool, TopK,
SAGPool) in place of the neural blob partitioner.

Example:
    >>> model = build_model(cfg, num_classes=31)
    >>> out = model(batch)  # a torch_geometric Batch with node_features, x, mask, batch
    >>> out.logits.shape, out.importance.shape, out.partition.memberships.shape
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

from bioblobs.modules.encoder import ProteinEncoder
from bioblobs.modules.gnn import ResidueGNN
from bioblobs.modules.mil_predictor import MILPredictor, blob_embeddings
from bioblobs.modules.partitioner import NeuralBlobPartitioner, Partition
from bioblobs.modules.pooling import AttentionPooling, DiffPool, MeanPooling, SelectionPool
from bioblobs.modules.rvq import Quantized, ResidualVQ


@dataclass
class ModelOutput:
    """Forward output for a batch of B proteins padded to N residues.

    Attributes:
        logits: Class logits, ``[B, C]``.
        residue_mask: Valid residues, ``[B, N]``.
        partition: Blob partition (None for residue-level heads).
        importance: MIL importance scores alpha, ``[B, K]`` (None for residue-level heads).
        residue_attention: Residue attention of attention pooling, ``[B, N]``.
        quantized: RVQ output when the quantizer is enabled.
    """

    logits: torch.Tensor
    residue_mask: torch.Tensor
    partition: Partition | None = None
    importance: torch.Tensor | None = None
    residue_attention: torch.Tensor | None = None
    quantized: Quantized | None = None


class BioBlobs(nn.Module):
    """Composes an encoder, an optional partitioner and RVQ, and a predictor.

    Args:
        encoder: Frozen protein encoder (precomputed embeddings).
        predictor: ``MILPredictor`` when a partitioner is set, otherwise ``MeanPooling`` or
            ``AttentionPooling``.
        partitioner: ``NeuralBlobPartitioner``, ``DiffPool``, ``SelectionPool`` or None.
        rvq: Optional residual vector quantizer between blob embeddings and the predictor.
        gnn: Optional residue GNN applied to the encoder output (comparators only).
    """

    def __init__(
        self,
        encoder: ProteinEncoder,
        predictor: nn.Module,
        partitioner: nn.Module | None = None,
        rvq: ResidualVQ | None = None,
        gnn: ResidueGNN | None = None,
    ) -> None:
        super().__init__()
        if (partitioner is None) != (not isinstance(predictor, MILPredictor)):
            raise ValueError("The MIL predictor requires a partitioner, and only it does.")
        self.encoder = encoder
        self.gnn = gnn
        self.partitioner = partitioner
        self.rvq = rvq
        self.predictor = predictor

    def set_epoch(self, epoch: int) -> None:
        """Advance the temperature and RVQ warmup schedules."""
        for module in (self.partitioner, self.rvq):
            if hasattr(module, "set_epoch"):
                module.set_epoch(epoch)

    def forward(self, batch: Batch) -> ModelOutput:
        """Run the model on a batch.

        Args:
            batch: Collated proteins with ``node_features`` ``[N_total, D]``, C-alpha
                coordinates ``x`` ``[N_total, 3]``, residue ``mask`` ``[N_total]``, ``batch``
                and, for graph comparators, ``edge_index``.

        Returns:
            The model output; see ``ModelOutput``.
        """
        features = self.encoder(batch.node_features)
        if self.gnn is not None:
            features = self.gnn(features, batch.edge_index)
        mask = batch.mask.bool() if getattr(batch, "mask", None) is not None else None
        if mask is None:
            mask = torch.ones(features.shape[0], dtype=torch.bool, device=features.device)
        dense, slots = to_dense_batch(features, batch.batch)
        residue_mask = slots & to_dense_batch(mask, batch.batch)[0]

        if self.partitioner is None:
            logits, attention = self.predictor(features, batch.batch, mask)
            return ModelOutput(logits, residue_mask, residue_attention=attention)

        if isinstance(self.partitioner, NeuralBlobPartitioner):
            coords = to_dense_batch(batch.x, batch.batch)[0]
            partition = self.partitioner(dense, coords, residue_mask)
        else:
            partition = self.partitioner(
                features, batch.edge_index, batch.batch, mask, dense.shape[1]
            )
        blobs = partition.blob_features
        if blobs is None:
            blobs = blob_embeddings(partition.memberships, dense)
        quantized = None
        if self.rvq is not None:
            b, k, d = blobs.shape
            quantized = self.rvq(blobs.reshape(b * k, d), partition.blob_mask.reshape(-1))
            blobs = quantized.embeddings.reshape(b, k, -1)
        logits, importance = self.predictor(blobs, partition.blob_mask)
        return ModelOutput(logits, residue_mask, partition, importance, quantized=quantized)


def build_model(cfg: DictConfig, num_classes: int) -> BioBlobs:
    """Build a model from the ``encoders``, ``partitioners``, ``codebooks`` and ``decoders`` groups.

    Args:
        cfg: Composed Hydra config.
        num_classes: Number of output logits.

    Returns:
        The assembled model.
    """
    enc = cfg.encoders
    encoder = ProteinEncoder(enc.name, enc.model_name, enc.get("output_dim"))
    width = encoder.output_dim
    gnn = None
    if enc.get("gnn") is not None:
        gnn = ResidueGNN(input_dim=width, **enc.gnn)
        width = gnn.output_dim

    part_cfg = cfg.partitioners
    partitioner: nn.Module | None = None
    if part_cfg.name == "bioblobs":
        partitioner = NeuralBlobPartitioner(
            width,
            num_blobs=part_cfg.num_blobs,
            seed_hidden_dim=part_cfg.seed_hidden_dim,
            attn_dim=part_cfg.attn_dim,
            seed_radius=part_cfg.seed_radius,
            proximity_bias=part_cfg.proximity_bias,
            seed_tau=tuple(part_cfg.seed_tau),
            membership_tau=tuple(part_cfg.membership_tau),
        )
    elif part_cfg.name == "diffpool":
        partitioner = DiffPool(
            width,
            max_clusters=part_cfg.max_clusters,
            hidden_dim=part_cfg.hidden_dim,
            temperature=part_cfg.temperature,
            entropy_weight=part_cfg.entropy_weight,
            link_weight=part_cfg.link_weight,
        )
    elif part_cfg.name in {"topk", "sagpool"}:
        partitioner = SelectionPool(width, method=part_cfg.name, ratio=part_cfg.ratio)
    elif part_cfg.name != "none":
        raise ValueError(f"Unknown partitioner {part_cfg.name!r}")

    rvq = None
    if partitioner is not None and cfg.get("codebooks", {}).get("enabled", False):
        params = {k: v for k, v in cfg.codebooks.items() if k not in {"name", "enabled"}}
        rvq = ResidualVQ(width, **params)
        width = rvq.output_dim

    dec = cfg.decoders
    common = {"hidden_multipliers": list(dec.hidden_multipliers), "dropout": dec.dropout}
    if dec.name == "mil":
        predictor: nn.Module = MILPredictor(
            width,
            num_classes,
            mil_dim=dec.mil_dim,
            use_blob_interaction=dec.use_blob_interaction,
            **common,
        )
    elif dec.name == "mean_pool":
        predictor = MeanPooling(
            width,
            num_classes,
            input_layernorm=dec.input_layernorm,
            masked=dec.masked,
            proj_dim=dec.get("proj_dim"),
            **common,
        )
    elif dec.name == "attention_pool":
        predictor = AttentionPooling(
            width,
            num_classes,
            input_layernorm=dec.input_layernorm,
            proj_dim=dec.get("proj_dim"),
            **common,
        )
    else:
        raise ValueError(f"Unknown decoder {dec.name!r}")
    return BioBlobs(encoder, predictor, partitioner, rvq, gnn)

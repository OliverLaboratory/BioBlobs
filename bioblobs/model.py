"""BioBlobs model: protein encoder -> neural blob partitioner -> (RVQ) -> MIL predictor.

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
from bioblobs.modules.mil_predictor import MILPredictor, blob_embeddings
from bioblobs.modules.partitioner import NeuralBlobPartitioner, Partition
from bioblobs.modules.rvq import Quantized, ResidualVQ


@dataclass
class ModelOutput:
    """Forward output for a batch of B proteins padded to N residues.

    Attributes:
        logits: Class logits, ``[B, C]``.
        residue_mask: Valid residues, ``[B, N]``.
        partition: Blob partition.
        importance: MIL importance scores alpha, ``[B, K]``.
        quantized: RVQ output when the quantizer is enabled.
    """

    logits: torch.Tensor
    residue_mask: torch.Tensor
    partition: Partition
    importance: torch.Tensor
    quantized: Quantized | None = None


class BioBlobs(nn.Module):
    """Composes an encoder, the blob partitioner, optional RVQ and the MIL predictor.

    Args:
        encoder: Frozen protein encoder (precomputed embeddings).
        predictor: MIL predictor over blob embeddings.
        partitioner: Neural blob partitioner.
        rvq: Optional residual vector quantizer between blob embeddings and the predictor.
    """

    def __init__(
        self,
        encoder: ProteinEncoder,
        predictor: MILPredictor,
        partitioner: NeuralBlobPartitioner,
        rvq: ResidualVQ | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
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
                coordinates ``x`` ``[N_total, 3]``, residue ``mask`` ``[N_total]`` and ``batch``.

        Returns:
            The model output; see ``ModelOutput``.
        """
        features = self.encoder(batch.node_features)
        mask = batch.mask.bool() if getattr(batch, "mask", None) is not None else None
        if mask is None:
            mask = torch.ones(features.shape[0], dtype=torch.bool, device=features.device)
        dense, slots = to_dense_batch(features, batch.batch)
        residue_mask = slots & to_dense_batch(mask, batch.batch)[0]

        coords = to_dense_batch(batch.x, batch.batch)[0]
        partition = self.partitioner(dense, coords, residue_mask)
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

    part_cfg = cfg.partitioners
    if part_cfg.name != "bioblobs":
        raise ValueError(f"Unknown partitioner {part_cfg.name!r}")
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

    rvq = None
    if cfg.get("codebooks", {}).get("enabled", False):
        params = {k: v for k, v in cfg.codebooks.items() if k not in {"name", "enabled"}}
        rvq = ResidualVQ(width, **params)
        width = rvq.output_dim

    dec = cfg.decoders
    if dec.name != "mil":
        raise ValueError(f"Unknown decoder {dec.name!r}")
    predictor = MILPredictor(
        width,
        num_classes,
        mil_dim=dec.mil_dim,
        use_blob_interaction=dec.use_blob_interaction,
        hidden_multipliers=list(dec.hidden_multipliers),
        dropout=dec.dropout,
    )
    return BioBlobs(encoder, predictor, partitioner, rvq)

"""
Modular BioBlobs Framework

A flexible framework orchestrating protein encoders, partitioners, and decoders
for protein structure classification tasks using a pipeline architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from typing import Tuple, List

from bioblobs.modules.encoders.config import resolve_encoder_name, sanitize_encoder_cfg
from bioblobs.modules.pipeline_ops import EncoderOp, PoolingOp, DecoderOp, PartitionerOp

MIL_DECODER_TARGET = "bioblobs.modules.decoders.mil_decoder.MILDecoder"
# Targets that satisfy the partitioner-on branch. All must subclass MILDecoder
# (the framework checks isinstance later via _is_mil_decoder).
MIL_DECODER_TARGETS = {
    MIL_DECODER_TARGET,
    "bioblobs.modules.decoders.mil_decoder.LightAttentionMILDecoder",
}
MLP_DECODER_TARGET = "bioblobs.modules.decoders.mlp_decoder.MLPDecoder"
LIGHT_ATTN_DECODER_TARGET = (
    "bioblobs.modules.decoders.light_attention_decoder.LightAttentionDecoder"
)
ATTENTION_POOL_DECODER_TARGET = (
    "bioblobs.modules.decoders.attention_pool_decoder.AttentionPoolDecoder"
)
SIMPLE_ATTN_DECODER_TARGET = (
    "bioblobs.modules.decoders.simple_attn_decoder.SimpleAttnDecoder"
)
# Decoders that pool internally and so are valid for the no-partitioner branch
# alongside MLPDecoder.
NO_PARTITIONER_DECODER_TARGETS = {
    MLP_DECODER_TARGET,
    LIGHT_ATTN_DECODER_TARGET,
    ATTENTION_POOL_DECODER_TARGET,
    SIMPLE_ATTN_DECODER_TARGET,
}
DEFAULT_MIL_DECODER_CFG = {
    "_target_": MIL_DECODER_TARGET,
    "dropout": 0.1,
    "return_attention": True,
    "return_instance_predictions": False,
    "num_blobs_per_protein": None,
}
DEFAULT_MLP_DECODER_CFG = {
    "_target_": MLP_DECODER_TARGET,
    "hidden_multipliers": [4, 2],
    "drop_rate": 0.1,
}


class BioBlobsFramework(nn.Module):
    """
    Modular framework orchestrating encoder, partitioner, and decoder
    using a pipeline architecture.

    Components can be swapped via configuration:
    - Encoder: ESM2-static, SaProt-static (via Hydra instantiate)
    - Partitioner: Optional (can be disabled)
    - Decoder: MLP, attention pool variants, or MIL

    Pipelines:
        no partitioner, classic decoder:    encoder → pooling → decoder
        no partitioner, attention decoder:  encoder → decoder (pools internally)
        partitioner + MIL:                  encoder → partitioner → decoder
        partitioner + classic decoder:      encoder → partitioner → pooling → decoder
    """

    def __init__(self, cfg: DictConfig, num_classes: int):
        """
        Initialize framework from Hydra config and build processing pipeline.

        Args:
            cfg: Config with keys 'encoders', 'partitioners', 'decoders'
            num_classes: Number of output classes
        """
        super().__init__()

        # Build components using Hydra instantiate
        self.encoder_name = resolve_encoder_name(cfg.encoders)
        self.encoder = instantiate(sanitize_encoder_cfg(cfg.encoders))
        self.partitioner = self._build_partitioner(cfg.partitioners)
        self.decoder_selection_note = None
        self.decoder = self._build_decoder(cfg.decoders, num_classes)

        # Store configuration
        self.task_problem_type = cfg.get("tasks", {}).get("problem_type", "multi_class")
        self.pooling = cfg.get('pooling', 'mean')
        self.num_classes = num_classes

        # Build processing pipeline
        self.pipeline = self._build_pipeline()

        # Print pipeline info
        self._print_pipeline_info()

    def _build_partitioner(self, cfg):
        if cfg.get('enabled', False):
            return instantiate(cfg, input_dim=self.encoder.get_output_dim())
        return None

    def _build_decoder(self, cfg, num_classes):
        input_dim = self.encoder.get_output_dim()
        decoder_cfg = self._resolve_decoder_cfg(cfg)
        return instantiate(decoder_cfg, input_dim=input_dim, num_classes=num_classes)

    def _resolve_decoder_cfg(self, cfg):
        """
        Resolve decoder configuration based on the active pipeline.

        Partitioner on  -> must be MIL.
        Partitioner off -> any decoder that pools internally
                           (MLP, SimpleAttn, AttentionPool, LightAttention).
        """
        if self.partitioner is not None:
            if self._decoder_cfg_is_mil(cfg):
                return cfg
            self.decoder_selection_note = (
                "Partitioner enabled; overriding decoder selection to MILDecoder."
            )
            return OmegaConf.create(DEFAULT_MIL_DECODER_CFG)

        if cfg.get("_target_", "") in NO_PARTITIONER_DECODER_TARGETS:
            return cfg

        self.decoder_selection_note = (
            "Partitioner disabled and decoder unrecognized; "
            "overriding decoder selection to MLPDecoder."
        )
        return OmegaConf.create(DEFAULT_MLP_DECODER_CFG)

    def _decoder_cfg_is_mil(self, cfg) -> bool:
        return cfg.get("_target_", "") in MIL_DECODER_TARGETS

    def _build_pipeline(self) -> List:
        pipeline = []

        # 1. Encoder (always present)
        pipeline.append(EncoderOp(self.encoder))

        decoder_pools_internally = bool(
            getattr(self.decoder, "consumes_batch_data", False)
        )

        # 2. Branch based on partitioner and decoder type
        if self.partitioner is not None:
            pipeline.append(PartitionerOp(self.partitioner))

            if self._is_mil_decoder():
                # MIL path: partitioner → MIL decoder (handles pooling internally)
                pass
            else:
                # Standard path: partitioner → pooling → decoder
                pipeline.append(PoolingOp(self.pooling))
        else:
            if self._is_mil_decoder():
                raise ValueError(
                    "MIL decoder requires an enabled partitioner. "
                    "Set partitioners.enabled=true or use a baseline decoder."
                )
            if not decoder_pools_internally:
                # Standard pooling → decoder path.
                pipeline.append(PoolingOp(self.pooling))
            # else: decoder consumes batch_data directly (e.g. LightAttention).

        # 3. Decoder (always present)
        pipeline.append(DecoderOp(self.decoder))

        return pipeline

    def _is_mil_decoder(self) -> bool:
        from bioblobs.modules.decoders.mil_decoder import MILDecoder
        return isinstance(self.decoder, MILDecoder)

    def _print_pipeline_info(self):
        print("\n" + "=" * 70)
        print("BioBlobs Framework Pipeline")
        print("=" * 70)
        print(f"Encoder: {type(self.encoder).__name__}")
        print(f"  - Output dim: {self.encoder.get_output_dim()}")
        print(f"  - Required features: {self.encoder.get_required_features()}")

        if self.partitioner is not None:
            print(f"Partitioner: {type(self.partitioner).__name__} (enabled)")
        else:
            print("Partitioner: Disabled")

        if self._is_mil_decoder():
            print("Decoder: MIL (Multiple Instance Learning)")
            print("  - Combines attention-based pooling + classification")
        elif getattr(self.decoder, "consumes_batch_data", False):
            print(f"Decoder: {type(self.decoder).__name__}")
            print("  - Consumes batch_data; pools internally (no explicit pooling op)")
        else:
            print(f"  - Using pooling: {self.pooling}")
            print(f"Decoder: {type(self.decoder).__name__}")
        if self.decoder_selection_note is not None:
            print(f"  - {self.decoder_selection_note}")

        print(f"  - Num classes: {self.num_classes}")

        print(f"\nPipeline ({len(self.pipeline)} operations):")
        for i, op in enumerate(self.pipeline, 1):
            print(f"  {i}. {type(op).__name__}")
        print("=" * 70 + "\n")

    def forward(self, batch_data):
        extra = {}
        result = batch_data
        for operation in self.pipeline:
            result, extra = operation(result, extra)
        logits = result
        return logits, extra

    def compute_cross_entropy_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels)

    def predict_with_proba(self, batch_data) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            logits, _ = self.forward(batch_data)
            if self.task_problem_type == "multi_label":
                probabilities = torch.sigmoid(logits)
                predictions = probabilities >= 0.5
            else:
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
            return predictions, probabilities


# Backward compatibility alias
BioBlobsModel = BioBlobsFramework

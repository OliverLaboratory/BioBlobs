"""
Base Partitioner Interface

Defines the abstract interface for all partitioner modules in BioBlobs framework.
Partitioners transform node-level features into blob-level features.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence


class BasePartitioner(nn.Module, ABC):
    """
    Abstract base class for partitioner modules.
    
    Partitioners process batched protein graphs and aggregate node features
    into blob-level features. Graph-level pooling is handled separately.
    
    Expected Input (batch_data attributes):
        - node_features: [num_nodes, feature_dim] - Node embeddings from encoder
        - batch: [num_nodes] - Graph assignment for each node
        - (optional) edge_index: [2, num_edges] - Graph connectivity
        - (optional) mask: [num_nodes] - Valid node mask
    
    Expected Output (updated batch_data attributes):
        - blob_features: [num_blobs, feature_dim] - Blob-level representations
        - blob_batch: [num_blobs] - Graph assignment for each blob

    Pipeline Flow:
        node_features → blob pooling → blob_features
        (Graph pooling is handled by PoolingOp in the pipeline)
    """
    
    def __init__(
        self, pooling: str = 'mean', emit_interpretability: bool = False
    ):
        """
        Initialize base partitioner.
        
        Args:
            pooling: Pooling method for aggregation ('mean', 'max', 'sum')
            emit_interpretability: Emit per-graph assignment provenance for
                post-hoc interpretability consumers.
        """
        super().__init__()
        self.pooling = pooling
        self.emit_interpretability = emit_interpretability
    
    @abstractmethod
    def forward(self, batch_data):
        """
        Transform node features to cluster features via partitioning.
        
        Args:
            batch_data: PyTorch Geometric Batch object with:
                - node_features: [num_nodes, feature_dim]
                - partition: [num_nodes] - Local cluster IDs
                - batch: [num_nodes] - Graph indices
        
        Returns:
            batch_data: Updated with blob_features and blob_batch attributes
        
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    def get_pooling_type(self) -> str:
        """
        Get the pooling method used by this partitioner.
        
        Returns:
            str: Pooling type ('mean', 'max', 'sum')
        """
        return self.pooling

    def get_required_features(self) -> List[str]:
        """
        Get the batch_data features required by this partitioner.

        Returns:
            List[str]: Required batch_data attribute names
        """
        return ['node_features', 'batch']

    def _build_blob_feature_offsets(
        self, num_blobs_per_graph: torch.Tensor
    ) -> torch.Tensor:
        """
        Build prefix offsets for each graph's blob slice in blob_features.

        Args:
            num_blobs_per_graph: [batch_size] tensor with blob counts

        Returns:
            [batch_size + 1] prefix-sum tensor where graph g maps to
            blob_features[offsets[g]:offsets[g + 1]]
        """
        if num_blobs_per_graph.ndim != 1:
            raise ValueError(
                "num_blobs_per_graph must be a 1D tensor for interpretability"
            )

        prefix = torch.zeros(
            1,
            device=num_blobs_per_graph.device,
            dtype=num_blobs_per_graph.dtype,
        )
        return torch.cat([prefix, num_blobs_per_graph.cumsum(dim=0)], dim=0)

    def _set_interpretability_payload(
        self,
        batch_data,
        assignment_type: str,
        num_blobs_per_graph: torch.Tensor,
        valid_node_local_indices_per_graph,
        assignment_matrices_per_graph,
    ):
        """
        Attach a standardized interpretability payload to batch_data.

        Args:
            batch_data: Batch object to annotate
            assignment_type: 'soft' or 'hard'
            num_blobs_per_graph: [batch_size] tensor
            valid_node_local_indices_per_graph: List[Tensor[N_valid_g]]
            assignment_matrices_per_graph: List[Tensor[N_valid_g, K_g]]
        """
        if not self.emit_interpretability:
            return

        if assignment_type not in {"soft", "hard"}:
            raise ValueError(
                f"Unsupported assignment_type {assignment_type!r}; expected 'soft' or 'hard'"
            )

        if len(valid_node_local_indices_per_graph) != len(assignment_matrices_per_graph):
            raise ValueError(
                "Interpretability payload lists must have the same number of graphs"
            )

        batch_data.partitioner_interpretability = {
            "assignment_type": assignment_type,
            "num_blobs_per_graph": num_blobs_per_graph,
            "valid_node_local_indices_per_graph": (
                valid_node_local_indices_per_graph
            ),
            "assignment_matrices_per_graph": assignment_matrices_per_graph,
            "blob_feature_offsets": self._build_blob_feature_offsets(
                num_blobs_per_graph
            ),
        }

    def _clear_interpretability_payload(self, batch_data):
        """Remove any stale interpretability payload from a reused batch object."""
        if hasattr(batch_data, "partitioner_interpretability"):
            delattr(batch_data, "partitioner_interpretability")
    
    def _validate_inputs(
        self, batch_data, required: Optional[Sequence[str]] = None
    ):
        """
        Validate that batch_data has required attributes.
        
        Args:
            batch_data: Batch object to validate
            required: Optional explicit list of required attributes. Defaults to
                self.get_required_features().
        
        Raises:
            AttributeError: If required attributes are missing
        """
        required = list(required or self.get_required_features())
        for attr in required:
            if not hasattr(batch_data, attr):
                raise AttributeError(
                    f"batch_data missing required attribute '{attr}' for partitioner"
                )

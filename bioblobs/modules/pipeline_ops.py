"""
Pipeline Operations

Callable wrappers for framework components that can be chained together
in a processing pipeline.
"""

from torch_scatter import scatter_mean, scatter_sum, scatter_max


class EncoderOp:
    """Wraps encoder in a callable pipeline operation."""
    
    def __init__(self, encoder):
        """
        Args:
            encoder: ProteinEncoder instance
        """
        self.encoder = encoder
    
    def __call__(self, batch_data, extra):
        """
        Execute encoder operation.
        
        Args:
            batch_data: Batch object
            extra: Dict for auxiliary outputs
            
        Returns:
            batch_data: Updated with node_features
            extra: Unchanged
        """
        batch_data = self.encoder(batch_data)
        return batch_data, extra


class PoolingOp:
    """Pools node/blob features to graph level."""
    
    def __init__(self, pooling_type='mean'):
        """
        Args:
            pooling_type: 'mean', 'max', or 'sum'
        """
        self.pooling_type = pooling_type
    
    def __call__(self, batch_data, extra):
        """
        Execute pooling operation.
        
        Handles two cases:
        1. If blob_features exists: integrate with node_features then pool to graph
        2. Otherwise: pool node_features directly to graph
        
        Args:
            batch_data: Batch object with:
                - node_features and batch (always present)
                - blob_features and blob_batch (optional, from partitioner)
            extra: Dict for auxiliary outputs
            
        Returns:
            batch_data: Updated with graph_features
            extra: Unchanged
        """
        # Check if blob features are present (from partitioner)
        if hasattr(batch_data, 'blob_features') and batch_data.blob_features is not None:
            blob_features = batch_data.blob_features
            blob_batch = batch_data.blob_batch

            if self.pooling_type == "mean":
                graph_features = scatter_mean(blob_features, blob_batch, dim=0)
            elif self.pooling_type == "max":
                graph_features = scatter_max(blob_features, blob_batch, dim=0)[0]
            elif self.pooling_type == "sum":
                graph_features = scatter_sum(blob_features, blob_batch, dim=0)
            else:
                graph_features = scatter_mean(blob_features, blob_batch, dim=0)
        else:
            # Standard node-level pooling
            node_features = batch_data.node_features
            batch = batch_data.batch
            
            if self.pooling_type == "mean":
                graph_features = scatter_mean(node_features, batch, dim=0)
            elif self.pooling_type == "max":
                graph_features = scatter_max(node_features, batch, dim=0)[0]
            elif self.pooling_type == "sum":
                graph_features = scatter_sum(node_features, batch, dim=0)
            else:
                graph_features = scatter_mean(node_features, batch, dim=0)
        
        batch_data.graph_features = graph_features
        return batch_data, extra


class DecoderOp:
    """Wraps decoder in a callable pipeline operation."""

    def __init__(self, decoder):
        """
        Args:
            decoder: BaseDecoder instance (or any decoder that consumes
                ``batch_data`` directly, e.g. MILDecoder, LightAttentionDecoder)
        """
        self.decoder = decoder
        self.consumes_batch_data = self._consumes_batch_data(decoder)

    def __call__(self, batch_data, extra):
        """
        Execute decoder operation.

        Args:
            batch_data: Batch object with graph_features (standard)
                       OR full batch (decoders flagged ``consumes_batch_data``)
            extra: Dict for auxiliary outputs

        Returns:
            logits: Classification logits [batch_size, num_classes]
            extra: Updated with decoder-specific outputs if applicable
        """
        if self.consumes_batch_data:
            # Decoder pools internally and returns (logits, extra_dict).
            logits, decoder_extra = self.decoder(batch_data)
            if decoder_extra:
                extra.update(decoder_extra)
        else:
            # Standard decoder expects pooled graph features.
            graph_features = batch_data.graph_features
            logits = self.decoder(graph_features)

        return logits, extra

    @staticmethod
    def _consumes_batch_data(decoder) -> bool:
        """Decoders that pool internally set ``consumes_batch_data = True``."""
        return bool(getattr(decoder, "consumes_batch_data", False))


class PartitionerOp:
    """Wraps partitioner in a callable pipeline operation."""
    
    def __init__(self, partitioner):
        """
        Args:
            partitioner: Partitioner instance
        """
        self.partitioner = partitioner
    
    def __call__(self, batch_data, extra):
        """
        Execute partitioner operation.
        
        Args:
            batch_data: Batch object with node_features attribute
            extra: Dict for auxiliary outputs
            
        Returns:
            batch_data: Updated with blob_features and blob_batch
            extra: Updated with partitioner info for MIL decoder
        """
        # Call partitioner to pool node features → blob features
        # Partitioner updates batch_data in-place and returns it
        batch_data = self.partitioner(batch_data)
        
        # Store partitioner info for downstream components (e.g., MIL decoder)
        if hasattr(batch_data, 'blob_batch') and batch_data.blob_batch is not None:
            batch_size = batch_data.blob_batch.max().item() + 1
            num_blobs = [
                (batch_data.blob_batch == i).sum().item()
                for i in range(batch_size)
            ]
            partitioner_info = {
                'blob_batch': batch_data.blob_batch,
                'num_blobs_per_protein': num_blobs
            }
            if hasattr(batch_data, 'blob_seed_valid'):
                partitioner_info['blob_seed_valid'] = batch_data.blob_seed_valid
            if hasattr(batch_data, 'blob_seed_valid_per_graph'):
                partitioner_info['blob_seed_valid_per_graph'] = (
                    batch_data.blob_seed_valid_per_graph
                )
            if hasattr(batch_data, 'seed_indices_per_graph'):
                partitioner_info['seed_indices_per_graph'] = (
                    batch_data.seed_indices_per_graph
                )
            if hasattr(batch_data, 'seed_scores_per_graph'):
                partitioner_info['seed_scores_per_graph'] = (
                    batch_data.seed_scores_per_graph
                )
            if (
                hasattr(batch_data, 'partitioner_interpretability')
                and batch_data.partitioner_interpretability is not None
            ):
                partitioner_info.update(batch_data.partitioner_interpretability)
            extra['partitioner_info'] = partitioner_info

        if hasattr(batch_data, 'partitioner_loss'):
            extra['partitioner_loss'] = batch_data.partitioner_loss

        if hasattr(batch_data, 'partitioner_metrics'):
            extra['partitioner_metrics'] = batch_data.partitioner_metrics

        return batch_data, extra

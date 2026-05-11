import torch.nn as nn


class MLPDecoder(nn.Module):
    """Simple MLP decoder for classification."""

    def __init__(
        self,
        input_dim,
        num_classes,
        hidden_multipliers=[4, 2],
        drop_rate=0.1,
        proj_dim=None,
    ):
        """
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            hidden_multipliers: List of multipliers for hidden layer sizes
            drop_rate: Dropout rate
            proj_dim: Optional input projection width. When set, the decoder
                prepends Linear(input_dim, proj_dim) + LayerNorm(proj_dim) and
                the downstream MLP operates at proj_dim instead of input_dim.
                This mirrors MILDecoder.input_proj so baselines can match the
                BioBlobs MIL classifier capacity.
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_multipliers = hidden_multipliers
        self.drop_rate = drop_rate
        self.proj_dim = proj_dim

        if proj_dim is not None:
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, proj_dim),
                nn.LayerNorm(proj_dim),
            )
            mlp_input_dim = proj_dim
        else:
            self.input_proj = nn.Identity()
            mlp_input_dim = input_dim

        # Build MLP layers
        layers = []
        current_dim = mlp_input_dim

        for multiplier in hidden_multipliers:
            hidden_dim = int(mlp_input_dim * multiplier)
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(drop_rate)
            ])
            current_dim = hidden_dim

        # Final classification layer
        layers.append(nn.Linear(current_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

    def forward(self, features):
        """
        Args:
            features: [batch_size, input_dim]
        Returns:
            logits: [batch_size, num_classes]
        """
        return self.mlp(self.input_proj(features))

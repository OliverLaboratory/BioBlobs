from abc import ABC, abstractmethod
import torch.nn as nn


class BaseDecoder(ABC, nn.Module):
    """Base class for classification decoders."""
    
    @abstractmethod
    def forward(self, features):
        """Decode features to class logits."""
        pass

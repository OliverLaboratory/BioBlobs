from abc import ABC, abstractmethod
import torch.nn as nn
from typing import List


class ProteinEncoder(ABC, nn.Module):
    """Base class for protein structure encoders."""
    
    @abstractmethod
    def forward(self, batch_data):
        """
        Encode protein graph to node features.
        
        Args:
            batch_data: PyTorch Geometric Batch object with encoder-specific attributes.
                       Encoder should extract required features and validate.
        
        Returns:
            batch_data: Updated batch_data with node_features attribute added
        
        Raises:
            AttributeError: If required features are missing from batch_data
        """
        pass
    
    @abstractmethod
    def get_output_dim(self):
        """
        Return output feature dimension.
        
        Returns:
            int: Number of scalar features in encoder output
        """
        pass
    
    @abstractmethod
    def get_required_features(self) -> List[str]:
        """
        Return list of required attribute names in batch_data.
        
        Returns:
            List[str]: Names of required attributes in batch_data
        """
        pass
    
    def _validate_batch_data(self, batch_data):
        """
        Validate that batch_data contains all required features.
        
        Args:
            batch_data: Batch object to validate
            
        Raises:
            AttributeError: If required features are missing
        """
        required = self.get_required_features()
        missing = [f for f in required if not hasattr(batch_data, f)]
        
        if missing:
            raise AttributeError(
                f"{self.__class__.__name__} requires features {required}, "
                f"but batch_data is missing: {missing}"
            )

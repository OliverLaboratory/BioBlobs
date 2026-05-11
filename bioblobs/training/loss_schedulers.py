
from typing import Dict


class LossWeightScheduler:
    """Scheduler for gradually ramping loss weights during training."""
    
    def __init__(self, initial_weights: Dict[str, float], final_weights: Dict[str, float], ramp_epochs: int):
        self.initial_weights = initial_weights
        self.final_weights = final_weights
        self.ramp_epochs = ramp_epochs
        
    def get_weights(self, epoch: int) -> Dict[str, float]:
        """Get interpolated weights for current epoch."""
        if epoch >= self.ramp_epochs:
            return self.final_weights.copy()
            
        alpha = epoch / self.ramp_epochs
        weights = {}
        for key in self.initial_weights:
            initial_val = self.initial_weights[key]
            final_val = self.final_weights.get(key, initial_val)
            weights[key] = (1 - alpha) * initial_val + alpha * final_val
        return weights
"""
PyTorch Lightning callbacks for BioBlobs training.

This module provides custom callbacks for tracking and logging during training.
"""

import os
import time
import json
from pytorch_lightning.callbacks import Callback


class EpochTimingCallback(Callback):
    """Callback to track and log epoch training time to WandB and save to file."""
    
    def __init__(self, stage_idx, output_dir):
        super().__init__()
        self.stage_idx = stage_idx
        self.output_dir = output_dir
        self.epoch_start_time = None
        self.train_start_time = None
        self.val_start_time = None
        # Store times for each epoch
        self.train_times = []
        self.val_times = []
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Record start time of training epoch."""
        self.epoch_start_time = time.time()
        self.train_start_time = time.time()
    
    def on_validation_epoch_start(self, trainer, pl_module):
        """Record start time of validation."""
        self.val_start_time = time.time()
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Log validation time and store it."""
        if self.val_start_time is not None:
            val_time = time.time() - self.val_start_time
            self.val_times.append(val_time)
            pl_module.log(f'stage{self.stage_idx}/val_time_seconds', val_time, sync_dist=True)
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Log epoch timing at the end of each epoch and store it."""
        if self.epoch_start_time is not None:
            # Total epoch time (train + val)
            total_epoch_time = time.time() - self.epoch_start_time
            
            # Training time only (excludes validation)
            train_time = time.time() - self.train_start_time
            if self.val_start_time is not None:
                train_time = self.val_start_time - self.train_start_time
            
            # Store training time
            self.train_times.append(train_time)
            
            # Log metrics
            pl_module.log(f'stage{self.stage_idx}/epoch_time_seconds', total_epoch_time, sync_dist=True)
            pl_module.log(f'stage{self.stage_idx}/train_time_seconds', train_time, sync_dist=True)
            
            # Print to console
            val_time_str = f", Val={self.val_times[-1]:.2f}s" if self.val_times else ""
            print(f"  ⏱️  Epoch {trainer.current_epoch}: "
                  f"Total={total_epoch_time:.2f}s, Train={train_time:.2f}s{val_time_str}")
    
    def save_timing_data(self):
        """Save timing data to JSON file with averages."""
        if not self.train_times:
            return
        
        # Calculate averages
        avg_train_time = sum(self.train_times) / len(self.train_times)
        avg_val_time = sum(self.val_times) / len(self.val_times) if self.val_times else 0.0
        
        timing_data = {
            "stage": self.stage_idx,
            "epochs": len(self.train_times),
            "train_times_per_epoch": [round(t, 4) for t in self.train_times],
            "val_times_per_epoch": [round(t, 4) for t in self.val_times],
            "avg_train_time_seconds": round(avg_train_time, 4),
            "avg_val_time_seconds": round(avg_val_time, 4),
        }
        
        # Save to file
        timing_file = os.path.join(self.output_dir, f"stage{self.stage_idx}_timing.json")
        os.makedirs(self.output_dir, exist_ok=True)
        with open(timing_file, "w") as f:
            json.dump(timing_data, f, indent=2)
        
        print(f"✓ Timing data saved to: {timing_file}")
        print(f"  Average train time: {avg_train_time:.4f}s")
        print(f"  Average validation time: {avg_val_time:.4f}s")
        
        return timing_data

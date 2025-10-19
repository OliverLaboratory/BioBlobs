"""
FMax metric for multi-label classification evaluation.

This class implements the F-max score which finds the optimal threshold
that maximizes the F1 score across different threshold values.

The F-max score is commonly used for protein function prediction tasks
and other multi-label classification problems where the optimal threshold
is not known a priori.
"""

import numpy as np
import torch

class FMaxMetric:
    """
    F-max metric for multi-label classification evaluation.
    
    This metric computes the maximum F1-score across all threshold values.
    """
    
    def __init__(self, num_thresholds=100):
        self.num_thresholds = num_thresholds
        self.thresholds = np.linspace(0, 1, num_thresholds)
    
    def _convert_to_numpy(self, tensor):
        """Convert tensor to numpy array if needed."""
        if hasattr(tensor, 'cpu'):
            return tensor.cpu().numpy()
        return tensor
    
    def _apply_sigmoid(self, logits):
        """Apply sigmoid to logits if they are not probabilities."""
        if logits.max() > 1.0 or logits.min() < 0.0:
            return 1 / (1 + np.exp(-np.clip(logits, -50, 50)))  # Clip to prevent overflow
        return logits
    
    def fmax(self, y_true, y_pred):
        """
        Compute F-max score.
        
        Args:
            y_true (torch.Tensor or np.ndarray): True binary labels [batch_size, num_classes]
            y_pred (torch.Tensor or np.ndarray): Predicted probabilities/logits [batch_size, num_classes]
            
        Returns:
            float: F-max score
        """
        # Convert torch tensors to numpy if needed
        y_true = self._convert_to_numpy(y_true)
        y_pred = self._convert_to_numpy(y_pred)
        
        # Apply sigmoid to logits if needed
        y_pred = self._apply_sigmoid(y_pred)
        
        fmax = 0
        for t in self.thresholds:
            prec = self.precision(y_true, y_pred, t)
            rec = self.recall(y_true, y_pred, t)
            if prec + rec == 0:
                continue
            f1 = (2 * prec * rec) / (prec + rec)
            fmax = max(fmax, f1)
        return fmax

    def precision(self, y_true, y_pred, threshold=0.5):
        """
        Compute precision at given threshold.
        
        Args:
            y_true (torch.Tensor or np.ndarray): True binary labels [batch_size, num_classes]
            y_pred (torch.Tensor or np.ndarray): Predicted probabilities/logits [batch_size, num_classes]
            threshold (float): Threshold value
            
        Returns:
            float: Precision score
        """
        # Convert torch tensors to numpy if needed
        y_true = self._convert_to_numpy(y_true)
        y_pred = self._convert_to_numpy(y_pred)
        
        # Apply sigmoid to logits if needed
        y_pred = self._apply_sigmoid(y_pred)
        
        batch_size = y_true.shape[0]
        if batch_size == 0:
            return 0.0
        
        y_pred_thresh = (y_pred >= threshold).astype(np.float32)
        
        # Calculate per-sample precision
        tp = np.logical_and(y_true, y_pred_thresh).sum(axis=1).astype(np.float32)  # True positives per sample
        fp = np.logical_and(1 - y_true, y_pred_thresh).sum(axis=1).astype(np.float32)  # False positives per sample
        
        # Precision = TP / (TP + FP) per sample, then average across samples
        precision_per_sample = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
        return precision_per_sample.mean()

    def recall(self, y_true, y_pred, threshold=0.5):
        """
        Compute recall at given threshold.
        
        Args:
            y_true (torch.Tensor or np.ndarray): True binary labels [batch_size, num_classes]
            y_pred (torch.Tensor or np.ndarray): Predicted probabilities/logits [batch_size, num_classes]
            threshold (float): Threshold value
            
        Returns:
            float: Recall score
        """
        # Convert torch tensors to numpy if needed
        y_true = self._convert_to_numpy(y_true)
        y_pred = self._convert_to_numpy(y_pred)
        
        # Apply sigmoid to logits if needed
        y_pred = self._apply_sigmoid(y_pred)
        
        batch_size = y_true.shape[0]
        if batch_size == 0:
            return 0.0
        
        y_pred_thresh = (y_pred >= threshold).astype(np.float32)
        
        # Calculate per-sample recall
        tp = np.logical_and(y_true, y_pred_thresh).sum(axis=1).astype(np.float32)  # True positives per sample
        fn = np.logical_and(y_true, 1 - y_pred_thresh).sum(axis=1).astype(np.float32)  # False negatives per sample
        
        # Recall = TP / (TP + FN) per sample, then average across samples
        recall_per_sample = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
        return recall_per_sample.mean()
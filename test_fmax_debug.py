"""
Quick test to debug FMax metric calculation with known correct values.
"""
import torch
import numpy as np
import sys
import os
sys.path.append('.')

from utils.fmax_metric import FMaxMetric

def test_fmax_with_known_values():
    """Test FMax with simple known cases."""
    print("🔍 Testing FMax metric with known correct values...")
    
    metric = FMaxMetric(num_thresholds=11)  # Use fewer thresholds for debugging
    print(f"Thresholds: {metric.thresholds}")
    
    # Test Case 1: Perfect predictions
    print("\n📊 Test Case 1: Perfect predictions")
    y_true = torch.tensor([[1, 0, 1, 0, 1]], dtype=torch.float32)  # [batch_size=1, num_classes=5]
    y_pred = torch.tensor([[0.9, 0.1, 0.8, 0.2, 0.95]], dtype=torch.float32)  # Strong predictions
    
    print(f"True labels: {y_true}")
    print(f"Predictions: {y_pred}")
    
    fmax = metric.fmax(y_true, y_pred)
    prec_05 = metric.precision(y_true, y_pred, 0.5)
    rec_05 = metric.recall(y_true, y_pred, 0.5)
    
    print(f"FMax: {fmax:.4f} (should be ~1.0)")
    print(f"Precision @ 0.5: {prec_05:.4f} (should be 1.0)")
    print(f"Recall @ 0.5: {rec_05:.4f} (should be 1.0)")
    
    # Test Case 2: All zeros prediction
    print("\n📊 Test Case 2: All zeros prediction")
    y_true = torch.tensor([[1, 0, 1, 0, 1]], dtype=torch.float32)
    y_pred = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    
    print(f"True labels: {y_true}")
    print(f"Predictions: {y_pred}")
    
    fmax = metric.fmax(y_true, y_pred)
    prec_05 = metric.precision(y_true, y_pred, 0.5)
    rec_05 = metric.recall(y_true, y_pred, 0.5)
    
    print(f"FMax: {fmax:.4f} (should be 0.0)")
    print(f"Precision @ 0.5: {prec_05:.4f} (should be 0.0)")
    print(f"Recall @ 0.5: {rec_05:.4f} (should be 0.0)")
    
    # Test Case 3: Random logits (what the real model produces)
    print("\n📊 Test Case 3: Random logits (simulating model output)")
    y_true = torch.tensor([[1, 0, 1, 0, 1]], dtype=torch.float32)
    y_pred = torch.tensor([[-0.5, 2.0, 1.5, -1.0, 0.8]], dtype=torch.float32)  # Raw logits
    
    print(f"True labels: {y_true}")
    print(f"Raw logits: {y_pred}")
    
    # Apply sigmoid manually to see what happens
    y_pred_sigmoid = torch.sigmoid(y_pred)
    print(f"After sigmoid: {y_pred_sigmoid}")
    
    fmax = metric.fmax(y_true, y_pred)
    prec_05 = metric.precision(y_true, y_pred, 0.5)
    rec_05 = metric.recall(y_true, y_pred, 0.5)
    
    print(f"FMax: {fmax:.4f} (should be 0-1)")
    print(f"Precision @ 0.5: {prec_05:.4f} (should be 0-1)")
    print(f"Recall @ 0.5: {rec_05:.4f} (should be 0-1)")
    
def test_individual_components():
    """Test precision and recall calculation step by step."""
    print("\n🔧 Testing individual components step by step...")
    
    metric = FMaxMetric()
    
    # Simple case
    y_true = np.array([[1, 0, 1]], dtype=np.float32)  # 1 sample, 3 classes
    y_pred = np.array([[0.8, 0.3, 0.9]], dtype=np.float32)  # probabilities
    threshold = 0.5
    
    print(f"Input shapes - y_true: {y_true.shape}, y_pred: {y_pred.shape}")
    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")
    print(f"threshold: {threshold}")
    
    # Manual calculation
    y_pred_thresh = (y_pred >= threshold).astype(np.float32)
    print(f"y_pred_thresh: {y_pred_thresh}")
    
    # For precision: TP / (TP + FP) per sample
    tp_and_fp = y_pred_thresh.sum(axis=1)  # Predicted positives per sample
    tp = np.logical_and(y_true, y_pred_thresh).sum(axis=1)  # True positives per sample
    print(f"Predicted positives per sample: {tp_and_fp}")
    print(f"True positives per sample: {tp}")
    
    # Calculate precision per sample
    precision_per_sample = np.divide(tp.astype(float), tp_and_fp.astype(float), out=np.zeros_like(tp, dtype=float), where=tp_and_fp!=0)
    precision_avg = precision_per_sample.mean()
    print(f"Precision per sample: {precision_per_sample}")
    print(f"Average precision: {precision_avg}")
    
    # For recall: TP / (TP + FN) per sample  
    tp_and_fn = y_true.sum(axis=1)  # Actual positives per sample
    # Calculate recall per sample
    recall_per_sample = np.divide(tp.astype(float), tp_and_fn.astype(float), out=np.zeros_like(tp, dtype=float), where=tp_and_fn!=0)
    recall_avg = recall_per_sample.mean()
    print(f"Actual positives per sample: {tp_and_fn}")
    print(f"Recall per sample: {recall_per_sample}")
    print(f"Average recall: {recall_avg}")
    
    # Expected F1
    if precision_avg + recall_avg > 0:
        f1_expected = 2 * precision_avg * recall_avg / (precision_avg + recall_avg)
    else:
        f1_expected = 0
    print(f"Expected F1: {f1_expected}")
    
    # Test our implementation
    prec_calc = metric.precision(y_true, y_pred, threshold)
    rec_calc = metric.recall(y_true, y_pred, threshold)
    print(f"Our precision: {prec_calc}")
    print(f"Our recall: {rec_calc}")
    
def test_with_real_model_output():
    """Test with shapes similar to real model output."""
    print("\n🧬 Testing with real model output shapes...")
    
    metric = FMaxMetric(num_thresholds=21)
    
    # Simulate batch_size=2, num_classes=10 (smaller than real 5127 for debugging)
    batch_size, num_classes = 2, 10
    
    # Create realistic targets (sparse, like real protein labels)
    y_true = torch.zeros(batch_size, num_classes, dtype=torch.float32)
    y_true[0, [1, 4, 7]] = 1  # First protein has 3 functions
    y_true[1, [2, 5]] = 1     # Second protein has 2 functions
    
    # Create realistic predictions (random logits from model)
    y_pred = torch.randn(batch_size, num_classes) * 2  # Logits in range roughly [-4, 4]
    
    print(f"Batch size: {batch_size}, Num classes: {num_classes}")
    print(f"True labels:\n{y_true}")
    print(f"Raw logits:\n{y_pred}")
    print(f"After sigmoid:\n{torch.sigmoid(y_pred)}")
    
    fmax = metric.fmax(y_true, y_pred)
    prec = metric.precision(y_true, y_pred, 0.5)
    rec = metric.recall(y_true, y_pred, 0.5)
    
    print(f"FMax: {fmax:.4f}")
    print(f"Precision @ 0.5: {prec:.4f}")
    print(f"Recall @ 0.5: {rec:.4f}")
    print(f"Valid ranges: FMax ∈ [0,1]: {0 <= fmax <= 1}, Precision ∈ [0,1]: {0 <= prec <= 1}, Recall ∈ [0,1]: {0 <= rec <= 1}")

if __name__ == "__main__":
    test_fmax_with_known_values()
    test_individual_components() 
    test_with_real_model_output()
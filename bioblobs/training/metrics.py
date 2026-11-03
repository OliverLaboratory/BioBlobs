"""Protein-level classification metrics: accuracy, macro-F1 and Fmax."""

from __future__ import annotations

import numpy as np
from sklearn import metrics as skm


def multiclass_metrics(
    labels: np.ndarray, logits: np.ndarray, num_classes: int
) -> dict[str, float]:
    """Accuracy, macro precision/recall/F1 and MCC for single-label classification.

    ``macro_f1`` averages over the classes present in labels or predictions (class union).
    ``macro_f1_fixed_vocab`` averages over all ``num_classes`` classes.

    Args:
        labels: True class indices, ``[M]``.
        logits: Class logits, ``[M, C]``.
        num_classes: Size of the class vocabulary.

    Returns:
        Metric name to value.
    """
    y = np.asarray(labels, dtype=int).reshape(-1)
    pred = np.asarray(logits).argmax(axis=1)
    macro = {"average": "macro", "zero_division": 0}
    return {
        "accuracy": float(skm.accuracy_score(y, pred)),
        "macro_precision": float(skm.precision_score(y, pred, **macro)),
        "macro_recall": float(skm.recall_score(y, pred, **macro)),
        "macro_f1": float(skm.f1_score(y, pred, **macro)),
        "macro_f1_fixed_vocab": float(
            skm.f1_score(y, pred, labels=list(range(num_classes)), **macro)
        ),
        "mcc": float(skm.matthews_corrcoef(y, pred)),
    }


def fmax(labels: np.ndarray, logits: np.ndarray, num_thresholds: int = 21) -> dict[str, float]:
    """Protein-centric Fmax over thresholds ``linspace(0, 1, num_thresholds)``.

    Precision is averaged over proteins with at least one prediction above the threshold,
    recall over all proteins.

    Args:
        labels: Binary label matrix, ``[M, C]``.
        logits: Label logits, ``[M, C]``.
        num_thresholds: Number of thresholds.

    Returns:
        ``{"fmax", "precision_at_fmax", "recall_at_fmax", "threshold"}``.
    """
    y = np.asarray(labels) > 0.5
    prob = 1.0 / (1.0 + np.exp(-np.clip(np.asarray(logits, dtype=np.float64), -50, 50)))
    best = {"fmax": 0.0, "precision_at_fmax": 0.0, "recall_at_fmax": 0.0, "threshold": 0.0}
    if y.shape[0] == 0:
        return best
    for t in np.linspace(0, 1, num_thresholds):
        hit = prob >= t
        tp = (y & hit).sum(1).astype(np.float32)
        predicted = hit.sum(1).astype(np.float32)
        positives = y.sum(1).astype(np.float32)
        any_pred = prob.max(1) >= t
        prec_each = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted != 0)
        rec_each = np.divide(tp, positives, out=np.zeros_like(tp), where=positives != 0)
        precision = float(prec_each[any_pred].sum() / any_pred.sum()) if any_pred.any() else 0.0
        recall = float(rec_each.mean())
        if precision + recall == 0:
            continue
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best["fmax"]:
            best = {
                "fmax": f1,
                "precision_at_fmax": precision,
                "recall_at_fmax": recall,
                "threshold": float(t),
            }
    return best

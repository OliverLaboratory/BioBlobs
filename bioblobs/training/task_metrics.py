"""Task-specific metric helpers for BioBlobs training."""

from __future__ import annotations

import numpy as np
from sklearn import metrics

from .fmax_metric import FMaxMetric


def compute_multiclass_metrics(y_true, logits) -> dict[str, float]:
    y_true_np = np.asarray(y_true, dtype=int).reshape(-1)
    logits_np = np.asarray(logits)
    if logits_np.ndim == 1:
        y_pred = logits_np.astype(int)
    else:
        y_pred = logits_np.argmax(axis=1).astype(int)

    return {
        "accuracy": float(metrics.accuracy_score(y_true_np, y_pred)),
        "macro_precision": float(
            metrics.precision_score(
                y_true_np,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            metrics.recall_score(
                y_true_np,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            metrics.f1_score(
                y_true_np,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "mcc": float(metrics.matthews_corrcoef(y_true_np, y_pred)),
    }


def compute_multilabel_metrics(y_true, logits, metric: FMaxMetric | None = None) -> dict[str, float]:
    fmax_metric = metric or FMaxMetric()
    stats = fmax_metric.best_stats(y_true, logits)
    return {
        "fmax": float(stats["fmax"]),
        "precision_at_fmax": float(stats["precision"]),
        "recall_at_fmax": float(stats["recall"]),
    }

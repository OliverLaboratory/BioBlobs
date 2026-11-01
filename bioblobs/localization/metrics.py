"""Residue localization metrics: per-protein AUROC, AUPRC and F1@L, aggregated by median.

F1@L ranks residues by score and takes the top L, where L is the number of annotated site
residues; with predicted and true set sizes equal, F1, precision and recall coincide. Ties are
broken by residue order (stable sort).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def residue_ranking_metrics(
    scores: torch.Tensor | np.ndarray,
    sites: torch.Tensor | np.ndarray,
    valid: torch.Tensor | np.ndarray | None = None,
) -> dict[str, float] | None:
    """Score one protein's residue ranking against its annotated sites.

    Args:
        scores: Residue scores, higher means more likely a site, ``[N]``.
        sites: Binary site annotation, ``[N]``.
        valid: Residues to include, ``[N]`` (all when None).

    Returns:
        ``{"auroc", "auprc", "f1_at_l", "n_res", "n_site"}``, or None when the protein has no
        site residue or only site residues.
    """
    s = _numpy(scores).astype(np.float64)
    y = (_numpy(sites) > 0.5).astype(int)
    if valid is not None:
        keep = _numpy(valid).astype(bool)
        s, y = s[keep], y[keep]
    n_site = int(y.sum())
    if n_site == 0 or n_site == y.size:
        return None
    top = np.argsort(-s, kind="stable")[:n_site]
    return {
        "auroc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
        "f1_at_l": float(y[top].mean()),
        "n_res": int(y.size),
        "n_site": n_site,
    }


def median_metrics(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    """Median of each metric over proteins (the per-seed summary used in the paper)."""
    rows = list(rows)
    keys = ("auroc", "auprc", "f1_at_l")
    return {k: float(np.median([r[k] for r in rows])) for k in keys} | {"n_proteins": len(rows)}


def _numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)

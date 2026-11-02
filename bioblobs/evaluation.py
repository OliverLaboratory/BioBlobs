"""Test-set evaluation of a trained model: classification metrics and residue localization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

from bioblobs.data import ProteinGraphDataset, make_loader
from bioblobs.localization.baselines import burial_scores, integrated_gradients, random_scores
from bioblobs.localization.metrics import median_metrics, residue_ranking_metrics
from bioblobs.localization.scores import max_membership, weighted_membership
from bioblobs.model import BioBlobs
from bioblobs.training.metrics import fmax, multiclass_metrics

SCORES = (
    "weighted_membership",
    "max_membership",
    "attention",
    "integrated_gradients",
    "burial",
    "random",
)


@torch.no_grad()
def _model_scores(model: BioBlobs, batch: Batch, score: str) -> tuple[torch.Tensor, torch.Tensor]:
    out = model(batch)
    if score == "attention":
        if out.residue_attention is None:
            raise ValueError("--score attention needs a decoders=attention_pool checkpoint")
        return out.logits, out.residue_attention
    if score not in {"weighted_membership", "max_membership"}:
        raise ValueError(f"Unknown residue score {score!r}")
    if out.partition is None:
        raise ValueError(f"--score {score} needs a checkpoint with a partitioner")
    if score == "weighted_membership":
        return out.logits, weighted_membership(out.partition.memberships, out.importance)
    return out.logits, max_membership(out.partition.memberships, out.partition.blob_mask)


def residue_scores(
    model: BioBlobs, batch: Batch, score: str, seed: int = 0, batch_index: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Logits ``[B, C]`` and dense residue scores ``[B, N]`` for one batch.

    Args:
        model: Trained model in eval mode.
        batch: Collated proteins.
        score: One of ``SCORES``.
        seed: Seed of the random baseline; batch ``b`` uses ``1000 * seed + b``.
        batch_index: Position of the batch in the loader (random baseline only).

    Returns:
        Logits and residue scores.
    """
    if score in {"weighted_membership", "max_membership", "attention"}:
        return _model_scores(model, batch, score)
    with torch.no_grad():
        logits = model(batch).logits
    if score == "burial":
        flat = torch.cat(
            [burial_scores(batch.x[batch.batch == g]) for g in range(batch.num_graphs)]
        )
    elif score == "random":
        generator = torch.Generator().manual_seed(1000 * seed + batch_index)
        flat = random_scores(batch.num_nodes, generator).to(batch.x.device)
    elif score == "integrated_gradients":

        def forward(x: torch.Tensor) -> torch.Tensor:
            shifted = batch.clone()
            shifted.node_features = x
            return model(shifted).logits

        flat = integrated_gradients(forward, batch.node_features.float(), logits.argmax(-1), 16)
    else:
        raise ValueError(f"Unknown score {score!r}")
    return logits, to_dense_batch(flat, batch.batch)[0]


def evaluate(
    model: BioBlobs,
    dataset: ProteinGraphDataset,
    cfg: DictConfig,
    num_classes: int,
    score: str | None,
    out_dir: str | Path,
    batch_size: int = 32,
    seed: int = 0,
) -> dict[str, Any]:
    """Evaluate one model on one split and write ``metrics.json`` and ``per_protein.jsonl``.

    Localization uses proteins that carry a ``residue_target``; per protein, residues with a
    valid backbone are ranked by ``score``; the summary is the median over proteins.

    Args:
        model: Trained model in eval mode.
        dataset: Evaluation split.
        cfg: Config of the model's training run.
        num_classes: Number of classes.
        score: Residue score (``SCORES``), or None for classification only.
        out_dir: Output directory.
        batch_size: Proteins per batch.
        seed: Seed of the random baseline.

    Returns:
        The summary metrics.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    logits_all, labels_all, rows = [], [], []
    for b, batch in enumerate(make_loader(dataset, batch_size, shuffle=False)):
        batch = batch.to(device)
        if score is None:
            with torch.no_grad():
                logits, dense = model(batch).logits, None
        else:
            logits, dense = residue_scores(model, batch, score, seed=seed, batch_index=b)
        logits_all.append(logits.float().cpu())
        labels_all.append(batch.y.cpu())
        if dense is None or getattr(batch, "residue_target", None) is None:
            continue
        sites = to_dense_batch(batch.residue_target, batch.batch)[0]
        valid = to_dense_batch(batch.mask.bool(), batch.batch)[0]
        for i, name in enumerate(batch.name):
            metrics = residue_ranking_metrics(dense[i], sites[i], valid[i])
            if metrics is not None:
                rows.append({"name": name, **metrics, "scores": dense[i][valid[i]].tolist()})
    logits = torch.cat(logits_all).numpy()
    labels = torch.cat(labels_all)
    if cfg.tasks.problem_type == "multi_label":
        summary: dict[str, Any] = fmax(labels.view(logits.shape).numpy(), logits)
    else:
        summary = multiclass_metrics(labels.view(-1).numpy(), logits, num_classes)
    if rows:
        summary["localization"] = {"score": score, **median_metrics(rows)}
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    with (out_dir / "per_protein.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    logger.success("{}", json.dumps({k: v for k, v in summary.items()}, default=_jsonable))
    return summary


def _jsonable(x: Any) -> Any:
    return x.tolist() if isinstance(x, np.ndarray) else str(x)


@torch.no_grad()
def evaluate_mcsa(
    model: BioBlobs,
    structures: list[dict],
    ids: set[str] | None,
    score: str,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Catalytic-residue localization on M-CSA enzymes, one protein per forward pass.

    Every residue of a chain is ranked (no validity mask), and the summary is the median
    over enzymes of AUROC, AUPRC and F1@L.

    Args:
        model: Trained model in eval mode.
        structures: Output of ``bioblobs.data.mcsa.load_mcsa``.
        ids: Enzymes to evaluate (for example the 74 ids of ``data/mcsa/evaluation_ids.txt``);
            all when None.
        score: ``"max_membership"`` or ``"weighted_membership"``.
        out_dir: Output directory for ``metrics.json`` and ``per_protein.jsonl``.

    Returns:
        The summary metrics.
    """
    from bioblobs.data.mcsa import to_graph

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    rows = []
    for structure in structures:
        if ids is not None and structure["name"] not in ids:
            continue
        graph = to_graph(structure)
        _, dense = _model_scores(model, Batch.from_data_list([graph]).to(device), score)
        metrics = residue_ranking_metrics(dense[0], graph.residue_target)
        if metrics is not None:
            rows.append({"name": structure["name"], **metrics, "scores": dense[0].tolist()})
    summary = {"score": score, **median_metrics(rows)}
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    with (out_dir / "per_protein.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    logger.success("M-CSA {}", json.dumps(summary))
    return summary

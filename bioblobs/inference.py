"""Predict functions, blobs and residue scores for user-provided structures.

Example:
    >>> model, cfg, classes = load_pretrained("bioblobs-mcsa-rvq-k8")
    >>> graphs = featurize_pdbs(["1abc.pdb"], cfg, device="cuda")
    >>> results = predict(model, graphs, classes)
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Batch, Data

from bioblobs.data.structure import parse_backbone, to_graph
from bioblobs.localization.scores import max_membership, weighted_membership
from bioblobs.model import BioBlobs, build_model


def load_pretrained(path: str | Path, device: str = "cpu") -> tuple[BioBlobs, DictConfig, list]:
    """Load a released model directory or a training checkpoint.

    Args:
        path: Directory with ``config.yaml``, ``model.safetensors`` and optionally
            ``classes.json``, or a ``.ckpt`` file written by ``train_bioblobs.py``.
        device: Torch device.

    Returns:
        The model in eval mode, its config and the class names (indices if unknown).
    """
    path = Path(path)
    if path.is_dir():
        from safetensors.torch import load_file

        cfg = OmegaConf.load(path / "config.yaml")
        model = build_model(cfg, int(cfg.num_classes))
        model.load_state_dict(load_file(path / "model.safetensors"))
        names_file = path / "classes.json"
        classes = json.loads(names_file.read_text()) if names_file.exists() else None
    else:
        from bioblobs.training.trainer import load_module

        module = load_module(path)
        model, cfg, classes = module.model, module.cfg, None
        cfg.num_classes = module.num_classes
    if classes is None:
        classes = list(range(int(cfg.num_classes)))
    elif isinstance(classes, dict):
        classes = [name for name, _ in sorted(classes.items(), key=lambda item: item[1])]
    return model.to(device).eval(), cfg, classes


def featurize_pdbs(
    pdb_paths: Sequence[str | Path],
    cfg: DictConfig,
    device: str = "cpu",
    chains: Sequence[str | None] | None = None,
    foldseek_bin: str | None = None,
) -> list[Data]:
    """Parse PDB files and embed them with the model's protein encoder.

    Args:
        pdb_paths: Structures to featurize.
        cfg: Model config (``encoders`` group).
        device: Device for the protein language model.
        chains: Optional chain per structure (all chains when None).
        foldseek_bin: Foldseek binary (SaProt only).

    Returns:
        Graph samples named by file stem (and chain).
    """
    from bioblobs.modules.plm import embed_esm2, embed_saprot

    chains = list(chains) if chains is not None else [None] * len(pdb_paths)
    parsed = [parse_backbone(p, c) for p, c in zip(pdb_paths, chains, strict=True)]
    sequences = [seq for seq, _, _ in parsed]
    if cfg.encoders.name == "saprot_static":
        embeddings = embed_saprot(
            sequences,
            pdb_paths,
            chains=chains,
            foldseek_bin=foldseek_bin,
            model_name=cfg.encoders.model_name,
            device=device,
            local_files_only=False,
        )
    else:
        embeddings = embed_esm2(
            sequences, model_name=cfg.encoders.model_name, device=device, local_files_only=False
        )
    graphs = []
    for path, chain, (seq, coords, resnums), emb in zip(
        pdb_paths, chains, parsed, embeddings, strict=True
    ):
        name = Path(path).name.split(".")[0] + (f"_{chain}" if chain else "")
        # Training and evaluation read float16 caches; round the same way here.
        structure = {"name": name, "seq": seq, "coords": coords, "resnum": resnums}
        graph = to_graph(structure, emb.half().float())
        graph.resnum = torch.tensor(resnums)
        graphs.append(graph)
    return graphs


@torch.no_grad()
def predict(
    model: BioBlobs,
    graphs: Sequence[Data],
    classes: Sequence,
    top_k: int = 5,
    batch_size: int = 16,
    membership_threshold: float = 0.5,
    multi_label: bool = False,
) -> list[dict[str, Any]]:
    """Run the model and collect predictions, blobs and residue scores per protein.

    Args:
        model: Model in eval mode.
        graphs: Featurized proteins.
        classes: Class name per output index.
        top_k: Number of top classes to report.
        batch_size: Proteins per forward pass.
        membership_threshold: Membership at which a residue is listed as part of a blob.
        multi_label: Report sigmoid instead of softmax probabilities.

    Returns:
        One dict per protein with ``top_classes``, ``blobs`` (ranked by importance score, each
        with its seed and member residue numbers) and per-residue ``max_membership`` and
        ``weighted_membership`` scores.
    """
    device = next(model.parameters()).device
    results = []
    for start in range(0, len(graphs), batch_size):
        chunk = list(graphs[start : start + batch_size])
        out = model(Batch.from_data_list(chunk).to(device))
        probs = out.logits.sigmoid() if multi_label else out.logits.softmax(-1)
        part = out.partition
        for i, graph in enumerate(chunk):
            n = graph.num_nodes
            resnums = graph.resnum.tolist()
            top = probs[i].topk(min(top_k, probs.shape[-1]))
            record: dict[str, Any] = {
                "name": graph.name,
                "num_residues": n,
                "residue_numbers": resnums,
                "top_classes": [
                    {"class": classes[int(c)], "probability": float(p)}
                    for p, c in zip(top.values, top.indices, strict=True)
                ],
            }
            if part is not None and out.importance is not None:
                record.update(
                    _blob_record(part, out.importance, i, n, resnums, membership_threshold)
                )
            results.append(record)
    return results


def _blob_record(part, importance, i: int, n: int, resnums: list, threshold: float) -> dict:
    memberships = part.memberships[i, :, :n]
    valid = part.blob_mask[i]
    order = torch.argsort(importance[i].masked_fill(~valid, -1.0), descending=True, stable=True)
    blobs = []
    for rank, k in enumerate(order[: int(valid.sum())].tolist(), start=1):
        members = (memberships[k] >= threshold).nonzero(as_tuple=True)[0].tolist()
        seed = int(part.seed_indices[i, k]) if part.seed_indices is not None else None
        blobs.append(
            {
                "rank": rank,
                "blob": k,
                "importance": float(importance[i, k]),
                "seed_residue": resnums[seed] if seed is not None and seed >= 0 else None,
                "residues": [resnums[j] for j in members],
            }
        )
    return {
        "blobs": blobs,
        "max_membership": max_membership(part.memberships, part.blob_mask)[i, :n].tolist(),
        "weighted_membership": weighted_membership(part.memberships, importance)[i, :n].tolist(),
    }


def write_results(results: Sequence[dict[str, Any]], out_dir: str | Path) -> None:
    """Write ``predictions.csv`` and one ``proteins/<name>.json`` per protein."""
    out_dir = Path(out_dir)
    (out_dir / "proteins").mkdir(parents=True, exist_ok=True)
    with (out_dir / "predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "num_residues", "top_class", "probability"])
        for r in results:
            best = r["top_classes"][0]
            writer.writerow(
                [r["name"], r["num_residues"], best["class"], f"{best['probability']:.4f}"]
            )
            (out_dir / "proteins" / f"{r['name']}.json").write_text(json.dumps(r, indent=1))
    logger.success("Wrote {} predictions to {}", len(results), out_dir)

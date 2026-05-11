#!/usr/bin/env python
"""Inference for a trained BioBlobs model on user-provided PDB files.

Loads a checkpoint produced by ``train_bioblobs.py``, parses each PDB in
process to extract backbone coords + sequence, embeds the sequence with ESM2
on the same GPU as the model, then runs the BioBlobs forward pass with the
partitioner emitting interpretability payload.

The script is fully self-contained: no prebuilt caches required. The user
brings PDB files; everything else (parsing, embedding, forward) happens
in-process. The HuggingFace ESM2 weights are downloaded automatically the
first time and cached at ``~/.cache/huggingface/`` (~600 MB, one-time).

Inputs (what the user provides):
    --checkpoint  best.ckpt (or last.ckpt) from train_bioblobs.py
    --config      config.json (default: alongside the checkpoint)
    --pdb         one or more .pdb / .pdb.gz file paths   OR
    --pdb-dir     a directory of PDBs (globs *.pdb / *.pdb.gz)

The EC-L3 string ↔ index mapping (``token_map.json``) is read from
``--data-dir`` and is needed to render ``pred_ec_l3`` strings. Ship it
alongside the checkpoint when distributing the model.

Always-on outputs:
    <out-dir>/predictions.csv          id, pred_label, pred_ec_l3, top1_prob (true_label=-1)
    <out-dir>/inference.pt             list[dict] per protein, full per-blob structure
    <out-dir>/inference_summary.json   {num_proteins, elapsed_seconds}

With ``--interpret``:
    <out-dir>/manifest.json            top-level index of per-protein artifacts
    <out-dir>/proteins/<name>.json     posthoc-interpretability schema (rank, attention,
                                       residue_numbers, instance probabilities, …)
    <out-dir>/proteins/<name>.pdb      decompressed PDB copy referenced by JSON/HTML

With ``--save-html`` (implies ``--interpret``):
    <out-dir>/proteins/<name>.html     standalone 3Dmol.js viewer (open in any browser)

Per-protein dict in inference.pt::

    {
        "name":             str  (PDB filename stem),
        "true_label":       int  (always -1 — PDBs carry no labels),
        "true_ec_l3":       None,
        "pred_label":       int,
        "pred_ec_l3":       str,
        "logits":           Tensor [num_classes],
        "top_k_probs":      Tensor [k],   softmax top-k
        "top_k_classes":    Tensor [k],
        "seed_indices":     Tensor [K],   residue idx per blob
        "seed_scores":      Tensor [K],
        "blob_assignment":  Tensor [N]    hard argmax over K (per residue)
        "blob_assignment_soft": Tensor [N, K] soft probs (only if --save-soft)
        "blob_embeddings":  Tensor [K, D] float16 (only if --save-blob-embeddings)
        "blob_valid":       Tensor [K] bool (only if --save-blob-embeddings)
    }

Usage::

    # Single PDB
    .venv/bin/python inference.py \\
        --checkpoint outputs/bioblobs/single_<TS>/best.ckpt \\
        --out-dir    outputs/bioblobs/single_<TS>/inference \\
        --pdb my_protein.pdb

    # Directory of PDBs + interpretability JSON + 3D HTML viewer
    .venv/bin/python inference.py \\
        --checkpoint outputs/bioblobs/single_<TS>/best.ckpt \\
        --out-dir    outputs/bioblobs/single_<TS>/inference \\
        --pdb-dir    /path/to/pdbs/ \\
        --interpret --save-html

Cost: ~1 s/protein on H200 (PDB parse + ESM2-150M forward + BioBlobs forward).
ESM2 occupies ~1.5 GB of additional VRAM beyond the BioBlobs model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running from anywhere.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm

from bioblobs.datasets.featurizers import (
    ESM2StaticGraphFeaturizer,
    LETTER_TO_NUM,
    ProteinData,
)
from bioblobs.datasets.task_dataset import TaskDataset
from bioblobs_framework import BioBlobsFramework


# ---------------------------------------------------------------------------
# CLI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True, type=Path,
                   help="Path to .ckpt file from train_bioblobs.py.")
    p.add_argument("--config", type=Path, default=None,
                   help="Path to config.json (default: alongside --checkpoint).")
    p.add_argument("--data-dir", type=Path, default=Path("data/ec_proteinshake"))
    p.add_argument("--structures-pt", type=Path, default=None,
                   help="Default: <data-dir>/cache/structures.pt")
    p.add_argument("--mmap-anchor-dataset-name", default="bioblobs")
    p.add_argument("--input-csv", type=Path, default=None,
                   help="Defaults to <data-dir>/train_split_val.csv (the mmseqs2 "
                        "30%%-id holdout). Falls back to <data-dir>/train_split.csv "
                        "if the val split is missing. Provides labels if present.")
    p.add_argument("--ids", default=None,
                   help="Comma-separated UniProt ids to score (subset of input-csv).")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--save-soft", action="store_true",
                   help="Also store the soft [N,K] assignment matrix per protein.")
    p.add_argument("--save-blob-embeddings", action="store_true",
                   help="Also store per-protein blob embeddings (float16 [K, D]) "
                        "plus a [K] bool mask of which blob slots are valid. "
                        "Read from the partitioner's batch_data.blob_features "
                        "after the forward pass — these are the membership-"
                        "weighted means of per-residue features that the MIL "
                        "decoder consumes. ~K*D*2 bytes per protein.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap number of proteins (debug).")

    # --- Interpretability outputs (per-protein JSON + optional 3D HTML) ---
    p.add_argument("--interpret", action="store_true",
                   help="Write a rich per-protein JSON under <out-dir>/proteins/ "
                        "(same schema as bioblobs/analysis/posthoc_blob_interpretability) "
                        "+ a top-level manifest.json. Also flips the decoder's "
                        "return_attention / return_instance_predictions on at load time.")
    p.add_argument("--save-html", action="store_true",
                   help="In addition to JSON, render a standalone HTML 3D viewer "
                        "(3Dmol.js) per protein. Requires --interpret. The PDB "
                        "is decompressed and co-located alongside the HTML.")

    # --- Mode B: ad-hoc PDB inference (user-provided structures, no caches) ---
    p.add_argument("--pdb", nargs="+", default=None, type=Path,
                   help="One or more PDB file paths (.pdb or .pdb.gz). Triggers "
                        "ad-hoc mode: each PDB is parsed + ESM2-embedded in-process. "
                        "No label is attached (true_label=-1).")
    p.add_argument("--pdb-dir", type=Path, default=None,
                   help="Directory of PDBs; globs *.pdb and *.pdb.gz. Combined "
                        "with --pdb if both are given.")
    p.add_argument("--min-completion", type=float, default=0.95,
                   help="Backbone-completion threshold for ad-hoc PDBs. Same "
                        "default as the unified builder.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data assembly (mirrors training, minus the LengthBucketBatchSampler)


def load_inference_structures(args: argparse.Namespace) -> tuple[list[dict], dict[str, int]]:
    """Load structures, attach phony esm2_cache_path, optionally restrict to --ids."""
    data_dir = args.data_dir.resolve()
    structures_pt = args.structures_pt or data_dir / "cache" / "structures.pt"
    if args.input_csv is not None:
        input_csv = args.input_csv
    elif (data_dir / "train_split_val.csv").exists():
        input_csv = data_dir / "train_split_val.csv"
    else:
        input_csv = data_dir / "train_split.csv"
    token_map_json = data_dir / "token_map.json"

    if not structures_pt.exists():
        sys.exit(f"ERROR: structure cache not found at {structures_pt}")
    if not input_csv.exists():
        sys.exit(f"ERROR: input CSV not found at {input_csv}")
    if not token_map_json.exists():
        sys.exit(f"ERROR: {token_map_json} not found")

    logger.info("Loading structure cache: {}", structures_pt)
    structures = torch.load(structures_pt, weights_only=False, mmap=True)
    name_to_struct = {s["name"]: s for s in structures}
    logger.info("  {:,} structures available", len(name_to_struct))

    logger.info("Reading {}", input_csv)
    df = pd.read_csv(input_csv)
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        df = df[df["id"].astype(str).isin(wanted)]
        logger.info("  filtered to {} proteins via --ids", len(df))

    if args.limit is not None:
        df = df.head(args.limit)
        logger.info("  limited to {} proteins via --limit", len(df))

    with token_map_json.open() as fh:
        token_map = json.load(fh)
    inv_token_map = {v: k for k, v in token_map.items()}

    data_root = data_dir.parent.resolve()
    encoder_cache_dir = (
        data_root / args.mmap_anchor_dataset_name
        / "esm2_static_cache" / "anchor" / "embeddings"
    )

    structures_for_inference: list[dict] = []
    missing_struct = 0
    for _, row in df.iterrows():
        name = str(row["id"])
        if name not in name_to_struct:
            missing_struct += 1
            continue
        s = dict(name_to_struct[name])  # shallow copy off mmap
        # Label may be missing for ad-hoc inference inputs; default to -1.
        if "label" in df.columns and not pd.isna(row.get("label")):
            s["label"] = int(row["label"])
        else:
            s["label"] = -1
        if "ec_l3" in df.columns and not pd.isna(row.get("ec_l3")):
            s["true_ec_l3"] = str(row["ec_l3"])
        else:
            s["true_ec_l3"] = None
        s["esm2_cache_path"] = str(encoder_cache_dir / f"{name}.pt")
        # PDB path for interpretability HTML rendering — resolved against the
        # data root if relative (the the dataset builder CSV writes
        # paths like 'alphafold_pdbs/AF-<id>-F1-model_v6.pdb.gz').
        if "pdb_path" in df.columns and not pd.isna(row.get("pdb_path")):
            s["pdb_path"] = str(data_root / str(row["pdb_path"]))
        structures_for_inference.append(s)

    logger.info(
        "{:,} proteins ready for inference ({:,} skipped — missing in structure cache)",
        len(structures_for_inference), missing_struct,
    )
    if not structures_for_inference:
        sys.exit("ERROR: no proteins to score")
    return structures_for_inference, inv_token_map


# ---------------------------------------------------------------------------
# Model load


def load_model(args: argparse.Namespace) -> tuple[torch.nn.Module, "OmegaConf"]:
    cfg_path = args.config or args.checkpoint.parent / "config.json"
    if not cfg_path.exists():
        sys.exit(f"ERROR: config not found at {cfg_path}; pass --config explicitly.")
    cfg = OmegaConf.load(cfg_path)
    num_classes = int(cfg.tasks.get("num_classes", 0))
    if num_classes <= 0:
        sys.exit(f"ERROR: cfg.tasks.num_classes is invalid ({num_classes})")

    logger.info("Loading checkpoint: {}", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)  # support raw state_dict too

    model = BioBlobsFramework(cfg, num_classes)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("missing keys: {} (first 5)", missing[:5])
    if unexpected:
        logger.warning("unexpected keys: {} (first 5)", unexpected[:5])
    model.eval()

    # Turn on interpretability emission so the partitioner attaches blob payload.
    partitioner = getattr(model, "partitioner", None)
    if partitioner is None:
        sys.exit("ERROR: framework has no .partitioner — cannot extract blobs")
    if hasattr(partitioner, "emit_interpretability"):
        partitioner.emit_interpretability = True

    # When --interpret is set, also flip the MIL decoder flags so the forward
    # produces attention_weights, blob_mask, instance_logits/probabilities in extra.
    if getattr(args, "interpret", False):
        decoder = getattr(model, "decoder", None)
        if decoder is None:
            sys.exit("ERROR: framework has no .decoder — cannot enable --interpret")
        decoder.return_attention = True
        decoder.return_instance_predictions = True

    return model, cfg


# ---------------------------------------------------------------------------
# Inference loop


def collate_to_batch(data_list: list) -> Batch:
    return Batch.from_data_list(data_list)


def extract_per_graph(batch: Batch) -> list[dict]:
    """Pull partitioner_interpretability for each graph in the batch.

    Returns a list of dicts (one per graph) with keys:
        valid_node_local_indices: LongTensor [N_valid]
        assignment_matrix:        Tensor [N_valid, K]   (soft)
        seed_indices:             LongTensor [K] or None
        seed_scores:              Tensor [K] or None
    """
    payload = getattr(batch, "partitioner_interpretability", None)
    if not payload:
        return []
    valid_node = payload.get("valid_node_local_indices_per_graph", [])
    assign = payload.get("assignment_matrices_per_graph", [])
    seed_idx = payload.get("seed_indices_per_graph", None)
    seed_sc = payload.get("seed_scores_per_graph", None)

    out = []
    for g in range(len(valid_node)):
        item = {
            "valid_node_local_indices": valid_node[g].detach().cpu()
            if torch.is_tensor(valid_node[g]) else torch.tensor(valid_node[g]),
            "assignment_matrix": assign[g].detach().cpu()
            if torch.is_tensor(assign[g]) else torch.tensor(assign[g]),
        }
        if seed_idx is not None:
            item["seed_indices"] = seed_idx[g].detach().cpu()
        if seed_sc is not None:
            item["seed_scores"] = seed_sc[g].detach().cpu()
        out.append(item)
    return out


def extract_blob_embeddings(batch: Batch) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Pull per-graph blob embeddings out of a forwarded batch.

    Returns a list aligned with batch order, each entry ``(features, valid)``:
        features: [K, D] float16 — membership-weighted mean residue feature
                  per blob slot, computed by the partitioner forward pass and
                  stashed on ``batch_data.blob_features`` (always present).
        valid:    [K] bool — True where the blob slot has a real seed (i.e.
                  ``blob_seed_valid_per_graph[g]``); False for masked-out slots.

    Empty list if the partitioner didn't run (shouldn't happen here).
    """
    feats = getattr(batch, "blob_features", None)
    blob_batch = getattr(batch, "blob_batch", None)
    valid_per_graph = getattr(batch, "blob_seed_valid_per_graph", None)
    if feats is None or blob_batch is None or valid_per_graph is None:
        return []
    B = int(valid_per_graph.shape[0])
    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for g in range(B):
        f = feats[blob_batch == g].detach().cpu().to(torch.float16)
        v = valid_per_graph[g].detach().cpu().bool()
        out.append((f, v))
    return out


def hard_blob_assignment(num_residues: int, valid_idx: torch.Tensor,
                         soft: torch.Tensor) -> torch.Tensor:
    """Build a [N] long tensor: blob index per residue, -1 for invalid."""
    out = torch.full((num_residues,), -1, dtype=torch.long)
    if valid_idx.numel() > 0 and soft.numel() > 0:
        hard = soft.argmax(dim=-1)  # [N_valid]
        out[valid_idx] = hard
    return out


# ---------------------------------------------------------------------------
# Interpretability outputs (per-protein JSON + optional 3D HTML)
# ---------------------------------------------------------------------------


def _write_protein_pdb_copy(src: Path, dst: Path) -> None:
    """Copy a PDB to ``dst``, decompressing on the fly if src ends with .gz."""
    import gzip
    import shutil
    dst.parent.mkdir(parents=True, exist_ok=True)
    if str(src).endswith(".gz"):
        with gzip.open(src, "rb") as fin, dst.open("wb") as fout:
            shutil.copyfileobj(fin, fout)
    else:
        shutil.copy(src, dst)


def _per_protein_extra_slice(extra: dict, i: int) -> dict:
    """Slice the batched ``extra`` dict at batch position ``i``.

    Expected keys when --interpret is on (from MILDecoder + BioBlobs partitioner):
        attention_weights:        [B, K]
        blob_mask:                [B, K]  (bool)
        instance_logits:          [B, K, num_classes]
        instance_probabilities:   [B, K, num_classes]
    Returns CPU tensors with the leading B dim dropped.
    """
    out = {}
    for key in ("attention_weights", "blob_mask",
                "instance_logits", "instance_probabilities"):
        if key in extra and extra[key] is not None:
            v = extra[key]
            out[key] = v[i].detach().cpu() if torch.is_tensor(v) else v[i]
    return out


def _write_interpretability_artifacts(
    *,
    out_dir: Path,
    protein_id: str,
    pdb_src: Path | None,           # source PDB (gz allowed); skipped if None
    blob_payload: dict,             # one element from extract_per_graph
    extra_per_protein: dict,        # _per_protein_extra_slice output
    bag_logits: torch.Tensor,       # [num_classes]
    bag_probabilities: torch.Tensor,
    pred_label_index: int,
    true_label_index: int | None,
    label_decoder: dict[int, str],
    residue_numbers: list,
    checkpoint_path: str,
    config_path: str,
    input_mode: str,
    save_html: bool,
) -> dict:
    """Build + write the per-protein JSON; optionally write HTML + PDB copy.

    Returns one manifest row pointing at the artifacts that landed.
    """
    from bioblobs.analysis.posthoc_blob_interpretability import build_protein_export

    proteins_dir = out_dir / "proteins"
    proteins_dir.mkdir(parents=True, exist_ok=True)

    # PDB copy/decompress so blob_viewer can read it as plain text.
    pdb_dst = proteins_dir / f"{protein_id}.pdb"
    if pdb_src is not None and pdb_src.exists():
        _write_protein_pdb_copy(pdb_src, pdb_dst)
        pdb_path_for_export = str(pdb_dst.resolve())
    else:
        pdb_path_for_export = ""

    valid_node_local_indices = blob_payload["valid_node_local_indices"]
    assignment_matrix = blob_payload["assignment_matrix"]

    # Required by build_protein_export — supply zeros if the decoder didn't expose
    # them (i.e., user passed --interpret but the model was trained without these
    # decoder flags. We turned them on at load time so this should not trigger.)
    K = int(assignment_matrix.shape[1]) if assignment_matrix.numel() else 0
    num_classes = int(bag_logits.numel())
    attention_weights = extra_per_protein.get(
        "attention_weights", torch.ones(K) / max(K, 1),
    )
    blob_mask = extra_per_protein.get(
        "blob_mask", torch.ones(K, dtype=torch.bool),
    ).bool()
    instance_logits = extra_per_protein.get(
        "instance_logits", torch.zeros(K, num_classes),
    )
    instance_probabilities = extra_per_protein.get(
        "instance_probabilities", torch.zeros(K, num_classes),
    )

    export = build_protein_export(
        protein_id=protein_id,
        input_mode=input_mode,
        split="inference",
        pdb_path=pdb_path_for_export,
        residue_numbers=list(residue_numbers),
        valid_node_local_indices=valid_node_local_indices,
        assignment_matrix=assignment_matrix,
        assignment_type="soft",
        attention_weights=attention_weights,
        blob_mask=blob_mask,
        instance_logits=instance_logits,
        instance_probabilities=instance_probabilities,
        bag_logits=bag_logits,
        bag_probabilities=bag_probabilities,
        pred_label_index=pred_label_index,
        true_label_index=true_label_index,
        label_decoder=label_decoder,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
    )

    json_path = proteins_dir / f"{protein_id}.json"
    with json_path.open("w") as fh:
        json.dump(export, fh, indent=2, sort_keys=True)

    html_path: Path | None = None
    if save_html and pdb_path_for_export:
        from bioblobs.analysis.blob_viewer import render_blob_html
        html_text = render_blob_html(str(json_path))
        html_path = proteins_dir / f"{protein_id}.html"
        html_path.write_text(html_text)

    row = {
        "name": protein_id,
        "pred_label": pred_label_index,
        "pred_ec_l3": export["prediction"]["pred_label_text"],
        "true_label": true_label_index if true_label_index is not None else -1,
        "true_ec_l3": export["prediction"]["true_label_text"],
        "confidence": export["prediction"]["confidence"],
        "json_path": str(json_path.relative_to(out_dir)),
        "pdb_path": str(pdb_dst.relative_to(out_dir)) if pdb_path_for_export else None,
        "html_path": str(html_path.relative_to(out_dir)) if html_path else None,
    }
    return row


# ---------------------------------------------------------------------------
# Mode B: ad-hoc PDB inference
# ---------------------------------------------------------------------------


def _collect_pdb_paths(args: argparse.Namespace) -> list[Path]:
    """Resolve --pdb + --pdb-dir into a deduped, ordered list of PDB paths."""
    paths: list[Path] = []
    seen: set[Path] = set()
    if args.pdb:
        for p in args.pdb:
            p = p.resolve()
            if p not in seen:
                seen.add(p); paths.append(p)
    if args.pdb_dir:
        d = args.pdb_dir.resolve()
        if not d.is_dir():
            sys.exit(f"ERROR: --pdb-dir {d} is not a directory")
        for pat in ("*.pdb", "*.pdb.gz"):
            for p in sorted(d.glob(pat)):
                if p not in seen:
                    seen.add(p); paths.append(p)
    if not paths:
        sys.exit("ERROR: --pdb / --pdb-dir matched no files")
    return paths


def _build_data_from_struct(struct: dict, embedding: torch.Tensor) -> ProteinData:
    """Construct a single ProteinData object from a parsed PDB + freshly-computed
    ESM2 embedding. Mirrors ``ESM2StaticGraphFeaturizer.featurize`` but bypasses
    the mmap and uses an empty edge_index (BioBlobs partitioner doesn't read edges).
    """
    coords = torch.as_tensor(struct["coords"], dtype=torch.float32)
    if coords.ndim != 3 or coords.shape[1:] != (4, 3):
        raise ValueError(f"bad coords shape for {struct['name']}: {tuple(coords.shape)}")
    mask = torch.isfinite(coords.sum(dim=(1, 2)))
    coords[~mask] = float("inf")
    x_ca = coords[:, 1]
    seq = torch.as_tensor(
        [LETTER_TO_NUM.get(a, 20) for a in struct["seq"]],
        dtype=torch.long,
    )
    if embedding.size(0) != seq.size(0):
        raise ValueError(
            f"embedding rows {embedding.size(0)} != seq length {seq.size(0)} "
            f"for {struct['name']}"
        )
    label = int(struct.get("label", -1))
    return ProteinData(
        x=x_ca,
        seq=seq,
        name=struct["name"],
        resnum=struct["resnum"],
        edge_index=torch.empty((2, 0), dtype=torch.long),
        mask=mask,
        node_features=embedding.float(),
        num_nodes=int(seq.size(0)),
        y=torch.tensor([label], dtype=torch.long),
    )


def _record_one(
    name: str, struct: dict,
    logits: torch.Tensor, top_p: torch.Tensor, top_c: torch.Tensor,
    inv_token_map: dict,
) -> tuple[dict, dict]:
    """Build one row for predictions.csv + one record for inference.pt.

    Pure label/logits work — no blob handling. The caller attaches blob fields
    by indexing ``extract_per_graph(batch)`` at the correct batch position.
    """
    pred_label = int(logits.argmax().item())
    pred_ec = inv_token_map.get(pred_label, str(pred_label))
    true_label = int(struct.get("label", -1))
    true_ec = struct.get("true_ec_l3")

    row = {
        "id": name,
        "true_label": true_label,
        "true_ec_l3": true_ec,
        "pred_label": pred_label,
        "pred_ec_l3": pred_ec,
        "top1_prob": float(top_p[0].item()),
    }
    rec = {
        "name": name,
        "true_label": true_label,
        "true_ec_l3": true_ec,
        "pred_label": pred_label,
        "pred_ec_l3": pred_ec,
        "logits": logits.detach().cpu(),
        "top_k_probs": top_p.detach().cpu(),
        "top_k_classes": top_c.detach().cpu(),
    }
    return row, rec


def _attach_blob_fields(
    rec: dict, blob_payload: dict, seq_len: int, save_soft: bool,
    blob_embedding: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> None:
    """Mutate ``rec`` to add seed_indices/scores + blob_assignment(_soft).

    When ``blob_embedding`` is provided (``--save-blob-embeddings``), also adds
    ``blob_embeddings`` ([K, D] float16) and ``blob_valid`` ([K] bool).
    """
    rec["seed_indices"] = blob_payload.get("seed_indices")
    rec["seed_scores"] = blob_payload.get("seed_scores")
    valid_idx = blob_payload["valid_node_local_indices"]
    soft = blob_payload["assignment_matrix"]
    rec["blob_assignment"] = hard_blob_assignment(seq_len, valid_idx, soft)
    if save_soft:
        K = soft.size(1) if soft.numel() else 0
        full = torch.zeros((seq_len, K), dtype=soft.dtype)
        if K > 0 and valid_idx.numel() > 0:
            full[valid_idx] = soft
        rec["blob_assignment_soft"] = full
    if blob_embedding is not None:
        feats, valid = blob_embedding
        rec["blob_embeddings"] = feats
        rec["blob_valid"] = valid


def infer_from_pdbs(
    args: argparse.Namespace, model, cfg, num_classes: int,
    inv_token_map: dict[int, str],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Mode B: parse user-provided PDBs, embed in-process, run model forward.

    Returns (rows, results, manifest_rows). ``manifest_rows`` is empty unless
    ``--interpret`` is set.
    """
    from bioblobs.datasets.parallel import _process_single_pdb
    from bioblobs.datasets.esm2_cache import (
        embed_sequence_with_esm2,
        sanitize_sequence,
        _load_esm2_model_and_tokenizer,
    )

    pdb_paths = _collect_pdb_paths(args)
    logger.info("ad-hoc PDB inference: {} files", len(pdb_paths))

    device = next(model.parameters()).device
    model_name = str(cfg.encoders.model_name)
    repr_layer = int(cfg.encoders.get("repr_layer", -1))
    window_size = int(cfg.encoders.get("window_size", 1022))
    window_overlap = int(cfg.encoders.get("window_overlap", 128))
    max_batch_tokens = int(cfg.encoders.get("max_batch_tokens", 8192))

    logger.info("loading ESM2 model ({}) for embedding…", model_name)
    tokenizer, esm_model = _load_esm2_model_and_tokenizer(model_name, device)
    esm_model.eval()

    cfg_path = str(args.config or args.checkpoint.parent / "config.json")
    rows: list[dict] = []
    results: list[dict] = []
    manifest: list[dict] = []
    skipped_low_completion = 0
    for pdb_path in tqdm(pdb_paths, desc="ad-hoc inference", unit="pdb"):
        # Use the filename (sans .pdb / .pdb.gz) as the protein name.
        name = pdb_path.name
        for ext in (".pdb.gz", ".pdb"):
            if name.endswith(ext):
                name = name[: -len(ext)]
                break

        struct, err = _process_single_pdb(
            (name, str(pdb_path), None, args.min_completion)
        )
        if struct is None:
            logger.warning("skipped {}: {}", name, err)
            skipped_low_completion += 1
            continue

        # Embed PDB-derived sequence with ESM2 (keep grad disabled).
        san_seq = sanitize_sequence(struct["seq"])
        with torch.no_grad():
            embedding = embed_sequence_with_esm2(
                sequence=san_seq, tokenizer=tokenizer, model=esm_model,
                repr_layer=repr_layer, device=device,
                window_size=window_size, window_overlap=window_overlap,
                max_batch_tokens=max_batch_tokens,
            )

        data = _build_data_from_struct(struct, embedding)
        batch = Batch.from_data_list([data]).to(device)
        with torch.no_grad():
            logits, extra = model(batch)
            probs = F.softmax(logits, dim=-1)
            top_k = min(args.top_k, num_classes)
            top_p, top_c = probs.topk(top_k, dim=-1)

        row, rec = _record_one(
            name=name, struct=struct,
            logits=logits[0], top_p=top_p[0], top_c=top_c[0],
            inv_token_map=inv_token_map,
        )
        blobs = extract_per_graph(batch)
        embs = extract_blob_embeddings(batch) if args.save_blob_embeddings else []
        if blobs:
            _attach_blob_fields(
                rec, blobs[0], len(struct["seq"]), args.save_soft,
                blob_embedding=(embs[0] if embs else None),
            )
        rows.append(row); results.append(rec)

        if args.interpret and blobs and isinstance(extra, dict):
            manifest.append(_write_interpretability_artifacts(
                out_dir=args.out_dir,
                protein_id=name,
                pdb_src=pdb_path,
                blob_payload=blobs[0],
                extra_per_protein=_per_protein_extra_slice(extra, 0),
                bag_logits=logits[0].detach().cpu(),
                bag_probabilities=probs[0].detach().cpu(),
                pred_label_index=int(probs[0].argmax().item()),
                true_label_index=None,
                label_decoder=inv_token_map,
                residue_numbers=list(struct["resnum"]) if not torch.is_tensor(struct["resnum"]) else struct["resnum"].tolist(),
                checkpoint_path=str(args.checkpoint),
                config_path=cfg_path,
                input_mode="pdb",
                save_html=args.save_html,
            ))

    if skipped_low_completion:
        logger.warning("skipped {:,} PDB(s) for low backbone completion",
                       skipped_low_completion)
    return rows, results, manifest


def _infer_from_cached_corpus(
    args: argparse.Namespace, model, cfg, num_classes: int,
    inv_token_map: dict[int, str],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Mode A: score proteins already in structures.pt + esm2_static_cache_mmap.

    Returns (rows, results, manifest_rows). ``manifest_rows`` is empty unless
    ``--interpret`` is set.
    """
    device = next(model.parameters()).device
    structures, _ = load_inference_structures(args)

    # Edges aren't consumed by the BioBlobs partitioner (it uses CA coords +
    # spatial seed_radius). edge_value=0 short-circuits to an empty edge_index.
    featurizer = ESM2StaticGraphFeaturizer(
        edge_method="knn", edge_value=0, device=torch.device("cpu"),
    )
    dataset = TaskDataset(structures, num_classes=num_classes, featurizer=featurizer)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_to_batch, num_workers=args.num_workers,
        pin_memory=True,
    )

    cfg_path = str(args.config or args.checkpoint.parent / "config.json")
    rows: list[dict] = []
    results: list[dict] = []
    manifest: list[dict] = []
    cursor = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="inference"):
            batch = batch.to(device)
            logits, extra = model(batch)
            probs = F.softmax(logits, dim=-1)
            top_k = min(args.top_k, num_classes)
            top_p, top_c = probs.topk(top_k, dim=-1)
            blobs = extract_per_graph(batch)
            embs = extract_blob_embeddings(batch) if args.save_blob_embeddings else []

            B = logits.size(0)
            for i in range(B):
                meta = structures[cursor + i]
                row, rec = _record_one(
                    name=meta["name"], struct=meta,
                    logits=logits[i], top_p=top_p[i], top_c=top_c[i],
                    inv_token_map=inv_token_map,
                )
                if i < len(blobs):
                    _attach_blob_fields(
                        rec, blobs[i], len(meta["seq"]), args.save_soft,
                        blob_embedding=(embs[i] if i < len(embs) else None),
                    )
                rows.append(row); results.append(rec)

                if args.interpret and i < len(blobs) and isinstance(extra, dict):
                    pdb_src = Path(meta["pdb_path"]) if meta.get("pdb_path") else None
                    manifest.append(_write_interpretability_artifacts(
                        out_dir=args.out_dir,
                        protein_id=meta["name"],
                        pdb_src=pdb_src,
                        blob_payload=blobs[i],
                        extra_per_protein=_per_protein_extra_slice(extra, i),
                        bag_logits=logits[i].detach().cpu(),
                        bag_probabilities=probs[i].detach().cpu(),
                        pred_label_index=int(probs[i].argmax().item()),
                        true_label_index=int(meta["label"]) if int(meta.get("label", -1)) >= 0 else None,
                        label_decoder=inv_token_map,
                        residue_numbers=list(meta["resnum"]) if not torch.is_tensor(meta["resnum"]) else meta["resnum"].tolist(),
                        checkpoint_path=str(args.checkpoint),
                        config_path=cfg_path,
                        input_mode="cached",
                        save_html=args.save_html,
                    ))
            cursor += B
    return rows, results, manifest


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model, cfg = load_model(args)
    model.to(device)
    num_classes = int(cfg.tasks.num_classes)

    # token_map for inverse-lookup is read once for either mode.
    token_map_json = args.data_dir.resolve() / "token_map.json"
    if not token_map_json.exists():
        sys.exit(f"ERROR: {token_map_json} not found (need it for ec_l3 string lookup)")
    with token_map_json.open() as fh:
        inv_token_map = {v: k for k, v in json.load(fh).items()}

    if args.save_html and not args.interpret:
        logger.warning("--save-html implies --interpret; enabling --interpret")
        args.interpret = True

    ad_hoc = args.pdb is not None or args.pdb_dir is not None
    t0 = time.time()
    if ad_hoc:
        logger.info("mode: ad-hoc PDB{}",
                    " (+interpret)" if args.interpret else "")
        rows, results, manifest_rows = infer_from_pdbs(
            args, model, cfg, num_classes, inv_token_map,
        )
    else:
        logger.info("mode: cached corpus{}",
                    " (+interpret)" if args.interpret else "")
        rows, results, manifest_rows = _infer_from_cached_corpus(
            args, model, cfg, num_classes, inv_token_map,
        )

    if args.interpret and manifest_rows:
        manifest_path = args.out_dir / "manifest.json"
        with manifest_path.open("w") as fh:
            json.dump({
                "n_proteins": len(manifest_rows),
                "checkpoint": str(args.checkpoint),
                "config": str(args.config or args.checkpoint.parent / "config.json"),
                "rows": manifest_rows,
            }, fh, indent=2, sort_keys=True)
        logger.success("wrote {} ({:,} rows, JSONs in proteins/)",
                       manifest_path, len(manifest_rows))

    elapsed = time.time() - t0
    logger.info("inference done in {:.1f}s ({:.1f} prot/s)",
                elapsed, len(results) / max(elapsed, 1e-6))

    pred_df = pd.DataFrame(rows)
    pred_csv = args.out_dir / "predictions.csv"
    pred_df.to_csv(pred_csv, index=False)
    logger.success("wrote {} ({:,} rows)", pred_csv, len(pred_df))

    inf_pt = args.out_dir / "inference.pt"
    torch.save(results, inf_pt)
    size_gib = inf_pt.stat().st_size / (1024 ** 3)
    logger.success("wrote {} ({:.2f} GiB)", inf_pt, size_gib)

    summary = {
        "num_proteins": len(results),
        "elapsed_seconds": elapsed,
        "blob_embeddings_saved": bool(args.save_blob_embeddings),
    }
    have_labels = pred_df["true_label"].ge(0).any()
    if have_labels:
        labeled = pred_df[pred_df["true_label"] >= 0]
        summary["num_labeled"] = int(len(labeled))
        summary["accuracy"] = float(
            accuracy_score(labeled["true_label"], labeled["pred_label"])
        )
        summary["macro_f1"] = float(
            f1_score(
                labeled["true_label"], labeled["pred_label"],
                average="macro", labels=list(range(num_classes)), zero_division=0,
            )
        )
        logger.info(
            "labeled metrics: accuracy={:.3f} macro_f1={:.3f} (n={:,})",
            summary["accuracy"], summary["macro_f1"], summary["num_labeled"],
        )

    summary_path = args.out_dir / "inference_summary.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    logger.success("wrote {}", summary_path)


if __name__ == "__main__":
    main()

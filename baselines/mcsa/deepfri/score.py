#!/usr/bin/env python
"""Grad-CAM residue saliency from a DeepFRI GCN enzyme-function model.

DeepFRI (Gligorijevic et al., 2021) combines sequence features with graph
convolutions over a residue contact graph for protein function prediction.
This wrapper reproduces the M-CSA localization baseline used in the
BioBlobs paper: Grad-CAM (Selvaraju et al., 2017) on the
"GCNN_concatenate" layer of the released EC-prediction GCN, with
``use_guided_grads=False``, and a C-alpha-distance contact map thresholded
at 10 angstrom.

Target-class selection, per protein:
    1. the exact annotated EC code, if present in the model's goterms
       vocabulary;
    2. otherwise, its 3-level parent ``"<a>.<b>.<c>.-"``, if present;
    3. otherwise (including proteins with no annotated EC), the model's
       own top-predicted class for that protein.

The released Grad-CAM implementation rescales its heatmap to [0, 1]
without rectifying it (no absolute value / ReLU is applied beyond what the
upstream ``GradCAM.heatmap`` already does). A protein is skipped if it has
no (or misaligned) C-alpha coordinates, or if its heatmap is constant or
non-finite.

Typical usage example:

    python score.py \\
        --inputs mcsa_inputs.jsonl \\
        --upstream-dir /path/to/DeepFRI \\
        --out scores_deepfri.json

``--upstream-dir`` is the cloned DeepFRI repository
(github.com/flatironinstitute/DeepFRI) with ``newest_trained_models.tar.gz``
(from the upstream README) extracted into it, so that
``<upstream-dir>/trained_models/model_config.json`` exists.

Must run in the separate DeepFRI environment (TensorFlow 2.3.1,
Python 3.7; see ../README.md). Output: a JSON object
``{chain_id: [score, ...]}`` with one float per residue of ``seq``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_records, write_scores  # noqa: E402

LOGGER = logging.getLogger("deepfri_score")

# Residue alphabet accepted by deepfrier.utils.seq2onehot; anything else is
# mapped to 'X', matching the original scoring script.
VOCAB = set([
    "-", "D", "G", "U", "L", "N", "T", "K", "H", "Y", "W", "C", "P",
    "V", "S", "O", "I", "E", "F", "X", "Q", "A", "B", "Z", "R", "M",
])


def sanitize(seq: str) -> str:
    """Replaces characters outside DeepFRI's residue vocabulary with 'X'."""
    return "".join(c if c in VOCAB else "X" for c in seq.upper())


def ca_distmat(ca: List[List[float]]) -> np.ndarray:
    """Computes the pairwise C-alpha Euclidean distance matrix.

    Args:
        ca: List of ``[x, y, z]`` coordinates, one per residue.

    Returns:
        An ``(L, L)`` array of pairwise distances.
    """
    x = np.asarray(ca, dtype=np.float64)
    d = x[:, None, :] - x[None, :, :]
    return np.sqrt((d * d).sum(-1))


def pick_class(ec: Optional[str], go_index: Dict[str, int], yhat: np.ndarray) -> Tuple[int, str]:
    """Chooses the target EC class index, reproducing the original back-off rule.

    Args:
        ec: Annotated EC number (e.g. "5.1.1.3"), or None if unknown.
        go_index: Mapping from EC string (as stored in the model's
            ``goterms``) to class index.
        yhat: The model's predicted score for every class, for this
            protein; used only as a fallback.

    Returns:
        A tuple ``(class_index, path)`` where ``path`` is one of "exact",
        "backoff" or "fallback".
    """
    if ec is not None and ec in go_index:
        return go_index[ec], "exact"
    if ec is not None:
        parts = ec.split(".")
        if len(parts) >= 3:
            three = ".".join(parts[:3] + ["-"])
            if three in go_index:
                return go_index[three], "backoff"
    return int(np.argmax(yhat)), "fallback"


def score_records(records: List[dict], upstream_dir: str) -> Dict[str, List[float]]:
    """Computes the DeepFRI Grad-CAM score for every input record.

    Args:
        records: Input records with "id", "seq", "ca" and "ec" fields (see
            ../common.py / ../README.md).
        upstream_dir: Path to the cloned (and weight-populated) DeepFRI
            repository.

    Returns:
        Mapping from chain id to a list of per-residue Grad-CAM scores.
    """
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # force CPU
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    cwd = os.getcwd()
    sys.path.insert(0, upstream_dir)
    os.chdir(upstream_dir)  # model_config.json paths are relative to the repo root
    try:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        from deepfrier.Predictor import GradCAM, Predictor
        from deepfrier.utils import seq2onehot

        cfg = json.load(open("trained_models/model_config.json"))
        predictor = Predictor(cfg["gcn"]["models"]["ec"], gcn=True)
        goterms = list(predictor.goterms)
        go_index = {g: i for i, g in enumerate(goterms)}
        gradcam = GradCAM(predictor.model, layer_name="GCNN_concatenate")
        LOGGER.info("loaded DeepFRI EC model: %d GO/EC terms", len(goterms))

        out: Dict[str, List[float]] = {}
        for rec in records:
            cid = rec["id"]
            seq = sanitize(rec["seq"])
            length = len(seq)
            ca = rec.get("ca")
            if ca is None or len(ca) != length:
                LOGGER.warning("%s: missing or misaligned Ca coordinates, skipping", cid)
                continue

            dmat = ca_distmat(ca)
            a = np.double(dmat < 10.0).reshape(1, length, length)
            s = seq2onehot(seq)
            s = s.reshape(1, *s.shape)
            inputs = [a.astype(np.float32), s.astype(np.float32)]

            yhat = predictor.model(inputs, training=False).numpy()[:, :, 0].reshape(-1)
            cls, _ = pick_class(rec.get("ec"), go_index, yhat)

            heat = gradcam.heatmap(inputs, cls, use_guided_grads=False)
            heat = np.asarray(heat, dtype=np.float64).reshape(-1)
            if heat.shape[0] != length:
                LOGGER.warning("%s: heatmap length %d != sequence length %d, skipping",
                                cid, heat.shape[0], length)
                continue
            if not np.isfinite(heat).all() or heat.std() <= 0:
                LOGGER.warning("%s: constant or non-finite heatmap, skipping", cid)
                continue
            out[cid] = [float(v) for v in heat]
        return out
    finally:
        os.chdir(cwd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", required=True, help="Common-format input file (see ../README.md).")
    parser.add_argument("--upstream-dir", required=True,
                         help="Cloned DeepFRI repo with trained_models/ extracted in place.")
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    records = load_records(args.inputs)
    LOGGER.info("loaded %d records from %s", len(records), args.inputs)
    scores = score_records(records, args.upstream_dir)
    LOGGER.info("scored %d/%d proteins", len(scores), len(records))
    write_scores(args.out, scores)
    LOGGER.info("wrote %s", args.out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Class-activation-mapping (CAM) residue scores from a ProteInfer EC model.

ProteInfer (Sanderson et al., 2023) is a dilated convolutional network for
protein function prediction. This wrapper reproduces the M-CSA
localization baseline used in the BioBlobs paper: the per-residue
"representation" activation, dotted with the output layer's weight column
for a target EC class (Zhou et al., 2016 class-activation mapping), with
the sign retained.

Target-class selection, per protein:
    1. the exact annotated EC code, if present in the model vocabulary
       ("EC:<a>.<b>.<c>.<d>");
    2. otherwise, the 3-level parent node "EC:<a>.<b>.<c>.-" if present in
       the vocabulary, else the first vocabulary entry under that 3-level
       prefix;
    3. otherwise (including proteins with no annotated EC), the model's
       own top-predicted class for that sequence.

A protein is skipped if no target class resolves, if the per-residue
activation does not align 1:1 with the input sequence, or if the resulting
score is constant or non-finite.

Typical usage example:

    python score.py \\
        --inputs mcsa_inputs.jsonl \\
        --upstream-dir /path/to/external_baselines \\
        --out scores_proteinfer.json

``--upstream-dir`` must contain the cloned ProteInfer repository at
``<upstream-dir>/proteinfer`` and the unzipped SavedModel at
``<upstream-dir>/models/<model-name>`` (see ../README.md for the exact
model name and download URL); both can be overridden with ``--repo-dir``
and ``--model-dir``.

Must run in the separate ProteInfer environment (TensorFlow 1.15,
Python 3.7; see ../README.md). Output: a JSON object
``{chain_id: [score, ...]}`` with one float per residue of ``seq``, in the
same order.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_records, write_scores  # noqa: E402

LOGGER = logging.getLogger("proteinfer_score")

DEFAULT_MODEL_NAME = (
    "noxpnd_cnn_swissprot_ec_random_swiss-cnn_for_swissprot_ec_random-13685140"
)


def _densify(x):
    """Converts a (possibly sparse) ProteInfer activation to a dense array."""
    return np.asarray(x.todense()) if hasattr(x, "todense") else np.asarray(x)


def pick_class(
    ec: Optional[str],
    vocab_index: Dict[str, int],
    vocab: List[str],
) -> Tuple[Optional[int], str]:
    """Chooses the target EC class index, reproducing the original back-off rule.

    Args:
        ec: Annotated EC number (e.g. "5.1.1.3"), or None if unknown.
        vocab_index: Mapping from vocabulary label (e.g. "EC:5.1.1.3") to
            class index.
        vocab: The model's label vocabulary, in index order.

    Returns:
        A tuple ``(class_index_or_None, path)`` where ``path`` is one of
        "exact", "backoff" or "fallback" (the last meaning the caller must
        resolve the class from the model's own top prediction).
    """
    if ec is None:
        return None, "fallback"
    full = "EC:" + ec
    if full in vocab_index:
        return vocab_index[full], "exact"
    fields = ec.split(".")
    if len(fields) >= 3:
        prefix = "EC:" + ".".join(fields[:3])
        cand = prefix + ".-"
        if cand in vocab_index:
            return vocab_index[cand], "backoff"
        matches = [i for i, v in enumerate(vocab) if v == prefix or v.startswith(prefix + ".")]
        if matches:
            return matches[0], "backoff"
    return None, "fallback"


def score_records(records: List[dict], repo_dir: str, model_dir: str) -> Dict[str, List[float]]:
    """Computes the ProteInfer CAM score for every input record.

    Args:
        records: Input records with "id", "seq" and "ec" fields (see
            ../common.py / ../README.md).
        repo_dir: Path to the cloned ProteInfer repository (importable
            ``inference`` module).
        model_dir: Path to the unzipped ProteInfer SavedModel directory.

    Returns:
        Mapping from chain id to a list of per-residue CAM scores.
    """
    sys.path.insert(0, repo_dir)
    import inference  # noqa: E402  (from the cloned ProteInfer repo)

    inf = inference.Inferrer(model_dir, batch_size=1, activation_type="representation")
    kernel = inf.get_variable("logits/kernel/read:0")  # [emb, n_classes]
    vocab = [v.decode() for v in inf.get_variable("label_vocab:0")]
    vocab_index = {v: i for i, v in enumerate(vocab)}
    LOGGER.info("loaded ProteInfer model: emb=%d n_classes=%d vocab=%d",
                kernel.shape[0], kernel.shape[1], len(vocab))

    target: Dict[str, Optional[int]] = {}
    fallback_ids: List[str] = []
    counts = {"exact": 0, "backoff": 0, "fallback": 0}
    for rec in records:
        c, path = pick_class(rec.get("ec"), vocab_index, vocab)
        target[rec["id"]] = c
        counts[path] += 1
        if path == "fallback":
            fallback_ids.append(rec["id"])
    LOGGER.info("class selection (pre-fallback resolve): %s", counts)

    if fallback_ids:
        inf_pred = inference.Inferrer(model_dir, batch_size=16, activation_type="serving_default")
        by_id = {r["id"]: r for r in records}
        seqs = [by_id[cid]["seq"] for cid in fallback_ids]
        preds = inf_pred.get_activations(seqs)
        for cid, p in zip(fallback_ids, preds):
            probs = _densify(p).reshape(-1)
            target[cid] = int(np.argmax(probs))

    out: Dict[str, List[float]] = {}
    for rec in records:
        cid, seq = rec["id"], rec["seq"]
        c = target[cid]
        if c is None:
            LOGGER.warning("%s: no target class resolved, skipping", cid)
            continue
        rep = _densify(inf.get_activations([seq])[0])  # [L, emb]
        if rep.shape[0] != len(seq):
            LOGGER.warning("%s: activation length %d != sequence length %d, skipping",
                            cid, rep.shape[0], len(seq))
            continue
        score = rep.dot(kernel[:, c])  # [L], signed
        if not np.isfinite(score).all():
            LOGGER.warning("%s: non-finite score, skipping", cid)
            continue
        if float(np.std(score)) <= 0:
            LOGGER.warning("%s: constant score, skipping", cid)
            continue
        out[cid] = [float(v) for v in score]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", required=True, help="Common-format input file (see ../README.md).")
    parser.add_argument("--upstream-dir", required=True,
                         help="Directory holding the cloned ProteInfer repo (./proteinfer) "
                              "and the unzipped SavedModel (./models/<name>).")
    parser.add_argument("--repo-dir", default=None,
                         help="Override: path to the cloned ProteInfer repo "
                              "(default: <upstream-dir>/proteinfer).")
    parser.add_argument("--model-dir", default=None,
                         help="Override: path to the unzipped SavedModel "
                              "(default: <upstream-dir>/models/%s)." % DEFAULT_MODEL_NAME)
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    repo_dir = args.repo_dir or os.path.join(args.upstream_dir, "proteinfer")
    model_dir = args.model_dir or os.path.join(args.upstream_dir, "models", DEFAULT_MODEL_NAME)

    records = load_records(args.inputs)
    LOGGER.info("loaded %d records from %s", len(records), args.inputs)
    scores = score_records(records, repo_dir, model_dir)
    LOGGER.info("scored %d/%d proteins", len(scores), len(records))
    write_scores(args.out, scores)
    LOGGER.info("wrote %s", args.out)


if __name__ == "__main__":
    main()

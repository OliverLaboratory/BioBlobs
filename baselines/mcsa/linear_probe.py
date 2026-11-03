#!/usr/bin/env python
"""Linear probe: the supervised ceiling, trained on M-CSA catalytic-residue labels.

An L2 logistic regression with balanced class weights predicts catalytic residues from
layer-29 ESM2-T30 embeddings (standardized on each training fold). Five-fold cross-validation
grouped by 30%-identity cluster (``clusters_30.tsv`` from
``preprocessing/mcsa/build_homology_excluded_train.py``) scores every chain with a model that never
saw its cluster. The probe uses every M-CSA chain of 10 to 800 residues with catalytic and
non-catalytic residues, in ``data/mcsa/mcsa_single_chain.json`` order (duplicates included).
It is the only baseline that sees M-CSA labels.

Usage::

    python baselines/mcsa/linear_probe.py --pdb-dir data/mcsa_raw/pdbs \\
        --clusters preprocessing_work/mcsa_homology_filter/clusters_30.tsv \\
        --out scores/linear_probe.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from inhouse import DEFAULT_LABELS, chain_sequence, esm2_layer, load_esm2, site_labels, write_scores
from sklearn.linear_model import LogisticRegression


def group_folds(group: np.ndarray, n_splits: int = 5) -> np.ndarray:
    """Fold of each residue: scikit-learn's ``GroupKFold`` rule (largest group to the lightest
    fold) with a stable sort, so tied group sizes get the same folds on every platform."""
    names, index = np.unique(group, return_inverse=True)
    sizes = np.bincount(index)
    load, fold_of = np.zeros(n_splits), np.zeros(len(names), dtype=int)
    for g in np.argsort(sizes, kind="stable")[::-1]:
        fold_of[g] = np.argmin(load)
        load[fold_of[g]] += sizes[g]
    return fold_of[index]


def cluster_of(clusters: str) -> dict[str, str]:
    """M-CSA chain id -> its cluster representative."""
    out = {}
    with open(clusters) as handle:
        next(handle)
        for line in handle:
            rep, member = line.rstrip("\n").split("\t")
            if member.startswith("MCSA__"):
                out[member.removeprefix("MCSA__")] = rep
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pdb-dir", required=True)
    p.add_argument("--clusters", required=True, help="clusters_30.tsv (rep, member)")
    p.add_argument("--labels", default=str(DEFAULT_LABELS))
    p.add_argument("--layer", type=int, default=29)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    rep = cluster_of(args.clusters)
    tokenizer, esm = load_esm2(args.device)
    xs, ys, groups, spans = [], [], [], []
    for entry in json.load(open(args.labels)):
        key = f"{entry['pdb_id']}_{entry['chain_used']}"
        parsed = chain_sequence(args.pdb_dir, entry)
        if parsed is None or not 10 <= len(parsed[0]) <= 800:
            continue
        y = site_labels(entry, parsed[1])
        if not 0 < y.sum() < len(y):
            continue
        xs.append(esm2_layer(tokenizer, esm, parsed[0], args.layer).cpu().numpy())
        ys.append(y)
        groups.append(np.full(len(y), rep.get(key, key)))
        spans.append(key)
    x, y, group = np.concatenate(xs), np.concatenate(ys), np.concatenate(groups)
    print(f"{len(spans)} chains, {len(y)} residues")

    oof = np.zeros(len(y))
    folds = group_folds(group)
    for fold in range(5):
        train, test = folds != fold, folds == fold
        mu, sd = x[train].mean(0), x[train].std(0) + 1e-6
        clf = LogisticRegression(max_iter=300, C=1.0, class_weight="balanced")
        clf.fit((x[train] - mu) / sd, y[train])
        oof[test] = clf.decision_function((x[test] - mu) / sd)
    scores, start = {}, 0
    for key, labels in zip(spans, ys, strict=True):
        scores[key] = oof[start : start + len(labels)].tolist()  # a duplicate keeps its last copy
        start += len(labels)
    write_scores(args.out, scores)
    print(f"wrote {len(scores)} chains to {args.out}")


if __name__ == "__main__":
    main()

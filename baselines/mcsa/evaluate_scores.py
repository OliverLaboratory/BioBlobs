#!/usr/bin/env python
"""M-CSA localization metrics of any set of residue-score files.

Each ``--score NAME=PATH`` is a ``{id: [score, ...]}`` JSON file in the residue order of the
M-CSA feature file (every baseline here), or the ``per_protein.jsonl`` that ``evaluate.py``
writes for BioBlobs. Labels come from ``data/mcsa/mcsa_single_chain.json``. Residues with a
non-finite score are dropped. The cohort is the requested ids that every method scores with
catalytic and non-catalytic residues. Per protein: AUROC, AUPRC (average precision) and AUPRC
enrichment (AUPRC over the catalytic fraction). The table reports medians over proteins and a
95% protein-bootstrap interval of the median enrichment (shared resamples, seed 0).

Usage::

    python baselines/mcsa/evaluate_scores.py --features data/mcsa/features.pt \\
        --score BioBlobs=results/mcsa/per_protein.jsonl \\
        --score "ESM2 attention=scores/esm2_attention.json" --out results/mcsa_table
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from inhouse import DEFAULT_IDS, DEFAULT_LABELS, load_features, read_ids, site_labels
from sklearn.metrics import average_precision_score, roc_auc_score


def read_scores(path: str) -> dict[str, list[float]]:
    if path.endswith(".jsonl"):
        rows = [json.loads(line) for line in open(path) if line.strip()]
        return {row["name"]: row["scores"] for row in rows}
    return json.load(open(path))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--features", required=True, help="M-CSA feature file (residue order)")
    p.add_argument("--score", action="append", required=True, metavar="NAME=PATH")
    p.add_argument("--ids", default=str(DEFAULT_IDS))
    p.add_argument("--labels", default=str(DEFAULT_LABELS))
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="output directory")
    args = p.parse_args()

    features = load_features(args.features)
    entries = {f"{e['pdb_id']}_{e['chain_used']}": e for e in json.load(open(args.labels))}
    ids = read_ids(args.ids)
    pairs = {}  # method -> id -> (labels, scores)
    for spec in args.score:
        name, path = spec.split("=", 1)
        scores, pairs[name] = read_scores(path), {}
        for key in ids:
            if key not in scores:
                continue
            y = site_labels(entries[key], features[key]["resnums"])
            s = np.asarray(scores[key], dtype=float)
            if len(s) != len(y):
                print(f"{name} {key}: {len(s)} scores for {len(y)} residues, skipped")
                continue
            keep = np.isfinite(s)
            if 0 < y[keep].sum() < keep.sum():
                pairs[name][key] = (y[keep], s[keep])
    cohort = sorted(k for k in ids if all(k in d for d in pairs.values()))
    print(f"{len(cohort)} of {len(ids)} chains scored by every method")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics, rows = {}, []
    for name, d in pairs.items():
        m = np.array([
            (roc_auc_score(y, s), average_precision_score(y, s), y.mean())
            for y, s in (d[k] for k in cohort)
        ])  # fmt: skip
        metrics[name] = {"auroc": m[:, 0], "auprc": m[:, 1], "enrichment": m[:, 1] / m[:, 2]}
        for i, key in enumerate(cohort):
            rows.append({"method": name, "id": key, **{k: v[i] for k, v in metrics[name].items()}})
    with open(out / "per_protein.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    resamples = np.random.default_rng(args.seed).integers(
        0, len(cohort), (args.bootstrap, len(cohort))
    )
    table = []
    for name, v in metrics.items():
        boot = np.median(v["enrichment"][resamples], axis=1)
        table.append({
            "method": name, "n": len(cohort),
            "enrichment": np.median(v["enrichment"]),
            "enrichment_lo": np.percentile(boot, 2.5), "enrichment_hi": np.percentile(boot, 97.5),
            "auprc": np.median(v["auprc"]), "auroc": np.median(v["auroc"]),
        })  # fmt: skip
    with open(out / "table.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"{'method':<40} {'AUPRC enrichment [95% CI]':>27} {'AUPRC':>7} {'AUROC':>7}")
    for r in table:
        ci = f"{r['enrichment']:.1f}x [{r['enrichment_lo']:.1f}, {r['enrichment_hi']:.1f}]"
        print(f"{r['method']:<40} {ci:>27} {r['auprc']:7.3f} {r['auroc']:7.3f}")


if __name__ == "__main__":
    main()

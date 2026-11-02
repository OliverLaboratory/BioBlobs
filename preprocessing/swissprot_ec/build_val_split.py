#!/usr/bin/env python
"""Class-stratified train/val split via MMseqs2 30%-identity clustering.

Splits ``train_split.csv`` (from ``build_structures.py``) into a train-only and
val-only CSV so that every val protein has ``< --min-seq-id`` sequence identity to
every train protein, and each EC-L3 class's clusters are split in proportion to
``--val-fraction`` so val class coverage mirrors the full corpus.

Pipeline:
    1. Write ``input.fasta`` from the split CSV.
    2. Run ``mmseqs easy-cluster`` at ``--min-seq-id`` / ``-c`` / ``--cov-mode``.
       Each protein maps to its cluster representative id.
    3. Per cluster, assign one EC-L3 class by majority vote of its members
       (stratification only; per-protein labels stay the protein's own EC-L3).
    4. Per EC-L3 class, shuffle its clusters with a fixed ``--seed`` and assign
       ``ceil(val_fraction * n_clusters)`` to val (a singleton-cluster class goes
       entirely to train, since it has no val signal to offer).
    5. Materialize the protein-level split: a cluster's members move together, so
       no val protein shares a cluster with any train protein.

Requires MMseqs2 (https://github.com/soedinglab/MMseqs2): ``conda install -c
bioconda mmseqs2``, or the static binary from https://mmseqs.com.

Usage::

    .venv/bin/python preprocessing/swissprot_ec/build_val_split.py \\
        --input-csv data/swissprot_ec/train_split.csv \\
        --val-fraction 0.05 --min-seq-id 0.3 --cov 0.8 --cov-mode 0 \\
        --threads 32 --seed 0
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input-csv", type=Path, default=Path("data/swissprot_ec/train_split.csv"))
    p.add_argument("--id-col", default="id")
    p.add_argument("--seq-col", default="seq")
    p.add_argument("--label-col", default="ec_l3", help="Column to stratify on.")
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.05,
        help="Fraction of clusters per class assigned to val.",
    )
    p.add_argument("--min-seq-id", type=float, default=0.3, help="mmseqs --min-seq-id.")
    p.add_argument("--cov", type=float, default=0.8, help="mmseqs -c (coverage threshold).")
    p.add_argument("--cov-mode", type=int, default=0, help="mmseqs --cov-mode (0=bidirectional).")
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mmseqs-bin", default="mmseqs", help="Path to the mmseqs binary.")
    p.add_argument(
        "--out-train", type=Path, default=None, help="Default: <input-dir>/train_split_train.csv"
    )
    p.add_argument(
        "--out-val", type=Path, default=None, help="Default: <input-dir>/train_split_val.csv"
    )
    p.add_argument(
        "--out-report", type=Path, default=None, help="Default: <input-dir>/val_split_report.json"
    )
    return p.parse_args()


def write_fasta(df: pd.DataFrame, fasta_path: Path, id_col: str, seq_col: str) -> None:
    """Write one FASTA record per row."""
    with fasta_path.open("w") as fh:
        for pid, seq in zip(df[id_col].values, df[seq_col].values, strict=True):
            fh.write(f">{pid}\n{seq}\n")


def run_mmseqs_cluster(
    fasta_path: Path,
    work_dir: Path,
    mmseqs_bin: str,
    min_seq_id: float,
    cov: float,
    cov_mode: int,
    threads: int,
) -> Path:
    """Run ``mmseqs easy-cluster``; returns the path to ``<prefix>_cluster.tsv``."""
    prefix = work_dir / "cluster"
    cmd = [
        mmseqs_bin,
        "easy-cluster",
        str(fasta_path),
        str(prefix),
        str(work_dir / "tmp"),
        "--min-seq-id",
        str(min_seq_id),
        "-c",
        str(cov),
        "--cov-mode",
        str(cov_mode),
        "--threads",
        str(threads),
    ]
    logger.info("$ {}", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit(f"mmseqs exited {res.returncode}\nlast stderr:\n{res.stderr[-2000:]}")
    cluster_tsv = Path(f"{prefix}_cluster.tsv")
    if not cluster_tsv.exists():
        sys.exit(f"mmseqs did not produce {cluster_tsv}")
    return cluster_tsv


def parse_cluster_tsv(cluster_tsv: Path) -> dict[str, str]:
    """Return ``{member_id: representative_id}``."""
    out: dict[str, str] = {}
    with cluster_tsv.open() as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                out[parts[1]] = parts[0]
    return out


def split_clusters(
    cluster_to_class: dict[str, str], val_fraction: float, seed: int
) -> tuple[set[str], set[str], dict[str, dict[str, int]]]:
    """Per-class stratified train/val assignment of cluster representatives."""
    rng = np.random.default_rng(seed)
    class_to_clusters: dict[str, list[str]] = defaultdict(list)
    for rep, cls in cluster_to_class.items():
        class_to_clusters[cls].append(rep)

    val_clusters: set[str] = set()
    train_clusters: set[str] = set()
    per_class: dict[str, dict[str, int]] = {}
    for cls in sorted(class_to_clusters):
        clusters = sorted(class_to_clusters[cls])
        clusters = [clusters[i] for i in rng.permutation(len(clusters))]
        n_total = len(clusters)
        if n_total == 1:
            n_val = 0
        else:
            n_val = min(max(1, int(np.ceil(val_fraction * n_total))), n_total - 1)
        val_clusters.update(clusters[:n_val])
        train_clusters.update(clusters[n_val:])
        per_class[cls] = {
            "n_clusters": n_total,
            "n_val_clusters": n_val,
            "n_train_clusters": n_total - n_val,
        }
    return train_clusters, val_clusters, per_class


def main() -> None:
    args = parse_args()
    if shutil.which(args.mmseqs_bin) is None:
        sys.exit(
            f"ERROR: '{args.mmseqs_bin}' not found in PATH.\n"
            "Install via `conda install -c bioconda mmseqs2` or "
            "https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz"
        )

    in_csv = args.input_csv.resolve()
    out_train = (args.out_train or in_csv.parent / "train_split_train.csv").resolve()
    out_val = (args.out_val or in_csv.parent / "train_split_val.csv").resolve()
    out_report = (args.out_report or in_csv.parent / "val_split_report.json").resolve()

    logger.info("Reading {}", in_csv)
    df = pd.read_csv(in_csv)
    df[args.id_col] = df[args.id_col].astype(str)
    df[args.seq_col] = df[args.seq_col].astype(str)
    df[args.label_col] = df[args.label_col].astype(str)
    logger.info("  {:,} proteins, {} classes", len(df), df[args.label_col].nunique())

    with tempfile.TemporaryDirectory(prefix="mmseqs_") as tmpdir:
        tmp = Path(tmpdir)
        fasta = tmp / "input.fasta"
        write_fasta(df, fasta, args.id_col, args.seq_col)
        cluster_tsv = run_mmseqs_cluster(
            fasta, tmp, args.mmseqs_bin, args.min_seq_id, args.cov, args.cov_mode, args.threads
        )
        member_to_repr = parse_cluster_tsv(cluster_tsv)

    logger.info(
        "mmseqs: {:,} clusters across {:,} mapped sequences",
        len(set(member_to_repr.values())),
        len(member_to_repr),
    )

    id_to_label = dict(zip(df[args.id_col], df[args.label_col], strict=True))
    cluster_to_members: dict[str, list[str]] = defaultdict(list)
    for mid, rid in member_to_repr.items():
        if mid in id_to_label:
            cluster_to_members[rid].append(mid)
    cluster_to_class = {}
    for rid, members in cluster_to_members.items():
        tally = Counter(id_to_label[m] for m in members)
        winners = sorted(lbl for lbl, c in tally.items() if c == max(tally.values()))
        cluster_to_class[rid] = winners[0]

    train_clusters, val_clusters, per_class = split_clusters(
        cluster_to_class, args.val_fraction, args.seed
    )
    logger.info(
        "Split: {:,} train clusters, {:,} val clusters", len(train_clusters), len(val_clusters)
    )

    df["__cluster"] = df[args.id_col].map(member_to_repr)
    in_val = df["__cluster"].isin(val_clusters)
    in_train = df["__cluster"].isin(train_clusters)
    df_val = df[in_val].drop(columns="__cluster").reset_index(drop=True)
    df_train = df[in_train].drop(columns="__cluster").reset_index(drop=True)
    n_dropped = len(df) - len(df_train) - len(df_val)
    logger.info(
        "Proteins: train={:,} val={:,} ({:.2%}); dropped={}",
        len(df_train),
        len(df_val),
        len(df_val) / max(len(df), 1),
        n_dropped,
    )

    df_train.to_csv(out_train, index=False)
    df_val.to_csv(out_val, index=False)
    logger.success("wrote {} ({:,} rows)", out_train, len(df_train))
    logger.success("wrote {} ({:,} rows)", out_val, len(df_val))

    report = {
        "input_csv": str(in_csv),
        "n_input": len(df),
        "n_train": len(df_train),
        "n_val": len(df_val),
        "n_dropped": n_dropped,
        "n_clusters_train": len(train_clusters),
        "n_clusters_val": len(val_clusters),
        "args": {k: str(v) for k, v in vars(args).items()},
        "per_class_top10_by_cluster_count": dict(
            sorted(per_class.items(), key=lambda kv: -kv[1]["n_clusters"])[:10]
        ),
    }
    with out_report.open("w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    logger.success("wrote {}", out_report)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""A SwissProt EC-L3 train split with M-CSA homologs removed.

Clusters SwissProt EC-L3 train sequences together with the curated M-CSA chains at
``--min-seq-id`` identity (MMseqs2), then drops every train protein that lands in a
cluster containing an M-CSA chain. Retraining on the result removes close
train/M-CSA-test homologs, so an M-CSA recovery signal cannot be attributed to
leakage.

The M-CSA held-out/evaluation ids shipped in ``data/mcsa/`` were derived from a
similar (not necessarily identical-snapshot) clustering; see
``data/mcsa/heldout_ids.txt`` for the exact caveat.

Outputs:
    <out-csv>                        filtered train rows (same columns as --train-csv)
    <workdir>/clusters_30.tsv        raw MMseqs2 cluster assignments (rep, member)
    <workdir>/homology_exclusion_report.json

Usage::

    .venv/bin/python preprocessing/mcsa/build_homology_excluded_train.py \\
        --train-csv data/swissprot_ec/train_split_train.csv \\
        --mcsa-labels data/mcsa/mcsa_single_chain.json \\
        --pdb-dir data/mcsa_raw/pdbs \\
        --out-csv data/swissprot_ec/train_split_train_homfiltered.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from subprocess import run

# Allow `python preprocessing/mcsa/build_homology_excluded_train.py ...` from any working
# directory: the repo isn't installed as a package (pyproject.toml: package = false), so
# `bioblobs` is only importable once the repo root is on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402

from bioblobs.data.structure import THREE_TO_ONE  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--train-csv", type=Path, required=True, help="SwissProt EC-L3 train split (id, seq)."
    )
    p.add_argument(
        "--mcsa-labels", type=Path, required=True, help="data/mcsa/mcsa_single_chain.json."
    )
    p.add_argument("--pdb-dir", type=Path, required=True, help="Directory of <pdb_id>.pdb files.")
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--workdir", type=Path, default=Path("preprocessing_work/mcsa_homology_filter"))
    p.add_argument("--min-seq-id", type=float, default=0.3)
    p.add_argument("--coverage", type=float, default=0.5)
    p.add_argument("--cov-mode", type=int, default=1)
    p.add_argument("--mmseqs-bin", default="mmseqs")
    return p.parse_args()


def chain_sequence(pdb_path: Path, chain: str) -> str | None:
    """One-letter sequence of ``chain`` in ``pdb_path``, in residue-number order."""
    from biopandas.pdb import PandasPdb

    atoms = PandasPdb().read_pdb(str(pdb_path)).df["ATOM"]
    atoms = atoms[atoms["chain_id"] == chain]
    if atoms.empty:
        return None
    residues = atoms.drop_duplicates(subset=["residue_number"]).sort_values("residue_number")
    return "".join(THREE_TO_ONE.get(name, "X") for name in residues["residue_name"])


def mcsa_sequences(labels_path: Path, pdb_dir: Path) -> dict[str, str]:
    """``{"MCSA__<pdb>_<chain>": sequence}`` for every curated entry with a local PDB."""
    seqs: dict[str, str] = {}
    for entry in json.loads(labels_path.read_text()):
        pdb_path = pdb_dir / f"{entry['pdb_id'].lower()}.pdb"
        if not pdb_path.exists():
            continue
        try:
            seq = chain_sequence(pdb_path, entry["chain_used"])
        except Exception as exc:  # noqa: BLE001 - report and continue
            logger.warning("sequence extraction failed for {}: {}", entry["pdb_id"], exc)
            continue
        if seq:
            seqs[f"MCSA__{entry['pdb_id']}_{entry['chain_used']}"] = seq
    return seqs


def write_fasta(path: Path, id_to_seq: dict[str, str]) -> None:
    with path.open("w") as fh:
        for name, seq in id_to_seq.items():
            fh.write(f">{name}\n{seq}\n")


def main() -> None:
    args = parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.train_csv)
    train_seqs = {
        str(r.id): str(r.seq) for r in df.itertuples() if isinstance(r.seq, str) and r.seq
    }
    logger.info("SwissProt EC-L3 train: {:,} sequences", len(train_seqs))

    mcsa = mcsa_sequences(args.mcsa_labels, args.pdb_dir)
    logger.info("M-CSA: {:,} chain sequences extracted", len(mcsa))

    fasta = args.workdir / "combined.fasta"
    write_fasta(fasta, {**train_seqs, **mcsa})

    prefix = args.workdir / "clust"
    cmd = [
        args.mmseqs_bin,
        "easy-cluster",
        str(fasta),
        str(prefix),
        str(args.workdir / "tmp"),
        "--min-seq-id",
        str(args.min_seq_id),
        "-c",
        str(args.coverage),
        "--cov-mode",
        str(args.cov_mode),
        "-v",
        "1",
    ]
    logger.info("$ {}", " ".join(cmd))
    run(cmd, check=True)

    cluster_tsv = Path(f"{prefix}_cluster.tsv")
    cl = pd.read_csv(cluster_tsv, sep="\t", header=None, names=["rep", "member"])
    clusters: dict[str, list[str]] = {}
    for rep, member in zip(cl["rep"], cl["member"], strict=True):
        clusters.setdefault(rep, []).append(member)

    contaminated_train_ids: set[str] = set()
    n_contaminated_clusters = 0
    for members in clusters.values():
        if any(m.startswith("MCSA__") for m in members):
            n_contaminated_clusters += 1
            contaminated_train_ids.update(m for m in members if not m.startswith("MCSA__"))

    keep = df[~df["id"].astype(str).isin(contaminated_train_ids)].copy()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    keep.to_csv(args.out_csv, index=False)
    cl.to_csv(args.workdir / "clusters_30.tsv", sep="\t", index=False)

    report = {
        "min_seq_id": args.min_seq_id,
        "coverage": args.coverage,
        "cov_mode": args.cov_mode,
        "n_train_in": len(df),
        "n_mcsa_seqs": len(mcsa),
        "n_clusters": len(clusters),
        "n_contaminated_clusters": n_contaminated_clusters,
        "n_train_removed": len(contaminated_train_ids),
        "n_train_kept": len(keep),
        "frac_removed": len(contaminated_train_ids) / max(1, len(df)),
    }
    with (args.workdir / "homology_exclusion_report.json").open("w") as fh:
        json.dump(report, fh, indent=2)
    logger.success(
        "removed {:,}/{:,} train proteins ({:.1%}) homologous to M-CSA -> {}",
        report["n_train_removed"],
        report["n_train_in"],
        report["frac_removed"],
        args.out_csv,
    )


if __name__ == "__main__":
    main()

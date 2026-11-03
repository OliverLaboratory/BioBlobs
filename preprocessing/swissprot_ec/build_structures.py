#!/usr/bin/env python
"""Build the SwissProt EC-L3 structure cache and label artifacts.

CPU-only: parses structures and assigns EC-L3 labels; it does not embed
sequences. Residue embeddings are built separately, over this script's
output, by ``preprocessing/prepare_plm_cache.py``.

Pipeline:
    1. Read the input CSV, keep rows with a complete EC string at ``--require-ec-level``
       and an existing PDB file.
    2. Parse each PDB's backbone (``bioblobs.data.structure.load_structures``); this
       also applies the backbone-completeness filter.
    3. Drop proteins whose backbone-derived sequence exceeds ``--max-len``.
    4. Dedup proteins with an identical sanitized sequence; the canonical id is the
       lexicographically smallest id in the group, and its label is the strict-majority
       EC-L<level> token among the group (a tie drops the group).
    5. Tensorize coordinates/residue numbers and save the structure cache.
    6. Write the derived split/label artifacts.

Input CSV columns: ``id``, ``seq``, ``ec_number``, ``pdb_path`` (``seq`` is advisory;
the saved sequence is re-derived from the PDB backbone). This script does not assert
the CSV's provenance — see ``preprocessing/swissprot_ec/README.md``.

Usage::

    .venv/bin/python preprocessing/swissprot_ec/build_structures.py \\
        --input-csv data/swissprot_ec/train_100_with_pdb.csv \\
        --pdb-root data \\
        --out-data-dir data/swissprot_ec \\
        --pdb-num-workers 32 --require-ec-level 3 --max-len 1500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

# Allow `python preprocessing/swissprot_ec/build_structures.py ...` from any working
# directory: the repo isn't installed as a package (pyproject.toml: package = false), so
# `bioblobs` is only importable once the repo root is on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402
from tqdm import tqdm  # noqa: E402

from bioblobs.data.structure import Structure, load_structures  # noqa: E402

VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYX")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input-csv", required=True, type=Path)
    p.add_argument("--pdb-root", type=Path, default=Path("data"))
    p.add_argument("--out-data-dir", type=Path, default=Path("data/swissprot_ec"))
    p.add_argument("--id-col", default="id")
    p.add_argument("--ec-col", default="ec_number")
    p.add_argument("--pdb-col", default="pdb_path")

    p.add_argument("--require-ec-level", type=int, default=3)
    p.add_argument("--max-len", type=int, default=1500)
    p.add_argument("--min-completion", type=float, default=0.95)
    p.add_argument("--pdb-num-workers", type=int, default=8)
    return p.parse_args()


def sanitize_sequence(sequence: str) -> str:
    """Replace non-standard amino acids with ``X`` (matches the dev ESM2 cache)."""
    return "".join(aa if aa in VALID_AMINO_ACIDS else "X" for aa in sequence)


def is_complete_ec(ec: str, level: int) -> bool:
    """Whether ``ec`` has at least ``level`` dot-separated numeric components."""
    parts = str(ec).split(".")
    return len(parts) >= level and all(p.isdigit() for p in parts[:level])


def truncate_ec(ec: str, level: int) -> str:
    """The first ``level`` dot-separated components of an EC number."""
    return ".".join(str(ec).split(".")[:level])


def build_pdb_tasks(args: argparse.Namespace) -> list[tuple]:
    """Read the input CSV and build ``(id, pdb_path, ec_number, min_completion)`` tasks."""
    df = pd.read_csv(args.input_csv, usecols=[args.id_col, args.ec_col, args.pdb_col])
    df = df.dropna(subset=[args.id_col, args.ec_col, args.pdb_col])
    df[args.id_col] = df[args.id_col].astype(str)
    df[args.ec_col] = df[args.ec_col].astype(str)
    n_raw = len(df)

    df = df[df[args.ec_col].map(lambda s: is_complete_ec(s, args.require_ec_level))]
    n_after_ec = len(df)

    tasks, n_missing_pdb = [], 0
    for _, row in df.iterrows():
        pdb_path = args.pdb_root / str(row[args.pdb_col])
        if not pdb_path.exists():
            n_missing_pdb += 1
            continue
        tasks.append((str(row[args.id_col]), pdb_path, str(row[args.ec_col]), args.min_completion))
    logger.info(
        "raw={:,} ec-complete={:,} with-pdb={:,} (dropped: {} ec-incomplete, {} missing-pdb)",
        n_raw,
        n_after_ec,
        len(tasks),
        n_raw - n_after_ec,
        n_missing_pdb,
    )
    return tasks


def filter_max_len(structures: list[Structure], max_len: int | None) -> list[Structure]:
    """Drop proteins whose sanitized sequence exceeds ``max_len``; attach ``__san``."""
    if max_len is None:
        for s in structures:
            s["__san"] = sanitize_sequence(s["seq"])
        return structures
    out = []
    for s in structures:
        san = sanitize_sequence(s["seq"])
        if len(san) <= max_len:
            s["__san"] = san
            out.append(s)
    logger.info("max-len {}: {:,} -> {:,}", max_len, len(structures), len(out))
    return out


def resolve_label(labels: list[str]) -> str | None:
    """Strict-majority vote; ``None`` on a tie."""
    counts = Counter(labels)
    winners = [lbl for lbl, c in counts.items() if c == max(counts.values())]
    return winners[0] if len(winners) == 1 else None


def dedup_and_label(
    structures: list[Structure], ec_level: int
) -> tuple[list[Structure], dict[str, int], dict[str, int]]:
    """Dedup on sanitized sequence; assign each kept protein its EC-L<level> label.

    Returns:
        ``(kept, token_map, class_counts)``.
    """
    by_san: dict[str, list[Structure]] = {}
    for s in structures:
        by_san.setdefault(s["__san"], []).append(s)

    kept: list[Structure] = []
    n_tied = 0
    for group in by_san.values():
        ec_truncated = [truncate_ec(s["target_payload"], ec_level) for s in group]
        label = resolve_label(ec_truncated)
        if label is None:
            n_tied += 1
            continue
        canonical = next(dict(s) for s, lt in zip(group, ec_truncated, strict=True) if lt == label)
        canonical["ec_l3"] = label
        canonical.pop("__san", None)
        kept.append(canonical)

    logger.info(
        "dedup: {:,} -> {:,} ({} collapsed, {} tied-drops)",
        len(structures),
        len(kept),
        len(structures) - len(kept) - n_tied,
        n_tied,
    )

    sorted_labels = sorted({s["ec_l3"] for s in kept})
    token_map = {lbl: i for i, lbl in enumerate(sorted_labels)}
    counts = {lbl: sum(1 for s in kept if s["ec_l3"] == lbl) for lbl in sorted_labels}
    return kept, token_map, counts


def write_artifacts(
    structures: list[Structure],
    token_map: dict[str, int],
    counts: dict[str, int],
    pdb_paths: dict[str, str],
    out_dir: Path,
) -> None:
    """Write ``train_split.csv``, ``labels.csv``, ``token_map.json``, ``class_priors.json``."""
    rows = [{
        "id": s["name"], "seq": s["seq"], "ec_number": s["target_payload"],
        "ec_l3": s["ec_l3"], "label": int(token_map[s["ec_l3"]]),
        "pdb_path": pdb_paths.get(s["name"], ""),
    } for s in structures]  # fmt: skip
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "train_split.csv", index=False)
    labels_df = pd.DataFrame({"pdb_id": df["id"], "label": df["label"]})
    labels_df.to_csv(out_dir / "labels.csv", index=False)
    with (out_dir / "token_map.json").open("w") as fh:
        json.dump(token_map, fh, indent=2, sort_keys=True)
    n = len(df)
    priors = {"N": n, "counts": counts, "priors": {lbl: counts[lbl] / n for lbl in counts}}
    with (out_dir / "class_priors.json").open("w") as fh:
        json.dump(priors, fh, indent=2, sort_keys=True)
    logger.success(
        "wrote train_split.csv, labels.csv, token_map.json, class_priors.json -> {}", out_dir
    )


def main() -> None:
    args = parse_args()
    out_dir = args.out_data_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    tasks = build_pdb_tasks(args)
    pdb_paths = {str(t[0]): str(t[1]) for t in tasks}

    structures = load_structures(tasks, num_workers=args.pdb_num_workers, desc="structures")
    structures = filter_max_len(structures, args.max_len)
    structures, token_map, counts = dedup_and_label(structures, args.require_ec_level)
    if not structures:
        sys.exit("ERROR: no proteins survived parsing/dedup; check --input-csv and --pdb-root")

    logger.info("tensorizing coords/resnum for {:,} proteins", len(structures))
    for s in tqdm(structures, desc="tensorize", unit="prot"):
        s["coords"] = torch.as_tensor(s["coords"], dtype=torch.float32)
        s["resnum"] = torch.as_tensor(s["resnum"], dtype=torch.long)

    cache_dir = out_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    structures_pt = cache_dir / "structures.pt"
    tmp = structures_pt.with_suffix(".pt.tmp")
    torch.save(structures, tmp)
    tmp.rename(structures_pt)
    logger.success(
        "structure cache: {:.2f} GiB -> {}", structures_pt.stat().st_size / (1024**3), structures_pt
    )

    write_artifacts(structures, token_map, counts, pdb_paths, out_dir)

    report = {
        "n_proteins": len(structures),
        "n_classes": len(token_map),
        "require_ec_level": args.require_ec_level,
        "max_len": args.max_len,
        "min_completion": args.min_completion,
        "elapsed_seconds": time.time() - t0,
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    with (out_dir / "build_quality_report.json").open("w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    logger.success(
        "done in {:.1f}m; {:,} proteins, {} classes",
        report["elapsed_seconds"] / 60,
        len(structures),
        len(token_map),
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Build a residue-embedding cache (``embeddings.bin`` + ``meta.pt``) for a prepared dataset.

A thin CLI over ``bioblobs.modules.plm`` (model loading, windowing, Foldseek) and
``bioblobs.data.plm_cache.write_embedding_cache`` (the on-disk format every dataset loader
reads). ``--structures`` is a torch ``.pt`` file holding either ``list[dict]`` (ProteinShake,
SwissProt) or ``dict[str, dict]`` (VenusX, keyed by ``protein_uid``) of structure dicts with at
least a sequence field and a cache-key field (``--seq-field``/``--key-field``).

Length policy: ``--mode window`` (default, used for every reported result) embeds sequences longer
than ``--window`` in overlapping windows and averages the overlaps (see
``bioblobs.modules.plm``). ``--max-len`` is an optional admission filter: longer proteins are
skipped, not truncated.

A cache that already covers every key of the given structures is left untouched unless
``--overwrite`` is passed; otherwise it is rebuilt from scratch.

Usage:
    # ESM2 over the three split caches of a ProteinShake task (name -> embeddings)
    python preprocessing/prepare_plm_cache.py \\
        --structures data/ec_proteinshake/cache/sequence_{train,val,test}_knn_30_0.7_cache.pt \\
        --encoder esm2_static --out data/ec_proteinshake/esm2_static_cache_mmap

    # SaProt needs the matching PDB files for Foldseek
    python preprocessing/prepare_plm_cache.py \\
        --structures data/ec_proteinshake/cache/sequence_{train,val,test}_knn_30_0.7_cache.pt \\
        --encoder saprot_static --pdb-dir data/ec_proteinshake/pdb \\
        --foldseek-bin data/foldseek/bin/foldseek \\
        --out data/ec_proteinshake/saprot_static_cache_mmap

    # VenusX uid-keyed structure cache
    python preprocessing/prepare_plm_cache.py \\
        --structures data/venusx_site_fragment/Act/MF50/cache/uid_structures_*.pt \\
        --encoder esm2_static --key-field protein_uid \\
        --out data/venusx_site_fragment/Act/MF50/esm2_static_cache_mmap

    # M-CSA features are built by preprocessing/mcsa/build_features.py instead.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Any

import torch
from loguru import logger

# Allow `python preprocessing/prepare_plm_cache.py ...` from any working directory: the
# repo isn't installed as a package (pyproject.toml: package = false), so `bioblobs` is only
# importable once the repo root is on sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bioblobs.data.plm_cache import write_embedding_cache  # noqa: E402
from bioblobs.modules.plm import embed_esm2, embed_saprot, sanitize_sequence  # noqa: E402

DEFAULT_MODEL_NAME = {
    "esm2_static": "facebook/esm2_t30_150M_UR50D",
    "saprot_static": "westlake-repl/SaProt_650M_PDB",
}


def parse_args() -> argparse.Namespace:
    """CLI arguments for building one encoder's embedding cache for one dataset."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--structures",
        required=True,
        type=Path,
        nargs="+",
        help="torch .pt files: list[dict] or dict[str, dict] of structure dicts.",
    )
    p.add_argument("--encoder", required=True, choices=["esm2_static", "saprot_static"])
    p.add_argument("--out", required=True, type=Path, help="Output cache directory.")
    p.add_argument(
        "--key-field",
        default="name",
        help="Cache-key field: 'name' (ProteinShake, SwissProt, M-CSA) or 'protein_uid' (VenusX).",
    )
    p.add_argument("--seq-field", default="seq")
    p.add_argument("--model-name", default=None, help="Default depends on --encoder.")
    p.add_argument("--repr-layer", type=int, default=-1)
    p.add_argument("--window", type=int, default=1022)
    p.add_argument("--overlap", type=int, default=128)
    p.add_argument(
        "--mode",
        choices=["window", "truncate"],
        default="window",
        help="See 'Length policy' above. No reported result uses 'truncate'.",
    )
    p.add_argument(
        "--max-len",
        type=int,
        default=None,
        help="Admission filter: drop proteins longer than this. Independent of --mode.",
    )
    p.add_argument("--max-batch-tokens", type=int, default=8192)
    p.add_argument("--cache-dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8, help="Capped at 8.")
    p.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Proteins embedded per model call; bounds peak memory.",
    )
    p.add_argument(
        "--pdb-dir",
        type=Path,
        default=None,
        help="Directory of <key>.pdb files, for --encoder saprot_static (unless "
        "--pdb-path-field names a field already holding each path).",
    )
    p.add_argument(
        "--pdb-path-field",
        default=None,
        help="Structure field holding an explicit PDB path, overriding --pdb-dir.",
    )
    p.add_argument(
        "--chain-field",
        default=None,
        help="Structure field naming the PDB chain (default: Foldseek's first).",
    )
    p.add_argument("--foldseek-bin", default=None)
    p.add_argument(
        "--saprot-pad-token",
        default="Xd",
        help="2-char pad for a Foldseek/sequence length mismatch; dev default for "
        "ProteinShake caches. VenusX used 'X#'.",
    )
    p.add_argument(
        "--overwrite", action="store_true", help="Rebuild even if --out already covers every key."
    )
    p.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only use the local Hugging Face cache (default: download weights when missing).",
    )
    return p.parse_args()


def _load_structures(path: Path) -> list[dict[str, Any]]:
    """Load a torch ``.pt`` structure cache as ``list[dict]``, accepting dict-keyed caches too."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        structures = list(obj.values())
    elif isinstance(obj, list):
        structures = obj
    else:
        raise TypeError(f"{path}: expected list[dict] or dict[str, dict], got {type(obj)}")
    if structures and not isinstance(structures[0], dict):
        raise TypeError(f"{path}: expected structure dicts, got {type(structures[0])}")
    return structures


def _dedupe(structures: list[dict], key_field: str) -> list[dict]:
    """First occurrence per cache key, preserving order."""
    seen: set[str] = set()
    unique = []
    for s in structures:
        key = str(s[key_field])
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


def _existing_keys(out: Path) -> set[str]:
    meta_path = out / "meta.pt"
    if not meta_path.exists():
        return set()
    meta = torch.load(meta_path, map_location="cpu", weights_only=True)
    return set(meta.get("names", []))


def _pdb_path(structure: dict, args: argparse.Namespace) -> Path:
    if args.pdb_path_field and args.pdb_path_field in structure:
        return Path(structure[args.pdb_path_field])
    if args.pdb_dir is None:
        raise ValueError(
            "--encoder saprot_static needs --pdb-dir (or --pdb-path-field naming a field "
            "that already holds each protein's PDB path)."
        )
    return args.pdb_dir / f"{structure[args.key_field]}.pdb"


def _iter_embeddings(structures: list[dict], args: argparse.Namespace):
    """Yield ``(key, embedding)`` pairs, embedding ``--chunk-size`` proteins at a time."""
    for start in range(0, len(structures), args.chunk_size):
        chunk = structures[start : start + args.chunk_size]
        sequences = [str(s[args.seq_field]) for s in chunk]
        keys = [str(s[args.key_field]) for s in chunk]
        if args.encoder == "esm2_static":
            embeddings = embed_esm2(
                sequences,
                model_name=args.model_name,
                repr_layer=args.repr_layer,
                window=args.window,
                overlap=args.overlap,
                mode=args.mode,
                max_batch_tokens=args.max_batch_tokens,
                device=args.device,
                local_files_only=args.local_files_only,
            )
        else:
            pdb_paths = [_pdb_path(s, args) for s in chunk]
            chains = [s.get(args.chain_field) if args.chain_field else None for s in chunk]
            embeddings = embed_saprot(
                sequences,
                pdb_paths,
                chains=chains,
                foldseek_bin=args.foldseek_bin,
                model_name=args.model_name,
                repr_layer=args.repr_layer,
                window=args.window,
                overlap=args.overlap,
                mode=args.mode,
                max_batch_tokens=args.max_batch_tokens,
                device=args.device,
                local_files_only=args.local_files_only,
                pad_token=args.saprot_pad_token,
            )
        logger.info(
            "Embedded {}/{} proteins",
            min(start + args.chunk_size, len(structures)),
            len(structures),
        )
        yield from zip(keys, embeddings, strict=True)


def main() -> None:
    """Build one encoder's embedding cache for one dataset's structures."""
    args = parse_args()
    if args.threads > 8:
        logger.warning("--threads {} > 8; capping at 8", args.threads)
    torch.set_num_threads(max(1, min(args.threads, 8)))
    if args.model_name is None:
        args.model_name = DEFAULT_MODEL_NAME[args.encoder]

    structures = [s for path in args.structures for s in _load_structures(path)]
    if not structures:
        raise RuntimeError(f"No structures found in {args.structures}")
    if args.key_field not in structures[0] or args.seq_field not in structures[0]:
        raise KeyError(
            f"--key-field {args.key_field!r} / --seq-field {args.seq_field!r} not both found "
            f"in structures (have {sorted(structures[0])})"
        )
    unique = _dedupe(structures, args.key_field)
    logger.info(
        "{}: {} structures, {} unique by {!r}",
        args.structures,
        len(structures),
        len(unique),
        args.key_field,
    )

    if args.max_len is not None:
        before = len(unique)
        unique = [
            s for s in unique if len(sanitize_sequence(str(s[args.seq_field]))) <= args.max_len
        ]
        logger.info(
            "--max-len {} admission filter: {} -> {} proteins", args.max_len, before, len(unique)
        )
    if not unique:
        raise RuntimeError("No proteins left to embed after filtering.")

    required = {str(s[args.key_field]) for s in unique}
    if not args.overwrite:
        existing = _existing_keys(args.out)
        if existing and required <= existing:
            logger.info(
                "{} already covers all {} proteins; nothing to do (pass --overwrite to rebuild).",
                args.out,
                len(required),
            )
            return
        if existing:
            logger.info(
                "{} of {} proteins missing from {}; rebuilding the full cache.",
                len(required - existing),
                len(required),
                args.out,
            )

    logger.info(
        "encoder={} model={} proteins={} device={} mode={} window={} overlap={}",
        args.encoder,
        args.model_name,
        len(unique),
        args.device,
        args.mode,
        args.window,
        args.overlap,
    )

    pairs = _iter_embeddings(unique, args)
    try:
        first = next(pairs)
    except StopIteration:
        raise RuntimeError("No embeddings were produced.") from None
    hidden_dim = int(first[1].shape[1])
    n_written = write_embedding_cache(
        args.out, itertools.chain([first], pairs), hidden_dim=hidden_dim, dtype=args.cache_dtype
    )
    logger.success(
        "Wrote {} proteins ({} dim, {}) to {}", n_written, hidden_dim, args.cache_dtype, args.out
    )


if __name__ == "__main__":
    main()


"""Direct-to-mmap PLM embedding cache builders for VenusX.

Builds a single ``embeddings.bin`` + ``meta.pt`` per (target, encoder) pair,
without writing per-protein ``.pt`` intermediates. Pattern adapted from
``gearnet_static_cache.prepare_gearnet_static_embedding_cache`` on the
``gearnet-ssl-pretrained`` branch.

Layout:
    <data_dir>/venusx_site_fragment/<target>/<split_strategy>/
        esm2_static_cache_mmap/
            embeddings.bin            # float16/float32, shape (total_rows, hidden_dim)
            meta.pt                   # {names, offsets, hidden_dim, dtype, total_rows, cache_key}
        saprot_static_cache_mmap/
            embeddings.bin
            meta.pt

``names`` indexes by ``protein_uid`` (deduped), ``offsets`` is an int64 array
of length ``len(names) + 1``, so protein i's residues live at rows
``offsets[i]:offsets[i+1]`` and have length ``len(seq_i)``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from .esm2_cache import (
    build_esm2_cache_key,
    embed_sequence_with_esm2,
    get_esm2_cache_root,
    resolve_cache_dtype,
    _load_esm2_model_and_tokenizer,
)
from .saprot_cache import (
    _find_foldseek_binary,
    _get_saprot_cache_root,
    _load_saprot_model_and_tokenizer,
    _resolve_pdb_dir,
    build_saprot_cache_key,
    embed_sequence_with_saprot,
    get_struc_seq,
)

ESM2_MMAP_DIR_NAME = "esm2_static_cache_mmap"
SAPROT_MMAP_DIR_NAME = "saprot_static_cache_mmap"


# ─── Shared helpers ─────────────────────────────────────────────────────────


def _np_dtype_for(cache_dtype: str) -> np.dtype:
    if cache_dtype == "float16":
        return np.dtype(np.float16)
    if cache_dtype == "float32":
        return np.dtype(np.float32)
    raise ValueError(f"Unsupported cache_dtype {cache_dtype!r} (use float16 or float32)")


def _check_existing_cache(
    bin_path: Path,
    meta_path: Path,
    required_uids: set[str],
    cache_key: str,
    overwrite: bool,
) -> bool:
    """Return True if existing cache fully covers required_uids and key matches."""
    if overwrite or not (bin_path.exists() and meta_path.exists()):
        return False
    try:
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read existing meta.pt ({}); rebuilding.", exc)
        return False
    existing_key = meta.get("cache_key")
    if existing_key is not None and existing_key != cache_key:
        logger.warning(
            "Cache key mismatch (existing={!r}, expected={!r}); rebuilding.",
            existing_key, cache_key,
        )
        return False
    existing_uids = set(meta.get("names", []))
    missing = required_uids - existing_uids
    if missing:
        logger.warning(
            "Existing cache is missing {} of {} required proteins; rebuilding.",
            len(missing), len(required_uids),
        )
        return False
    return True


def _dedupe_structures(structures: Sequence[dict]) -> list[dict]:
    """Keep first occurrence of each protein_uid, preserving order."""
    seen: set[str] = set()
    unique: list[dict] = []
    for s in structures:
        uid = str(s.get("protein_uid", s["name"]))
        if uid in seen:
            continue
        seen.add(uid)
        unique.append(s)
    return unique


def _write_meta(
    meta_path: Path,
    *,
    names: list[str],
    offsets: np.ndarray,
    hidden_dim: int,
    cache_dtype: str,
    cache_key: str,
) -> None:
    torch.save(
        {
            "names": names,
            "offsets": torch.tensor(offsets, dtype=torch.long),
            "hidden_dim": int(hidden_dim),
            "dtype": cache_dtype,
            "total_rows": int(offsets[-1]),
            "cache_key": cache_key,
        },
        meta_path,
    )


# ─── ESM2 builder ───────────────────────────────────────────────────────────


def prepare_esm2_static_mmap_cache(
    structures: Sequence[dict],
    cfg,
    device: torch.device | None = None,
) -> Path:
    """Build a direct-to-mmap ESM2 residue embedding cache for VenusX.

    ``structures`` is the union of train+val+test from
    ``venusx_dataset.load_prepared_structures``. Proteins are deduped by
    ``protein_uid`` so a single mmap covers every split.
    """
    if not structures:
        raise ValueError("structures is empty")

    cache_dtype = cfg.encoders.cache_dtype
    np_dtype = _np_dtype_for(cache_dtype)

    cache_key = build_esm2_cache_key(
        model_name=cfg.encoders.model_name,
        repr_layer=cfg.encoders.repr_layer,
        window_size=cfg.encoders.window_size,
        window_overlap=cfg.encoders.window_overlap,
        cache_dtype=cache_dtype,
    )

    # Dataset-rooted layout: bypass cfg.encoders.cache_dir so we land at
    # <prepared_root>/esm2_static_cache_mmap/ regardless of cache_dir name.
    prepared_root = get_esm2_cache_root(cfg).parent
    mmap_dir = prepared_root / ESM2_MMAP_DIR_NAME
    bin_path = mmap_dir / "embeddings.bin"
    meta_path = mmap_dir / "meta.pt"

    unique = _dedupe_structures(structures)
    required_uids = {str(s.get("protein_uid", s["name"])) for s in unique}
    overwrite = bool(cfg.encoders.get("overwrite_cache", False))

    if _check_existing_cache(bin_path, meta_path, required_uids, cache_key, overwrite):
        logger.info(
            "Reusing ESM2 mmap cache at {} ({} proteins already covered)",
            mmap_dir, len(required_uids),
        )
        return mmap_dir

    if not cfg.encoders.get("precompute_missing", True):
        raise FileNotFoundError(
            f"ESM2 mmap cache missing or incomplete at {mmap_dir} and "
            "encoders.precompute_missing=false."
        )

    if device is None:
        device = torch.device(cfg.encoders.get("precompute_device", "cpu"))

    names = [str(s.get("protein_uid", s["name"])) for s in unique]
    seq_lens = [len(s["seq"]) for s in unique]
    offsets = np.concatenate([[0], np.cumsum(seq_lens, dtype=np.int64)])
    total_rows = int(offsets[-1])

    logger.info("Loading ESM2 model {} on {}", cfg.encoders.model_name, device)
    tokenizer, model = _load_esm2_model_and_tokenizer(cfg.encoders.model_name, device)
    repr_layer = int(cfg.encoders.repr_layer)
    hidden_dim = int(getattr(model.config, "hidden_size"))

    mmap_dir.mkdir(parents=True, exist_ok=True)
    tmp_bin = bin_path.with_suffix(".bin.tmp")
    if tmp_bin.exists():
        tmp_bin.unlink()

    est_size_gib = (total_rows * hidden_dim * np_dtype.itemsize) / (1024 ** 3)
    logger.info(
        "Building ESM2 mmap cache → {} (proteins={}, rows={}, hidden={}, dtype={}, ~{:.2f} GiB)",
        mmap_dir, len(unique), total_rows, hidden_dim, cache_dtype, est_size_gib,
    )

    mmap_out = np.memmap(tmp_bin, dtype=np_dtype, mode="w+", shape=(total_rows, hidden_dim))

    save_dtype = resolve_cache_dtype(cache_dtype)
    with torch.no_grad():
        for i, s in enumerate(tqdm(unique, desc="ESM2 mmap", unit="prot")):
            emb = embed_sequence_with_esm2(
                sequence=str(s["seq"]),
                tokenizer=tokenizer,
                model=model,
                repr_layer=repr_layer,
                device=device,
                window_size=int(cfg.encoders.window_size),
                window_overlap=int(cfg.encoders.window_overlap),
                max_batch_tokens=int(cfg.encoders.max_batch_tokens),
            )
            expected = seq_lens[i]
            if emb.size(0) != expected:
                raise RuntimeError(
                    f"ESM2 returned {emb.size(0)} rows for {names[i]} "
                    f"(expected {expected})"
                )
            mmap_out[offsets[i]:offsets[i + 1]] = (
                emb.to(save_dtype).cpu().numpy().astype(np_dtype, copy=False)
            )

    mmap_out.flush()
    del mmap_out
    os.replace(tmp_bin, bin_path)

    _write_meta(
        meta_path,
        names=names,
        offsets=offsets,
        hidden_dim=hidden_dim,
        cache_dtype=cache_dtype,
        cache_key=cache_key,
    )

    logger.success(
        "Wrote ESM2 mmap cache: {} proteins, {} rows, {:.2f} GiB at {}",
        len(unique), total_rows, bin_path.stat().st_size / (1024 ** 3), bin_path,
    )
    return mmap_dir


# ─── SaProt builder ─────────────────────────────────────────────────────────


def _build_sa_sequence(
    foldseek_bin: str,
    pdb_path: Path,
    expected_seq: str,
    process_id: int = 0,
) -> str:
    """Run Foldseek on a PDB and return the SA sequence (AA+3Di interleaved).

    Pads or truncates so that ``len(sa_seq) // 2 == len(expected_seq)``.
    """
    seq_dict = get_struc_seq(foldseek_bin, str(pdb_path), process_id=process_id)
    if not seq_dict:
        raise RuntimeError(f"Foldseek returned no chains for {pdb_path}")

    # Pick the first chain (matches saprot_cache.py behavior).
    _aa_seq, _struc_seq, sa_seq = next(iter(seq_dict.values()))

    expected_len = len(expected_seq)
    sa_residue_len = len(sa_seq) // 2
    if sa_residue_len < expected_len:
        # Pad with "X#" (unknown AA + unknown 3Di) to expected length.
        sa_seq = sa_seq + "X#" * (expected_len - sa_residue_len)
    elif sa_residue_len > expected_len:
        sa_seq = sa_seq[: expected_len * 2]
    return sa_seq


def prepare_saprot_static_mmap_cache(
    structures: Sequence[dict],
    cfg,
    device: torch.device | None = None,
) -> Path:
    """Build a direct-to-mmap SaProt residue embedding cache for VenusX."""
    if not structures:
        raise ValueError("structures is empty")

    cache_dtype = cfg.encoders.cache_dtype
    np_dtype = _np_dtype_for(cache_dtype)

    cache_key = build_saprot_cache_key(
        model_name=cfg.encoders.model_name,
        repr_layer=cfg.encoders.repr_layer,
        window_size=cfg.encoders.window_size,
        window_overlap=cfg.encoders.window_overlap,
        cache_dtype=cache_dtype,
    )

    prepared_root = _get_saprot_cache_root(cfg).parent
    mmap_dir = prepared_root / SAPROT_MMAP_DIR_NAME
    bin_path = mmap_dir / "embeddings.bin"
    meta_path = mmap_dir / "meta.pt"

    unique = _dedupe_structures(structures)
    required_uids = {str(s.get("protein_uid", s["name"])) for s in unique}
    overwrite = bool(cfg.encoders.get("overwrite_cache", False))

    if _check_existing_cache(bin_path, meta_path, required_uids, cache_key, overwrite):
        logger.info(
            "Reusing SaProt mmap cache at {} ({} proteins already covered)",
            mmap_dir, len(required_uids),
        )
        return mmap_dir

    if not cfg.encoders.get("precompute_missing", True):
        raise FileNotFoundError(
            f"SaProt mmap cache missing or incomplete at {mmap_dir} and "
            "encoders.precompute_missing=false."
        )

    if device is None:
        device = torch.device(cfg.encoders.get("precompute_device", "cpu"))

    foldseek_bin = _find_foldseek_binary(cfg.encoders.get("foldseek_bin"))
    pdb_dir = _resolve_pdb_dir(cfg)

    names = [str(s.get("protein_uid", s["name"])) for s in unique]
    seq_lens = [len(s["seq"]) for s in unique]
    offsets = np.concatenate([[0], np.cumsum(seq_lens, dtype=np.int64)])
    total_rows = int(offsets[-1])

    logger.info("Loading SaProt model {} on {}", cfg.encoders.model_name, device)
    tokenizer, model = _load_saprot_model_and_tokenizer(cfg.encoders.model_name, device)
    repr_layer = int(cfg.encoders.repr_layer)
    hidden_dim = int(getattr(model.config, "hidden_size"))

    mmap_dir.mkdir(parents=True, exist_ok=True)
    tmp_bin = bin_path.with_suffix(".bin.tmp")
    if tmp_bin.exists():
        tmp_bin.unlink()

    est_size_gib = (total_rows * hidden_dim * np_dtype.itemsize) / (1024 ** 3)
    logger.info(
        "Building SaProt mmap cache → {} (proteins={}, rows={}, hidden={}, dtype={}, ~{:.2f} GiB)",
        mmap_dir, len(unique), total_rows, hidden_dim, cache_dtype, est_size_gib,
    )

    mmap_out = np.memmap(tmp_bin, dtype=np_dtype, mode="w+", shape=(total_rows, hidden_dim))

    save_dtype = resolve_cache_dtype(cache_dtype)
    skipped_uids: list[str] = []
    with torch.no_grad():
        for i, s in enumerate(tqdm(unique, desc="SaProt mmap", unit="prot")):
            uid = names[i]
            seq = str(s["seq"])
            pdb_path = pdb_dir / f"{uid}.pdb"
            if not pdb_path.exists():
                raise FileNotFoundError(f"PDB missing for {uid}: {pdb_path}")
            try:
                sa_seq = _build_sa_sequence(foldseek_bin, pdb_path, seq, process_id=i)
                emb = embed_sequence_with_saprot(
                    sa_sequence=sa_seq,
                    tokenizer=tokenizer,
                    model=model,
                    repr_layer=repr_layer,
                    device=device,
                    window_size=int(cfg.encoders.window_size),
                    window_overlap=int(cfg.encoders.window_overlap),
                    max_batch_tokens=int(cfg.encoders.max_batch_tokens),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "SaProt embedding failed for {} ({}); writing zeros.", uid, exc,
                )
                skipped_uids.append(uid)
                emb = torch.zeros((seq_lens[i], hidden_dim), dtype=torch.float32)

            expected = seq_lens[i]
            if emb.size(0) < expected:
                pad = torch.zeros(expected - emb.size(0), emb.size(1), dtype=emb.dtype)
                emb = torch.cat([emb, pad], dim=0)
            elif emb.size(0) > expected:
                emb = emb[:expected]

            mmap_out[offsets[i]:offsets[i + 1]] = (
                emb.to(save_dtype).cpu().numpy().astype(np_dtype, copy=False)
            )

    mmap_out.flush()
    del mmap_out
    os.replace(tmp_bin, bin_path)

    _write_meta(
        meta_path,
        names=names,
        offsets=offsets,
        hidden_dim=hidden_dim,
        cache_dtype=cache_dtype,
        cache_key=cache_key,
    )

    if skipped_uids:
        logger.warning(
            "SaProt cache wrote zero embeddings for {} proteins (foldseek/embed failure): {}",
            len(skipped_uids), skipped_uids[:5],
        )

    logger.success(
        "Wrote SaProt mmap cache: {} proteins, {} rows, {:.2f} GiB at {}",
        len(unique), total_rows, bin_path.stat().st_size / (1024 ** 3), bin_path,
    )
    return mmap_dir

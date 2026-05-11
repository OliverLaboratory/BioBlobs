"""Offline SaProt residue embedding cache utilities.

SaProt is a structure-aware protein language model that takes Foldseek 3Di
structural alphabet tokens interleaved with amino acids.  This module handles:
  1. Converting PDB structures to structure-aware (SA) sequences via Foldseek.
  2. Embedding SA sequences with SaProt (HuggingFace ``EsmModel``).
  3. Caching per-residue embeddings as ``.pt`` files for downstream use.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
from loguru import logger
from tqdm import tqdm

from .esm2_cache import (
    CACHE_DTYPES,
    _resolve_hidden_state_index,
    _write_cache_metadata,
    resolve_cache_dtype,
    stitch_windows,
)


# ── Foldseek utilities ──────────────────────────────────────────────────────


def _find_foldseek_binary(foldseek_bin: str | None) -> str:
    """Resolve the Foldseek binary path.

    Resolution order: explicit path → ``$FOLDSEEK_BIN`` → ``foldseek`` in PATH.
    """
    if foldseek_bin:
        if os.path.isfile(foldseek_bin) and os.access(foldseek_bin, os.X_OK):
            return foldseek_bin
        raise FileNotFoundError(
            f"Foldseek binary not found or not executable: {foldseek_bin}"
        )

    env_bin = os.environ.get("FOLDSEEK_BIN")
    if env_bin:
        if os.path.isfile(env_bin) and os.access(env_bin, os.X_OK):
            return env_bin
        raise FileNotFoundError(
            f"$FOLDSEEK_BIN is set to {env_bin!r} but the file is not found or "
            "not executable"
        )

    which_bin = shutil.which("foldseek")
    if which_bin:
        return which_bin

    raise FileNotFoundError(
        "Foldseek binary not found. Install Foldseek and ensure it is in your "
        "PATH, or set the $FOLDSEEK_BIN environment variable, or specify "
        "encoders.foldseek_bin in the config.\n"
        "  Install via conda:  conda install -c conda-forge -c bioconda foldseek\n"
        "  Or download from:   https://github.com/steineggerlab/foldseek"
    )


def get_struc_seq(
    foldseek_bin: str,
    pdb_path: str,
    chains: list[str] | None = None,
    process_id: int = 0,
) -> dict[str, tuple[str, str, str]]:
    """Run Foldseek to extract 3Di structural alphabet sequences from a PDB file.

    Adapted from SaProt ``utils/foldseek_util.get_struc_seq``.

    Returns:
        ``{chain_id: (aa_seq, struc_seq, combined_sa_seq)}`` where
        ``combined_sa_seq`` interleaves amino acids with lowercase 3Di tokens
        (e.g. ``"MdKpVq"``).
    """
    pdb_path = str(pdb_path)
    tmp_dir = tempfile.mkdtemp(prefix=f"saprot_foldseek_{process_id}_")
    tmp_tsv = os.path.join(tmp_dir, "output.tsv")

    try:
        cmd = [
            foldseek_bin,
            "structureto3didescriptor",
            "-v", "0",
            "--threads", "1",
            "--chain-name-mode", "1",
            pdb_path,
            tmp_tsv,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"Foldseek failed on {pdb_path} (exit code {result.returncode}):\n"
                f"  stderr: {result.stderr.strip()}"
            )

        pdb_basename = os.path.basename(pdb_path)
        seq_dict: dict[str, tuple[str, str, str]] = {}

        with open(tmp_tsv, "r") as fh:
            for line in fh:
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                desc, aa_seq, struc_seq = parts[0], parts[1], parts[2].strip()

                # Parse chain ID from descriptor: "{filename}_{chain}" format.
                name_chain = desc.split(" ")[0]
                chain = name_chain.replace(pdb_basename, "").split("_")[-1]

                if chains is not None and chain not in chains:
                    continue

                if chain not in seq_dict:
                    # Ensure aa_seq and struc_seq have equal length.
                    # Foldseek can occasionally produce mismatched lengths.
                    if len(aa_seq) != len(struc_seq):
                        min_len = min(len(aa_seq), len(struc_seq))
                        aa_seq = aa_seq[:min_len]
                        struc_seq = struc_seq[:min_len]
                    combined = "".join(
                        a + b.lower() for a, b in zip(aa_seq, struc_seq)
                    )
                    seq_dict[chain] = (aa_seq, struc_seq, combined)

        return seq_dict

    finally:
        # Foldseek creates the output .tsv and a .dbtype sidecar file.
        for suffix in ("", ".dbtype"):
            path = tmp_tsv + suffix
            if os.path.exists(path):
                os.remove(path)
        if os.path.isdir(tmp_dir):
            os.rmdir(tmp_dir)


# ── SA sequence windowing ───────────────────────────────────────────────────


def window_sa_sequence(
    sa_seq: str,
    window_size: int = 1022,
    window_overlap: int = 128,
) -> List[Tuple[int, int, str]]:
    """Split an SA sequence into overlapping windows in residue space.

    Each residue occupies 2 characters in the SA sequence (AA + 3Di token).
    ``window_size`` and ``window_overlap`` are in **residue count**, matching
    the ESM2 convention.

    Returns:
        List of ``(start_residue, end_residue, sa_window_str)`` tuples.
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if window_overlap < 0:
        raise ValueError("window_overlap must be non-negative")
    if window_overlap >= window_size:
        raise ValueError("window_overlap must be smaller than window_size")
    if len(sa_seq) % 2 != 0:
        raise ValueError(
            f"SA sequence length must be even (got {len(sa_seq)}). "
            "Each residue should be a 2-char bigram."
        )

    if not sa_seq:
        return []

    num_residues = len(sa_seq) // 2
    windows: list[tuple[int, int, str]] = []
    start = 0

    while True:
        end = min(start + window_size, num_residues)
        char_start = start * 2
        char_end = end * 2
        windows.append((start, end, sa_seq[char_start:char_end]))
        if end >= num_residues:
            break
        start = end - window_overlap

    return windows


# ── Model loading & embedding ───────────────────────────────────────────────


def _load_saprot_model_and_tokenizer(model_name: str, device: torch.device):
    """Load SaProt model and tokenizer from HuggingFace."""
    try:
        from transformers import EsmModel, EsmTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "transformers is required for SaProt static embedding precomputation"
        ) from exc

    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def _run_saprot_window_batch(
    sa_windows: Sequence[str],
    tokenizer,
    model,
    repr_layer: int,
    device: torch.device,
) -> List[torch.Tensor]:
    """Embed a batch of SA-sequence windows and return per-window residue tensors.

    Each SaProt token is a bigram (AA + 3Di), so the residue count for a window
    is ``len(sa_window) // 2``.  Hidden state extraction strips the CLS token
    (position 0) and takes the next ``residue_count`` positions.
    """
    if not sa_windows:
        return []

    tokens = tokenizer(
        list(sa_windows),
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    tokens = {key: value.to(device) for key, value in tokens.items()}

    with torch.no_grad():
        outputs = model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            output_hidden_states=True,
        )

    hidden_states = outputs.hidden_states
    hidden_state_index = _resolve_hidden_state_index(hidden_states, model, repr_layer)
    hidden = hidden_states[hidden_state_index]

    # Use attention mask to determine actual content token count per row.
    # This is more robust than assuming len(sa_window) // 2, since the
    # tokenizer may produce a different number of tokens for some bigrams.
    attention_mask = tokens["attention_mask"]

    embeddings = []
    for row_idx, sa_window in enumerate(sa_windows):
        # Actual attended positions minus CLS and EOS special tokens.
        actual_tokens = int(attention_mask[row_idx].sum().item()) - 2
        expected_tokens = len(sa_window) // 2
        if actual_tokens != expected_tokens:
            logger.debug(
                "SaProt tokenizer produced {} tokens for window with {} expected "
                "residues (len={} chars). Using actual count.",
                actual_tokens,
                expected_tokens,
                len(sa_window),
            )
        residue_embedding = hidden[row_idx, 1 : actual_tokens + 1, :].detach().cpu()
        embeddings.append(residue_embedding)
    return embeddings


def embed_sequence_with_saprot(
    sa_sequence: str,
    tokenizer,
    model,
    repr_layer: int,
    device: torch.device,
    window_size: int,
    window_overlap: int,
    max_batch_tokens: int,
) -> torch.Tensor:
    """Embed a full SA sequence with SaProt, using windowing when necessary.

    Returns:
        Tensor of shape ``[num_residues, hidden_dim]``.
    """
    windows = window_sa_sequence(
        sa_sequence,
        window_size=window_size,
        window_overlap=window_overlap,
    )
    num_residues = len(sa_sequence) // 2

    if not windows:
        output_dim = getattr(model.config, "hidden_size", 0)
        return torch.empty((0, output_dim), dtype=torch.float32)

    if max_batch_tokens <= 0:
        raise ValueError("max_batch_tokens must be positive")

    window_embeddings: list[torch.Tensor] = []
    batch_windows: list[tuple[int, int, str]] = []
    batch_residues = 0

    for window in windows:
        _, _, sa_window_str = window
        window_residues = len(sa_window_str) // 2
        if batch_windows and batch_residues + window_residues > max_batch_tokens:
            window_embeddings.extend(
                _run_saprot_window_batch(
                    [item[2] for item in batch_windows],
                    tokenizer=tokenizer,
                    model=model,
                    repr_layer=repr_layer,
                    device=device,
                )
            )
            batch_windows = []
            batch_residues = 0

        batch_windows.append(window)
        batch_residues += window_residues

    if batch_windows:
        window_embeddings.extend(
            _run_saprot_window_batch(
                [item[2] for item in batch_windows],
                tokenizer=tokenizer,
                model=model,
                repr_layer=repr_layer,
                device=device,
            )
        )

    # The SaProt tokenizer may produce slightly fewer tokens than
    # len(sa_window) // 2 for some proteins.  Pad or trim each window
    # embedding so it matches the window bounds expected by stitch_windows.
    adjusted_embeddings = []
    for emb, (start, end, _) in zip(window_embeddings, windows):
        expected_len = end - start
        if emb.size(0) != expected_len:
            if emb.size(0) < expected_len:
                pad = torch.zeros(
                    expected_len - emb.size(0), emb.size(1), dtype=emb.dtype
                )
                emb = torch.cat([emb, pad], dim=0)
            else:
                emb = emb[:expected_len]
        adjusted_embeddings.append(emb)

    stitched = stitch_windows(adjusted_embeddings, windows, num_residues)
    return stitched.float()


# ── Cache path helpers ──────────────────────────────────────────────────────


def _canonicalize_saprot_model_name(model_name: str) -> str:
    return model_name.split("/")[-1]


def build_saprot_cache_key(
    model_name: str,
    repr_layer: int,
    window_size: int,
    window_overlap: int,
    cache_dtype: str,
) -> str:
    """Create a deterministic cache key from the embedding configuration."""
    canonical_name = _canonicalize_saprot_model_name(model_name)
    metadata = {
        "model_name": canonical_name,
        "repr_layer": int(repr_layer),
        "window_size": int(window_size),
        "window_overlap": int(window_overlap),
        "cache_dtype": cache_dtype,
    }
    payload = json.dumps(metadata, sort_keys=True).encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:12]
    return (
        f"{canonical_name}_layer{repr_layer}_"
        f"w{window_size}_o{window_overlap}_{cache_dtype}_{digest}"
    )


def _get_saprot_cache_root(cfg) -> Path:
    """Resolve the base cache directory for static SaProt embeddings.

    Mirrors ``get_esm2_cache_root`` but reads ``cfg.encoders.cache_dir``
    (which defaults to ``saprot_static_cache``).
    """
    dataset_name = cfg.datasets.get("dataset_name", "ec")

    if dataset_name == "venusx_site_fragment":
        prepared_root = (
            Path(cfg.datasets.data_dir)
            / dataset_name
            / cfg.datasets.target
            / cfg.datasets.split_strategy
        )
    else:
        from .proteinshake.task_registry import get_prepared_root_name

        try:
            go_branch = cfg.datasets.get("go_branch") if dataset_name == "go" else None
            prepared_root = Path(cfg.datasets.data_dir) / get_prepared_root_name(dataset_name, go_branch)
        except ValueError:
            prepared_root = Path(cfg.datasets.data_dir) / dataset_name

    return prepared_root / cfg.encoders.cache_dir


def _get_saprot_cache_dir(cfg) -> Path:
    """Resolve the shared embedding cache directory for static SaProt embeddings."""
    cache_key = build_saprot_cache_key(
        model_name=cfg.encoders.model_name,
        repr_layer=cfg.encoders.repr_layer,
        window_size=cfg.encoders.window_size,
        window_overlap=cfg.encoders.window_overlap,
        cache_dtype=cfg.encoders.cache_dtype,
    )
    return _get_saprot_cache_root(cfg) / cache_key / "embeddings"


def _build_saprot_cache_path(split_cache_dir: Path, protein_name: str) -> Path:
    safe_name = protein_name.replace("/", "_")
    return split_cache_dir / f"{safe_name}.pt"


def _resolve_pdb_dir(cfg) -> Path:
    """Resolve the PDB directory from dataset config.

    PDB files are stored at ``{prepared_root}/pdb/`` during dataset preparation.
    """
    dataset_name = cfg.datasets.get("dataset_name", "ec")

    if dataset_name == "venusx_site_fragment":
        prepared_root = (
            Path(cfg.datasets.data_dir)
            / dataset_name
            / cfg.datasets.target
            / cfg.datasets.split_strategy
        )
    else:
        from .proteinshake.task_registry import get_prepared_root_name

        try:
            go_branch = cfg.datasets.get("go_branch") if dataset_name == "go" else None
            prepared_root = Path(cfg.datasets.data_dir) / get_prepared_root_name(dataset_name, go_branch)
        except ValueError:
            prepared_root = Path(cfg.datasets.data_dir) / dataset_name

    return prepared_root / "pdb"


def _resolve_precompute_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        logger.warning(
            "CUDA requested for SaProt precompute but unavailable; "
            "falling back to CPU"
        )
        return torch.device("cpu")
    return torch.device(device_name)


# ── Main cache preparation ──────────────────────────────────────────────────


def _get_saprot_cache_metadata(cfg) -> dict:
    return {
        "model_name": _canonicalize_saprot_model_name(cfg.encoders.model_name),
        "repr_layer": int(cfg.encoders.repr_layer),
        "window_size": int(cfg.encoders.window_size),
        "window_overlap": int(cfg.encoders.window_overlap),
        "max_batch_tokens": int(cfg.encoders.max_batch_tokens),
        "cache_dtype": cfg.encoders.cache_dtype,
        "output_dim": int(cfg.encoders.output_dim),
    }


def _iter_missing_structures(
    structures: Sequence[dict],
    overwrite_cache: bool,
) -> Iterable[dict]:
    seen_cache_paths: set[Path] = set()
    for structure in structures:
        cache_path = Path(structure["saprot_cache_path"])
        if cache_path in seen_cache_paths:
            continue
        seen_cache_paths.add(cache_path)
        if overwrite_cache or not cache_path.exists():
            yield structure


def prepare_saprot_embedding_cache(
    structures: Sequence[dict],
    cfg,
    split_name: str,
) -> Path:
    """Attach cache paths and precompute missing SaProt residue embeddings.

    For each structure dict, this function:
      1. Attaches ``structure["saprot_cache_path"]``.
      2. If the cache file is missing and ``precompute_missing=True``:
         a. Finds the PDB file on disk.
         b. Runs Foldseek to get the 3Di sequence.
         c. Builds the structure-aware (SA) sequence.
         d. Embeds with SaProt and saves the ``.pt`` cache file.
    """
    cache_dir = _get_saprot_cache_dir(cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not structures:
        return cache_dir

    # Attach cache paths.
    for structure in structures:
        structure["saprot_cache_path"] = str(
            _build_saprot_cache_path(
                cache_dir,
                structure.get("protein_uid", structure["name"]),
            )
        )

    metadata_root = cache_dir.parent
    metadata = _get_saprot_cache_metadata(cfg)
    _write_cache_metadata(metadata_root, metadata)

    missing_structures = list(
        _iter_missing_structures(
            structures,
            overwrite_cache=cfg.encoders.overwrite_cache,
        )
    )
    if not missing_structures:
        logger.info(
            "Reusing static SaProt cache for {} split from {}",
            split_name,
            cache_dir,
        )
        return cache_dir

    if not cfg.encoders.precompute_missing:
        missing_names = [s["name"] for s in missing_structures[:5]]
        raise FileNotFoundError(
            "Missing static SaProt cache files while precompute_missing=false. "
            f"First missing proteins: {missing_names}"
        )

    # Resolve dependencies.
    foldseek_bin = _find_foldseek_binary(
        getattr(cfg.encoders, "foldseek_bin", None)
    )
    pdb_dir = _resolve_pdb_dir(cfg)
    device = _resolve_precompute_device(cfg.encoders.precompute_device)
    tokenizer, model = _load_saprot_model_and_tokenizer(
        cfg.encoders.model_name, device
    )
    save_dtype = resolve_cache_dtype(cfg.encoders.cache_dtype)

    logger.info(
        "Precomputing static SaProt residue embeddings for {} {} proteins into {}",
        split_name,
        len(missing_structures),
        cache_dir,
    )

    for structure in tqdm(
        missing_structures,
        desc=f"Precomputing {split_name} SaProt",
        leave=False,
    ):
        protein_name = structure.get("protein_uid", structure["name"])
        pdb_path = pdb_dir / f"{protein_name}.pdb"
        if not pdb_path.exists():
            raise FileNotFoundError(
                f"PDB file not found for protein {protein_name!r}: {pdb_path}"
            )

        # Run Foldseek to get the SA sequence.
        seq_dict = get_struc_seq(
            foldseek_bin,
            str(pdb_path),
            process_id=os.getpid(),
        )
        if not seq_dict:
            raise RuntimeError(
                f"Foldseek returned no chains for {pdb_path}. "
                "The PDB file may be empty or malformed."
            )

        # Pick the first chain (ProteinShake structures are single-chain).
        chain_id = next(iter(seq_dict))
        aa_seq, _struc_seq, sa_sequence = seq_dict[chain_id]

        # Validate that the SA sequence matches the stored sequence length.
        # Use len(sa_sequence) // 2 as the ground truth for residue count
        # (aa_seq may differ from struc_seq length; get_struc_seq truncates).
        expected_len = len(structure["seq"])
        sa_residue_len = len(sa_sequence) // 2
        if sa_residue_len != expected_len:
            logger.warning(
                "SA sequence residue count ({}) != stored sequence length ({}) "
                "for {}. Truncating/padding SA sequence to match.",
                sa_residue_len,
                expected_len,
                protein_name,
            )
            if sa_residue_len > expected_len:
                sa_sequence = sa_sequence[: expected_len * 2]
            else:
                # Pad with masked residues (AA='X', 3Di='d' — a common 3Di token).
                pad_count = expected_len - sa_residue_len
                sa_sequence = sa_sequence + "Xd" * pad_count

        # Embed the SA sequence.
        embedding = embed_sequence_with_saprot(
            sa_sequence=sa_sequence,
            tokenizer=tokenizer,
            model=model,
            repr_layer=cfg.encoders.repr_layer,
            device=device,
            window_size=cfg.encoders.window_size,
            window_overlap=cfg.encoders.window_overlap,
            max_batch_tokens=cfg.encoders.max_batch_tokens,
        )

        if embedding.size(0) != expected_len:
            raise AssertionError(
                f"SaProt cache length mismatch for {protein_name}: "
                f"expected {expected_len}, received {embedding.size(0)}"
            )

        cache_path = Path(structure["saprot_cache_path"])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embedding.to(dtype=save_dtype, device="cpu"), cache_path)

    return cache_dir

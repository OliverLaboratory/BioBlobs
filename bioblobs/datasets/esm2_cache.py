"""Offline ESM2 residue embedding cache utilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
from loguru import logger
from tqdm import tqdm


ESM2_MODEL_OUTPUT_DIMS = {
    "esm2_t6_8M_UR50D": 320,
    "esm2_t12_35M_UR50D": 480,
    "esm2_t30_150M_UR50D": 640,
    "esm2_t33_650M_UR50D": 1280,
    "esm2_t36_3B_UR50D": 2560,
    "esm2_t48_15B_UR50D": 5120,
}

VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYX")
CACHE_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def canonicalize_esm2_model_name(model_name: str) -> str:
    """Normalize model names to the short HuggingFace checkpoint ID."""
    return model_name.split("/")[-1]


def infer_esm2_output_dim(model_name: str, output_dim: int | None = None) -> int:
    """Infer the ESM2 output dimension from the configured checkpoint."""
    if output_dim is not None:
        return int(output_dim)

    canonical_name = canonicalize_esm2_model_name(model_name)
    if canonical_name not in ESM2_MODEL_OUTPUT_DIMS:
        raise ValueError(
            f"Unsupported ESM2 checkpoint {model_name!r}. "
            f"Supported checkpoints: {sorted(ESM2_MODEL_OUTPUT_DIMS)}"
        )
    return ESM2_MODEL_OUTPUT_DIMS[canonical_name]


def resolve_cache_dtype(cache_dtype: str) -> torch.dtype:
    """Map a string cache dtype to a torch dtype."""
    if cache_dtype not in CACHE_DTYPES:
        raise ValueError(
            f"Unsupported cache dtype {cache_dtype!r}. "
            f"Supported values: {sorted(CACHE_DTYPES)}"
        )
    return CACHE_DTYPES[cache_dtype]


def sanitize_sequence(sequence: str) -> str:
    """Replace non-standard amino acids with X for ESM2 tokenization."""
    return "".join(aa if aa in VALID_AMINO_ACIDS else "X" for aa in sequence)


def build_esm2_cache_key(
    model_name: str,
    repr_layer: int,
    window_size: int,
    window_overlap: int,
    cache_dtype: str,
) -> str:
    """Create a deterministic cache key from the embedding configuration."""
    canonical_name = canonicalize_esm2_model_name(model_name)
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


def window_sequence(
    sequence: str,
    window_size: int = 1022,
    window_overlap: int = 128,
) -> List[Tuple[int, int, str]]:
    """Split a sequence into overlapping windows suitable for ESM2."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if window_overlap < 0:
        raise ValueError("window_overlap must be non-negative")
    if window_overlap >= window_size:
        raise ValueError("window_overlap must be smaller than window_size")

    if not sequence:
        return []

    seq_len = len(sequence)
    windows = []
    start = 0
    while True:
        end = min(start + window_size, seq_len)
        windows.append((start, end, sequence[start:end]))
        if end >= seq_len:
            break
        start = end - window_overlap

    return windows


def stitch_windows(
    window_embeddings: Sequence[torch.Tensor],
    windows: Sequence[Tuple[int, int, str]],
    seq_len: int,
) -> torch.Tensor:
    """Average overlapping window embeddings back into a full-length tensor."""
    if len(window_embeddings) != len(windows):
        raise ValueError("window_embeddings and windows must have the same length")

    if seq_len == 0:
        if window_embeddings:
            raise ValueError("Non-empty embeddings provided for an empty sequence")
        return torch.empty((0, 0), dtype=torch.float32)

    if not window_embeddings:
        raise ValueError("window_embeddings cannot be empty when seq_len > 0")

    embed_dim = int(window_embeddings[0].size(-1))
    accum = torch.zeros((seq_len, embed_dim), dtype=window_embeddings[0].dtype)
    counts = torch.zeros((seq_len, 1), dtype=window_embeddings[0].dtype)

    for embedding, (start, end, _) in zip(window_embeddings, windows):
        expected_len = end - start
        if embedding.ndim != 2:
            raise ValueError("Each window embedding must be a 2D tensor")
        if embedding.size(0) != expected_len:
            raise ValueError(
                f"Window embedding length mismatch: expected {expected_len}, "
                f"received {embedding.size(0)}"
            )
        accum[start:end] += embedding.cpu()
        counts[start:end] += 1

    if not torch.all(counts > 0):
        raise ValueError("Window stitching left uncovered residue positions")

    return accum / counts


def get_esm2_cache_root(cfg) -> Path:
    """Resolve the base cache directory for static ESM2 embeddings."""
    dataset_name = cfg.datasets.get("dataset_name", "ec")

    if dataset_name == "venusx_site_fragment":
        prepared_root = (
            Path(cfg.datasets.data_dir)
            / dataset_name
            / cfg.datasets.target
            / cfg.datasets.split_strategy
        )
    else:
        # All ProteinShake datasets (ec, go, go_mf, go_cc, pfam, scop_fam, …)
        from .proteinshake.task_registry import get_prepared_root_name

        try:
            go_branch = cfg.datasets.get("go_branch") if dataset_name == "go" else None
            prepared_root = Path(cfg.datasets.data_dir) / get_prepared_root_name(dataset_name, go_branch)
        except ValueError:
            prepared_root = Path(cfg.datasets.data_dir) / dataset_name

    return prepared_root / cfg.encoders.cache_dir


def get_esm2_cache_dir(cfg) -> Path:
    """Resolve the shared embedding cache directory for static ESM2 embeddings.

    All proteins are stored in a single flat directory regardless of their
    train/val/test assignment, since embeddings depend only on the sequence.
    """
    cache_key = build_esm2_cache_key(
        model_name=cfg.encoders.model_name,
        repr_layer=cfg.encoders.repr_layer,
        window_size=cfg.encoders.window_size,
        window_overlap=cfg.encoders.window_overlap,
        cache_dtype=cfg.encoders.cache_dtype,
    )
    return get_esm2_cache_root(cfg) / cache_key / "embeddings"


def get_esm2_cache_metadata(cfg) -> dict:
    """Build a metadata dict describing the cache contents."""
    return {
        "model_name": canonicalize_esm2_model_name(cfg.encoders.model_name),
        "repr_layer": int(cfg.encoders.repr_layer),
        "window_size": int(cfg.encoders.window_size),
        "window_overlap": int(cfg.encoders.window_overlap),
        "max_batch_tokens": int(cfg.encoders.max_batch_tokens),
        "cache_dtype": cfg.encoders.cache_dtype,
        "output_dim": infer_esm2_output_dim(
            cfg.encoders.model_name,
            getattr(cfg.encoders, "output_dim", None),
        ),
    }


def build_esm2_cache_path(split_cache_dir: Path, protein_name: str) -> Path:
    """Build the per-protein cache path."""
    safe_name = protein_name.replace("/", "_")
    return split_cache_dir / f"{safe_name}.pt"


def _write_cache_metadata(cache_root: Path, metadata: dict) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_root / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def _resolve_precompute_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested for ESM2 precompute but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_name)


def _load_esm2_model_and_tokenizer(model_name: str, device: torch.device):
    """Load tokenizer and model lazily so tests can run without transformers."""
    try:
        from transformers import EsmModel, EsmTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "transformers is required for ESM2 static embedding precomputation"
        ) from exc

    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def _resolve_hidden_state_index(
    hidden_states: Sequence[torch.Tensor],
    model,
    repr_layer: int,
) -> int:
    """Resolve repr_layer into a valid hidden-state index."""
    if repr_layer == -1:
        return len(hidden_states) - 1

    num_hidden_layers = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    if num_hidden_layers is None:
        num_hidden_layers = len(hidden_states) - 1

    if repr_layer < 1 or repr_layer > num_hidden_layers:
        raise ValueError(
            f"repr_layer must be in [1, {num_hidden_layers}] or -1, got {repr_layer}"
        )
    return repr_layer


def _run_esm2_window_batch(
    sequences: Sequence[str],
    tokenizer,
    model,
    repr_layer: int,
    device: torch.device,
) -> List[torch.Tensor]:
    """Embed a batch of sequence windows and return per-window residue tensors."""
    if not sequences:
        return []

    tokens = tokenizer(
        list(sequences),
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

    embeddings = []
    for row_idx, sequence in enumerate(sequences):
        residue_count = len(sequence)
        residue_embedding = hidden[row_idx, 1 : residue_count + 1, :].detach().cpu()
        embeddings.append(residue_embedding)
    return embeddings


def embed_sequence_with_esm2(
    sequence: str,
    tokenizer,
    model,
    repr_layer: int,
    device: torch.device,
    window_size: int,
    window_overlap: int,
    max_batch_tokens: int,
) -> torch.Tensor:
    """Embed a full sequence with ESM2, using windowing when necessary."""
    sanitized_sequence = sanitize_sequence(sequence)
    windows = window_sequence(
        sanitized_sequence,
        window_size=window_size,
        window_overlap=window_overlap,
    )
    if not windows:
        output_dim = getattr(model.config, "hidden_size", 0)
        return torch.empty((0, output_dim), dtype=torch.float32)

    if max_batch_tokens <= 0:
        raise ValueError("max_batch_tokens must be positive")

    window_embeddings = []
    batch_windows = []
    batch_tokens = 0

    for window in windows:
        _, _, window_sequence_str = window
        window_len = len(window_sequence_str)
        if batch_windows and batch_tokens + window_len > max_batch_tokens:
            window_embeddings.extend(
                _run_esm2_window_batch(
                    [item[2] for item in batch_windows],
                    tokenizer=tokenizer,
                    model=model,
                    repr_layer=repr_layer,
                    device=device,
                )
            )
            batch_windows = []
            batch_tokens = 0

        batch_windows.append(window)
        batch_tokens += window_len

    if batch_windows:
        window_embeddings.extend(
            _run_esm2_window_batch(
                [item[2] for item in batch_windows],
                tokenizer=tokenizer,
                model=model,
                repr_layer=repr_layer,
                device=device,
            )
        )

    stitched = stitch_windows(window_embeddings, windows, len(sanitized_sequence))
    return stitched.float()


def _iter_missing_structures(
    structures: Sequence[dict],
    overwrite_cache: bool,
) -> Iterable[dict]:
    seen_cache_paths: set[Path] = set()
    for structure in structures:
        cache_path = Path(structure["esm2_cache_path"])
        if cache_path in seen_cache_paths:
            continue
        seen_cache_paths.add(cache_path)
        if overwrite_cache or not cache_path.exists():
            yield structure


def prepare_esm2_embedding_cache(
    structures: Sequence[dict],
    cfg,
    split_name: str,
) -> Path:
    """Attach cache paths and precompute missing ESM2 residue embeddings.

    Embeddings are stored in a single flat directory shared across all
    train/val/test partitions.  ``split_name`` is used only for logging.
    """
    cache_dir = get_esm2_cache_dir(cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not structures:
        return cache_dir

    for structure in structures:
        structure["esm2_cache_path"] = str(
            build_esm2_cache_path(
                cache_dir,
                structure.get("protein_uid", structure["name"]),
            )
        )

    metadata_root = cache_dir.parent
    metadata = get_esm2_cache_metadata(cfg)
    _write_cache_metadata(metadata_root, metadata)

    missing_structures = list(
        _iter_missing_structures(
            structures,
            overwrite_cache=cfg.encoders.overwrite_cache,
        )
    )
    if not missing_structures:
        logger.info(
            "Reusing static ESM2 cache for {} split from {}",
            split_name,
            cache_dir,
        )
        return cache_dir

    if not cfg.encoders.precompute_missing:
        missing_names = [structure["name"] for structure in missing_structures[:5]]
        raise FileNotFoundError(
            "Missing static ESM2 cache files while precompute_missing=false. "
            f"First missing proteins: {missing_names}"
        )

    device = _resolve_precompute_device(cfg.encoders.precompute_device)
    tokenizer, model = _load_esm2_model_and_tokenizer(cfg.encoders.model_name, device)
    save_dtype = resolve_cache_dtype(cfg.encoders.cache_dtype)

    logger.info(
        "Precomputing static ESM2 residue embeddings for {} {} proteins into {}",
        split_name,
        len(missing_structures),
        cache_dir,
    )

    for structure in tqdm(
        missing_structures,
        desc=f"Precomputing {split_name} ESM2",
        leave=False,
    ):
        embedding = embed_sequence_with_esm2(
            sequence=structure["seq"],
            tokenizer=tokenizer,
            model=model,
            repr_layer=cfg.encoders.repr_layer,
            device=device,
            window_size=cfg.encoders.window_size,
            window_overlap=cfg.encoders.window_overlap,
            max_batch_tokens=cfg.encoders.max_batch_tokens,
        )
        expected_len = len(structure["seq"])
        if embedding.size(0) != expected_len:
            raise AssertionError(
                f"ESM2 cache length mismatch for {structure['name']}: "
                f"expected {expected_len}, received {embedding.size(0)}"
            )

        cache_path = Path(structure["esm2_cache_path"])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embedding.to(dtype=save_dtype, device="cpu"), cache_path)

    return cache_dir

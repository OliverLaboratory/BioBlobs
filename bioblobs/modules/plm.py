"""ESM2 and SaProt residue-embedding computation (Section 3.1 encoder).

Computes the per-residue protein-language-model (PLM) embeddings that
``preprocessing/prepare_plm_cache.py`` writes to an on-disk cache (see
``bioblobs.data.plm_cache``); this module does not read or write a cache itself.

Both encoders are ``transformers`` ``EsmModel``/``EsmTokenizer`` checkpoints run with no
gradient. Long sequences use the sliding-window scheme behind every reported result:
``window=1022``, ``overlap=128``, averaging embeddings over overlapping windows
(``stitch_windows``). ``mode="truncate"`` (embed only the first ``window`` residues, no
stitching) is also offered for completeness -- **no reported result uses it**. See
``preprocessing/prepare_plm_cache.py`` for how the M-CSA caches (``feat_cache.pt`` vs
``feat_cache_maxlen1500.pt``) differ: both use ``mode="window"``; they only differ in a
chain-length admission filter applied before embedding (1022 vs 1500), not in the algorithm.

SaProt additionally needs Foldseek to turn a PDB structure into 3Di structural-alphabet
tokens, interleaved with the amino-acid sequence into a structure-aware (SA) bigram string
(``"MdKpVq..."``); windowing then operates in residue space (2 chars/residue).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from loguru import logger

from bioblobs.data.foldseek import build_sa_sequence, resolve_foldseek_binary

ESM2_OUTPUT_DIMS: dict[str, int] = {
    "esm2_t6_8M_UR50D": 320,
    "esm2_t12_35M_UR50D": 480,
    "esm2_t30_150M_UR50D": 640,
    "esm2_t33_650M_UR50D": 1280,
    "esm2_t36_3B_UR50D": 2560,
    "esm2_t48_15B_UR50D": 5120,
}

# ESM2's 20 standard amino acids plus the tokenizer's "unknown residue" token.
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYX")

# (model_name, device, local_files_only) -> (tokenizer, model), so repeated calls from a CLI
# loop over many proteins pay the HF load cost once.
_MODEL_CACHE: dict[tuple[str, str, bool], tuple] = {}


def canonical_model_name(model_name: str) -> str:
    """Short checkpoint id from a possibly-namespaced HuggingFace model name."""
    return model_name.split("/")[-1]


def infer_esm2_output_dim(model_name: str, output_dim: int | None = None) -> int:
    """Resolve an ESM2 checkpoint's hidden size D (``output_dim`` overrides when given)."""
    if output_dim is not None:
        return int(output_dim)
    name = canonical_model_name(model_name)
    if name not in ESM2_OUTPUT_DIMS:
        raise ValueError(f"Unknown ESM2 checkpoint {model_name!r}; pass output_dim explicitly.")
    return ESM2_OUTPUT_DIMS[name]


def sanitize_sequence(sequence: str) -> str:
    """Replace non-standard amino acids with ``X`` (the ESM2 tokenizer's unknown-residue token)."""
    return "".join(c if c in VALID_AMINO_ACIDS else "X" for c in sequence)


def window_spans(num_residues: int, window: int, overlap: int) -> list[tuple[int, int]]:
    """Overlapping ``[start, end)`` residue spans covering a sequence of ``num_residues``.

    A single ``(0, num_residues)`` span is returned when the sequence fits in one window.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= window:
        raise ValueError("overlap must be smaller than window")
    if num_residues <= 0:
        return []
    spans, start = [], 0
    while True:
        end = min(start + window, num_residues)
        spans.append((start, end))
        if end >= num_residues:
            return spans
        start = end - overlap


def stitch_windows(
    window_embeddings: Sequence[torch.Tensor],
    spans: Sequence[tuple[int, int]],
    num_residues: int,
) -> torch.Tensor:
    """Average overlapping per-window embeddings (one ``[end - start, D]`` tensor per span in
    ``spans``) back into one ``[num_residues, D]`` float32 tensor.
    """
    if len(window_embeddings) != len(spans):
        raise ValueError("window_embeddings and spans must have the same length")
    if not window_embeddings:
        raise ValueError("stitch_windows requires at least one window")
    dim = int(window_embeddings[0].size(-1))
    accum = torch.zeros((num_residues, dim), dtype=torch.float32)
    counts = torch.zeros((num_residues, 1), dtype=torch.float32)
    for embedding, (start, end) in zip(window_embeddings, spans, strict=True):
        if embedding.size(0) != end - start:
            raise ValueError(
                f"window covers {end - start} residues, embedding has {embedding.size(0)}"
            )
        accum[start:end] += embedding.float().cpu()
        counts[start:end] += 1
    if not torch.all(counts > 0):
        raise ValueError("stitch_windows left uncovered residue positions")
    return accum / counts


def _load_esm_family_model(model_name: str, device: torch.device, local_files_only: bool) -> tuple:
    """Load (and cache) an ``EsmTokenizer``/``EsmModel`` pair, offline by default."""
    cache_key = (model_name, str(device), local_files_only)
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    from transformers import EsmModel, EsmTokenizer

    try:
        tokenizer = EsmTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        model = EsmModel.from_pretrained(model_name, local_files_only=local_files_only)
    except OSError as exc:
        raise FileNotFoundError(
            f"{model_name!r} is not in the local HuggingFace cache and local_files_only=True "
            "forbids downloading it. Pre-fetch the checkpoint or pass local_files_only=False."
        ) from exc
    model.to(device).eval()
    _MODEL_CACHE[cache_key] = (tokenizer, model)
    logger.info("Loaded {} on {} (hidden_size={})", model_name, device, model.config.hidden_size)
    return tokenizer, model


def _resolve_hidden_index(num_hidden_states: int, num_hidden_layers: int, repr_layer: int) -> int:
    """Map a ``repr_layer`` config value (``-1`` = last) to a ``hidden_states`` index."""
    if repr_layer == -1:
        return num_hidden_states - 1
    if repr_layer < 1 or repr_layer > num_hidden_layers:
        raise ValueError(f"repr_layer must be in [1, {num_hidden_layers}] or -1, got {repr_layer}")
    return repr_layer


def _run_token_batch(
    texts: Sequence[str],
    residue_counts: Sequence[int],
    tokenizer,
    model,
    repr_layer: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Embed one batch of (window) strings; strip CLS/EOS via the attention mask.

    Uses the attention-mask content length rather than assuming ``len(text)`` tokens, since the
    SaProt tokenizer can occasionally emit a different token count for a bigram string (ported
    from the dev SaProt cache builder, which found this necessary for robustness).
    """
    tokens = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=False)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        outputs = model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            output_hidden_states=True,
        )
    hidden_states = outputs.hidden_states
    num_hidden_layers = int(getattr(model.config, "num_hidden_layers", len(hidden_states) - 1))
    hidden = hidden_states[_resolve_hidden_index(len(hidden_states), num_hidden_layers, repr_layer)]
    attention_mask = tokens["attention_mask"]

    embeddings = []
    for row, expected in enumerate(residue_counts):
        actual = int(attention_mask[row].sum().item()) - 2  # minus CLS and EOS
        if actual != expected:
            logger.debug(
                "tokenizer produced {} residues, expected {}; using actual", actual, expected
            )
        embeddings.append(hidden[row, 1 : actual + 1, :].detach().cpu())
    return embeddings


def _embed_windows(
    window_texts: list[str],
    residue_counts: list[int],
    tokenizer,
    model,
    repr_layer: int,
    device: torch.device,
    max_batch_tokens: int,
) -> list[torch.Tensor]:
    """Batch windows up to ``max_batch_tokens`` residues per forward pass."""
    if max_batch_tokens <= 0:
        raise ValueError("max_batch_tokens must be positive")
    embeddings: list[torch.Tensor] = []
    batch_texts: list[str] = []
    batch_counts: list[int] = []
    batch_tokens = 0
    for text, count in zip(window_texts, residue_counts, strict=True):
        if batch_texts and batch_tokens + count > max_batch_tokens:
            embeddings.extend(
                _run_token_batch(batch_texts, batch_counts, tokenizer, model, repr_layer, device)
            )
            batch_texts, batch_counts, batch_tokens = [], [], 0
        batch_texts.append(text)
        batch_counts.append(count)
        batch_tokens += count
    if batch_texts:
        embeddings.extend(
            _run_token_batch(batch_texts, batch_counts, tokenizer, model, repr_layer, device)
        )
    return embeddings


def _embed_text(
    text: str,
    *,
    is_bigram: bool,
    tokenizer,
    model,
    repr_layer: int,
    device: torch.device,
    window: int,
    overlap: int,
    mode: str,
    max_batch_tokens: int,
    hidden_dim: int,
) -> torch.Tensor:
    """Embed one (possibly long) sequence, windowing or truncating as configured."""
    num_residues = len(text) // 2 if is_bigram else len(text)
    if num_residues == 0:
        return torch.empty((0, hidden_dim), dtype=torch.float32)

    if mode == "window":
        spans = window_spans(num_residues, window, overlap)
    elif mode == "truncate":
        spans = [(0, min(window, num_residues))]
    else:
        raise ValueError(f"mode must be 'window' or 'truncate', got {mode!r}")

    window_texts = [text[s * 2 : e * 2] if is_bigram else text[s:e] for s, e in spans]
    residue_counts = [e - s for s, e in spans]
    window_embeddings = _embed_windows(
        window_texts, residue_counts, tokenizer, model, repr_layer, device, max_batch_tokens
    )
    if mode == "truncate":
        return window_embeddings[0].float()
    return stitch_windows(window_embeddings, spans, num_residues).float()


def embed_esm2(
    sequences: Sequence[str],
    *,
    model_name: str = "facebook/esm2_t30_150M_UR50D",
    repr_layer: int = -1,
    window: int = 1022,
    overlap: int = 128,
    mode: str = "window",
    max_batch_tokens: int = 8192,
    device: str | torch.device = "cpu",
    local_files_only: bool = True,
) -> list[torch.Tensor]:
    """Embed sequences with ESM2, returning one ``[len(sequence), D]`` float32 tensor each.

    Non-standard residues are replaced with ``X`` (``sanitize_sequence``) before tokenization,
    matching every dev ESM2 cache. ``mode="window"`` (the default, and the only mode any
    reported result uses) splits sequences longer than ``window`` into overlapping windows and
    averages the residue embeddings where windows overlap (``stitch_windows``); ``"truncate"``
    embeds only the first ``window`` residues. ``local_files_only=True`` never downloads.
    """
    device = torch.device(device)
    tokenizer, model = _load_esm_family_model(model_name, device, local_files_only)
    hidden_dim = int(model.config.hidden_size)
    return [
        _embed_text(
            sanitize_sequence(seq),
            is_bigram=False,
            tokenizer=tokenizer,
            model=model,
            repr_layer=repr_layer,
            device=device,
            window=window,
            overlap=overlap,
            mode=mode,
            max_batch_tokens=max_batch_tokens,
            hidden_dim=hidden_dim,
        )
        for seq in sequences
    ]


def embed_saprot(
    sequences: Sequence[str],
    pdb_paths: Sequence[str | Path],
    *,
    chains: Sequence[str | None] | None = None,
    foldseek_bin: str | None = None,
    model_name: str = "westlake-repl/SaProt_650M_PDB",
    repr_layer: int = -1,
    window: int = 1022,
    overlap: int = 128,
    mode: str = "window",
    max_batch_tokens: int = 8192,
    device: str | torch.device = "cpu",
    local_files_only: bool = True,
    pad_token: str = "Xd",
) -> list[torch.Tensor]:
    """Embed sequences with SaProt, one ``[len(sequence), D]`` float32 tensor each.

    ``pdb_paths``/``chains`` (one per sequence) are run through Foldseek to get each protein's
    structure-aware bigram sequence (``build_sa_sequence``, padded/truncated to match
    ``len(sequence)``); that derived sequence, not ``sequences`` itself, is what gets embedded.
    Windowing is identical to ``embed_esm2`` but operates on bigram (AA + 3Di) spans; see its
    docstring for ``mode``. ``foldseek_bin=None`` resolves via ``resolve_foldseek_binary``.
    """
    if len(sequences) != len(pdb_paths):
        raise ValueError("sequences and pdb_paths must have the same length")
    chains = list(chains) if chains is not None else [None] * len(sequences)
    if len(chains) != len(sequences):
        raise ValueError("chains must have the same length as sequences")

    device = torch.device(device)
    resolved_bin = resolve_foldseek_binary(foldseek_bin)
    tokenizer, model = _load_esm_family_model(model_name, device, local_files_only)
    hidden_dim = int(model.config.hidden_size)

    embeddings = []
    for seq, pdb_path, chain in zip(sequences, pdb_paths, chains, strict=True):
        if len(seq) == 0:
            embeddings.append(torch.empty((0, hidden_dim), dtype=torch.float32))
            continue
        sa_seq = build_sa_sequence(
            resolved_bin, pdb_path, len(seq), chain=chain, pad_token=pad_token
        )
        embeddings.append(
            _embed_text(
                sa_seq,
                is_bigram=True,
                tokenizer=tokenizer,
                model=model,
                repr_layer=repr_layer,
                device=device,
                window=window,
                overlap=overlap,
                mode=mode,
                max_batch_tokens=max_batch_tokens,
                hidden_dim=hidden_dim,
            )
        )
    return embeddings

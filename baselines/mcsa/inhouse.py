"""Shared helpers for the in-house M-CSA baselines, which run in this repository's environment.

Every in-house scorer writes ``{id: [score, ...]}`` (``common.write_scores``) in the residue
order of the M-CSA feature file built by ``preprocessing/mcsa/build_features.py``. That is the
order ``evaluate_scores.py`` and ``bioblobs.data.mcsa`` use. Sequences for the ESM2 baselines are
parsed from the PDB files with the same parser as that feature file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[1]
for _path in (REPO, REPO / "preprocessing" / "mcsa", _HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from build_features import chain_backbone  # noqa: E402
from common import write_scores  # noqa: E402,F401

from bioblobs.data.mcsa import DEFAULT_LABELS, catalytic_indices  # noqa: E402,F401

DEFAULT_IDS = REPO / "data" / "mcsa" / "evaluation_ids.txt"
ESM2 = "facebook/esm2_t30_150M_UR50D"


def read_ids(path: str | Path = DEFAULT_IDS) -> list[str]:
    """Ids of a newline-separated file, ignoring blank lines and ``#`` comments."""
    lines = (line.split("#", 1)[0].strip() for line in Path(path).read_text().splitlines())
    return [line for line in lines if line]


def load_features(path: str | Path) -> dict[str, dict]:
    """The M-CSA feature file ``{pdb_chain: {"nf", "ca", "resnums"}}``."""
    return torch.load(path, map_location="cpu", weights_only=False)


def site_labels(entry: dict, resnums: list[int]) -> np.ndarray:
    """0/1 catalytic labels of an M-CSA entry in the given residue order."""
    y = np.zeros(len(resnums), dtype=int)
    y[catalytic_indices(entry, [int(r) for r in resnums])] = 1
    return y


def chain_sequence(pdb_dir: str | Path, entry: dict) -> tuple[str, list[int]] | None:
    """Sequence and residue numbers of an entry's chain, parsed like the feature file.

    None when the PDB file or the chain is missing or cannot be parsed.
    """
    path = Path(pdb_dir) / f"{entry['pdb_id'].lower()}.pdb"
    try:
        parsed = chain_backbone(path, entry["chain_used"]) if path.exists() else None
    except Exception:  # noqa: BLE001 - an unreadable file counts as missing
        parsed = None
    if parsed is None:
        return None
    sequence, _coords, resnums = parsed
    return sequence, resnums


def load_esm2(device: str, eager_attention: bool = False):
    """Tokenizer and ``EsmModel`` of ESM2-T30 (150M), in eval mode."""
    from transformers import AutoTokenizer, EsmModel

    kwargs = {"attn_implementation": "eager"} if eager_attention else {}
    model = EsmModel.from_pretrained(ESM2, **kwargs).to(device).eval()
    return AutoTokenizer.from_pretrained(ESM2), model


@torch.no_grad()
def esm2_layer(tokenizer, model, sequence: str, layer: int) -> torch.Tensor:
    """Hidden states of one ESM2 layer (0 = embeddings, 30 = final), ``[L, 640]`` float32."""
    device = next(model.parameters()).device
    tokens = tokenizer(sequence, return_tensors="pt").to(device)
    states = model(**tokens, output_hidden_states=True).hidden_states[layer]
    return states[0, 1 : len(sequence) + 1].float()

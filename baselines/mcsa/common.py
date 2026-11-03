"""Shared I/O helpers for the M-CSA localization-baseline wrappers.

All three baseline scripts (``proteinfer/score.py``, ``deepfri/score.py``,
``protdetr/score.py``) read the same per-chain input format and write the
same per-chain score format, defined here once so the format only has to
be described in one place (see ``../README.md``).

Input record (one JSON object per protein chain)::

    {
        "id": str,                        # e.g. "1b73_A" (pdb_id_chain)
        "seq": str,                       # one-letter amino-acid sequence
        "ca": list[list[float]] | None,   # one [x, y, z] C-alpha coordinate
                                           # per residue of `seq`, in the
                                           # same order, or null if no
                                           # structure is available
        "ec": str | None,                 # annotated EC number, e.g.
                                           # "5.1.1.3", or null if unknown
    }

A file of these records is either a JSON array (``*.json``) or
newline-delimited JSON with one record per line (``*.jsonl``); both are
accepted regardless of the file extension actually used.

Output: a JSON object ``{id: [score, ...]}`` with one float per residue of
``seq``, in the same order, for every chain the baseline could score.
Chains a baseline had to skip (e.g. a constant or non-finite saliency map)
are simply absent from the output -- each script logs why.

This module is imported by scripts running under Python 3.7 (TensorFlow
baselines) as well as Python 3.12 (ProtDETR), so it is intentionally
limited to the standard library and old-style typing syntax.
"""
from __future__ import annotations

import json
from typing import Dict, List


def load_records(path: str) -> List[dict]:
    """Loads per-chain input records from a ``.json`` or ``.jsonl`` file.

    Args:
        path: Path to a JSON array of records, or a file with one JSON
            record per line.

    Returns:
        The list of input records, each a dict with at least "id" and
        "seq" keys.

    Raises:
        ValueError: If a record is missing a required field.
    """
    with open(path) as fh:
        text = fh.read()
    stripped = text.lstrip()
    if stripped.startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    for rec in records:
        if "id" not in rec or "seq" not in rec:
            raise ValueError("record missing required 'id'/'seq' field: %r" % (rec,))
    return records


def write_scores(path: str, scores: Dict[str, List[float]]) -> None:
    """Writes per-chain residue scores as a single JSON object.

    Args:
        path: Output file path.
        scores: Mapping from chain id to a list of per-residue scores.
    """
    with open(path, "w") as fh:
        json.dump(scores, fh)

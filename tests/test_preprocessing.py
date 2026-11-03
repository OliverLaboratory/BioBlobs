"""Preprocessing logic that runs without external tools."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "preprocessing/mcsa/build_homology_excluded_train.py"


def load_script():
    spec = importlib.util.spec_from_file_location("build_homology_excluded_train", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_homology_selection() -> None:
    """Train proteins sharing a cluster with M-CSA are removed; held-out = one per clean cluster."""
    rows = [
        ("P1", "P1"),  # training only
        ("P2", "P2"),  # training + M-CSA: P2 and P3 removed, nothing held out
        ("P2", "P3"),
        ("P2", "MCSA__1abc_A"),
        ("MCSA__2xyz_B", "MCSA__2xyz_B"),  # M-CSA only, two chains: one held out
        ("MCSA__2xyz_B", "MCSA__9zzz_A"),
        ("MCSA__1aaa_C", "MCSA__1aaa_C"),  # M-CSA singleton
    ]
    selected = load_script().select_from_clusters(pd.DataFrame(rows, columns=["rep", "member"]))
    assert selected["n_clusters"] == 4
    assert selected["n_contaminated_clusters"] == 3
    assert selected["train_removed"] == {"P2", "P3"}
    assert selected["heldout"] == ["1aaa_C", "2xyz_B"]

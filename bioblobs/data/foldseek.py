"""Foldseek 3Di structural-alphabet tokens for SaProt's structure-aware sequences."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from loguru import logger


def resolve_foldseek_binary(foldseek_bin: str | None) -> str:
    """Resolve the Foldseek binary: explicit path -> ``$FOLDSEEK_BIN`` -> ``foldseek`` in PATH."""
    if foldseek_bin:
        if os.path.isfile(foldseek_bin) and os.access(foldseek_bin, os.X_OK):
            return foldseek_bin
        raise FileNotFoundError(f"Foldseek binary not found or not executable: {foldseek_bin}")
    env_bin = os.environ.get("FOLDSEEK_BIN")
    if env_bin:
        if os.path.isfile(env_bin) and os.access(env_bin, os.X_OK):
            return env_bin
        raise FileNotFoundError(f"$FOLDSEEK_BIN={env_bin!r} is not an executable file")
    which_bin = shutil.which("foldseek")
    if which_bin:
        return which_bin
    raise FileNotFoundError(
        "Foldseek binary not found. Pass foldseek_bin, set $FOLDSEEK_BIN, or install it on PATH "
        "(conda install -c conda-forge -c bioconda foldseek)."
    )


def get_struc_seq(
    foldseek_bin: str, pdb_path: str | Path, chains: list[str] | None = None
) -> dict[str, tuple[str, str, str]]:
    """Run Foldseek on a (possibly multi-chain) PDB file to get 3Di structural-alphabet tokens.

    Returns:
        ``{chain_id: (aa_seq, struc_seq, sa_seq)}``, where ``sa_seq`` interleaves each amino
        acid with its lowercase 3Di token (``"MdKpVq..."``), the form SaProt's tokenizer expects.
        ``chains`` restricts to the given chain ids; ``None`` keeps every chain found.
    """
    pdb_path = str(pdb_path)
    tmp_dir = tempfile.mkdtemp(prefix="saprot_foldseek_")
    tmp_tsv = os.path.join(tmp_dir, "output.tsv")
    try:
        cmd = [
            foldseek_bin,
            "structureto3didescriptor",
            "-v",
            "0",
            "--threads",
            "1",
            "--chain-name-mode",
            "1",
            pdb_path,
            tmp_tsv,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Foldseek failed on {pdb_path}: {result.stderr.strip()}")

        pdb_basename = os.path.basename(pdb_path)
        seq_dict: dict[str, tuple[str, str, str]] = {}
        with open(tmp_tsv) as handle:
            for line in handle:
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                desc, aa_seq, struc_seq = parts[0], parts[1], parts[2].strip()
                name_chain = desc.split(" ")[0]
                chain = name_chain.replace(pdb_basename, "").split("_")[-1]
                if chains is not None and chain not in chains:
                    continue
                if chain in seq_dict:
                    continue
                if len(aa_seq) != len(struc_seq):
                    min_len = min(len(aa_seq), len(struc_seq))
                    aa_seq, struc_seq = aa_seq[:min_len], struc_seq[:min_len]
                combined = "".join(a + b.lower() for a, b in zip(aa_seq, struc_seq, strict=True))
                seq_dict[chain] = (aa_seq, struc_seq, combined)
        return seq_dict
    finally:
        for suffix in ("", ".dbtype"):
            path = tmp_tsv + suffix
            if os.path.exists(path):
                os.remove(path)
        if os.path.isdir(tmp_dir):
            os.rmdir(tmp_dir)


def build_sa_sequence(
    foldseek_bin: str,
    pdb_path: str | Path,
    expected_length: int,
    chain: str | None = None,
    pad_token: str = "Xd",
) -> str:
    """Build SaProt's SA bigram sequence for one chain, padded/truncated to ``expected_length``.

    Foldseek occasionally resolves a different residue count than the stored amino-acid sequence
    (missing density, alternate conformers); the dev caches padded or truncated the SA sequence
    to match. Two dev lineages used different 2-char pad tokens (``"Xd"`` for the ProteinShake
    caches, ``"X#"`` for VenusX) -- pass ``pad_token`` to match whichever you are reproducing; it
    only matters when a mismatch actually occurs. ``chain=None`` picks Foldseek's first chain.
    Returns an SA sequence of exactly ``2 * expected_length`` characters.
    """
    seq_dict = get_struc_seq(foldseek_bin, pdb_path, chains=[chain] if chain else None)
    if not seq_dict:
        raise RuntimeError(f"Foldseek returned no chains for {pdb_path}")
    if chain is not None:
        if chain not in seq_dict:
            raise KeyError(f"Chain {chain!r} not found by Foldseek in {pdb_path}")
        _, _, sa_seq = seq_dict[chain]
    else:
        _, _, sa_seq = next(iter(seq_dict.values()))

    residues = len(sa_seq) // 2
    if residues < expected_length:
        logger.warning(
            "{}: Foldseek gave {} residues, expected {}; padding",
            pdb_path,
            residues,
            expected_length,
        )
        sa_seq = sa_seq + pad_token * (expected_length - residues)
    elif residues > expected_length:
        logger.warning(
            "{}: Foldseek gave {} residues, expected {}; truncating",
            pdb_path,
            residues,
            expected_length,
        )
        sa_seq = sa_seq[: expected_length * 2]
    return sa_seq

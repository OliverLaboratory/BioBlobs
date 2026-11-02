"""Stage VenusX AlphaFold2 PDB files from a HuggingFace dataset repo.

The per-target VenusX PDB repos (``AI4Protein/VenusX_<target>_AlphaFold2_PDB``) publish
their structures in one of two shapes: either one ``.pdb`` file per protein directly in the
repo tree, or a single ``.zip`` archive of them. Both are handled here so
``preprocessing/prepare_venusx.py`` does not need to know which one a given target uses.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from pathlib import Path
from zipfile import ZipFile

from loguru import logger
from tqdm import tqdm


def build_remote_pdb_map(repo_files: Iterable[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Map each ``.pdb`` basename (exact and lowercased) to its path in the HF repo tree."""
    exact_map: dict[str, str] = {}
    lower_map: dict[str, str] = {}
    for repo_path in repo_files:
        if not repo_path.lower().endswith(".pdb"):
            continue
        stem = Path(repo_path).stem
        if stem in exact_map and exact_map[stem] != repo_path:
            raise ValueError(f"Duplicate PDB basename in VenusX repo: {stem!r}")
        exact_map[stem] = repo_path
        lower_map.setdefault(stem.lower(), repo_path)
    if not exact_map:
        raise ValueError("No .pdb files found in the VenusX AlphaFold2 repo")
    return exact_map, lower_map


def resolve_remote_pdb_path(
    uid: str, *, exact_map: dict[str, str], lower_map: dict[str, str]
) -> str:
    """Repo-relative path of ``uid``'s PDB file, falling back to a case-insensitive match."""
    if uid in exact_map:
        return exact_map[uid]
    if uid.lower() in lower_map:
        return lower_map[uid.lower()]
    raise FileNotFoundError(f"Missing PDB for VenusX uid {uid!r}")


def build_archive_member_candidates(
    member_names: Iterable[str],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Map each archive member's trailing ``_<uid>`` token (exact and lowercased) to its paths."""
    exact_map: dict[str, list[str]] = {}
    lower_map: dict[str, list[str]] = {}
    for member_name in member_names:
        if not member_name.lower().endswith(".pdb"):
            continue
        uid = Path(member_name).stem.rsplit("_", 1)[-1]
        exact_map.setdefault(uid, []).append(member_name)
        lower_map.setdefault(uid.lower(), []).append(member_name)
    if not exact_map:
        raise ValueError("No .pdb members found inside the VenusX AlphaFold2 archive")
    return exact_map, lower_map


def resolve_archive_member(
    uid: str, member_candidates: tuple[dict[str, list[str]], dict[str, list[str]]]
) -> str:
    """Lexicographically-first archive member matching ``uid`` (exact, then case-insensitive)."""
    exact_map, lower_map = member_candidates
    candidates = exact_map.get(uid) or lower_map.get(uid.lower())
    if not candidates:
        raise FileNotFoundError(f"Missing archived PDB for VenusX uid {uid!r}")
    return sorted(candidates)[0]


def stage_pdb_files(
    required_uids: list[str], pdb_repo_id: str, pdb_dir: Path, hf_cache_dir: Path
) -> int:
    """Download or extract one ``<uid>.pdb`` per required protein into ``pdb_dir``.

    Tries direct per-protein ``.pdb`` files in the repo tree first; if the repo instead
    ships one ``<repo_name>.zip`` archive, downloads and extracts from that.

    Args:
        required_uids: Protein UIDs that need a staged PDB file.
        pdb_repo_id: HuggingFace dataset repo holding the AlphaFold2 structures.
        pdb_dir: Output directory; existing ``<uid>.pdb`` files are left untouched.
        hf_cache_dir: HuggingFace hub cache directory.

    Returns:
        Number of required UIDs (all are guaranteed staged on success).
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    repo_files = list_repo_files(repo_id=pdb_repo_id, repo_type="dataset")
    pdb_repo_files = [path for path in repo_files if path.lower().endswith(".pdb")]
    pdb_dir.mkdir(parents=True, exist_ok=True)

    if pdb_repo_files:
        exact_map, lower_map = build_remote_pdb_map(pdb_repo_files)
        for uid in tqdm(required_uids, desc="Staging VenusX PDBs"):
            output_path = pdb_dir / f"{uid}.pdb"
            if output_path.exists():
                continue
            remote_path = resolve_remote_pdb_path(uid, exact_map=exact_map, lower_map=lower_map)
            cached_path = hf_hub_download(
                repo_id=pdb_repo_id, filename=remote_path, repo_type="dataset",
                cache_dir=str(hf_cache_dir),
            )  # fmt: skip
            shutil.copy2(cached_path, output_path)
        return len(required_uids)

    archive_name = f"{pdb_repo_id.split('/')[-1]}.zip"
    if archive_name not in repo_files:
        raise ValueError(
            f"No direct .pdb files or expected archive {archive_name!r} found in {pdb_repo_id!r}"
        )
    archive_path = Path(
        hf_hub_download(
            repo_id=pdb_repo_id,
            filename=archive_name,
            repo_type="dataset",
            cache_dir=str(hf_cache_dir),
        )  # fmt: skip
    )
    logger.info("Extracting VenusX PDBs from archive {}", archive_path)
    with ZipFile(archive_path) as archive:
        member_candidates = build_archive_member_candidates(archive.namelist())
        for uid in tqdm(required_uids, desc="Extracting VenusX PDBs"):
            output_path = pdb_dir / f"{uid}.pdb"
            if output_path.exists():
                continue
            member_name = resolve_archive_member(uid, member_candidates)
            with archive.open(member_name) as source, output_path.open("wb") as dest:
                shutil.copyfileobj(source, dest)
    return len(required_uids)

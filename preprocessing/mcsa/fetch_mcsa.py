#!/usr/bin/env python
"""Fetch M-CSA catalytic-site entries and their reference PDB files.

Standalone CLI, no other part of this release required. Network access needed.

The M-CSA API embeds one reference PDB per entry; an entry can still have several
chains (``catalytic_residues[*].chain``). Picking a single ``chain_used`` per entry
— the curation step behind the shipped ``data/mcsa/mcsa_single_chain.json`` — is a
separate step not reproduced here (see ``preprocessing/mcsa/README.md``).

Outputs:
    <out>/mcsa_entries.json   raw entries: mcsa_id, enzyme_name, ec_number, pdb_id,
                              catalytic_residues[{chain, resid, amino_acid, role, ...}]
    <out>/pdbs/<pdb_id>.pdb   one file per entry's reference PDB (lowercase id)

Usage::

    .venv/bin/python preprocessing/mcsa/fetch_mcsa.py --out data/mcsa_raw
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

from loguru import logger

MCSA_ENTRIES_API = "https://www.ebi.ac.uk/thornton-srv/m-csa/api/entries/?format=json"
RCSB_PDB_DOWNLOAD = "https://files.rcsb.org/download/{pdb_id}.pdb"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", type=Path, default=Path("data/mcsa_raw"))
    p.add_argument(
        "--entries-cache",
        type=Path,
        default=None,
        help="Reuse/write raw entries here (default: <out>/mcsa_entries.json).",
    )
    p.add_argument("--skip-pdbs", action="store_true", help="Fetch entries only; skip PDBs.")
    p.add_argument("--max-entries", type=int, default=None, help="Smoke-test entry limit.")
    p.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout, seconds.")
    return p.parse_args()


def _fetch_paginated(url: str, timeout: float) -> list[dict]:
    """Fetch every page of a paginated M-CSA API endpoint."""
    results: list[dict] = []
    next_url: str | None = url
    while next_url:
        with urllib.request.urlopen(next_url, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        if isinstance(data, list):
            results.extend(data)
            break
        results.extend(data.get("results", []))
        next_url = data.get("next")
    return results


def _parse_entry(record: dict) -> dict[str, Any] | None:
    """Extract id, name, EC, reference PDB and catalytic residues from one raw M-CSA record."""
    mcsa_id = str(record.get("mcsa_id", "")).strip()
    if not mcsa_id:
        return None
    all_ecs = record.get("all_ecs") or []
    pdb_id, catalytic_residues = "", []
    for res in record.get("residues") or []:
        roles_summary = res.get("roles_summary") or ""
        function_location = res.get("function_location_abv") or ""
        for chain_rec in res.get("residue_chains") or []:
            if not chain_rec.get("is_reference", False):
                continue
            candidate_pdb = (chain_rec.get("pdb_id") or "").strip().lower()
            if candidate_pdb and not pdb_id:
                pdb_id = candidate_pdb
            resid = chain_rec.get("auth_resid")
            if resid is None:
                resid = chain_rec.get("resid")
            if resid is None:
                continue
            catalytic_residues.append(
                {
                    "chain": chain_rec.get("chain_name") or "",
                    "resid": int(resid),
                    "amino_acid": chain_rec.get("code") or "",
                    "role": roles_summary,
                    "function_location": function_location,
                }
            )
    if not pdb_id:
        return None
    return {
        "mcsa_id": mcsa_id,
        "enzyme_name": record.get("enzyme_name") or "",
        "ec_number": all_ecs[0] if all_ecs else "",
        "pdb_id": pdb_id,
        "catalytic_residues": catalytic_residues,
    }


def fetch_mcsa_entries(cache_path: Path, timeout: float) -> list[dict]:
    """Load cached entries if present, else fetch and cache them."""
    if cache_path.exists():
        logger.info("Loading M-CSA entries from cache: {}", cache_path)
        return json.loads(cache_path.read_text())
    logger.info("Fetching M-CSA entries from {}", MCSA_ENTRIES_API)
    raw = _fetch_paginated(MCSA_ENTRIES_API, timeout)
    entries = [e for e in (_parse_entry(r) for r in raw) if e is not None]
    logger.info(
        "Parsed {} entries ({} skipped: no reference PDB)", len(entries), len(raw) - len(entries)
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(entries, indent=2))
    logger.success("wrote {}", cache_path)
    return entries


def download_pdb(pdb_id: str, dest_dir: Path, timeout: float) -> bool:
    """Download one PDB file from RCSB into ``dest_dir``; ``True`` if present or fetched."""
    dest = dest_dir / f"{pdb_id.lower()}.pdb"
    if dest.exists():
        return True
    url = RCSB_PDB_DOWNLOAD.format(pdb_id=pdb_id.upper())
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            dest.write_bytes(response.read())
        return True
    except Exception as exc:  # noqa: BLE001 - report and keep going
        logger.warning("failed to download {}: {}", pdb_id, exc)
        return False


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cache_path = args.entries_cache or (args.out / "mcsa_entries.json")

    t0 = time.time()
    entries = fetch_mcsa_entries(cache_path, args.timeout)
    if args.max_entries is not None:
        entries = entries[: args.max_entries]

    if args.skip_pdbs:
        logger.success("done in {:.0f}s ({} entries, PDBs skipped)", time.time() - t0, len(entries))
        return

    pdb_dir = args.out / "pdbs"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    pdb_ids = sorted({e["pdb_id"] for e in entries})
    ok = sum(download_pdb(pdb_id, pdb_dir, args.timeout) for pdb_id in pdb_ids)
    logger.success(
        "done in {:.0f}s: {} entries, {}/{} PDBs downloaded -> {}",
        time.time() - t0,
        len(entries),
        ok,
        len(pdb_ids),
        pdb_dir,
    )


if __name__ == "__main__":
    main()

# M-CSA catalytic-site evaluation data

Scripts that reproduce the raw-data side of the M-CSA pipeline. The curated
single-chain annotations and id lists consumed by `bioblobs.data.mcsa.load_mcsa` are
already shipped under `data/mcsa/` — these scripts are for regenerating or extending
that raw data, not required to use `load_mcsa`.

## `fetch_mcsa.py`

Fetches every M-CSA entry (mechanism + catalytic residues) from the public M-CSA API
and downloads each entry's reference PDB from RCSB. Network access required.

```bash
.venv/bin/python preprocessing/mcsa/fetch_mcsa.py --out data/mcsa_raw
```

Writes `mcsa_entries.json` (one record per M-CSA entry, every annotated chain) and
`pdbs/<pdb_id>.pdb`.

## The missing step: single-chain curation

`data/mcsa/mcsa_single_chain.json` (892 entries) adds one field, `chain_used`, not
present in `mcsa_entries.json` (998 entries): the single PDB chain picked per entry for
evaluation. The chain-selection step is not part of this release — only its output is
shipped, as `data/mcsa/mcsa_single_chain.json`. (106 of the 998 raw entries are dropped
between the two files — presumably ones with no resolvable single chain — but the
exact rule is not reproduced here.)

## `build_features.py`

Parses each curated chain's backbone from its PDB file and embeds its sequence with
ESM2, writing the `{pdb_chain: {"nf", "ca", "resnums"}}` feature cache
`bioblobs.data.mcsa.load_mcsa` reads via `cfg.feat_cache`.

```bash
.venv/bin/python preprocessing/mcsa/build_features.py \
    --labels data/mcsa/mcsa_single_chain.json \
    --pdb-dir data/mcsa_raw/pdbs \
    --out data/mcsa_raw/feat_cache.pt --max-len 1022

.venv/bin/python preprocessing/mcsa/build_features.py \
    --labels data/mcsa/mcsa_single_chain.json \
    --pdb-dir data/mcsa_raw/pdbs \
    --existing data/mcsa_raw/feat_cache.pt \
    --out data/mcsa_raw/feat_cache_maxlen1500.pt --max-len 1500
```

`--max-len` is an admission filter (chains longer than this are dropped, not
truncated); see `bioblobs/data/mcsa.py`'s module docstring for which evaluation used
the 1022 cache and which used the 1500 one.

## `build_homology_excluded_train.py`

Clusters SwissProt EC-L3 train sequences together with the curated M-CSA chains at 30%
identity (MMseqs2) and drops every train protein whose cluster also contains an M-CSA
chain.

```bash
.venv/bin/python preprocessing/mcsa/build_homology_excluded_train.py \
    --train-csv data/swissprot_ec/train_split_train.csv \
    --mcsa-labels data/mcsa/mcsa_single_chain.json \
    --pdb-dir data/mcsa_raw/pdbs \
    --out-csv data/swissprot_ec/train_split_train_homfiltered.csv
```

`data/mcsa/heldout_ids.txt` lists the representative of each cluster that holds M-CSA
chains but no training protein (79 chains). It is the list used for the paper. The script
also writes this list to `<workdir>/heldout_ids.txt`, and a full re-run with MMseqs2
18.8cc5c reproduces the shipped file exactly. A different MMseqs2 version may cluster
differently.

## Required external tools

- **MMseqs2** for `build_homology_excluded_train.py`: `conda install -c bioconda mmseqs2`.
- Network access for `fetch_mcsa.py` (M-CSA API + RCSB PDB download).

## What this does not do

- Does not build `data/mcsa/evaluation_ids.txt` (the 74-chain evaluation cohort) from
  scratch — it is an intersection over per-method raw score files that are not part of
  this release; see its file header for the exact selection rule.

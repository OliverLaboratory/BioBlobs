# SwissProt EC-L3 pretraining data

Builds the dataset consumed by `bioblobs.data.swissprot` from raw inputs. Two steps,
both CPU-only and embedding-free (residue embeddings are built separately by
`preprocessing/prepare_plm_cache.py`, over this pipeline's output):

1. **`build_structures.py`** — parses PDB backbones, filters/dedups/labels proteins by
   EC number, and writes the structure cache + label artifacts:
   `cache/structures.pt`, `train_split.csv`, `labels.csv`, `token_map.json`,
   `class_priors.json`, `build_quality_report.json`.
2. **`build_val_split.py`** — MMseqs2 30%-identity clustering to carve a class-stratified,
   homology-disjoint val set out of `train_split.csv`: `train_split_train.csv`,
   `train_split_val.csv`, `val_split_report.json`.

```bash
.venv/bin/python preprocessing/swissprot_ec/build_structures.py \
    --input-csv data/swissprot_ec/train_100_with_pdb.csv \
    --pdb-root data --out-data-dir data/swissprot_ec \
    --pdb-num-workers 32 --require-ec-level 3 --max-len 1500

.venv/bin/python preprocessing/swissprot_ec/build_val_split.py \
    --input-csv data/swissprot_ec/train_split.csv \
    --val-fraction 0.05 --min-seq-id 0.3 --cov 0.8 --cov-mode 0 \
    --threads 32 --seed 0
```

## Required external tools

- **MMseqs2** (`build_val_split.py`): `conda install -c bioconda mmseqs2`, or the static
  binary from https://mmseqs.com.
- **The AFDB v6 bulk tarball** (for `--pdb-root`, if starting from UniProt ids rather than
  already-extracted PDB files): `download_alphafold_pdbs.sh` extracts matching
  `AF-<id>-F1-model_v6.pdb.gz` entries from a local `swissprot_pdb_v6.tar` (download it from
  https://alphafold.ebi.ac.uk/download — "Swiss-Prot" bulk archive — separately; this script
  does not fetch it).

## Input CSV

`build_structures.py --input-csv` is plain `id, seq, ec_number` data (`download_alphafold_pdbs.sh`
adds the matched `pdb_path` column), for example `train_100.csv`. Its provenance is not asserted
by this release; supply your own CSV with these columns.

## Note on structure parsing

Structures were parsed with a different PDB parser for the paper; re-running this script from the
raw PDB files may differ in rare residues (e.g. alternate locations). The released split files are
the ones used for the paper.

## What this does not do

- Does not build residue embeddings (ESM2/SaProt) — see `preprocessing/prepare_plm_cache.py`.
- Does not build the homology-excluded train variant used for the M-CSA leakage control —
  see `preprocessing/mcsa/build_homology_excluded_train.py`.

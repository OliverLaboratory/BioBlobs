# Reproducing the paper

Run every command from the repository root inside the `uv` environment (see the README). Each
training run uses one GPU. ProteinShake runs use seeds 0, 1 and 2, and VenusX runs use seeds 1, 2
and 3. A run writes `outputs/<name>/seed<k>/` with `best.ckpt`, `metrics.json`, `config.yaml` and
its logs. The paper reports the mean and sample standard deviation over three seeds. This guide
covers BioBlobs. The comparison methods and baselines are on the
[`baselines`](https://github.com/OliverLaboratory/BioBlobs/tree/baselines) branch.

## 1. Prepare the data

The ProteinShake and VenusX structures, labels and splits, and the VenusX embeddings behind the
paper, are in the Hugging Face dataset `xwang112358/BioBlobs-data`. Download it as `data/` and
unpack the structures:

```bash
hf download xwang112358/BioBlobs-data --repo-type dataset --local-dir data
for f in data/*_proteinshake/pdb.tar.gz data/venusx_site_fragment/*/MF50/pdb.tar.gz; do
  tar -xzf "$f" -C "$(dirname "$f")"
done
```

The download is 38.2 GB, of which 33.0 GB are the VenusX embeddings. Add
`--exclude "*_static_cache_mmap/*"` to skip them. The preparation scripts below find the unpacked
files and skip their own download.

Do this once per dataset and encoder. ESM2-T30 (`facebook/esm2_t30_150M_UR50D`) and SaProt-650M
(`westlake-repl/SaProt_650M_PDB`) download from Hugging Face. SaProt also needs Foldseek.

**ProteinShake** (`ec`, `pfam`, `go_mf`, `scop_fam`, `scop_sf`, each with a `sequence` and a
`structure` split):

```bash
python preprocessing/prepare_proteinshake.py --dataset ec --split sequence
python preprocessing/prepare_plm_cache.py \
    --structures data/ec_proteinshake/cache/sequence_{train,val,test}_knn_30_0.7_cache.pt \
    --encoder esm2_static --out data/ec_proteinshake/esm2_static_cache_mmap
```

For SaProt, use `--encoder saprot_static --out data/<task>_proteinshake/saprot_static_cache_mmap`
and add `--pdb-dir data/<task>_proteinshake/pdb --foldseek-bin <path to foldseek>`.

**VenusX** (targets `Act`, `BindI`, `Evo` and `Motif`):

```bash
python preprocessing/prepare_venusx.py --target Act --split-strategy MF50 --data-dir data
```

This parses the structures. Training then reads the downloaded ESM2-T30 and SaProt-650M embeddings.
They lack 84 proteins (11, 10, 53 and 10 for `Act`, `BindI`, `Evo` and `Motif`) whose AlphaFold2
structure has residues past the end of the VenusX site annotation, which the paper's embedding build
skipped. Training leaves these proteins out of every split, which gives the paper's test sets of
186, 203, 1,291 and 249 samples. To build the embeddings yourself, run the command below for each
encoder. Training needs both encoders' embeddings to cover the same proteins. A fresh build covers
every protein and gives test sets of 186, 205, 1,297 and 251 samples.

```bash
python preprocessing/prepare_plm_cache.py \
    --structures data/venusx_site_fragment/Act/MF50/cache/uid_structures_min_completion_0p95.pt \
    --key-field protein_uid --encoder esm2_static \
    --out data/venusx_site_fragment/Act/MF50/esm2_static_cache_mmap
```

**Swiss-Prot EC (the M-CSA model's training data)**: the input is the EC-Bench training set, a
CSV with `id`, `seq` and `ec_number`. [preprocessing/swissprot_ec/README.md](../preprocessing/swissprot_ec/README.md)
describes how to attach AlphaFold DB v6 structures and build the labels.
`data/swissprot_ec/` ships the training and validation ids used for the paper, split at 30% sequence
identity with MMseqs2 (copy the two id files along if you move `datasets.data_dir`). Then build the
ESM2 cache:

```bash
python preprocessing/prepare_plm_cache.py --structures data/swissprot_ec/cache/structures.pt \
    --encoder esm2_static --max-len 1500 --out data/swissprot_ec/esm2_static_cache_mmap
```

The cache admits proteins up to 1,500 residues, and training keeps those up to 1,000
(`datasets.max_len`).

**M-CSA**: `data/mcsa/` ships the chain chosen for each entry, its catalytic residues, the 79
enzymes held out from training and the 74 evaluated in the paper. Build the embedding file with
`python preprocessing/mcsa/build_features.py --labels data/mcsa/mcsa_single_chain.json --pdb-dir <pdbs> --max-len 1022 --out data/mcsa/features.pt`.

## 2. Commands for each result

Train three seeds of one configuration with Hydra's multirun, for example:

```bash
python train_bioblobs.py -m +experiment=proteinshake/ec_sequence_esm2 training.seed=0,1,2
```

| Result | Configuration | Then |
| --- | --- | --- |
| Table 1, BioBlobs with ESM2-T30 | `+experiment=proteinshake/<task>_<split>_esm2` | test metrics are in `metrics.json` |
| Table 1, BioBlobs with SaProt-650M | `+experiment=proteinshake/<task>_<split>_saprot` | as above |
| Tables 2 and 10, BioBlobs | `+experiment=venusx/<target>_<encoder>_labelfree` | `evaluate.py --checkpoint <run>/best.ckpt --score weighted_membership --out <dir>` |
| Tables 2 and 10, BioBlobs with seed alignment | `+experiment=venusx/<target>_<encoder>_seedalign` | as above |
| Figure 2 and Tables 8 and 9, M-CSA model | `+experiment=mcsa/rvq_k8` | `evaluate.py` in M-CSA mode, below |

Table 1 reports Fmax for GO-MF and macro-F1 for the other tasks. The experiment files set the
matching task type and selection metric.

**Hyperparameters.** The ESM2-T30 and SaProt-650M files in `conf/experiment/proteinshake/` hold the
settings behind the printed Table 1 cells, from the search grids in the appendix. Figure 3 and
the appendix grids vary `partitioners.num_blobs` (K), `partitioners.seed_radius` (r) and
`objective.hoyer_weight` (λ_H) on top of an experiment config.

**M-CSA** (Figure 2 and Table 9). Evaluate the released model, or your own `mcsa/rvq_k8` run, on
the 74 held-out enzymes:

```bash
python evaluate.py --checkpoint bioblobs-mcsa-rvq-k8 --mcsa-features data/mcsa/features.pt \
    --mcsa-ids data/mcsa/evaluation_ids.txt --score max_membership --out results/mcsa
```

This reproduces the BioBlobs row of Table 9.

## 3. Notes

- `prepare_proteinshake.py` writes the data caches in split-file order, which differs from the order
  of the original runs. The proteins, sequences and coordinates are the same, but retrained models will
  not match the paper's checkpoints bit for bit.
- The ESM2-T30 seed-alignment rows of the VenusX tables were trained with an earlier version of
  the seed-alignment code. The released code is the version that produced the SaProt-650M rows.
- The M-CSA chain selection, the 79 held-out enzymes and the 74 evaluated enzymes ship as fixed
  files in `data/mcsa/`. The held-out set is the MMseqs2 representative of each cluster that holds
  M-CSA chains but no training protein, as described in the header of `data/mcsa/heldout_ids.txt`.
  `preprocessing/mcsa/build_homology_excluded_train.py` also writes this list, and with MMseqs2
  18.8cc5c it reproduces the shipped file exactly. A different MMseqs2 version may cluster
  differently.
- Compared with the May 2026 release, several config names changed: `datasets=go` is `go_mf`, `tasks=multi-class` is `multi_class`,
  `partitioners.num_blobs_per_protein` is `partitioners.num_blobs`, and the loss weights moved to
  the `objective` block (for example `objective.hoyer_weight`).

# Reproducing the paper

Run every command from the repository root inside the `uv` environment (see the README). Each
training run uses one GPU. ProteinShake runs use seeds 0, 1 and 2, and VenusX runs use seeds 1, 2
and 3. A run writes `outputs/<name>/seed<k>/` with `best.ckpt`, `metrics.json`, `config.yaml` and
its logs. The paper reports the mean and sample standard deviation over three seeds.

## 1. Prepare the data

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
python preprocessing/prepare_plm_cache.py \
    --structures data/venusx_site_fragment/Act/MF50/cache/uid_structures_min_completion_0p95.pt \
    --key-field protein_uid --encoder esm2_static \
    --out data/venusx_site_fragment/Act/MF50/esm2_static_cache_mmap
```

**Swiss-Prot EC (the M-CSA model's training data)**: the input is the EC-Bench training set, a
CSV with `id`, `seq` and `ec_number`. [preprocessing/swissprot_ec/README.md](../preprocessing/swissprot_ec/README.md)
describes how to attach AlphaFold DB v6 structures and build the labels.
`data/swissprot_ec/` ships the training and validation ids used for the paper, split at 30% sequence
identity with MMseqs2. Then build the ESM2 cache:

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
| Table 1, mean pooling | `datasets=<task> datasets.split=<split> encoders=<esm2_static\|saprot_static> partitioners=none decoders=mean_pool training.epochs=50 tasks=<multi_class\|multi_label>` | as above |
| Table 1, attention pooling | as mean pooling, with `decoders=attention_pool` | as above |
| Table 1, GCN, GIN and GAT | `encoders=<gcn\|gin\|gat>_<esm2\|saprot> partitioners=none decoders=mean_pool datasets.edge_types=eps_8` | as above |
| Table 1, DiffPool, TopK and SAGPool | `partitioners=<diffpool\|topk\|sagpool> datasets.edge_types=eps_8` | as above |
| Tables 2, 10 and 11, BioBlobs | `+experiment=venusx/<target>_<encoder>_labelfree` | `evaluate.py --checkpoint <run>/best.ckpt --score weighted_membership --out <dir>` |
| Tables 2, 10 and 11, BioBlobs with seed alignment | `+experiment=venusx/<target>_<encoder>_seedalign` | as above |
| Figure 3 and Tables 8 and 9, M-CSA model | `+experiment=mcsa/rvq_k8` | `evaluate.py` in M-CSA mode, below |

Use `tasks=multi_label selection.metric=fmax` for GO-MF and `tasks=multi_class` for the other
ProteinShake tasks. Table 1 reports Fmax for GO-MF and macro-F1 for the other tasks. The BioBlobs
experiment files already set these.

**Hyperparameters.** The ESM2-T30 files in `conf/experiment/proteinshake/` hold the settings behind
the printed Table 1 cells. The SaProt-650M files hold the settings selected on validation from the
grid in the appendix. The comparators were tuned per cell on validation, with the search spaces
listed in the appendix. The config defaults are not those tuned values. For each cell, set
`training.lr`, `training.batch_size`, `training.epochs` and the comparator's own parameters
(`encoders.gnn.num_layers` and `encoders.gnn.dropout` for the GNNs, `partitioners.max_clusters` for
DiffPool, `partitioners.ratio` for TopK and SAGPool) to the selected values. The EC and SCOP-FAM
comparators trained at batch size 32 for up to 100 epochs, and the Pfam, GO-MF and SCOP-SF
comparators at batch size 16 for up to 60 epochs. Figure 2 and the appendix grids vary `partitioners.num_blobs` (K),
`partitioners.seed_radius` (r) and `objective.hoyer_weight` (λ_H) on top of an experiment config.

**VenusX baselines** (Tables 2, 10 and 11). Train mean or attention pooling with

```bash
python train_bioblobs.py -m datasets=venusx datasets.target=Act datasets.site_supervised=true \
    encoders=esm2_static training=venusx training.lr=3e-4 training.epochs=60 partitioners=none \
    decoders=attention_pool decoders.input_layernorm=true decoders.proj_dim=null \
    objective.hoyer_weight=0 selection.patience=15 'selection.early_stop_any_of=[macro_f1,loss]' \
    training.seed=1,2,3
```

For mean pooling, use `decoders=mean_pool decoders.masked=true`. Score residues with
`evaluate.py --score attention` for attention pooling and with `--score integrated_gradients` on
the mean-pooling model. `--score burial` and `--score random` give the two controls.

**M-CSA** (Figure 3 and Table 9). Evaluate the released model, or your own `mcsa/rvq_k8` run, on
the 74 held-out enzymes:

```bash
python evaluate.py --checkpoint bioblobs-mcsa-rvq-k8 --mcsa-features data/mcsa/features.pt \
    --mcsa-ids data/mcsa/evaluation_ids.txt --score max_membership --out results/mcsa
```

This reproduces the BioBlobs row of Table 9. The third-party baselines
(ProtDETR, ProteInfer and DeepFRI) run in their own environments, as described in
[baselines/mcsa/README.md](../baselines/mcsa/README.md).

## 3. Hardware and run times

| Runs | Original hardware | Time per run |
| --- | --- | --- |
| Table 1, BioBlobs with ESM2-T30 | 1× NVIDIA H100 80 GB | 7–27 min for EC, Pfam and SCOP, about 1 h for GO-MF |
| Table 1, BioBlobs with SaProt-650M | 1× NVIDIA RTX 5090 | up to about 1 h, and up to a few hours for GO-MF |
| Table 1, mean and attention pooling | 1× H100 or RTX 5090 | 3–46 min, longest for GO-MF |
| Table 1, GCN, GIN, GAT, DiffPool, TopK and SAGPool | 1× H200 or H100 | not recorded |
| VenusX, BioBlobs | 1× RTX 5090 or H100 | 3–58 min, longest for Evo |
| M-CSA model | 1 GPU | about 70 s per epoch, 100 epochs |
| M-CSA evaluation | CPU | minutes |

GO-MF needs 35–50 GB of host memory per run, plus room to cache its 25 GB embedding file.

## 4. Notes

- The ESM2-T30 seed-alignment rows of the VenusX tables were trained with an earlier version of
  the seed-alignment code. The released code is the version that produced the SaProt-650M rows.
- The M-CSA chain selection, the 79 held-out enzymes and the 74 evaluated enzymes ship as fixed
  files in `data/mcsa/`. Re-deriving the held-out set from the MMseqs2 clustering gives one extra
  chain (1znv_B), as noted in `data/mcsa/heldout_ids.txt`.
- Compared with the May 2026 release, several config names changed: `decoders=mlp` is now
  `mean_pool`, `datasets=go` is `go_mf`, `tasks=multi-class` is `multi_class`,
  `partitioners.num_blobs_per_protein` is `partitioners.num_blobs`, and the loss weights moved to
  the `objective` block (for example `objective.hoyer_weight`).

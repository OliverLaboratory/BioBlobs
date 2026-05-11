# BioBlobs: Unsupervised Discovery of Functional Substructures for Protein Function Prediction
Anonymous code release accompanying the paper *BioBlobs: Unsupervised Discovery of Functional Substructures for Protein Function Prediction*.

## Repository layout

```
.
├── train_bioblobs.py          # Single training entrypoint (BioBlobs + baselines)
├── inference.py               # Inference + blob interpretability on user PDBs
├── preprocessing/             # Direct-to-mmap PLM caches + VenusX dataset prep
│   ├── prepare_plm_cache.py              # ProteinShake (ec/go/pfam)
│   ├── prepare_venusx_site_fragment.py   # VenusX download + staging
│   └── prepare_venusx_plm_cache.py       # VenusX PLM mmap
├── conf/                      # Hydra configs
│   ├── train_bioblobs.yaml    # Default training config
│   ├── datasets/              # ec, go, pfam (ProteinShake), venusx_site_fragment
│   ├── encoders/              # esm2_static (default), saprot_static
│   ├── partitioners/          # bioblobs, none
│   ├── decoders/              # mil, mil_light_attn, mlp,
│   │                          #   simple_attn, attention_pool, light_attn
│   ├── tasks/                 # multi-class, multi-label, venusx_fragment_multiclass
│   └── training/              # baseline.yaml
├── bioblobs/                  # Python package
│   ├── modules/               # Encoders, partitioners, decoders, pipeline ops
│   ├── datasets/              # ProteinShake loaders, PDB parsing, PLM mmap caches
│   │                          # (atom_representation, parallel, features live here)
│   └── training/              # Lightning module, shared runner, metrics
│                              # (experiments, fmax_metric, utils live here)
├── bioblobs_framework.py      # Top-level encoder → partitioner → decoder model
└── archive/                   # Variants and tooling not used by the paper
```

## Installation

The repo targets Linux x86_64, Python 3.12, PyTorch 2.7 + CUDA 12.8, and
PyTorch Geometric. Dependencies (and the matching CUDA / PyG wheel
indexes) are declared in `pyproject.toml`, so a single `uv sync` reproduces
the environment from `uv.lock`:

```bash
uv sync
source .venv/bin/activate
```

`uv sync` creates `./.venv/`, fetches CUDA 12.8 PyTorch wheels from the
PyTorch index, and pulls the matching PyG kernel extensions (`pyg_lib`,
`torch_scatter`, `torch_cluster`, `torch_sparse`, `torch_spline_conv`) from
the PyG flat index — no manual install ordering required. The lockfile
pins every transitive dependency for byte-exact reproducibility.


## Datasets

Training is wired to ProteinShake datasets (downloaded automatically on first
run into `./data/`):

| Dataset | Tag | Task |
| --- | --- | --- |
| Enzyme Commission | `ec`   | multi-class |
| Pfam              | `pfam` | multi-class |
| Gene Ontology     | `go`   | multi-label (`go_branch=molecular_function|biological_process|cellular_component`) |

The default config uses EC. Switch dataset with `datasets=<tag>`; switch GO
branch with `datasets.go_branch=<branch>`.

## Training

A single entrypoint, `train_bioblobs.py`, runs all three model variants by
toggling the partitioner / decoder via Hydra overrides.

### BioBlobs (default — partitioner + MIL decoder)

```bash
python train_bioblobs.py
```

This uses `conf/train_bioblobs.yaml`, which selects:

* `encoders=esm2_static`   (frozen ESM2-150M residue embeddings, mmap-cached)
* `partitioners=bioblobs`  (differentiable blob partitioner)
* `decoders=mil`           (attention-based MIL pooling + classification)
* `datasets=ec`, `tasks=multi-class`

Useful BioBlobs partitioner overrides (defaults in
`conf/partitioners/bioblobs.yaml`):

```bash
python train_bioblobs.py \
  partitioners.num_blobs_per_protein=5 \
  partitioners.seed_radius=12.0 \
  partitioners.proximity_bias=0.5 \
  partitioners.membership_hoyer=0.0
```

#### Swept hyperparameters

The four hyperparameters swept in the paper:

| Override | Symbol | Default | Where | What it controls |
| --- | --- | --- | --- | --- |
| `partitioners.num_blobs_per_protein` | `K` | `5` | `conf/partitioners/bioblobs.yaml` | Fixed blob-proposal budget per protein. Each protein is partitioned into exactly `K` (possibly empty) blobs, which the MIL decoder weighs and aggregates. Larger `K` lets the model recover finer-grained substructures at the cost of more padded/empty slots and a wider MIL head. |
| `partitioners.seed_radius` | `r` (Å) | `12.0` | `conf/partitioners/bioblobs.yaml` | Spatial radius (Å) of the locality window used when assigning residues to a blob seed. Smaller `r` produces tighter, more locally compact blobs; larger `r` allows blobs to span longer-range structural motifs. |
| `partitioners.membership_hoyer` | `λ_H` | `0.0` | `conf/partitioners/bioblobs.yaml` | Weight of the Hoyer-Square (HS) sparsity regularizer applied to the soft residue→blob membership matrix. `0` disables the term; larger values push each residue's `K`-way membership toward a one-hot assignment, yielding sharper, more disjoint blobs. |
| `decoders.use_blob_interaction` | — | `false` | `conf/decoders/mil.yaml` | If `true`, run a single self-attention layer across the `K` blobs before the MIL attention gate. Each blob attends to every other blob, and a `[B, K, K]` correlation matrix is exposed in the model's `extra` output for interpretability. |

Sweep example:

```bash
# K   ∈ {3, 5, 8, 12}
# r   ∈ {8.0, 12.0, 16.0}
# λ_H ∈ {0.0, 1e-4, 1e-3, 1e-2}
# use_blob_interaction ∈ {false, true}
python train_bioblobs.py \
  partitioners.num_blobs_per_protein=8 \
  partitioners.seed_radius=16.0 \
  partitioners.membership_hoyer=1e-3 \
  decoders.use_blob_interaction=true
```

### Baselines

The same `train_bioblobs.py` runs every baseline by toggling the partitioner
and decoder. All baselines run on the same frozen PLM encoder as BioBlobs
itself (default `esm2_static`).

| Variant | Override | Pooling op |
| --- | --- | --- |
| Mean-pool + MLP | `partitioners=none decoders=mlp pooling=mean` | explicit `PoolingOp(mean)` |
| Simple attention pool | `partitioners=none decoders=simple_attn` | inside the decoder (`Linear(D, 1)` + masked softmax) |
| Gated attention pool (Ilse 2018) | `partitioners=none decoders=attention_pool` | inside the decoder (additive `tanh(V·)·sigmoid(U·)` + softmax) |
| Light Attention (Stärk 2021) | `partitioners=none decoders=light_attn` | inside the decoder (Conv1d attention + max-pool concat) |
| BioBlobs + Light-Attn MIL head | `decoders=mil_light_attn`                | MIL head replaces gated attention with Light-Attention over `K` blobs |

Each non-MIL pooling decoder sets `consumes_batch_data = True`, so the
framework hands the full `batch_data` (with the residue mask) directly to
the decoder and skips the explicit pooling step.

### Encoder swap

The default encoder is ESM2-150M (`facebook/esm2_t30_150M_UR50D`).
SaProt-650M is also supported and reads from the same direct-to-mmap PLM
cache:

```bash
python train_bioblobs.py encoders=saprot_static
```

Build the cache with `preprocessing/prepare_plm_cache.py` (see below)
before the first training run.

### Other handy overrides

```bash
python train_bioblobs.py \
  datasets=pfam \
  training.batch_size=128 \
  training.epochs=100 \
  training.num_workers=8 \
  wandb.use_wandb=true wandb.job_name=bioblobs_pfam
```

## Preprocessing

Two dataset families are wired in. Each has its own staging script (download
+ PDB parsing) and a shared direct-to-mmap PLM embedding-cache builder:

| Dataset family | Stage script | PLM cache script | Output root |
| --- | --- | --- | --- |
| ProteinShake — EC | _auto, on first `train_bioblobs.py` run_ | `preprocessing/prepare_plm_cache.py datasets=ec` | `<data_dir>/ec_proteinshake/<split>/` |
| ProteinShake — GO | _auto_ | `preprocessing/prepare_plm_cache.py datasets=go` | `<data_dir>/go_proteinshake/<split>/` |
| ProteinShake — Pfam | _auto_ | `preprocessing/prepare_plm_cache.py datasets=pfam` | `<data_dir>/pfam_proteinshake/<split>/` |
| VenusX — site / fragment | `preprocessing/prepare_venusx_site_fragment.py datasets.target=<T> datasets.split_strategy=<S>` | `preprocessing/prepare_venusx_plm_cache.py datasets.target=<T>` | `<data_dir>/venusx_site_fragment/<T>/<S>/` |

PLM encoders supported by both cache scripts: `encoders=esm2_static`
(ESM2-150M, default) and `encoders=saprot_static` (SaProt-650M, requires a
local Foldseek binary). Add `encoders.precompute_device=cuda` to embed on
GPU. Each cache writes a single `embeddings.bin` + `meta.pt`; no
per-protein `.pt` intermediates.

VenusX fields:

| Field | Allowed values |
| --- | --- |
| `datasets.target`         | `Act`, `BindI`, `Evo`, `Motif`, `Dom` |
| `datasets.split_strategy` | `MF50`, `MF70`, `MF90` |

### ProteinShake PLM cache

For static ESM2 / SaProt residue embeddings on EC / GO / Pfam, build a single
direct-to-mmap cache (`embeddings.bin` + `meta.pt`) per (dataset × encoder)
pair before training:

```bash
python preprocessing/prepare_plm_cache.py \
  datasets=ec \
  encoders=esm2_static \
  encoders.precompute_device=cuda
```

The cache lives next to the prepared dataset under
`<data_dir>/<dataset>_proteinshake/<split>/{esm2_static_cache_mmap,saprot_static_cache_mmap}/`.

### VenusX (residue-level fragment classification)

VenusX is a fragment-classification benchmark hosted on HuggingFace
(`AI4Protein/VenusX_Res_<target>_<split_strategy>` and
`AI4Protein/VenusX_<target>_AlphaFold2_PDB`). Two preprocessing steps:

**1. Download + stage the dataset.** Pulls the HF parquet + PDB archive,
parses each PDB through the same backbone-completion checks used for
ProteinShake, and writes per-split CSVs under
`<data_dir>/venusx_site_fragment/<target>/<split_strategy>/`:

```bash
python preprocessing/prepare_venusx_site_fragment.py \
  datasets.target=Act \
  datasets.split_strategy=MF50
```

(See the table above for allowed `target` × `split_strategy` values.)

**2. Build the PLM mmap cache for that target.** Unions train+val+test,
dedupes by `protein_uid`, and writes one `embeddings.bin` + `meta.pt` next
to the staged target. Default encoder is ESM2; override
`encoders=saprot_static` for SaProt-650M:

```bash
python preprocessing/prepare_venusx_plm_cache.py \
  datasets.target=Act \
  encoders=esm2_static \
  encoders.precompute_device=cuda
```

After both steps the directory layout is:

```
<data_dir>/venusx_site_fragment/<target>/<split_strategy>/
├── pdb/
├── train_split.csv  val_split.csv  test_split.csv
├── samples.jsonl  metadata.json
└── esm2_static_cache_mmap/   # (or saprot_static_cache_mmap/)
    ├── embeddings.bin
    └── meta.pt
```

Train on a prepared VenusX target via the same `train_bioblobs.py`:

```bash
python train_bioblobs.py \
  datasets=venusx_site_fragment \
  datasets.target=Act \
  datasets.split_strategy=MF50 \
  tasks=venusx_fragment_multiclass \
  encoders=esm2_static
```

## Inference + blob interpretability

`inference.py` loads a trained checkpoint and scores user-provided PDB files,
optionally exporting per-protein blob assignments and a standalone 3D HTML
viewer for inspection.

```bash
# Score one or more PDBs and write predictions + blob JSON + 3D viewer
python inference.py \
  --checkpoint outputs/<run>/best.ckpt \
  --out-dir    outputs/<run>/inference \
  --pdb        my_protein.pdb \
  --interpret --save-html
```

See [INFERENCE.md](INFERENCE.md) for the full CLI, output schema, and notes
on running batches over a directory of PDBs.


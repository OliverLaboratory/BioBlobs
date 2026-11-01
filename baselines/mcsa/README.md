# M-CSA localization baselines

Thin wrappers around three third-party models used as published-model
baselines for catalytic-residue localization on M-CSA (see the paper's
"Baselines for Functional Site Localization" appendix). Each wrapper loads
the model's released weights unchanged and reproduces one published
attribution method as a per-residue score. These scripts do not retrain or
fine-tune anything.

| Baseline | One-line description |
| --- | --- |
| [`protdetr/score.py`](protdetr/score.py) | ProtDETR's encoder self-attention and decoder cross-attention, read out of its released EC-classification transformer. |
| [`proteinfer/score.py`](proteinfer/score.py) | Class-activation mapping (CAM) on ProteInfer's dilated-CNN EC classifier. |
| [`deepfri/score.py`](deepfri/score.py) | Grad-CAM on DeepFRI's graph-convolutional EC classifier, over a Cα contact graph. |

Each baseline needs its **own** Python environment (they pin mutually
incompatible, and in two cases very old, dependency versions) and its own
clone of the upstream repository plus released weights; none of those are
vendored here.

## Upstream repositories, weights and licenses

| Baseline | Repo | Commit | Weights | License |
| --- | --- | --- | --- | --- |
| ProtDETR | <https://github.com/yangzhao1230/ProtDETR> | `aa49809` | `saved_models/ProtDETR_split100.pt`, from the upstream README's "Pretrained Models" Google Drive link | MIT |
| ProteInfer | <https://github.com/google-research/proteinfer> | `0acc7b1` | `noxpnd_cnn_swissprot_ec_random_swiss-cnn_for_swissprot_ec_random-13685140.tar.gz` from `https://storage.googleapis.com/brain-genomics-public/research/proteins/proteinfer/models/zipped_models/` | Apache-2.0 |
| DeepFRI | <https://github.com/flatironinstitute/DeepFRI> | `3bd6c7e` | `newest_trained_models.tar.gz` from `https://users.flatironinstitute.org/~renfrew/DeepFRI_data/newest_trained_models.tar.gz` (the CPU-compatible model bundle, per the upstream README) | BSD-3-Clause |

## Environment setup

Each environment is Python-version- and framework-locked by the upstream
project; install into a **separate** virtualenv/conda env per baseline, not
into this repo's own environment.

### ProtDETR

```bash
git clone https://github.com/yangzhao1230/ProtDETR
cd ProtDETR && git checkout aa49809
# Neither saved_models/ProtDETR_split100.pt nor data/multi_func/split100.csv
# is tracked by the repo. Get the checkpoint from the upstream README's
# "Pretrained Models" Google Drive link (place the saved_models/ directory
# at the repo root), and the CSV from its "Data Requirements" section
# (place it under data/multi_func/).

conda create -n protdetr python=3.12
conda activate protdetr
pip install torch==2.7.0 fair-esm==2.0.0 scikit-learn numpy
```

This reproduces the versions actually used to produce the paper's numbers
(torch 2.7.0+cu128, fair-esm 2.0.0, Python 3.12; a CPU-only torch build
also works, just slower). The upstream README instead lists torch 1.13.0
and scipy 1.10.1 against an older fair-esm; the model is a standard
encoder-decoder transformer with no version-pinned custom ops, so any
reasonably recent torch + fair-esm 2.0.0 combination is interchangeable.

### ProteInfer

```bash
git clone https://github.com/google-research/proteinfer
cd proteinfer && git checkout 0acc7b1
mkdir -p models && cd models
wget https://storage.googleapis.com/brain-genomics-public/research/proteins/proteinfer/models/zipped_models/noxpnd_cnn_swissprot_ec_random_swiss-cnn_for_swissprot_ec_random-13685140.tar.gz
tar xzf noxpnd_cnn_swissprot_ec_random_swiss-cnn_for_swissprot_ec_random-13685140.tar.gz

conda create -n proteinfer python=3.7.16
conda activate proteinfer
pip install tensorflow==1.15.5 tensorflow-estimator==1.15.1 tensorflow-hub==0.9.0 \
            numpy==1.16.5 scipy==1.7.3 protobuf==3.20.3 absl-py==2.1.0
```

Pins above are the exact versions found in the conda environment that
produced the paper's numbers (`pip freeze` on that environment). CPU only;
`CUDA_VISIBLE_DEVICES` is not touched by the model code itself, so run
under a GPU-less environment or set it to `""` yourself.

### DeepFRI

```bash
git clone https://github.com/flatironinstitute/DeepFRI
cd DeepFRI
wget https://users.flatironinstitute.org/~renfrew/DeepFRI_data/newest_trained_models.tar.gz
tar xzf newest_trained_models.tar.gz    # extracts into ./trained_models/

conda create -n deepfri python=3.7.16
conda activate deepfri
pip install tensorflow==2.3.1 tensorflow-estimator==2.3.0 \
            numpy==1.18.5 scipy==1.7.3 biopython==1.76 networkx==2.4
```

Pins above are the exact versions found in the conda environment that
produced the paper's numbers. CPU only (`score.py` sets
`CUDA_VISIBLE_DEVICES=""` itself).

## Building the common input file

All three `score.py` scripts read the same per-chain input file. Build it
from a curated single-chain M-CSA JSON (schema: a list of records with
`pdb_id`, `chain_used`, `ec_number`, `catalytic_residues`) and a directory
of PDB structure files:

```bash
python make_inputs.py \
    --mcsa-json /path/to/mcsa_single_chain.json \
    --pdb-dir /path/to/pdbs \
    --out mcsa_inputs.jsonl
```

This runs in any modern Python (3.9+) with no extra dependencies, and does
not need any of the three baseline environments above.

### Common input format

A JSON array (`*.json`) or newline-delimited JSON (`*.jsonl`) of records,
one per protein chain:

```json
{"id": "1b73_A", "seq": "MSLAKD...", "ca": [[12.1, 4.3, -8.0], ...], "ec": "5.1.1.3"}
```

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | string | A chain identifier, e.g. `"<pdb_id>_<chain>"`. Used as the key in every output file. |
| `seq` | string | One-letter amino-acid sequence. All output scores are aligned to this ordering. |
| `ca` | list of `[x, y, z]`, or `null` | One Cα coordinate per residue of `seq`, in the same order; `null` if no structure is available (only DeepFRI needs this field). |
| `ec` | string, or `null` | Annotated EC number, e.g. `"5.1.1.3"`; `null` if unknown (ProteInfer and DeepFRI use this for class selection; ProtDETR ignores it). |

## Running each baseline

One command per baseline, each run in its own environment:

```bash
# ProtDETR (torch env)
python protdetr/score.py \
    --inputs mcsa_inputs.jsonl \
    --upstream-dir /path/to/ProtDETR \
    --out scores_protdetr.json
# writes scores_protdetr.json (encoder) and scores_protdetr_dec.json (decoder)

# ProteInfer (TF 1.15 / py3.7 env)
python proteinfer/score.py \
    --inputs mcsa_inputs.jsonl \
    --upstream-dir /path/to/external_baselines \
    --out scores_proteinfer.json

# DeepFRI (TF 2.3.1 / py3.7 env)
python deepfri/score.py \
    --inputs mcsa_inputs.jsonl \
    --upstream-dir /path/to/DeepFRI \
    --out scores_deepfri.json
```

For ProteInfer, `--upstream-dir` should be a directory that contains the
cloned repo at `<upstream-dir>/proteinfer` and the unzipped SavedModel at
`<upstream-dir>/models/<model-name>` (matching the clone layout above);
use `--repo-dir` / `--model-dir` to point at different locations.

## Output format

Each script writes a JSON object mapping chain id to a list of per-residue
scores, aligned to `seq` (truncated to 1,022 residues for ProtDETR, which
also writes a second `*_dec.json` file for its decoder read-out):

```json
{"1b73_A": [0.01, -0.03, 0.42, ...], "...": [...]}
```

Chains a baseline could not score (misaligned activations, a missing
structure, or a constant/non-finite saliency map) are simply absent from
the output; each script logs the reason via the standard `logging` module.
Scores keep their original sign and scale from each upstream method (CAM
and encoder/decoder attention are signed or unit-norm; DeepFRI's Grad-CAM
is rescaled to `[0, 1]` by the upstream implementation itself) -- no
additional normalization is applied here.

# `inference.py` — usage guide

Self-contained inference for a trained BioBlobs model on
user-provided PDB files. No prebuilt caches needed: the script parses each
PDB in process, embeds the sequence with ESM2 on the same GPU as the model,
and runs the BioBlobs forward pass.

## Quick reference

| Goal | Command |
| --- | --- |
| Score a single PDB | `--pdb my.pdb` |
| Score multiple PDBs | `--pdb a.pdb b.pdb.gz` |
| Score a directory | `--pdb-dir /path/to/pdbs/` |
| + per-protein JSON | add `--interpret` |
| + 3D HTML viewer | add `--save-html` (implies `--interpret`) |
| + per-protein blob embeddings | add `--save-blob-embeddings` |

The script always writes `predictions.csv`, `inference.pt`, and
`inference_summary.json` to `--out-dir`. With `--interpret` it also writes
`manifest.json` and `proteins/<name>.{json,pdb,html?}`.

---

## What the user needs locally

| Artifact | Notes |
| --- | --- |
| `best.ckpt` (or `last.ckpt`) | Trained checkpoint from `train_bioblobs.py` |
| `config.json` | OmegaConf snapshot used to rebuild the model. By default looked up alongside the checkpoint; pass `--config` to override |
| `token_map.json` | EC-L3 string ↔ index dict, needed to render `pred_ec_l3` strings. Read from `--data-dir` (default: `data/ec_proteinshake/`). Ship it alongside the checkpoint |
| `.pdb` / `.pdb.gz` files | What you want classified |

The HuggingFace ESM2 model (`facebook/esm2_t30_150M_UR50D`, ~600 MB) is
downloaded automatically the first time and cached at `~/.cache/huggingface/`.

## What the user does NOT need

* `data/ec_proteinshake/cache/structures.pt` (the prebuilt structure cache)
* `data/ec_proteinshake/esm2_static_cache_mmap/` (the prebuilt ESM2 mmap)
* `train_split_*.csv` / `class_priors.json` / training data

Everything is computed from the PDB at inference time.

---

## Examples

```bash
# Single PDB
./.venv/bin/python inference.py \
    --checkpoint outputs/bioblobs/single_<TS>/best.ckpt \
    --out-dir    outputs/bioblobs/single_<TS>/inference \
    --pdb        my_protein.pdb

# Multiple files at once
./.venv/bin/python inference.py \
    --checkpoint outputs/bioblobs/single_<TS>/best.ckpt \
    --out-dir    outputs/bioblobs/single_<TS>/inference \
    --pdb        a.pdb b.pdb.gz c.pdb

# Whole directory + interpretability JSON + 3D HTML viewer
./.venv/bin/python inference.py \
    --checkpoint outputs/bioblobs/single_<TS>/best.ckpt \
    --out-dir    outputs/bioblobs/single_<TS>/inference \
    --pdb-dir    /path/to/pdbs/ \
    --interpret --save-html
```

The protein name is taken from the filename stem (`my_protein.pdb` →
`my_protein`; `.pdb.gz` is also accepted). `true_label` is always set to
`-1` since PDBs carry no labels — `inference_summary.json` therefore reports
only `num_proteins` and `elapsed_seconds`, no accuracy / macro_f1.

**Cost:** ~1 s/protein on H200 (PDB parse + ESM2-150M forward + BioBlobs
forward). ESM2 occupies ~1.5 GB of additional VRAM beyond the BioBlobs model
(model + ESM2 ≈ 7 GB total during inference).

---

## Always-on outputs

### `predictions.csv`

```csv
id,true_label,true_ec_l3,pred_label,pred_ec_l3,top1_prob
my_protein,-1,,120,2.4.1,0.9999
```

Use `pandas.read_csv` for downstream analysis.

### `inference.pt`

`torch.load(...)` returns a Python list. Each element is a dict per protein:

| Field | Description |
| --- | --- |
| `name` | Protein id (filename stem) |
| `true_label` / `true_ec_l3` | `-1` / `None` (no ground truth in PDB inputs) |
| `pred_label` / `pred_ec_l3` | Argmax prediction |
| `logits` | `[num_classes]` |
| `top_k_probs` / `top_k_classes` | Top-k softmax (default k=5) |
| `seed_indices` | `[K]` long — residue index of each blob's seed |
| `seed_scores` | `[K]` float |
| `blob_assignment` | `[seq_len]` long — hard blob index per residue (`-1` for invalid) |
| `blob_assignment_soft` | `[seq_len, K]` float (only with `--save-soft`) |
| `blob_embeddings` | `[K, D]` float16 (only with `--save-blob-embeddings`) — the membership-weighted mean of per-residue ESM2 features per blob slot, i.e. what the MIL decoder consumes. `D=640` for ESM2-150M |
| `blob_valid` | `[K]` bool (only with `--save-blob-embeddings`) — True where the blob slot has a valid seed; False slots have all-zero embeddings |

### `inference_summary.json`

```json
{
  "num_proteins": 12,
  "elapsed_seconds": 13.4,
  "blob_embeddings_saved": false
}
```

`blob_embeddings_saved` flips to `true` when `--save-blob-embeddings` is set,
so downstream loaders can detect the new fields without inspecting every
record.

---

## Interpretability outputs (`--interpret`)

Adds rich per-protein artifacts under `<out-dir>/proteins/<name>.{json,pdb,html?}`
plus a top-level `manifest.json`. **Default off** because the per-protein
JSON can be hundreds of KB per protein.

### What flips on

* `model.partitioner.emit_interpretability = True` — partitioner attaches
  per-graph blob payload (assignment matrix, seed indices/scores).
* `model.decoder.return_attention = True` — MIL decoder exposes
  `attention_weights[B, K]` and `blob_mask[B, K]`.
* `model.decoder.return_instance_predictions = True` — MIL decoder exposes
  `instance_logits[B, K, num_classes]` and `instance_probabilities`.

### Per-protein JSON (`proteins/<name>.json`)

Schema follows `bioblobs.analysis.posthoc_blob_interpretability.build_protein_export`:

| Top-level field | Type | Description |
| --- | --- | --- |
| `protein_id` | str | Same as the manifest row's `name` |
| `input_mode` | `"pdb"` | Always (this script is PDB-only) |
| `pdb_path` | str | Absolute path to the decompressed PDB copy (used by `blob_viewer.render_blob_html`) |
| `assignment_type` | `"soft"` | Soft membership matrix is always emitted by the BioBlobs partitioner |
| `assignment_matrix` | `[N_valid, K]` | Per-residue × per-blob soft assignment |
| `soft_assignments` | same as above | Duplicate field (legacy) |
| `hard_blob_index_per_valid_residue` | `[N_valid]` | Argmax (with tie-breaking) into the active blobs |
| `active_blob_indices` | list[int] | The K' ≤ K blobs whose mask is True |
| `ranked_blob_indices` | list[int] | Active blobs reordered by attention weight (descending) |
| `top_blob_attention` | float | Attention of the rank-1 blob |
| `residues.residue_numbers` | list[int] | PDB residue numbers (one per row of seq/coords) |
| `residues.valid_residue_mask` | list[bool] | True where coords are finite (backbone-complete) |
| `residues.valid_residue_indices` | list[int] | Indices into the full seq where coords are valid |
| `prediction.pred_label_index` | int | Argmax of bag_logits |
| `prediction.pred_label_text` | str | EC L3 string (e.g. `"2.4.1"`) |
| `prediction.true_label_index` | null | (no ground truth in PDB input mode) |
| `prediction.true_label_text` | null | |
| `prediction.bag_logits` | list[num_classes] | Pre-softmax classifier output |
| `prediction.bag_probabilities` | list[num_classes] | Softmax of bag_logits |
| `prediction.confidence` | float | Max bag probability |
| `blobs[i]` | per-blob record | See below |
| `checkpoint_path`, `config_path` | str | For traceability |

**Per-blob record:**

| Field | Type | Description |
| --- | --- | --- |
| `blob_index` | int | Position in the original K-blob layout |
| `rank` | int | 1 = highest attention (1-based) |
| `attention` | float | MIL attention weight |
| `size` | int | Number of residues whose hard assignment lands on this blob |
| `residue_indices` | list[int] | Indices into `residues.residue_numbers` |
| `residue_numbers` | list[int] | PDB-numbered residue ids |
| `soft_mass` | float | `Σ_i a_{i,k}` over the blob's column |
| `soft_count_0p5` | int | How many residues have membership > 0.5 in this blob |
| `effective_k_hoyer` | float | Hoyer effective support: `(Σ a)² / Σ a²` |
| `instance_logits` | list[num_classes] | The blob's own classification output (pre-softmax) |
| `instance_probabilities` | list[num_classes] | Softmax of instance_logits |

### 3D HTML viewer (`--save-html`, implies `--interpret`)

For each protein, also writes `proteins/<name>.html` — a self-contained
3Dmol.js viewer that loads the PDB inline, color-codes blobs by attention
rank, and shows a side panel with predicted EC, confidence, and per-blob
residue lists.

Open in any browser; no server needed. Internet access is required at
view time (3Dmol.js is loaded from a CDN); for offline use, save the HTML
once on a connected machine to populate the browser cache, or vendor the
script.

```bash
xdg-open outputs/bioblobs/single_<TS>/inference/proteins/my_protein.html
```

### Manifest (`manifest.json`)

Top-level index over all per-protein artifacts:

```json
{
  "n_proteins": 12,
  "checkpoint": "<absolute/path/to/best.ckpt>",
  "config":     "<absolute/path/to/config.json>",
  "rows": [
    {
      "name":        "my_protein",
      "pred_label":  120,
      "pred_ec_l3":  "2.4.1",
      "true_label":  -1,
      "true_ec_l3":  null,
      "confidence":  0.9999972581863403,
      "json_path":   "proteins/my_protein.json",
      "pdb_path":    "proteins/my_protein.pdb",
      "html_path":   "proteins/my_protein.html"
    }
  ]
}
```

Paths are relative to `<out-dir>` so the directory is portable.

---

## All flags

```text
--checkpoint PATH            [required] best.ckpt or last.ckpt from training
--config PATH                config.json (default: alongside the checkpoint)
--out-dir PATH               [required] where outputs land

# PDB inputs
--pdb file1.pdb [file2.pdb.gz ...]
--pdb-dir DIR                globs *.pdb and *.pdb.gz
--min-completion FLOAT       default: 0.95 — backbone-completeness threshold;
                             PDBs below this are skipped with a warning

# Interpretability
--interpret                  per-protein JSON under proteins/<name>.json + manifest.json
--save-html                  also render 3Dmol HTML per protein (implies --interpret)
--save-soft                  include soft [N,K] assignment matrix in inference.pt
--save-blob-embeddings       include per-protein blob embeddings ([K,D] float16)
                             plus a [K] bool validity mask in inference.pt.
                             Adds ~K*D*2 bytes per protein (~40 KB at K=16, D=640).
                             Useful for downstream KNN/clustering on blob-level
                             representations without rerunning the model.

# Inference compute
--device cuda:0
--top-k 5                    how many top classes to record per protein
--data-dir PATH              location of token_map.json (default: data/ec_proteinshake)
```

---

## Common pitfalls

* **`config.json` not next to the checkpoint** → pass `--config` explicitly.
* **`token_map.json` not in `--data-dir`** → ship it next to the checkpoint
  and pass `--data-dir <dir>` pointing there. Without it the script can't
  decode `pred_ec_l3` strings and exits with an error.
* **Wrong `--device cuda:N`** → if you also set `CUDA_VISIBLE_DEVICES=N`,
  use `--device cuda:0` (the only visible card is renumbered to 0 inside
  the masked process).
* **`--save-html` without internet** → 3Dmol.js is loaded from a CDN at
  view time; for offline use, save the HTML once on a connected machine
  (the JS gets cached) or vendor the script.
* **Backbone-completeness skips** → AlphaFold predictions usually pass; X-ray
  structures with disordered loops can fall below the default 0.95
  threshold. Lower with `--min-completion 0.8` if needed.

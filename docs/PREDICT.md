# Predict EC numbers and catalytic sites

`inference.py` runs a BioBlobs model on your own PDB files. The released model predicts each
enzyme's EC number to the third level and scores every residue as a candidate catalytic residue.
It parses each structure, computes residue embeddings with the model's protein language model, and
runs BioBlobs. No precomputed caches are needed.

## Models

- **Released model**, `bioblobs-mcsa-rvq-k8` (on [Hugging Face](https://huggingface.co/xwang112358/BioBlobs)): third-level EC
  classes (242 classes), ESM2-T30 embeddings, 8 blobs with residual vector quantization. It is the
  model behind the paper's M-CSA results. The folder holds `model.safetensors`, `config.yaml` and
  `classes.json`.
- **Your own model**: any `best.ckpt` written by `train_bioblobs.py`. Such a checkpoint has no
  `classes.json`, so classes are reported by index.

## Usage

```bash
# Download the released model
hf download xwang112358/BioBlobs --local-dir bioblobs-mcsa-rvq-k8

# Every .pdb file in a folder, with the released model
python inference.py --model bioblobs-mcsa-rvq-k8 --pdb-dir my_pdbs --out results

# One chain of one structure, with a model you trained
python inference.py --model outputs/<run>/seed0/best.ckpt --pdb 1abc.pdb --chain A --out results
```

| Option | Meaning |
| --- | --- |
| `--model` | released model folder or `.ckpt` file |
| `--pdb` / `--pdb-dir` | PDB files, or a folder of them |
| `--chain` | chain to use (default: all residues in the file) |
| `--out` | output folder |
| `--top-k` | number of top classes to report (default 5) |
| `--foldseek-bin` | Foldseek binary, needed for SaProt models |
| `--device` | `cuda` or `cpu` (default: `cuda` when available) |

## Outputs

`results/predictions.csv` has one row per protein: `name`, `num_residues`, `top_class` and
`probability`.

`results/proteins/<name>.json` holds, for each protein:

- `top_classes`: the top classes with their probabilities.
- `blobs`: the blobs ranked by importance score. Each lists its `rank`, `blob` index,
  `importance`, `seed_residue` and the `residues` with membership of at least 0.5.
- `residue_numbers`, `max_membership` and `weighted_membership`: per-residue scores, in the
  order of the PDB residue numbers.

Residue numbers follow the PDB file.

## Notes

- ESM2 and SaProt weights download from Hugging Face on first use. Proteins longer than 1,022
  residues are embedded in overlapping windows.
- Embeddings are stored in float16, as in training. Residue scores can therefore differ from the
  evaluation pipeline by about 1e-5.
- Probabilities are softmax scores, not calibrated confidence.

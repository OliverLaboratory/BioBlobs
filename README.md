[![arXiv](https://img.shields.io/badge/arXiv-2510.01632-b31b1b.svg)](https://arxiv.org/abs/2510.01632)
---

# BioBlobs: Differentiable Graph Partitioning for Protein Representation Learning

* Generate rich reisue-level and protein-level embeddings for proteins that capture protein structure modularity.
* Compute importance scores for protein substructures (blobs)
* Visualize learned function-critical blobs

![](https://raw.githubusercontent.com/OliverLaboratory/BioBlobs/refs/heads/main/blobs.png)

## Cite

```
@article{wang2025bioblobs,
  title={BioBlobs: Differentiable Graph Partitioning for Protein Representation Learning},
  author={Wang, Xin and Oliver, Carlos},
  journal={arXiv preprint arXiv:2510.01632},
  year={2025}
}
```

## Installation

Create environment:
```bash
conda create -n bioblobs python=3.12
conda activate bioblobs
```

Install dependencies:
```bash
# PyTorch with CUDA 12.8
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# PyTorch Geometric
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

# Other dependencies
pip install pytorch-lightning scikit-learn hydra-core omegaconf wandb tqdm numpy pandas matplotlib seaborn proteinshake
```

## Usage

## Inference

To run the model on a dataset and get node embeddings and  blob assignments:

```
python inference.py
```

### Multi-stage Training

Train BioBlobs with multi-stage training (BioBlobs + VQ Codebook):

Train on Enzyme Commission dataset:
```bash
python run_bioblobs_multistage.py \
    data.dataset_name=ec \  
    data.split=structure \  # data split 
    data.edge_types=knn_16 \ 
    train.stage0.epochs=120 \
    train.stage1.epochs=30 \
```

Train on Gene Ontology dataset:
```bash
python run_bioblobs_multistage.py \
    data.dataset_name=go \
```

## Output Structure

Training outputs are organized as follows:
```
outputs/bioblobs_multistage_test/{dataset}/{split}/YYYY-MM-DD-HH-MM-SS/
├── final_summary.json                  # Overall training summary
├── run_bioblobs_multistage.log         # Training logs
├── stage0/                             # Stage 0: BioBlobs training
│   ├── best-stage0-epoch=XX-val_*.ckpt # Best checkpoint (stage 0)
│   ├── last.ckpt                       # Last checkpoint (stage 0)
│   └── stage0_results.json             # Stage 0 results
├── stage1/                             # Stage 1: VQ Codebook training
│   ├── best-stage1-epoch=XX-val_*.ckpt # Best checkpoint (stage 1)
│   ├── last.ckpt                       # Last checkpoint (stage 1)
│   ├── stage1_results.json             # Stage 1 results
│   └── interpretability/               # Interpretability analysis
│       └── test_interpretability.json
├── codebook_initialization/            # VQ Codebook initialization
    └── initialization_stats.json

```





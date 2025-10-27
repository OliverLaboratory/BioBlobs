# BioBlobs

Differentiable  Graph Partitioning for Protein Representation Learning

## Installation

Create environment:
```bash
conda create -n bioblobs python=3.12
conda activate bioblobs
```

Install dependencies:
```bash
# PyTorch with CUDA 12.8
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# PyTorch Geometric
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

# Other dependencies
pip install pytorch-lightning scikit-learn hydra-core omegaconf wandb tqdm numpy pandas matplotlib seaborn proteinshake
```

## Usage

### Multi-stage Training

Train BioBlobs with multi-stage training (BioBlobs + VQ Codebook):

Train on Enzyme Commission dataset:
```bash
python run_bioblobs_multistage.py \
    data.dataset_name=ec \  
    data.split=structure \  # data split 
    multistage.stage0.epochs=120 \
    multistage.stage1.epochs=30 \
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





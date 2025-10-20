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

### 1. Train BioBlobs without codebook 

Train on Enzyme Commission dataset:
```bash
python run_partgvp.py \
    data.dataset_name=enzymecommission \
    data.split=random \
    train.epochs=120 \
    train.use_wandb=true
```

### 2. Resume Training with Codebook 

Resume from PartGVP checkpoint with VQ codebook:
```bash
python run_partoken_resume.py \
    data.dataset_name=enzymecommission \
    resume.partgvp_checkpoint_path=outputs/YYYY-MM-DD/HH-MM-SS/best-partgvp-*.ckpt \
    multistage.stage0.epochs=50
```

## Output Structure

Training outputs are organized as follows:
```
outputs/YYYY-MM-DD/HH-MM-SS/
├── results_summary.json          # Training metrics and test results
├── best-partgvp-*.ckpt          # Best checkpoint (PartGVP)  
├── last.ckpt                     # Last checkpoint
├── final_partgvp_model.ckpt      # Final saved model
├── interpretability/             # Analysis results
│   ├── test_interpretability.json
│   └── initial.json              # (resume training only)
├── codebook_initialization/      # (resume training only)
│   └── initialization_stats.json
└── .hydra/                       # Hydra configuration logs
    └── config.yaml
```

## Key Files

- `run_bioblobs.py`: Train BIOBLOS w/o codebook baseline
- `run_bioblobs_codebook_resume.py`: Resume training with VQ codebook to get BIOBLOS results
- `train_lightling.py`: Lightning modules for both methods
- `partoken_model.py`: Core model architecture
- `conf/`: Configuration files for different datasets/settings

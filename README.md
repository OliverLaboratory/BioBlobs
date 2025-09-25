# BioBlobs: Differentiable Graph Partitioning for Protein Representation Learning

BioBlobs is a multi-stage protein classification model that combines graph neural networks with vector quantization for interpretable protein function prediction. The model uses a two-stage training approach: baseline training followed by joint fine-tuning with codebook learning.

## Overview

The multi-stage training process consists of:

1. **Stage 0 (Baseline)**: Train the encoder and partitioner without vector quantization
2. **Stage 1 (Joint Fine-tuning)**: Enable codebook with K-means initialization and gradual loss ramping

## Quick Start

### Basic Usage

Run multi-stage training with default settings:

```bash
python run_partoken_multistage.py
```

### Custom Configuration

Override specific parameters using Hydra's command-line interface:

```bash
python run_partoken_multistage.py \
    data.dataset_name=enzymecommission \
    train.batch_size=64 \
    train.use_wandb=true \
    multistage.stage0.epochs=20 \
    multistage.stage1.epochs=15
```

## Configuration

The training is configured through `conf/config_partoken_multistage.yaml`. Key sections include:

### Data Configuration

```yaml
data:
  dataset_name: enzymecommission  # Available: enzymecommission, geneontology, proteinfamily, scop
  split: structure               # Split type: structure, random
  split_similarity_threshold: 0.7
  data_dir: ./data
```

### Model Parameters

```yaml
model:
  # GVP-GNN parameters
  node_in_dim: [6, 3]
  node_h_dim: [100, 16]
  edge_in_dim: [32, 1]
  edge_h_dim: [32, 1]
  num_layers: 3
  drop_rate: 0.1
  
  # Partitioner parameters
  max_clusters: 10
  cluster_size_max: 15
  termination_threshold: 0.95
  
  # Codebook parameters
  codebook_size: 512
  codebook_beta: 0.25
  codebook_decay: 0.99
```

### Multi-Stage Training

```yaml
multistage:
  # Stage 0: Baseline training (no codebook)
  stage0:
    epochs: 20
    lr: 1e-4
    bypass_codebook: true
    
  # Stage 1: Joint fine-tuning with codebook
  stage1:
    epochs: 20
    lr: 5e-5
    kmeans_init: true          # Initialize codebook with K-means
    loss_ramp:
      enabled: true
      ramp_epochs: 5           # Gradually increase VQ loss weights
```

## Command Line Examples

### Training on Different Datasets

```bash
# Enzyme Commission dataset
python run_partoken_multistage.py data.dataset_name=enzymecommission

# Gene Ontology dataset
python run_partoken_multistage.py data.dataset_name=geneontology

# Protein Family dataset
python run_partoken_multistage.py data.dataset_name=proteinfamily

# SCOP dataset
python run_partoken_multistage.py data.dataset_name=scop
```

### Adjusting Training Parameters

```bash
# Quick test run (fewer epochs)
python run_partoken_multistage.py \
    multistage.stage0.epochs=5 \
    multistage.stage1.epochs=5 \
    train.batch_size=32

# Extended training with larger codebook
python run_partoken_multistage.py \
    multistage.stage0.epochs=30 \
    multistage.stage1.epochs=25 \
    model.codebook_size=1024 \
    train.batch_size=64

# Enable Weights & Biases logging
python run_partoken_multistage.py \
    train.use_wandb=true \
    train.wandb_project=my-partoken-project
```

### GPU Configuration

```bash
# Specific GPU
export CUDA_VISIBLE_DEVICES=1
python run_partoken_multistage.py

# Multiple GPUs (modify in script if needed)
export CUDA_VISIBLE_DEVICES=0,1
python run_partoken_multistage.py
```

## Output Structure

The training creates organized outputs under `./outputs/partoken-multistage/{dataset}/multistage_{date}/`:

```
outputs/
└── partoken-multistage/
    └── enzymecommission/
        └── multistage_2025-09-08/
            ├── stage_0/
            │   ├── best-stage0-epoch=19-val_acc=0.850.ckpt
            │   ├── last.ckpt
            │   └── stage_0_final.ckpt
            ├── stage_1/
            │   ├── best-stage1-epoch=19-val_acc=0.875.ckpt
            │   ├── last.ckpt
            │   └── stage_1_final.ckpt
            ├── interpretability/
            │   └── test_interpretability.json
            ├── final_multistage_model.ckpt
            └── checkpoint_summary.json
```

## Key Features

### Multi-Stage Learning
- **Stage 0**: Focuses on learning good protein representations without quantization overhead
- **Stage 1**: Introduces codebook learning with proper initialization and gradual loss ramping

### K-means Initialization
The codebook is initialized using K-means clustering on protein representations from Stage 0, providing a better starting point than random initialization.

### Loss Ramping
Vector quantization losses are gradually increased during Stage 1 to avoid training instability:
- Start with reduced VQ loss weights
- Linearly increase to full weights over specified epochs

### Interpretability Analysis
After training, the model automatically runs interpretability analysis to understand:
- Cluster importance distributions
- Codebook utilization patterns
- Attention weights and protein motifs

### Checkpointing
- Stage-specific checkpoints for resuming individual stages
- Best model checkpoints based on validation accuracy
- Final combined model checkpoint
- Detailed checkpoint summary with metrics

## Testing Script

A test script is provided for quick validation:

```bash
./test_partoken_multistage.sh
```

This runs a shortened version of training suitable for testing the pipeline.

## Tips for Best Results

1. **Batch Size**: Start with 64-128, adjust based on GPU memory
2. **Learning Rates**: Stage 1 LR should be 2-5x lower than Stage 0
3. **Codebook Size**: 512-1024 works well for most protein datasets
4. **K-means Initialization**: Use 50-100 batches for stable initialization
5. **Loss Ramping**: 5-10 epochs usually sufficient for smooth transition

## Monitoring Training

### Console Output
The training provides detailed console logging for each stage:
- Stage progress and metrics
- K-means initialization status
- Interpretability analysis results

### Weights & Biases Integration
Enable W&B logging for detailed metrics tracking:
```bash
python run_partoken_multistage.py train.use_wandb=true
```

## Troubleshooting

### Common Issues

**CUDA Out of Memory**: Reduce batch size
```bash
python run_partoken_multistage.py train.batch_size=32
```

**Slow K-means Initialization**: Reduce number of batches
```bash
python run_partoken_multistage.py multistage.stage1.kmeans_batches=25
```

**Training Instability**: Enable or extend loss ramping
```bash
python run_partoken_multistage.py \
    multistage.stage1.loss_ramp.enabled=true \
    multistage.stage1.loss_ramp.ramp_epochs=10
```

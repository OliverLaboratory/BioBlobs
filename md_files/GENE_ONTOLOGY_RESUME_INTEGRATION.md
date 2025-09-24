# Gene Ontology Resume Training Integration

This document explains how to use the newly integrated Gene Ontology dataset support in the ParToken resume training workflow.

## Overview

The ParToken resume training system now supports multi-label classification for the Gene Ontology dataset. This integration allows you to:

1. Resume training from a PartGVP checkpoint trained on Gene Ontology
2. Add a VQ codebook to the model architecture
3. Perform joint training of the backbone and codebook
4. Use appropriate multi-label metrics (FMax, precision, recall)

## Key Components

### 1. Multi-label Lightning Module
- `ParTokenResumeTrainingMultiLabelLightning`: Extends the base resume training module with multi-label support
- Uses `BCEWithLogitsLoss` instead of `CrossEntropyLoss`
- Tracks FMax, precision, and recall metrics
- Accumulates predictions for epoch-level metric computation

### 2. Model Creation Function
- `create_partoken_resume_multilabel_model_from_checkpoint()`: Creates multi-label resume models
- Automatically detects and uses `PartGVPMultiLabelLightning` for weight transfer
- Preserves codebook parameters while loading backbone weights

### 3. Dataset Detection
- `run_partoken_resume.py` automatically detects Gene Ontology dataset
- Uses appropriate model class, metrics, and checkpoint monitoring
- Handles both single-label and multi-label classification seamlessly

## Usage Examples

### Basic Usage
```bash
# Resume from PartGVP checkpoint for Gene Ontology
python run_partoken_resume.py \
    --config-name=config_partoken_resume_geneontology \
    resume.partgvp_checkpoint_path=/path/to/partgvp_geneontology_checkpoint.ckpt
```

### Test Mode (Smaller Dataset)
```bash
# Resume with test mode for quick validation
python run_partoken_resume.py \
    --config-name=config_partoken_resume_geneontology \
    resume.partgvp_checkpoint_path=/path/to/checkpoint.ckpt \
    data.test_mode=true
```

### Custom Configuration
```bash
# Resume with custom settings
python run_partoken_resume.py \
    --config-name=config_partoken_resume_geneontology \
    resume.partgvp_checkpoint_path=/path/to/checkpoint.ckpt \
    multistage.stage0.epochs=80 \
    train.batch_size=128 \
    model.codebook_size=1024
```

### With Weights & Biases Logging
```bash
# Enable wandb logging
python run_partoken_resume.py \
    --config-name=config_partoken_resume_geneontology \
    resume.partgvp_checkpoint_path=/path/to/checkpoint.ckpt \
    train.use_wandb=true
```

## Configuration Details

### Key Parameters for Gene Ontology

```yaml
# Data configuration
data:
  dataset_name: geneontology  # Triggers multi-label mode
  split: structure
  batch_size: 256  # Larger batches work well for Gene Ontology

# Training configuration  
multistage:
  stage0:
    epochs: 60  # More epochs needed for complex multi-label task
    lr: 1e-3    # Higher learning rate for Gene Ontology

# Model configuration
model:
  max_clusters: 3          # Fewer clusters for Gene Ontology
  codebook_size: 512       # Reasonable codebook size
  lambda_vq: 1.0          # VQ loss weight
  lambda_psc: 0.01        # Protein structure consistency loss
```

## Expected Outputs

### Metrics Logged
- **Training**: `train_fmax`, `train_precision`, `train_recall`, `train_bce_loss`
- **Validation**: `val_fmax`, `val_precision`, `val_recall`, `val_bce_loss`  
- **Testing**: `test_fmax`, `test_precision`, `test_recall`, `test_bce_loss`
- **VQ Metrics**: `*_vq_loss`, `*_codebook_loss`, `*_commitment_loss`, `*_perplexity`

### Checkpoint Monitoring
- **Metric**: `val_fmax` (instead of `val_acc` for single-label)
- **Mode**: `max` (higher FMax is better)
- **Filename**: `best-partoken-{epoch:02d}-{val_fmax:.3f}.ckpt`

### Results Summary
The results summary will include:
```json
{
  "checkpoints": {
    "best": {
      "test_fmax": 0.8234,
      "test_precision": 0.7891,
      "test_recall": 0.8567,
      "test_bce_loss": 0.1234,
      "test_vq_loss": 0.0567
    }
  }
}
```

## Workflow Comparison

### Original PartGVP Training
```bash
python run_partgvp.py --config-name=config_partgvp_geneontology
```

### Resume Training with Codebook
```bash
# Step 1: Train PartGVP baseline (if not done)
python run_partgvp.py --config-name=config_partgvp_geneontology

# Step 2: Resume with codebook from the best checkpoint
python run_partoken_resume.py \
    --config-name=config_partoken_resume_geneontology \
    resume.partgvp_checkpoint_path=./outputs/partgvp_geneontology/.../best-partgvp-*.ckpt
```

## Integration Features

### Automatic Detection
- The system automatically detects `dataset_name: geneontology`
- Switches to multi-label model classes and metrics
- No manual configuration needed

### Weight Transfer
- Transfers all compatible weights from PartGVP to ParToken
- Skips codebook weights (newly initialized)
- Reports transfer statistics

### Comprehensive Logging
- All VQ-related metrics are logged
- Multi-label metrics computed at both step and epoch level
- Interpretability analysis with appropriate metrics

## Troubleshooting

### Common Issues
1. **Checkpoint not found**: Ensure the PartGVP checkpoint path is correct
2. **Shape mismatches**: Verify the model configurations match between PartGVP and resume training
3. **Memory issues**: Reduce batch size or number of workers
4. **Poor convergence**: Try adjusting learning rate or VQ loss weights

### Verification
Run the integration test to verify everything is working:
```bash
./test_gene_ontology_resume_integration.sh
```

This should show all components are correctly integrated and ready for use.
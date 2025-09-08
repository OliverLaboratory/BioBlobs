#!/bin/bash
# Test script for modified run_partoken.py with PyTorch Lightning

# Activate conda environment
conda activate partoken

# Run the partoken model with small test settings
python run_partoken.py \
    train.epochs=2 \
    train.batch_size=4 \
    train.use_wandb=false \
    model.max_clusters=5 \
    model.codebook_size=64

echo "Test completed successfully!"

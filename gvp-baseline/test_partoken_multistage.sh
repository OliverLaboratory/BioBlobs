#!/bin/bash

echo "🧬 Testing Multi-Stage ParToken Training"
echo "========================================"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Run multi-stage training
echo "🚀 Starting multi-stage training..."
python run_partoken_multistage.py \
    data.dataset_name=enzymecommission \
    train.batch_size=32 \
    train.use_wandb=false \
    multistage.stage0.epochs=5 \
    multistage.stage1.epochs=2 \
    multistage.stage2.epochs=5

echo "✅ Multi-stage training completed!"

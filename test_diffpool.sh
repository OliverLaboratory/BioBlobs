#!/bin/bash

# Test DiffPool with smaller configurations for quick validation
# Usage: bash test_diffpool.sh

echo "Running DiffPool test with small configuration..."

# Quick test on Enzyme Commission with test mode enabled
CUDA_VISIBLE_DEVICES=6 python run_diffpool.py \
    --config-path=conf \
    --config-name=config_diffpool \
    train.epochs=5 \
    data.test_mode=true \
    train.use_wandb=false \
    train.wandb_project=diffpool-test \
    data.dataset_name=enzymecommission \
    data.split=structure \
    model.max_clusters=10 \
    model.entropy_weight=0.1 \
    model.link_pred_weight=0.5 \
    train.batch_size=128

echo "DiffPool test completed!"

# Also test with different cluster configurations
echo "Testing different cluster configurations..."

# Test with more clusters
CUDA_VISIBLE_DEVICES=0 python run_diffpool.py \
    --config-path=conf \
    --config-name=config_diffpool \
    train.epochs=3 \
    data.test_mode=true \
    train.use_wandb=false \
    train.wandb_project=diffpool-test \
    data.dataset_name=enzymecommission \
    data.split=structure \
    model.max_clusters=50 \
    model.entropy_weight=0.2 \
    model.link_pred_weight=0.3 \
    train.batch_size=16

echo "All DiffPool tests completed!"

#!/bin/bash

# Test script for PartGVP implementation
echo "🧬 Testing PartGVP Implementation"
echo "=" * 50

# Test with minimal epochs for quick validation
python run_partgvp.py \
    --config-path=conf \
    --config-name=config_partgvp \
    train.epochs=2 \
    data.test_mode=true \
    train.use_wandb=false

echo "🎉 PartGVP test completed!"

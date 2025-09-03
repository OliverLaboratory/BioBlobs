#!/bin/bash
# Test script for cosine learning rate scheduler

echo "Testing ParToken with cosine learning rate schedule..."

# Test 1: Default (without cosine schedule)
echo "=== Test 1: Default training (no cosine schedule) ==="
python run_partoken.py \
    train.epochs=2 \
    train.batch_size=4 \
    train.use_wandb=false \
    model.max_clusters=5 \
    model.codebook_size=64

echo ""
echo "=== Test 2: With cosine learning rate schedule ==="
# Test 2: With cosine schedule
python run_partoken.py \
    train.epochs=2 \
    train.batch_size=4 \
    train.use_wandb=false \
    train.use_cosine_schedule=true \
    train.warmup_epochs=1 \
    model.max_clusters=5 \
    model.codebook_size=64

echo "All tests completed successfully!"

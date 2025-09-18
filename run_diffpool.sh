#!/bin/bash

# DiffPool experiments on different datasets and splits (sequential on GPU 6)
# Usage: bash run_diffpool.sh

echo "Starting DiffPool experiments sequentially on GPU 6..."

# Enzyme Commission - Structure Split
echo "Running DiffPool on Enzyme Commission (structure split)..."
CUDA_VISIBLE_DEVICES=6 python run_diffpool.py \
    --config-path=conf \
    --config-name=config_diffpool \
    train.epochs=100 \
    data.test_mode=false \
    train.use_wandb=true \
    train.wandb_project=diffpool-baseline \
    data.dataset_name=enzymecommission \
    data.split=structure \
    model.max_clusters=20 \
    model.entropy_weight=0.1 \
    model.link_pred_weight=0.5
echo "Completed: enzymecommission (structure)"

# Enzyme Commission - Random Split
echo "Running DiffPool on Enzyme Commission (random split)..."
CUDA_VISIBLE_DEVICES=6 python run_diffpool.py \
    --config-path=conf \
    --config-name=config_diffpool \
    train.epochs=100 \
    data.test_mode=false \
    train.use_wandb=true \
    train.wandb_project=diffpool-baseline \
    data.dataset_name=enzymecommission \
    data.split=random \
    model.max_clusters=20 \
    model.entropy_weight=0.1 \
    model.link_pred_weight=0.5
echo "Completed: enzymecommission (random)"

# SCOP - Structure Split
echo "Running DiffPool on SCOP (structure split)..."
CUDA_VISIBLE_DEVICES=6 python run_diffpool.py \
    --config-path=conf \
    --config-name=config_diffpool \
    train.epochs=100 \
    data.test_mode=false \
    train.use_wandb=true \
    train.wandb_project=diffpool-baseline \
    data.dataset_name=scope \
    data.split=structure \
    model.max_clusters=20 \
    model.entropy_weight=0.1 \
    model.link_pred_weight=0.5
echo "Completed: scope (structure)"

# SCOP - Random Split
echo "Running DiffPool on SCOP (random split)..."
CUDA_VISIBLE_DEVICES=6 python run_diffpool.py \
    --config-path=conf \
    --config-name=config_diffpool \
    train.epochs=100 \
    data.test_mode=false \
    train.use_wandb=true \
    train.wandb_project=diffpool-baseline \
    data.dataset_name=scope \
    data.split=random \
    model.max_clusters=20 \
    model.entropy_weight=0.1 \
    model.link_pred_weight=0.5
echo "Completed: scope (random)"

echo "All DiffPool experiments completed sequentially on GPU 6."

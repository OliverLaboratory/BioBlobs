#!/bin/bash

# PartGNN Training Script for Enzyme Commission and Structural Class Datasets
# Sequential execution on single GPU

python run_partgnn.py \
    --config-name=config_partgnn \
    train.max_epochs=120 \
    data.test_mode=false \
    use_wandb=true \
    data.dataset_name=structural_class \
    data.split=random \
    train.wandb_project=PartGNN

# python run_partgnn.py \
#     --config-name=config_partgnn \
#     train.max_epochs=120 \
#     data.test_mode=false \
#     use_wandb=true \
#     data.dataset_name=structural_class \
#     data.split=structure 

# python run_partgnn.py \
#     --config-name=config_partgnn \
#     train.max_epochs=120 \
#     data.test_mode=false \
#     use_wandb=true \
#     data.dataset_name=enzyme_commission \
#     data.split=random 

# python run_partgnn.py \
#     --config-name=config_partgnn \
#     train.max_epochs=120 \
#     data.test_mode=false \
#     use_wandb=true \
#     data.dataset_name=enzyme_commission \
#     data.split=structure 
#!/bin/bash



# python run_partgvp.py \
#     --config-name config_partgvp_geneontology \
#     data.test_mode=false \
#     train.use_wandb=true \
#     data.split="structure"
    

python run_partgvp.py \
    --config-name config_partgvp_geneontology \
    data.test_mode=false \
    train.use_wandb=true \
    data.split="random"
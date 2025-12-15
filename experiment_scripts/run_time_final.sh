#!/usr/bin/env bash
set -e

num_blob_list=(5 10 15 20 25)
k_hop_list=(1 2 3)

for num_blob in "${num_blob_list[@]}"; do
    for k_hop in "${k_hop_list[@]}"; do
        echo "Running with num_blob=${num_blob}, k_hop=${k_hop}"
        CUDA_VISIBLE_DEVICES=0 python run_bioblobs_multistage.py \
            data.dataset_name=ec \
            data.split=random \
            data.edge_types=knn_8 \
            train.stage0.epochs=5 \
            train.stage1.epochs=1 \
            train.use_wandb=true \
            train.wandb_project=bioblobs_runtime_poster \
            train.job_name="num_blob${num_blob}_khop${k_hop}" \
            model.k_hop="${k_hop}" \
            data.test_mode=false \
            model.max_clusters="${num_blob}" \
            model.cluster_size_max=30 \
            interpretability.enabled=false
    done
done


#!/usr/bin/env bash
set -e

max_cluster_list=(5 10 15 20 25)

for max_cluster in "${max_cluster_list[@]}"; do
    echo "Running with max_cluster=${max_cluster}"
    CUDA_VISIBLE_DEVICES=0 python run_bioblobs_multistage.py \
        data.dataset_name=ec \
        data.split=random \
        data.edge_types=knn_8 \
        train.stage0.epochs=2 \
        train.stage1.epochs=2 \
        train.use_wandb=true \
        train.wandb_project=bioblobs_runtime \
        train.job_name="num_clusters${max_cluster}" \
        model.k_hop=2 \
        model.cluster_size_max="${max_cluster}"
done


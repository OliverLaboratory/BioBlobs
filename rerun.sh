CUDA_VISIBLE_DEVICES=1 python run_bioblobs_multistage.py \
    data.dataset_name=scop-fa \
    data.split=structure \
    data.edge_types=knn_8 \
    train.stage0.epochs=120 \
    train.stage1.epochs=30 \
    model.k_hop=2 \
    train.wandb_project=bioblobs_scop_rerun \
    train.use_wandb=true &

sleep 900

CUDA_VISIBLE_DEVICES=0 python run_bioblobs_multistage.py \
    data.dataset_name=scop-fa \
    data.split=random \
    data.edge_types=knn_8 \ 
    train.stage0.epochs=120 \
    train.stage1.epochs=30 \
    model.k_hop=2 \
    train.wandb_project=bioblobs_scop_rerun \
    train.use_wandb=true &


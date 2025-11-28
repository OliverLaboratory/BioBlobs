CUDA_VISIBLE_DEVICES=0 python run_bioblobs_multistage.py \
    data.dataset_name=ec \
    data.split=structure \
    data.edge_types=knn_30 \
    train.stage0.epochs=120 \
    train.stage1.epochs=30 \
    model.k_hop=2 &

CUDA_VISIBLE_DEVICES=1 python run_bioblobs_multistage.py \
    data.dataset_name=ec \
    data.split=random \
    data.edge_types=knn_30 \
    train.stage0.epochs=120 \
    train.stage1.epochs=30 \
    model.k_hop=2 &
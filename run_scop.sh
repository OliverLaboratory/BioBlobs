python run_partgvp.py \
    train.epochs=2 \
    data.test_mode=false \
    train.use_wandb=false \
    model.seq_in=true \
    model.max_clusters=5 \
    model.cluster_size_max=15 \
    train.wandb_project=PartGVP \
    data.dataset_name=scope \
    data.split=structure \ 


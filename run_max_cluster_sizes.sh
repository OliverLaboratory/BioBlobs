cuda_devices=(0 1 2)
sizes=(5 10 15 25)

for i in "${!sizes[@]}"; do
    size=${sizes[$i]}
    cuda=${cuda_devices[$i]}
    echo "Current size: $size on CUDA_VISIBLE_DEVICES=$cuda"
    CUDA_VISIBLE_DEVICES=$cuda python run_partgvp.py \
        --config-path=conf \
        --config-name=config_partgvp \
        train.epochs=120 \
        data.test_mode=false \
        train.use_wandb=true \
        train.wandb_project=PartGVP-max-cluster-sizes \
        data.dataset_name=scope \
        model.seq_in=false \
        model.cluster_size_max=$size \
        data.split=random &
done

wait

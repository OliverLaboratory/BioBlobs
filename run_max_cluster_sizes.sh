cuda_devices=(1 2 3)
sizes=(5 10 25)

# Ensure arrays have the same length
if [ ${#sizes[@]} -ne ${#cuda_devices[@]} ]; then
    echo "Error: sizes and cuda_devices arrays must have the same length."
    exit 1
fi

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
        data.dataset_name=enzymecommission \
        model.seq_in=false \
        model.cluster_size_max=$size \
        data.split=random &
done

wait
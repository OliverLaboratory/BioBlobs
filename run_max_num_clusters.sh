Dataset=scope
Split=random
cuda_devices=(3 4 5)
max_num_clusters=(5 10 15 25)

for i in "${!max_num_clusters[@]}"; do
    size=${max_num_clusters[$i]}
    cuda=${cuda_devices[$i]}
    echo "Current size: $size on CUDA_VISIBLE_DEVICES=$cuda"
    CUDA_VISIBLE_DEVICES=$cuda python run_partgvp.py \
        --config-path=conf \
        --config-name=config_partgvp \
        train.epochs=120 \
        data.test_mode=false \
        train.use_wandb=true \
        train.wandb_project=PartGVP-max-num-clusters \
        data.dataset_name=$Dataset \
        model.seq_in=false \
        model.max_clusters=$size \
        data.split=$Split &
done

wait

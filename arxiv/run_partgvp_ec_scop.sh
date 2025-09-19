# little finetuning 
CUDA_VISIBLE_DEVICES=1 python run_partgvp.py \
    --config-path=conf \
    --config-name=config_partgvp \
    train.epochs=120 \
    data.test_mode=false \
    train.use_wandb=true \
    train.wandb_project=PartGVP-sequence \
    data.dataset_name=scope \
    data.split=structure &


CUDA_VISIBLE_DEVICES=3 python run_partgvp.py \
    --config-path=conf \
    --config-name=config_partgvp \
    train.epochs=120 \
    data.test_mode=false \
    train.use_wandb=true \
    train.wandb_project=PartGVP-sequence \
    data.dataset_name=enzymecommission \
    data.split=random &


CUDA_VISIBLE_DEVICES=5 python run_partgvp.py \
    --config-path=conf \
    --config-name=config_partgvp \
    train.epochs=120 \
    data.test_mode=false \
    train.use_wandb=true \
    train.wandb_project=PartGVP-sequence \
    data.dataset_name=enzymecommission \
    data.split=structure &

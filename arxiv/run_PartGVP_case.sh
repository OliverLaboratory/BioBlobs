project=PartGVP_case_study

CUDA_VISIBLE_DEVICES=0 python run_partgvp.py \
    train.epochs=120 \
    data.test_mode=false \
    train.use_wandb=true \
    model.max_clusters=3 \
    model.cluster_size_max=10 \
    train.wandb_project=PartGVP \
    data.dataset_name=enzymecommission \
    data.split=structure \
    train.wandb_project=$project \
    hydra.run.dir=./case_study/partgvp/enzymecommission/structure/maxC3_CSmax10/ &

CUDA_VISIBLE_DEVICES=1 python run_partgvp.py \
    train.epochs=120 \
    data.test_mode=false \
    train.use_wandb=true \
    model.max_clusters=3 \
    model.cluster_size_max=20 \
    train.wandb_project=PartGVP \
    data.dataset_name=enzymecommission \
    data.split=structure \
    train.wandb_project=$project \
    hydra.run.dir=./case_study/partgvp/enzymecommission/structure/maxC3_CSmax20/ &

CUDA_VISIBLE_DEVICES=2 python run_partgvp.py \
    train.epochs=120 \
    data.test_mode=false \
    train.use_wandb=true \
    model.max_clusters=10 \
    model.cluster_size_max=10 \
    train.wandb_project=PartGVP \
    data.dataset_name=enzymecommission \
    data.split=structure \
    train.wandb_project=$project \
    hydra.run.dir=./case_study/partgvp/enzymecommission/structure/maxC10_CSmax10/ &

CUDA_VISIBLE_DEVICES=3 python run_partgvp.py \
    train.epochs=120 \
    data.test_mode=false \
    train.use_wandb=true \
    model.max_clusters=10 \
    model.cluster_size_max=20 \
    train.wandb_project=PartGVP \
    data.dataset_name=enzymecommission \
    data.split=structure \
    train.wandb_project=$project \
    hydra.run.dir=./case_study/partgvp/enzymecommission/structure/maxC10_CSmax20/ &

wait

# python run_partgvp.py \
#     train.epochs=120 \
#     data.test_mode=false \
#     train.use_wandb=true \
#     model.max_clusters=5 \
#     model.cluster_size_max=15 \
#     train.wandb_project=PartGVP \
#     data.dataset_name=enzymecommission\
#     data.split=random 


python run_partgvp.py \
    train.epochs=120 \
    data.test_mode=false \
    train.use_wandb=true \
    model.max_clusters=5 \
    model.cluster_size_max=15 \
    train.wandb_project=PartGVP \
    data.dataset_name=enzymecommission\
    data.split=structure 

python run_partgvp.py \
    train.epochs=120 \
    data.test_mode=false \
    train.use_wandb=true \
    model.max_clusters=5 \
    model.cluster_size_max=15 \
    train.wandb_project=PartGVP \
    data.dataset_name=enzymecommission\
    data.split=sequence 
# Enzyme Commission Dataset 

## random split
python run_gvpgnn.py data.dataset_name=enzymecommission data.split=random train.batch_size=64 train.epochs=100 train.use_wandb=true train.wandb_project=gvpgnn-baseline914
## structure split
python run_gvpgnn.py data.dataset_name=enzymecommission data.split=structure train.batch_size=64 train.epochs=100 train.use_wandb=true train.wandb_project=gvpgnn-baseline914
## sequence split
python run_gvpgnn.py data.dataset_name=enzymecommission data.split=sequence train.batch_size=64 train.epochs=100 train.use_wandb=true train.wandb_project=gvpgnn-baseline914
# Protein Family Dataset

# ## random split
# python run_gvpgnn.py data.dataset_name=proteinfamily data.split=random train.batch_size=64 train.epochs=100 train.use_wandb=true train.wandb_project=gvpgnn-baseline914
# ## structure split
# python run_gvpgnn.py data.dataset_name=proteinfamily data.split=structure train.batch_size=64 train.epochs=100 train.use_wandb=true train.wandb_project=gvpgnn-baseline914
# ## sequence split
# python run_gvpgnn.py data.dataset_name=proteinfamily data.split=sequence train.batch_size=64 train.epochs=100 train.use_wandb=true train.wandb_project=gvpgnn-baseline914
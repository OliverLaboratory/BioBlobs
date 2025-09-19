
source ~/miniconda3/etc/profile.d/conda.sh
conda activate partoken

HOME_DIR="/home/wangx86/partoken/partoken-protein"

EC_RANDOM="$HOME_DIR/checkpoints/ec/random/best-partgvp-epoch=117-val_acc=0.856.ckpt"
EC_STRUCTURE="$HOME_DIR/checkpoints/ec/structure/best-partgvp-epoch=107-val_acc=0.574.ckpt"
SCOP_RANDOM="$HOME_DIR/checkpoints/scop/random/best-partgvp-epoch=105-val_acc=0.517.ckpt"
SCOP_STRUCTURE="$HOME_DIR/checkpoints/scop/structure/best-partgvp-epoch=112-val_acc=0.352.ckpt"



# enzyme commission 

CUDA_VISIBLE_DEVICES=2 python run_partoken_resume.py \
    resume.partgvp_checkpoint_path="$EC_RANDOM" \
    data.dataset_name="enzymecommission" \
    data.split="random" \
    data.test_mode=false \
    multistage.stage0.epochs=30 \
    train.batch_size=128 \
    train.use_wandb=true \
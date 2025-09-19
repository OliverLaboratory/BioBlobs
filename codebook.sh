

HOME_DIR="/home/wangx86/partoken/partoken-protein/"

EC_RANDOM="checkpoints/ec/random/best-partgvp-epoch=117-val_acc=0.856.ckpt"
EC_STRUCTURE="checkpoints/ec/structure/best-partgvp-epoch=107-val_acc=0.574.ckpt"
SCOP_RANDOM="checkpoints/scop/random/best-partgvp-epoch=105-val_acc=0.517.ckpt"
SCOP_STRUCTURE="checkpoints/scop/structure/best-partgvp-epoch=112-val_acc=0.352.ckpt"



# enzyme commission 

CUDA_VISIBLE_DEVICES=0 python run_partoken_resume.py \
    resume.partgvp_checkpoint_path="$PARTGVP_CHECKPOINT_PATH" \
    data.dataset_name="$DATASET" \
    data.split="$SPLIT" \
    data.test_mode=false \
    multistage.stage0.epochs=30 \
    train.batch_size=128 \
    train.use_wandb=true \

#!/bin/bash



HOME_DIR="/home/wangx86/partoken/partoken-protein"

# Gene Ontology PartGVP checkpoints
GO_RANDOM="$HOME_DIR/checkpoints/go/random/best-partgvp-epoch=98-val_fmax=0.682.ckpt"
GO_STRUCTURE="$HOME_DIR/checkpoints/go/structure/best-partgvp-epoch=59-val_fmax=0.542.ckpt"

# Run both training jobs in parallel
CUDA_VISIBLE_DEVICES=0 python run_partoken_resume.py \
    --config-name=config_partoken_resume_geneontology \
    "resume.partgvp_checkpoint_path='$GO_RANDOM'" \
    data.split=random \
    multistage.stage0.epochs=20 \
    train.use_wandb=true \
    train.wandb_project=codebook-resume-go &

CUDA_VISIBLE_DEVICES=7 python run_partoken_resume.py \
    --config-name=config_partoken_resume_geneontology \
    "resume.partgvp_checkpoint_path='$GO_STRUCTURE'" \
    data.split=structure \
    multistage.stage0.epochs=20 \
    train.use_wandb=true \
    train.wandb_project=codebook-resume-go &

wait

# Test mode

#!/bin/bash

# Test script for ParToken resume training
# Adapted to run four checkpoints on separate GPUs

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate partoken

echo "🧪 Testing ParToken Resume Training with Four Checkpoints"
echo "======================================================="

HOME_DIR="/home/wangx86/partoken/partoken-protein"

EC_RANDOM="$HOME_DIR/checkpoints/ec/random/best-partgvp-epoch=117-val_acc=0.856.ckpt"
EC_STRUCTURE="$HOME_DIR/checkpoints/ec/structure/best-partgvp-epoch=107-val_acc=0.574.ckpt"
SCOP_RANDOM="$HOME_DIR/checkpoints/scop/random/best-partgvp-epoch=105-val_acc=0.517.ckpt"
SCOP_STRUCTURE="$HOME_DIR/checkpoints/scop/structure/best-partgvp-epoch=112-val_acc=0.352.ckpt"

# Function to run a single test
run_test() {
    local checkpoint=$1
    local dataset=$2
    local split=$3
    local cuda=$4

    echo ""
    echo "🚀 Starting resume training for $dataset $split on GPU $cuda..."
    echo "   Checkpoint: $checkpoint"

    CUDA_VISIBLE_DEVICES=$cuda python run_partoken_resume.py \
        "resume.partgvp_checkpoint_path='$checkpoint'" \
        data.dataset_name="$dataset" \
        data.split="$split" \
        data.test_mode=false \
        multistage.stage0.epochs=30 \
        train.batch_size=128 \
        train.use_wandb=true 
}

# Run tests on separate GPUs
# run_test "$EC_RANDOM" "enzymecommission" "random" 7 &
# run_test "$EC_STRUCTURE" "enzymecommission" "structure" 6 &
run_test "$SCOP_RANDOM" "scope" "random" 5 &
run_test "$SCOP_STRUCTURE" "scope" "structure" 4 &

# Wait for all to finish
wait

echo ""
echo "🎉 All resume training tests completed!"

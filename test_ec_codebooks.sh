#!/bin/bash

# Test script for ParToken resume training with different codebook sizes
# EC Random checkpoint with codebook sizes: 64, 128, 256, 512

# Activate conda environment
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate partoken

echo "🧪 Testing ParToken Resume Training - EC Random with Different Codebook Sizes"
echo "============================================================================="

HOME_DIR="/home/wangx86/partoken/partoken-protein"

EC_RANDOM="$HOME_DIR/checkpoints/ec/random/best-partgvp-epoch=117-val_acc=0.856.ckpt"

# Function to run a single test with specific codebook size
run_test() {
    local checkpoint=$1
    local codebook_size=$2
    local cuda=$3

    echo ""
    echo "🚀 Starting resume training for EC Random with codebook size $codebook_size on GPU $cuda..."
    echo "   Checkpoint: $checkpoint"

    CUDA_VISIBLE_DEVICES=$cuda python run_partoken_resume.py \
        "resume.partgvp_checkpoint_path='$checkpoint'" \
        data.dataset_name="enzymecommission" \
        data.split="random" \
        data.test_mode=false \
        multistage.stage0.epochs=30 \
        train.batch_size=128 \
        train.use_wandb=true \
        model.codebook_size=$codebook_size
}

# Run tests with different codebook sizes on separate GPUs
run_test "$EC_RANDOM" 64 4 &
run_test "$EC_RANDOM" 128 5 &
run_test "$EC_RANDOM" 256 6 &
run_test "$EC_RANDOM" 512 7 &

# Wait for all to finish
wait

echo ""
echo "🎉 All EC Random codebook size tests completed!"
echo "   Tested codebook sizes: 64, 128, 256, 512"
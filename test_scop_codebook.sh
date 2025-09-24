#!/bin/bash

# Test script for ParToken resume training with different codebook sizes
# Testing SCOP Random split with codebook sizes: 64, 128, 256, 512

# Activate conda environment
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate partoken

echo "🧪 Testing ParToken SCOP Random with Different Codebook Sizes"
echo "============================================================="

HOME_DIR="/root/ICLR2026/partoken-protein"
SCOP_RANDOM="$HOME_DIR/checkpoints/scop/random/best-partgvp-epoch=105-val_acc=0.517.ckpt"

# Function to run a single test with specific codebook size
run_codebook_test() {
    local checkpoint=$1
    local codebook_size=$2

    echo ""
    echo "🚀 Starting resume training for SCOP random with codebook size $codebook_size..."
    echo "   Checkpoint: $checkpoint"

    python run_partoken_resume.py \
        "resume.partgvp_checkpoint_path='$checkpoint'" \
        data.dataset_name="scope" \
        data.split="random" \
        data.test_mode=false \
        multistage.stage0.epochs=30 \
        train.batch_size=128 \
        train.use_wandb=true \
        model.codebook_size=$codebook_size \
        train.wandb_project="case_study_codebook_size"
}

# Run tests with different codebook sizes sequentially
echo "📋 Running SCOP random with codebook sizes sequentially: 64, 128, 256, 512"

run_codebook_test "$SCOP_RANDOM" 64
run_codebook_test "$SCOP_RANDOM" 128
run_codebook_test "$SCOP_RANDOM" 256
run_codebook_test "$SCOP_RANDOM" 512

echo ""
echo "🎉 All SCOP codebook size tests completed!"
echo "📊 Tested codebook sizes sequentially: 64, 128, 256, 512"
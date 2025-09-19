#!/bin/bash

# Test script for ParToken resume training
# This script demonstrates how to resume ParToken training from a PartGVP checkpoint

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate partoken

echo "🧪 Testing ParToken Resume Training"
echo "=================================="

# Configuration
DATASET="enzymecommission"
SPLIT="random" 
OUTPUT_DIR="/home/wangx86/partoken/partoken-protein/outputs"
PARTGVP_CHECKPOINT_PATH="${OUTPUT_DIR}/partgvp-sequence/enzymecommission/random/2025-09-17-00-49-31/last.ckpt"

# Check if checkpoint exists (user needs to update this path)
if [ ! -f "$PARTGVP_CHECKPOINT_PATH" ]; then
    echo "❌ PartGVP checkpoint not found: $PARTGVP_CHECKPOINT_PATH"
    echo "📝 Please update PARTGVP_CHECKPOINT_PATH in this script to point to your actual checkpoint"
    echo ""
    echo "💡 To find available checkpoints, run:"
    echo "   find outputs/ -name '*.ckpt' -type f | grep partgvp"
    exit 1
fi

echo "✅ Found PartGVP checkpoint: $PARTGVP_CHECKPOINT_PATH"

# Test mode with reduced parameters for quick testing
echo ""
echo "🚀 Starting resume training test..."
echo "   (Using test_mode=true for quick execution)"

CUDA_VISIBLE_DEVICES=1 python run_partoken_resume.py \
    resume.partgvp_checkpoint_path="$PARTGVP_CHECKPOINT_PATH" \
    data.dataset_name="$DATASET" \
    data.split="$SPLIT" \
    data.test_mode=false \
    multistage.stage0.epochs=2 \
    train.batch_size=128 \
    train.use_wandb=true \
    resume.kmeans_max_batches=5 \
    evaluation.max_batches_interp=5

# echo ""
# echo "🎉 Resume training test completed!"
# echo ""
# echo "📁 Check the output directory for results:"
# echo "   - results_summary.json (overall results)"
# echo "   - codebook_initialization/ (K-means stats)"
# echo "   - interpretability/ (analysis results)"
# echo "   - checkpoints/ (trained models)"

# # Production mode example (commented out)
# echo ""
# echo "💼 For production training, use:"
# echo "python run_partoken_resume.py \\"
# echo "    resume.partgvp_checkpoint_path=\"$PARTGVP_CHECKPOINT_PATH\" \\"
# echo "    data.dataset_name=\"$DATASET\" \\"
# echo "    data.split=\"$SPLIT\" \\"
# echo "    data.test_mode=false \\"
# echo "    multistage.stage0.epochs=10 \\"
# echo "    multistage.stage1.epochs=30 \\"
# echo "    train.batch_size=16 \\"
# echo "    train.use_wandb=true"

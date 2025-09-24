#!/bin/bash
# Test ParToken checkpoints
# Usage: ./test_checkpoints_batch.sh checkpoint1.ckpt checkpoint2.ckpt ...

if [ $# -eq 0 ]; then
    echo "Usage: $0 <checkpoint_paths...>"
    echo "Example: $0 /path/to/best-partoken-*.ckpt"
    exit 1
fi

echo "🧪 Testing ${#} checkpoint(s)"
mkdir -p ./results

for checkpoint in "$@"; do
    [ -f "$checkpoint" ] || { echo "❌ Not found: $checkpoint"; continue; }
    
    echo "🔍 $(basename "$checkpoint")"
    output="./results/$(basename "$checkpoint" .ckpt).json"
    
    python test_partoken_checkpoints.py --checkpoint_path "$checkpoint" --output_file "$output"
    [ $? -eq 0 ] && echo "✅ Done" || echo "❌ Failed"
done

echo "📊 Results in ./results/"
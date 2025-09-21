#!/bin/bash

# Test run script for Gene Ontology multi-label classification
# This script runs a quick test training with limited data and epochs

echo "🧬 Starting Gene Ontology Multi-Label Classification Test Run"
echo "================================================================"
echo "Configuration:"
echo "  - Dataset: Gene Ontology (test mode - limited data)"
echo "  - Epochs: 2"
echo "  - Batch size: 16"
echo "  - Model: PartGVP with multi-label support"
echo "  - Metrics: FMax, Precision, Recall"
echo ""

# Run the training with overrides for test parameters
python run_partgvp.py \
    --config-name config_partgvp_geneontology \
    data.test_mode=true \
    train.epochs=2 \
    train.batch_size=16 \
    train.wandb_project="partoken_gene_ontology_test" \

echo ""
echo "🎉 Gene Ontology test run completed!"
echo "Check the output logs above for training metrics and results."
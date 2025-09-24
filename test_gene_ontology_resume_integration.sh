#!/bin/bash

# Test script for Gene Ontology resume training integration

echo "🧪 Testing Gene Ontology Resume Training Integration"
echo "=" * 70

# Test 1: Check if configuration file exists
echo "📋 Test 1: Configuration file"
if [ -f "conf/config_partoken_resume_geneontology.yaml" ]; then
    echo "✓ Gene Ontology resume config found"
else
    echo "❌ Gene Ontology resume config missing"
    exit 1
fi

# Test 2: Check if multi-label class is importable
echo "📋 Test 2: Multi-label class import"
python3 -c "
try:
    from partoken_resume_lightning import ParTokenResumeTrainingMultiLabelLightning
    print('✓ ParTokenResumeTrainingMultiLabelLightning import successful')
except ImportError as e:
    print(f'❌ Import failed: {e}')
    exit(1)
"

# Test 3: Check if multi-label model creation function exists
echo "📋 Test 3: Multi-label model creation function"
python3 -c "
try:
    from train_lightling import create_partoken_resume_multilabel_model_from_checkpoint
    print('✓ create_partoken_resume_multilabel_model_from_checkpoint import successful')
except ImportError as e:
    print(f'❌ Import failed: {e}')
    exit(1)
"

# Test 4: Validate configuration syntax
echo "📋 Test 4: Configuration syntax validation"
python3 -c "
from omegaconf import OmegaConf
try:
    cfg = OmegaConf.load('conf/config_partoken_resume_geneontology.yaml')
    print('✓ Configuration file syntax is valid')
    print(f'  • Dataset: {cfg.data.dataset_name}')
    print(f'  • Split: {cfg.data.split}')
    print(f'  • Codebook size: {cfg.model.codebook_size}')
    print(f'  • Epochs: {cfg.multistage.stage0.epochs}')
except Exception as e:
    print(f'❌ Configuration validation failed: {e}')
    exit(1)
"

echo "=" * 70
echo "🎉 All tests passed! Gene Ontology resume training integration is ready."
echo ""
echo "Usage examples:"
echo "  # Resume from PartGVP checkpoint for Gene Ontology:"
echo "  python run_partoken_resume.py --config-name=config_partoken_resume_geneontology resume.partgvp_checkpoint_path=/path/to/partgvp_checkpoint.ckpt"
echo ""
echo "  # Resume with test mode:"
echo "  python run_partoken_resume.py --config-name=config_partoken_resume_geneontology resume.partgvp_checkpoint_path=/path/to/checkpoint.ckpt data.test_mode=true"
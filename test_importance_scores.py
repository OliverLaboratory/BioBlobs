from train_lightling import MultiStageParTokenLightning
from omegaconf import DictConfig
import torch

print('Testing MultiStageParTokenLightning with new modules...')

# Create minimal config
model_cfg = DictConfig({
    'node_in_dim': [6, 3],
    'node_h_dim': [100, 16], 
    'edge_in_dim': [32, 1],
    'edge_h_dim': [32, 1],
    'seq_in': False,
    'num_layers': 3,
    'drop_rate': 0.1,
    'pooling': 'mean',
    'max_clusters': 5,
    'nhid': 50,
    'k_hop': 1,
    'cluster_size_max': 15,
    'termination_threshold': 0.95,
    'tau_init': 1.0,
    'tau_min': 0.1,
    'tau_decay': 0.95,
    'codebook_size': 512,
    'codebook_dim': None,
    'codebook_beta': 0.25,
    'codebook_decay': 0.99,
    'codebook_eps': 1e-5,
    'codebook_distance': 'l2',
    'codebook_cosine_normalize': False,
    'lambda_vq': 1.0,
    'lambda_ent': 1e-3,
    'lambda_psc': 1e-2,
    'psc_temp': 0.3
})

train_cfg = DictConfig({
    'batch_size': 16,
    'seed': 42,
    'use_cosine_schedule': True,
    'warmup_epochs': 1
})

multistage_cfg = DictConfig({
    'enabled': True,
    'stage0': DictConfig({
        'name': 'baseline',
        'epochs': 1,
        'lr': 1e-4,
        'bypass_codebook': True,
        'loss_weights': DictConfig({
            'lambda_vq': 0.0,
            'lambda_ent': 0.0,
            'lambda_psc': 0.0
        })
    }),
    'stage1': DictConfig({
        'name': 'joint_finetuning',
        'epochs': 1,
        'lr': 5e-5,
        'kmeans_init': True,
        'kmeans_batches': 10
    })
})

# Test lightning module initialization
lightning_model = MultiStageParTokenLightning(model_cfg, train_cfg, multistage_cfg, num_classes=2)
print('Lightning module initialized successfully!')

# Test stage setup
lightning_model.setup_stage(0, multistage_cfg.stage0)
print('Stage 0 setup successful!')

lightning_model.setup_stage(1, multistage_cfg.stage1)  
print('Stage 1 setup successful!')

print('All tests passed!')

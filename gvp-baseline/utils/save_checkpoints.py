"""
Checkpoint saving utilities for multi-stage ParToken training.

This module provides functions to save specific model components at different
training stages, allowing for fine-grained checkpoint management.
"""

import os
import torch
from typing import Dict, Any, Optional
from datetime import datetime


def save_stage0_checkpoint(model, output_dir: str, stage_info: Optional[Dict] = None) -> str:
    """
    Save Stage 0 checkpoint: encoder + partitioner + classifier components.
    
    Args:
        model: The ParToken Lightning model
        output_dir: Directory to save the checkpoint
        stage_info: Optional dictionary with stage metadata
        
    Returns:
        Path to saved checkpoint file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract the core model components needed for stage 0
    checkpoint_data = {
        # Node and edge encoders
        'node_encoder': model.model.node_encoder.state_dict(),
        'edge_encoder': model.model.edge_encoder.state_dict(),
        
        # GVP layers
        'gvp_layers': [layer.state_dict() for layer in model.model.gvp_layers],
        
        # Output projection and partitioner
        'output_projection': model.model.output_projection.state_dict(),
        'partitioner': model.model.partitioner.state_dict(),
        
        # Classifier components
        'classifier': model.model.classifier.state_dict(),
        
        # Sequence embedding if present
        'sequence_embedding': model.model.sequence_embedding.state_dict() if hasattr(model.model, 'sequence_embedding') else None,
        
        # Model configuration and metadata
        'model_config': {
            'node_in_dim': model.model.node_in_dim,
            'node_h_dim': model.model.node_h_dim,
            'edge_in_dim': model.model.edge_in_dim,
            'edge_h_dim': model.model.edge_h_dim,
            'num_classes': model.model.num_classes,
            'seq_in': model.model.seq_in,
            'num_layers': model.model.num_layers,
            'drop_rate': model.model.drop_rate,
            'pooling': model.model.pooling,
        },
        
        # Stage metadata
        'stage_info': {
            'stage': 0,
            'stage_name': 'baseline',
            'timestamp': datetime.now().isoformat(),
            'components': ['encoder', 'partitioner', 'classifier'],
            **(stage_info or {})
        }
    }
    
    checkpoint_path = os.path.join(output_dir, 'stage0_encoder_partitioner_classifier.ckpt')
    torch.save(checkpoint_data, checkpoint_path)
    
    print(f"✓ Stage 0 checkpoint saved: {checkpoint_path}")
    print(f"  Components: encoder, partitioner, classifier")
    
    return checkpoint_path


def save_stage1_checkpoint(model, output_dir: str, stage_info: Optional[Dict] = None) -> str:
    """
    Save Stage 1 checkpoint: codebook parameters.
    
    Args:
        model: The ParToken Lightning model
        output_dir: Directory to save the checkpoint
        stage_info: Optional dictionary with stage metadata
        
    Returns:
        Path to saved checkpoint file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract codebook-specific components
    checkpoint_data = {
        # Codebook parameters
        'codebook': model.model.codebook.state_dict(),
        
        # Codebook configuration
        'codebook_config': {
            'codebook_size': model.model.codebook_size,
            'codebook_dim': model.model.codebook_dim,
            'codebook_beta': model.model.codebook_beta,
            'codebook_decay': model.model.codebook_decay,
            'codebook_eps': model.model.codebook_eps,
            'codebook_distance': model.model.codebook_distance,
            'codebook_cosine_normalize': model.model.codebook_cosine_normalize,
        },
        
        # Loss weights at end of stage 1
        'loss_weights': {
            'lambda_vq': model.model.lambda_vq,
            'lambda_ent': model.model.lambda_ent,
            'lambda_psc': model.model.lambda_psc,
        },
        
        # Stage metadata
        'stage_info': {
            'stage': 1,
            'stage_name': 'codebook_warmup',
            'timestamp': datetime.now().isoformat(),
            'components': ['codebook'],
            **(stage_info or {})
        }
    }
    
    checkpoint_path = os.path.join(output_dir, 'stage1_codebook.ckpt')
    torch.save(checkpoint_data, checkpoint_path)
    
    print(f"✓ Stage 1 checkpoint saved: {checkpoint_path}")
    print(f"  Components: codebook")
    
    return checkpoint_path


def save_stage2_checkpoint(model, output_dir: str, stage_info: Optional[Dict] = None) -> str:
    """
    Save Stage 2 checkpoint: all updated model components.
    
    Args:
        model: The ParToken Lightning model
        output_dir: Directory to save the checkpoint
        stage_info: Optional dictionary with stage metadata
        
    Returns:
        Path to saved checkpoint file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save complete model state
    checkpoint_data = {
        # Complete model state dict
        'model_state_dict': model.model.state_dict(),
        
        # Complete model configuration
        'model_config': {
            'node_in_dim': model.model.node_in_dim,
            'node_h_dim': model.model.node_h_dim,
            'edge_in_dim': model.model.edge_in_dim,
            'edge_h_dim': model.model.edge_h_dim,
            'num_classes': model.model.num_classes,
            'seq_in': model.model.seq_in,
            'num_layers': model.model.num_layers,
            'drop_rate': model.model.drop_rate,
            'pooling': model.model.pooling,
            # Partitioner config
            'max_clusters': model.model.max_clusters,
            'nhid': model.model.nhid,
            'k_hop': model.model.k_hop,
            'cluster_size_max': model.model.cluster_size_max,
            'termination_threshold': model.model.termination_threshold,
            'tau_init': model.model.tau_init,
            'tau_min': model.model.tau_min,
            'tau_decay': model.model.tau_decay,
            # Codebook config
            'codebook_size': model.model.codebook_size,
            'codebook_dim': model.model.codebook_dim,
            'codebook_beta': model.model.codebook_beta,
            'codebook_decay': model.model.codebook_decay,
            'codebook_eps': model.model.codebook_eps,
            'codebook_distance': model.model.codebook_distance,
            'codebook_cosine_normalize': model.model.codebook_cosine_normalize,
            'lambda_vq': model.model.lambda_vq,
            'lambda_ent': model.model.lambda_ent,
            'lambda_psc': model.model.lambda_psc,
            'psc_temp': model.model.psc_temp,
        },
        
        # Training configuration
        'training_config': {
            'current_stage': model.current_stage,
            'total_epochs': model.total_epoch,
        },
        
        # Stage metadata
        'stage_info': {
            'stage': 2,
            'stage_name': 'joint_finetuning',
            'timestamp': datetime.now().isoformat(),
            'components': ['all_model_components'],
            **(stage_info or {})
        }
    }
    
    checkpoint_path = os.path.join(output_dir, 'stage2_complete_model.ckpt')
    torch.save(checkpoint_data, checkpoint_path)
    
    print(f"✓ Stage 2 checkpoint saved: {checkpoint_path}")
    print(f"  Components: complete model (all parameters updated)")
    
    return checkpoint_path


def save_stage_specific_checkpoint(
    model, 
    stage_idx: int, 
    output_dir: str, 
    stage_info: Optional[Dict] = None
) -> str:
    """
    Save checkpoint for specific stage using appropriate saving function.
    
    Args:
        model: The ParToken Lightning model
        stage_idx: Stage index (0, 1, or 2)
        output_dir: Directory to save the checkpoint
        stage_info: Optional dictionary with stage metadata
        
    Returns:
        Path to saved checkpoint file
    """
    if stage_idx == 0:
        return save_stage0_checkpoint(model, output_dir, stage_info)
    elif stage_idx == 1:
        return save_stage1_checkpoint(model, output_dir, stage_info)
    elif stage_idx == 2:
        return save_stage2_checkpoint(model, output_dir, stage_info)
    else:
        raise ValueError(f"Invalid stage index: {stage_idx}. Must be 0, 1, or 2.")


def load_stage_checkpoint(checkpoint_path: str, model=None) -> Dict[str, Any]:
    """
    Load a stage-specific checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model: Optional model to load state into
        
    Returns:
        Dictionary containing checkpoint data
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    stage_info = checkpoint.get('stage_info', {})
    stage = stage_info.get('stage', 'unknown')
    
    print(f"📂 Loading checkpoint from stage {stage}: {checkpoint_path}")
    print(f"   Components: {', '.join(stage_info.get('components', ['unknown']))}")
    
    if model is not None:
        if stage == 0:
            # Load stage 0 components
            model.model.node_encoder.load_state_dict(checkpoint['node_encoder'])
            model.model.edge_encoder.load_state_dict(checkpoint['edge_encoder'])
            
            for i, layer_state in enumerate(checkpoint['gvp_layers']):
                model.model.gvp_layers[i].load_state_dict(layer_state)
            
            model.model.output_projection.load_state_dict(checkpoint['output_projection'])
            model.model.partitioner.load_state_dict(checkpoint['partitioner'])
            model.model.classifier.load_state_dict(checkpoint['classifier'])
            
            if checkpoint['sequence_embedding'] is not None and hasattr(model.model, 'sequence_embedding'):
                model.model.sequence_embedding.load_state_dict(checkpoint['sequence_embedding'])
                
        elif stage == 1:
            # Load codebook components
            model.model.codebook.load_state_dict(checkpoint['codebook'])
            
            # Update loss weights
            loss_weights = checkpoint.get('loss_weights', {})
            model.model.lambda_vq = loss_weights.get('lambda_vq', model.model.lambda_vq)
            model.model.lambda_ent = loss_weights.get('lambda_ent', model.model.lambda_ent)
            model.model.lambda_psc = loss_weights.get('lambda_psc', model.model.lambda_psc)
            
        elif stage == 2:
            # Load complete model
            model.model.load_state_dict(checkpoint['model_state_dict'])
            
        print(f"✓ Model state loaded from stage {stage} checkpoint")
    
    return checkpoint


def create_checkpoint_summary(output_dir: str) -> str:
    """
    Create a summary file of all stage checkpoints.
    
    Args:
        output_dir: Base output directory containing stage subdirectories
        
    Returns:
        Path to summary file
    """
    summary_data = {
        'checkpoint_summary': {
            'created_at': datetime.now().isoformat(),
            'base_directory': output_dir,
            'stages': {}
        }
    }
    
    # Scan for stage checkpoints
    for stage_idx in range(3):
        stage_dir = os.path.join(output_dir, f"stage_{stage_idx}")
        stage_files = []
        
        if os.path.exists(stage_dir):
            for file in os.listdir(stage_dir):
                if file.endswith('.ckpt'):
                    file_path = os.path.join(stage_dir, file)
                    file_size = os.path.getsize(file_path)
                    stage_files.append({
                        'filename': file,
                        'full_path': file_path,
                        'size_mb': round(file_size / (1024*1024), 2)
                    })
        
        summary_data['checkpoint_summary']['stages'][f'stage_{stage_idx}'] = {
            'directory': stage_dir,
            'files': stage_files
        }
    
    # Save summary
    summary_path = os.path.join(output_dir, 'checkpoint_summary.json')
    import json
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"📋 Checkpoint summary created: {summary_path}")
    return summary_path

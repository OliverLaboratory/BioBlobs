"""
Interpretability utilities for ParToken model analysis.

This module provides functions for analyzing and interpreting ParToken model predictions,
including cluster importance scores, attention visualization, and biological insights.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def extract_cluster_info_batch(
    model,
    batch,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Extract cluster importance scores for a batch of proteins.
    
    Args:
        model: ParToken model (either raw model or lightning module)
        batch: Data batch
        device: Device to run inference on
        
    Returns:
        Dict containing predictions, probabilities, importance scores, and metadata
    """
    # Handle lightning module vs raw model
    if hasattr(model, 'model'):
        actual_model = model.model
    else:
        actual_model = model
    
    if device is not None:
        actual_model = actual_model.to(device)
    
    actual_model.eval()
    
    with torch.no_grad():
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        seq = batch.seq if hasattr(batch, 'seq') and hasattr(actual_model, 'sequence_embedding') else None
        
        # Get predictions, probabilities, importance scores, and statistics in one call
        predictions, probabilities, importance_scores, stats = actual_model.get_cluster_analysis(
            h_V, batch.edge_index, h_E, seq, batch.batch
        )
        
        return {
            'predictions': predictions.cpu().numpy(),
            'probabilities': probabilities.cpu().numpy(),
            'importance_scores': importance_scores.cpu().numpy(),
            'true_labels': batch.y.cpu().numpy(),
            'assignment_matrix': stats['assignment_matrix'].cpu().numpy(),
            'cluster_stats': {
                'avg_coverage': stats['avg_coverage'],
                'avg_clusters': stats['avg_clusters'],
                'avg_cluster_size': stats['avg_cluster_size'],
                'avg_max_importance': stats.get('avg_max_importance', 0.0),
                'avg_importance_entropy': stats.get('avg_importance_entropy', 0.0),
                'importance_concentration': stats.get('importance_concentration', 0.0)
            }
        }


def analyze_single_protein(
    importance_data: Dict[str, Any],
    protein_idx: int,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Analyze interpretability for a single protein.
    
    Args:
        importance_data: Output from extract_cluster_importance_batch
        protein_idx: Index of protein to analyze
        top_k: Number of top important clusters to highlight
        
    Returns:
        Dict with detailed analysis for the protein
    """
    pred = importance_data['predictions'][protein_idx]
    prob = importance_data['probabilities'][protein_idx]
    true_label = importance_data['true_labels'][protein_idx]
    importance = importance_data['importance_scores'][protein_idx]
    assignment = importance_data['assignment_matrix'][protein_idx]
    
    # Find valid (non-empty) clusters
    cluster_sizes = assignment.sum(axis=0)  # Sum over residues
    valid_clusters = cluster_sizes > 0
    valid_importance = importance[valid_clusters]
    valid_cluster_indices = np.where(valid_clusters)[0]
    
    # Rank clusters by importance
    importance_ranking = np.argsort(valid_importance)[::-1]  # Descending order
    top_clusters = valid_cluster_indices[importance_ranking[:top_k]]
    
    # Calculate confidence and correctness
    confidence = np.max(prob)
    is_correct = pred == true_label
    
    # Importance distribution analysis (concentration measures focus: 0=dispersed, 1=focused)
    importance_concentration = 1.0 - (
        (-np.sum(valid_importance * np.log(valid_importance + 1e-8))) / 
        np.log(len(valid_importance))
    )
    
    return {
        'protein_idx': protein_idx,
        'prediction': int(pred),
        'true_label': int(true_label),
        'probabilities': prob.tolist(),
        'confidence': float(confidence),
        'is_correct': bool(is_correct),
        'num_valid_clusters': int(valid_clusters.sum()),
        'cluster_sizes': cluster_sizes[valid_clusters].tolist(),
        'importance_scores': valid_importance.tolist(),
        'importance_concentration': float(importance_concentration),
        'top_clusters': {
            'indices': top_clusters.tolist(),
            'importance_scores': importance[top_clusters].tolist(),
            'sizes': cluster_sizes[top_clusters].tolist()
        },
        'cluster_composition': {
            int(cluster_idx): np.where(assignment[:, cluster_idx] > 0.5)[0].tolist()
            for cluster_idx in top_clusters
        }
    }


def dataset_inter_results(
    model,
    dataloader,
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run interpretability analysis on multiple batches.
    
    Args:
        model: ParToken model
        dataloader: DataLoader for analysis
        device: Device to run on
        max_batches: Maximum number of batches to process
        save_path: Optional path to save results
        
    Returns:
        Aggregated interpretability results
    """
    all_results = []
    batch_stats = []
    
    model.eval()
    
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
            
        if device is not None:
            batch = batch.to(device)
        
        # Extract importance data for this batch
        importance_data = extract_cluster_info_batch(model, batch, device)
        batch_stats.append(importance_data['cluster_stats'])
        
        # Analyze each protein in the batch
        batch_size = len(importance_data['predictions'])
        for protein_idx in range(batch_size):
            protein_analysis = analyze_single_protein(importance_data, protein_idx)
            protein_analysis['batch_idx'] = batch_idx
            protein_analysis['global_protein_idx'] = batch_idx * batch_size + protein_idx
            all_results.append(protein_analysis)
    
    # Aggregate statistics
    correct_predictions = [r for r in all_results if r['is_correct']]
    incorrect_predictions = [r for r in all_results if not r['is_correct']]
    
    aggregated_stats = {
        'total_proteins': len(all_results),
        'accuracy': len(correct_predictions) / len(all_results),
        'avg_confidence': np.mean([r['confidence'] for r in all_results]),
        'avg_clusters_per_protein': np.mean([r['num_valid_clusters'] for r in all_results]),
        'avg_importance_concentration': np.mean([r['importance_concentration'] for r in all_results]),
        'avg_coverage': np.mean([bs['avg_coverage'] for bs in batch_stats]),
        'avg_cluster_size': np.mean([bs['avg_cluster_size'] for bs in batch_stats]),
        'avg_max_importance': np.mean([bs['avg_max_importance'] for bs in batch_stats]),
        'correct_vs_incorrect': {
            'correct': {
                'count': len(correct_predictions),
                'avg_confidence': np.mean([r['confidence'] for r in correct_predictions]) if correct_predictions else 0,
                'avg_concentration': np.mean([r['importance_concentration'] for r in correct_predictions]) if correct_predictions else 0
            },
            'incorrect': {
                'count': len(incorrect_predictions),
                'avg_confidence': np.mean([r['confidence'] for r in incorrect_predictions]) if incorrect_predictions else 0,
                'avg_concentration': np.mean([r['importance_concentration'] for r in incorrect_predictions]) if incorrect_predictions else 0
            }
        }
    }
    
    results = {
        'protein_analyses': all_results,
        'aggregated_stats': aggregated_stats,
        'metadata': {
            'num_batches_processed': batch_idx + 1,
        }
    }
    
    if save_path:
        save_interpretability_results(results, save_path)
    
    return results


def save_interpretability_results(results: Dict[str, Any], save_path: str) -> None:
    """Save interpretability results to JSON file."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_numpy(results)
    
    with open(save_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"✓ Interpretability results saved to {save_path}")



def print_interpretability_summary(results: Dict[str, Any]) -> None:
    """Print a summary of interpretability analysis results."""
    stats = results['aggregated_stats']
    
    print("\n" + "="*60)
    print("📊 INTERPRETABILITY ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"📈 Overall Performance:")
    print(f"  • Total proteins analyzed: {stats['total_proteins']}")
    print(f"  • Accuracy: {stats['accuracy']:.3f}")
    print(f"  • Average confidence: {stats['avg_confidence']:.3f}")
    
    print(f"\n🧬 Clustering Analysis:")
    print(f"  • Average clusters per protein: {stats['avg_clusters_per_protein']:.1f}")
    print(f"  • Average cluster coverage: {stats['avg_coverage']:.3f}")
    print(f"  • Average cluster size: {stats['avg_cluster_size']:.1f}")
    
    print(f"\n🎯 Attention Analysis:")
    print(f"  • Average importance concentration: {stats['avg_importance_concentration']:.3f}")
    print(f"  • Average max cluster importance: {stats['avg_max_importance']:.3f}")
    
    print(f"\n✅ Correct vs ❌ Incorrect Predictions:")
    correct_stats = stats['correct_vs_incorrect']['correct']
    incorrect_stats = stats['correct_vs_incorrect']['incorrect']
    
    print(f"  ✅ Correct ({correct_stats['count']} proteins):")
    print(f"     • Average confidence: {correct_stats['avg_confidence']:.3f}")
    print(f"     • Average attention concentration: {correct_stats['avg_concentration']:.3f}")
    
    print(f"  ❌ Incorrect ({incorrect_stats['count']} proteins):")
    print(f"     • Average confidence: {incorrect_stats['avg_confidence']:.3f}")
    print(f"     • Average attention concentration: {incorrect_stats['avg_concentration']:.3f}")
    
    # Analysis insights
    conf_diff = correct_stats['avg_confidence'] - incorrect_stats['avg_confidence']
    conc_diff = correct_stats['avg_concentration'] - incorrect_stats['avg_concentration']
    
    print(f"\n🔍 Key Insights:")
    if conf_diff > 0.05:
        print(f"  • ✓ Correct predictions have notably higher confidence (+{conf_diff:.3f})")
    else:
        print(f"  • ⚠️  Similar confidence between correct/incorrect predictions ({conf_diff:+.3f})")
    
    if conc_diff > 0.05:
        print(f"  • ✓ Correct predictions show more focused attention (+{conc_diff:.3f})")
    elif conc_diff < -0.05:
        print(f"  • ⚠️  Incorrect predictions show more focused attention ({conc_diff:+.3f})")
    else:
        print(f"  • → Similar attention patterns between correct/incorrect ({conc_diff:+.3f})")
    
    print("="*60)


def load_interpretability_results(file_path: str) -> Dict[str, Any]:
    """Load interpretability results from JSON file."""
    with open(file_path, 'r') as f:
        results = json.load(f)
    print(f"✓ Interpretability results loaded from {file_path}")
    return results






# # Example usage function
# def run_interpretability_analysis_example(model, test_loader, device=None, output_dir="./interpretability_outputs"):
#     """
#     Example function showing how to run complete interpretability analysis.
    
#     Args:
#         model: Trained ParToken model
#         test_loader: Test DataLoader
#         device: Device to run on
#         output_dir: Directory to save outputs
#     """
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     print("🔍 Running interpretability analysis...")
    
#     # Run batch analysis
#     results = batch_interpretability_analysis(
#         model=model,
#         dataloader=test_loader,
#         device=device,
#         max_batches=10,  # Limit for example
#         save_path=str(output_dir / "interpretability_results.json")
#     )
    
#     # Print summary
#     print_interpretability_summary(results)
    
#     # Create visualizations
#     plot_importance_distribution(
#         results, 
#         save_path=str(output_dir / "importance_distribution.png")
#     )
    
#     print(f"\n✅ Interpretability analysis complete!")
#     print(f"📁 Results saved to: {output_dir}")
    
#     return results


# def plot_importance_distribution(results: Dict[str, Any], save_path: Optional[str] = None) -> None:
#     """Plot distribution of cluster importance scores."""
#     fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
#     # Extract data
#     concentrations = [p['importance_concentration'] for p in results['protein_analyses']]
#     confidences = [p['confidence'] for p in results['protein_analyses']]
#     num_clusters = [p['num_valid_clusters'] for p in results['protein_analyses']]
#     correct = [p['is_correct'] for p in results['protein_analyses']]
    
#     # Plot 1: Importance concentration distribution
#     axes[0, 0].hist(concentrations, bins=30, alpha=0.7, edgecolor='black')
#     axes[0, 0].set_xlabel('Importance Concentration')
#     axes[0, 0].set_ylabel('Frequency')
#     axes[0, 0].set_title('Distribution of Attention Concentration')
#     axes[0, 0].axvline(np.mean(concentrations), color='red', linestyle='--', label=f'Mean: {np.mean(concentrations):.3f}')
#     axes[0, 0].legend()
    
#     # Plot 2: Confidence vs Concentration
#     correct_mask = np.array(correct)
#     axes[0, 1].scatter(np.array(concentrations)[correct_mask], np.array(confidences)[correct_mask], 
#                       alpha=0.6, label='Correct', color='green')
#     axes[0, 1].scatter(np.array(concentrations)[~correct_mask], np.array(confidences)[~correct_mask], 
#                       alpha=0.6, label='Incorrect', color='red')
#     axes[0, 1].set_xlabel('Importance Concentration')
#     axes[0, 1].set_ylabel('Prediction Confidence')
#     axes[0, 1].set_title('Confidence vs Attention Concentration')
#     axes[0, 1].legend()
    
#     # Plot 3: Number of clusters distribution
#     axes[1, 0].hist(num_clusters, bins=range(1, max(num_clusters)+2), alpha=0.7, edgecolor='black')
#     axes[1, 0].set_xlabel('Number of Valid Clusters')
#     axes[1, 0].set_ylabel('Frequency')
#     axes[1, 0].set_title('Distribution of Cluster Count')
    
#     # Plot 4: Accuracy by concentration quartiles
#     concentration_quartiles = np.percentile(concentrations, [25, 50, 75])
#     quartile_labels = ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']
#     quartile_accuracies = []
    
#     for i in range(4):
#         if i == 0:
#             mask = np.array(concentrations) <= concentration_quartiles[0]
#         elif i == 3:
#             mask = np.array(concentrations) > concentration_quartiles[2]
#         else:
#             mask = (np.array(concentrations) > concentration_quartiles[i-1]) & \
#                    (np.array(concentrations) <= concentration_quartiles[i])
        
#         quartile_acc = np.mean(np.array(correct)[mask]) if mask.any() else 0
#         quartile_accuracies.append(quartile_acc)
    
#     axes[1, 1].bar(quartile_labels, quartile_accuracies, alpha=0.7, edgecolor='black')
#     axes[1, 1].set_ylabel('Accuracy')
#     axes[1, 1].set_title('Accuracy by Attention Concentration Quartile')
#     axes[1, 1].set_ylim(0, 1)
    
#     # Add values on bars
#     for i, acc in enumerate(quartile_accuracies):
#         axes[1, 1].text(i, acc + 0.02, f'{acc:.3f}', ha='center')
    
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"✓ Importance distribution plot saved to {save_path}")
    
#     plt.show()

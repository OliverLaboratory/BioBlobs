"""
Interpretability utilities for bioblobs model analysis.

This module provides functions for analyzing and interpreting bioblobs model predictions,
including blob importance scores, attention visualization, and biological insights.
"""

import torch
import numpy as np
from typing import Dict, Optional, Any
import json
from pathlib import Path
import os


def run_interpretability_analysis(final_model, test_loader, cfg, custom_output_dir):
    """
    Run interpretability analysis on the test set.
    
    Args:
        final_model: The best trained model
        test_loader: Test data loader
        cfg: Configuration object
        custom_output_dir: Directory to save interpretability results
    
    Returns:
        dict: Interpretability results summary for the results JSON
    """
    if not cfg.interpretability.get("enabled", True):
        print("Interpretability analysis disabled")
        return {"enabled": False}

    print("\nINTERPRETABILITY ANALYSIS")
    print("=" * 70)

    # Create interpretability output directory
    interp_output_dir = os.path.join(custom_output_dir, "interpretability")
    os.makedirs(interp_output_dir, exist_ok=True)

    # Run analysis on test set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Running interpretability analysis on test set...")
    interp_results = final_model.get_inter_info(
        test_loader,
        device=device,
        max_batches=cfg.interpretability.get("max_batches", None),
    )

    if interp_results is not None:
        # Save results
        results_path = os.path.join(interp_output_dir, "interpret_blob.json")
        save_interpretability_results(interp_results, results_path)

        # Print summary
        print_interpretability_summary(interp_results)

        return {
            "enabled": True,
            "results_path": results_path,
            "summary": interp_results["aggregated_stats"],
        }
    else:
        print("Interpretability analysis failed")
        return {
            "enabled": False,
            "error": "Analysis failed",
        }


def blob_info_batch(
    model, batch, device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Extract blob importance scores for a batch of proteins.

    Args:
        model: bioblobs model (either raw model or lightning module)
        batch: Data batch
        device: Device to run inference on

    Returns:
        Dict containing predictions, probabilities, importance scores, and metadata
    """
    # Handle lightning module vs raw model
    if hasattr(model, "model"):
        actual_model = model.model
    else:
        actual_model = model

    if device is not None:
        actual_model = actual_model.to(device)

    actual_model.eval()

    with torch.no_grad():
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        seq = (
            batch.seq
            if hasattr(batch, "seq") and hasattr(actual_model, "sequence_embedding")
            else None
        )

        # Get predictions, probabilities, importance scores, and statistics in one call
        predictions, probabilities, importance_scores, stats = (
            actual_model.get_cluster_analysis(
                h_V, batch.edge_index, h_E, seq, batch.batch
            )
        )

        # Extract codebook indices if codebook exists
        codebook_indices = None
        if hasattr(actual_model, "codebook"):
            # Recompute blob features to get codebook indices
            # Process through GVP layers to get node features
            if seq is not None and hasattr(actual_model, "sequence_embedding"):
                seq_emb = actual_model.sequence_embedding(seq)
                h_V_aug = (torch.cat([h_V[0], seq_emb], dim=-1), h_V[1])
            else:
                h_V_aug = h_V

            h_V_encoded = actual_model.node_encoder(h_V_aug)
            h_E_encoded = actual_model.edge_encoder(h_E)
            
            for layer in actual_model.gvp_layers:
                h_V_encoded = layer(h_V_encoded, batch.edge_index, h_E_encoded)
            
            node_features = actual_model.output_projection(h_V_encoded)
            
            # Convert to dense format for partitioning
            from torch_geometric.utils import to_dense_batch
            dense_x, mask = to_dense_batch(node_features, batch.batch)
            dense_index, _ = to_dense_batch(
                torch.arange(node_features.size(0), device=node_features.device), batch.batch
            )
            
            # Apply partitioner to get blob features
            blob_features, assignment_matrix = actual_model.partitioner(
                dense_x, None, mask, edge_index=batch.edge_index, batch_vec=batch.batch, dense_index=dense_index
            )
            
            # Get valid blob mask
            blob_valid_mask = (assignment_matrix.sum(dim=1) > 0)
            
            # Get codebook assignments
            _, code_indices, _, _ = actual_model.codebook(blob_features, mask=blob_valid_mask)
            codebook_indices = code_indices.cpu().numpy()
        else:
            # Model doesn't have codebook (bypass_codebook mode)
            # Create empty array with same shape as assignment matrix
            assignment_shape = stats["assignment_matrix"].shape
            codebook_indices = np.full((assignment_shape[0], assignment_shape[1]), -1, dtype=np.int64)

        return {
            "predictions": predictions.cpu().numpy(),
            "probabilities": probabilities.cpu().numpy(),
            "importance_scores": importance_scores.cpu().numpy(),
            "true_labels": batch.y.cpu().numpy(),
            "assignment_matrix": stats["assignment_matrix"].cpu().numpy(),
            "codebook_indices": codebook_indices,
            "blob_stats": {
                "avg_coverage": stats["avg_coverage"],
                "avg_blobs": stats["avg_clusters"],
                "avg_blob_size": stats["avg_cluster_size"],
                "avg_max_importance": stats.get("avg_max_importance", 0.0),
                "avg_importance_entropy": stats.get("avg_importance_entropy", 0.0),
                "importance_concentration": stats.get("importance_concentration", 0.0),
            },
        }


def _is_multilabel_prediction(prob):
    """
    Check if prediction is multi-label based on probability tensor shape and values.
    
    Multi-label predictions use sigmoid activation (probabilities don't sum to 1.0),
    while single-label predictions use softmax activation (probabilities sum to 1.0).
    
    Args:
        prob: Probability array or tensor for a single sample
        
    Returns:
        bool: True if multi-label classification, False if single-label
    """
    # Convert to numpy if tensor
    if isinstance(prob, torch.Tensor):
        prob = prob.cpu().numpy()
    
    # Convert to numpy array if not already
    prob = np.asarray(prob)
    
    # Check if any probability exceeds 1.0 (impossible for softmax)
    if np.any(prob > 1.0):
        return True
    
    # Check if probabilities sum deviates significantly from 1.0
    # For softmax (single-label): sum ≈ 1.0
    # For sigmoid (multi-label): sum can be arbitrary
    prob_sum = float(np.sum(prob))
    return abs(prob_sum - 1.0) > 0.1


def _compute_multilabel_metrics(pred_prob, true_labels, threshold=0.5):
    """Compute multi-label metrics using threshold."""
    pred_binary = pred_prob >= threshold

    # Compute per-sample metrics
    if isinstance(true_labels, np.ndarray) and true_labels.dtype == bool:
        true_binary = true_labels
    else:
        true_binary = true_labels >= threshold

    # Check if predictions exactly match true labels
    exact_match = np.array_equal(pred_binary, true_binary)

    # Compute F1 score for this sample
    intersection = np.logical_and(pred_binary, true_binary).sum()
    union = np.logical_or(pred_binary, true_binary).sum()

    if union == 0:
        f1_score = 1.0 if intersection == 0 else 0.0
    else:
        precision = intersection / pred_binary.sum() if pred_binary.sum() > 0 else 0.0
        recall = intersection / true_binary.sum() if true_binary.sum() > 0 else 0.0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

    return {
        "exact_match": exact_match,
        "f1_score": f1_score,
        "precision": precision if "precision" in locals() else 0.0,
        "recall": recall if "recall" in locals() else 0.0,
        "num_predicted_labels": pred_binary.sum(),
        "num_true_labels": true_binary.sum(),
    }


def analyze_single_protein(
    importance_data: Dict[str, Any], protein_idx: int
) -> Dict[str, Any]:
    """
    Analyze interpretability for a single protein.

    Args:
        importance_data: Output from blob_info_batch
        protein_idx: Index of protein to analyze

    Returns:
        Dict with detailed analysis for the protein
    """
    pred = importance_data["predictions"][protein_idx]
    prob = importance_data["probabilities"][protein_idx]
    true_label = importance_data["true_labels"][protein_idx]
    importance = importance_data["importance_scores"][protein_idx]
    assignment = importance_data["assignment_matrix"][protein_idx]
    codebook_indices = importance_data["codebook_indices"][protein_idx]

    # Find valid (non-empty) blobs
    blob_sizes = assignment.sum(axis=0)  
    valid_blobs = blob_sizes > 0
    valid_importance = importance[valid_blobs]
    valid_blob_indices = np.where(valid_blobs)[0]
    valid_codebook_indices = codebook_indices[valid_blobs]

    # Rank blobs by importance
    importance_ranking = np.argsort(valid_importance)[::-1]  # Descending order
    all_blobs = valid_blob_indices[importance_ranking]
    ranked_codebook_indices = valid_codebook_indices[importance_ranking]

    # Determine if multi-label or single-label classification
    is_multilabel = _is_multilabel_prediction(prob)

    if is_multilabel:
        # Multi-label classification metrics
        ml_metrics = _compute_multilabel_metrics(prob, true_label)
        # Confidence for multi-label: maximum probability across all labels
        # This represents the model's highest confidence for any single label
        confidence = prob.max()
        is_correct = ml_metrics["exact_match"]

        result = {
            "protein_idx": protein_idx,
            "prediction": (prob >= 0.5).astype(int).tolist(),  # Binary predictions
            "true_label": true_label.tolist()
            if hasattr(true_label, "tolist")
            else true_label,
            "probabilities": prob.tolist(),
            "confidence": float(confidence),
            "is_correct": bool(is_correct),
            "classification_type": "multi-label",
            "f1_score": ml_metrics["f1_score"],
            "precision": ml_metrics["precision"],
            "recall": ml_metrics["recall"],
            "num_predicted_labels": int(ml_metrics["num_predicted_labels"]),
            "num_true_labels": int(ml_metrics["num_true_labels"]),
        }
    else:
        # Single-label classification metrics
        confidence = np.max(prob)
        is_correct = pred == true_label

        result = {
            "protein_idx": protein_idx,
            "prediction": int(pred),
            "true_label": int(true_label),
            "probabilities": prob.tolist(),
            "confidence": float(confidence),
            "is_correct": bool(is_correct),
            "classification_type": "single-label",
        }

    # Add blob analysis (common to both types)
    importance_concentration = 1.0 - (
        (-np.sum(valid_importance * np.log(valid_importance + 1e-8)))
        / np.log(len(valid_importance))
    )

    result.update(
        {
            "num_valid_blobs": int(valid_blobs.sum()),
            "blob_sizes": blob_sizes[valid_blobs].tolist(),
            "importance_scores": valid_importance.tolist(),
            "importance_concentration": float(importance_concentration),
            "blob_indices": all_blobs.tolist(),  # All blobs
            "blob_importance": valid_importance[importance_ranking].tolist(),
            "blob_codebook_indices": ranked_codebook_indices.tolist(),  # Codebook indices in importance order
        }
    )

    # Blob composition analysis
    blob_composition = {}
    for blob_idx in all_blobs:  # Already importance-ranked
        # Since assignment is binary (0 or 1), we can use either > 0 or > 0.5
        residue_ids = np.where(assignment[:, blob_idx] > 0)[0]
        blob_composition[int(blob_idx)] = residue_ids.tolist()

    result["blob_composition"] = blob_composition

    return result


def dataset_inter_results(
    model,
    dataloader,
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run interpretability analysis on multiple batches.

    Args:
        model: bioblobs model
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
        importance_data = blob_info_batch(model, batch, device)
        batch_stats.append(importance_data["blob_stats"])

        # Analyze each protein in the batch
        batch_size = len(importance_data["predictions"])
        for protein_idx in range(batch_size):
            protein_analysis = analyze_single_protein(importance_data, protein_idx)
            protein_analysis["batch_idx"] = batch_idx
            protein_analysis["global_protein_idx"] = (
                batch_idx * batch_size + protein_idx
            )
            all_results.append(protein_analysis)

    # Aggregate statistics - detect classification type
    classification_types = [r["classification_type"] for r in all_results]
    is_multilabel = "multi-label" in classification_types

    # correct_predictions = [r for r in all_results if r["is_correct"]]
    # incorrect_predictions = [r for r in all_results if not r["is_correct"]]

    aggregated_stats = {
        "total_proteins": len(all_results),
        "classification_type": "multi-label" if is_multilabel else "single-label",
        "avg_confidence": np.mean([r["confidence"] for r in all_results]),
    }

    # # Add performance metrics based on classification type
    # if is_multilabel:
    #     # Multi-label metrics
    #     aggregated_stats.update(
    #         {
    #             "exact_match_accuracy": len(correct_predictions) / len(all_results),
    #             "avg_f1_score": np.mean([r["f1_score"] for r in all_results]),
    #             "avg_precision": np.mean([r["precision"] for r in all_results]),
    #             "avg_recall": np.mean([r["recall"] for r in all_results]),
    #             "avg_predicted_labels": np.mean(
    #                 [r["num_predicted_labels"] for r in all_results]
    #             ),
    #             "avg_true_labels": np.mean([r["num_true_labels"] for r in all_results]),
    #         }
    #     )
    # else:
    #     # Single-label metrics
    #     aggregated_stats.update(
    #         {"accuracy": len(correct_predictions) / len(all_results)}
    #     )

    aggregated_stats.update(
        {
            "avg_blobs_per_protein": np.mean(
                [r["num_valid_blobs"] for r in all_results]
            ),
            "avg_importance_concentration": np.mean(
                [r["importance_concentration"] for r in all_results]
            ),
            "avg_coverage": np.mean([bs["avg_coverage"] for bs in batch_stats]),
            "avg_blob_size": np.mean([bs["avg_blob_size"] for bs in batch_stats]),
            "avg_max_importance": np.mean(
                [bs["avg_max_importance"] for bs in batch_stats]
            ),
            # "correct_vs_incorrect": {
            #     "correct": {
            #         "count": len(correct_predictions),
            #         "avg_confidence": np.mean(
            #             [r["confidence"] for r in correct_predictions]
            #         )
            #         if correct_predictions
            #         else 0,
            #         "avg_concentration": np.mean(
            #             [r["importance_concentration"] for r in correct_predictions]
            #         )
            #         if correct_predictions
            #         else 0,
            #     },
            #     "incorrect": {
            #         "count": len(incorrect_predictions),
            #         "avg_confidence": np.mean(
            #             [r["confidence"] for r in incorrect_predictions]
            #         )
            #         if incorrect_predictions
            #         else 0,
            #         "avg_concentration": np.mean(
            #             [r["importance_concentration"] for r in incorrect_predictions]
            #         )
            #         if incorrect_predictions
            #         else 0,
            #     },
            # },
        }
    )

    results = {
        "protein_analyses": all_results,
        "aggregated_stats": aggregated_stats,
        "metadata": {
            "num_batches_processed": batch_idx + 1,
        },
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

    with open(save_path, "w") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"✓ Interpretability results saved to {save_path}")


def print_interpretability_summary(results: Dict[str, Any]) -> None:
    """Print a summary of interpretability analysis results."""
    stats = results["aggregated_stats"]

    print("\n" + "=" * 60)
    print("📊 INTERPRETABILITY ANALYSIS SUMMARY")
    print("=" * 60)

    # print("📈 Overall Performance:")
    # print(f"  • Total proteins analyzed: {stats['total_proteins']}")
    # print(f"  • Classification type: {stats['classification_type']}")

    # if stats["classification_type"] == "multi-label":
    #     print(f"  • Exact match accuracy: {stats['exact_match_accuracy']:.3f}")
    #     print(f"  • Average F1 score: {stats['avg_f1_score']:.3f}")
    #     print(f"  • Average precision: {stats['avg_precision']:.3f}")
    #     print(f"  • Average recall: {stats['avg_recall']:.3f}")
    #     print(f"  • Avg predicted labels: {stats['avg_predicted_labels']:.1f}")
    #     print(f"  • Avg true labels: {stats['avg_true_labels']:.1f}")
    # else:
    #     print(f"  • Accuracy: {stats['accuracy']:.3f}")

    # print(f"  • Average confidence: {stats['avg_confidence']:.3f}")

    print("\n🧬 Blob Analysis:")
    print(f"  • Average blobs per protein: {stats['avg_blobs_per_protein']:.1f}")
    print(f"  • Average blob coverage: {stats['avg_coverage']:.3f}")
    print(f"  • Average blob size: {stats['avg_blob_size']:.1f}")

    print("\n🎯 Attention Analysis:")
    print(
        f"  • Average importance concentration: {stats['avg_importance_concentration']:.3f}"
    )
    print(f"  • Average max blob importance: {stats['avg_max_importance']:.3f}")

    # print("\n✅ Correct vs ❌ Incorrect Predictions:")
    # correct_stats = stats["correct_vs_incorrect"]["correct"]
    # incorrect_stats = stats["correct_vs_incorrect"]["incorrect"]

    # print(f"  ✅ Correct ({correct_stats['count']} proteins):")
    # print(f"     • Average confidence: {correct_stats['avg_confidence']:.3f}")
    # print(
    #     f"     • Average attention concentration: {correct_stats['avg_concentration']:.3f}"
    # )

    # print(f"  ❌ Incorrect ({incorrect_stats['count']} proteins):")
    # print(f"     • Average confidence: {incorrect_stats['avg_confidence']:.3f}")
    # print(
    #     f"     • Average attention concentration: {incorrect_stats['avg_concentration']:.3f}"
    # )

    # Analysis insights
    # conf_diff = correct_stats["avg_confidence"] - incorrect_stats["avg_confidence"]
    # conc_diff = (
    #     correct_stats["avg_concentration"] - incorrect_stats["avg_concentration"]
    # )

    # print("\n🔍 Key Insights:")
    # if conf_diff > 0.05:
    #     print(
    #         f"  • ✓ Correct predictions have notably higher confidence (+{conf_diff:.3f})"
    #     )
    # else:
    #     print(
    #         f"  • ⚠️  Similar confidence between correct/incorrect predictions ({conf_diff:+.3f})"
    #     )

    # if conc_diff > 0.05:
    #     print(
    #         f"  • ✓ Correct predictions show more focused attention (+{conc_diff:.3f})"
    #     )
    # elif conc_diff < -0.05:
    #     print(
    #         f"  • ⚠️  Incorrect predictions show more focused attention ({conc_diff:+.3f})"
    #     )
    # else:
    #     print(
    #         f"  • → Similar attention patterns between correct/incorrect ({conc_diff:+.3f})"
        # )

    print("=" * 60)


def load_interpretability_results(file_path: str) -> Dict[str, Any]:
    """Load interpretability results from JSON file."""
    with open(file_path, "r") as f:
        results = json.load(f)
    print(f"✓ Interpretability results loaded from {file_path}")
    return results


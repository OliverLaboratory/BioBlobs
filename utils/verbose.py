
def print_final_results(results_summary):
    """
    Print comprehensive final results summary.
    
    Args:
        results_summary: Dictionary containing all results and metrics
    """
    print("\nBIOBLOBS TRAINING COMPLETED!")
    print("=" * 70)

    # Print final results summary for checkpoints
    if "best" in results_summary["checkpoints"]:
        best_checkpoint = results_summary['checkpoints']['best']
        if "test_accuracy" in best_checkpoint:
            print(f"Best checkpoint test accuracy: {best_checkpoint['test_accuracy']:.4f}")
        elif "test_fmax" in best_checkpoint:
            print(f"Best checkpoint test FMax: {best_checkpoint['test_fmax']:.4f}")
            print(f"Best precision: {best_checkpoint['test_precision']:.4f}")
            print(f"Best recall: {best_checkpoint['test_recall']:.4f}")

    if "last" in results_summary["checkpoints"]:
        last_checkpoint = results_summary['checkpoints']['last']
        if "test_accuracy" in last_checkpoint:
            print(f"Last checkpoint test accuracy: {last_checkpoint['test_accuracy']:.4f}")
        elif "test_fmax" in last_checkpoint:
            print(f"Last checkpoint test FMax: {last_checkpoint['test_fmax']:.4f}")

    # Print interpretability results
    if results_summary["interpretability"]["enabled"]:
        print("Interpretability analysis completed")
        if "summary" in results_summary["interpretability"]:
            print(
                f"Overall test accuracy: {results_summary['interpretability']['summary']['accuracy']:.4f}"
            )
            print(
                f"Average attention concentration: {results_summary['interpretability']['summary']['avg_importance_concentration']:.3f}"
            )

    print("=" * 70)
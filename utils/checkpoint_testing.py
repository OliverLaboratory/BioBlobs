import os
import json
from datetime import datetime
from omegaconf import OmegaConf


def test_checkpoints_and_save_results(trainer, model, test_loader, checkpoint_callback, 
                                     output_dir, cfg, model_class, wandb_logger=None):
    """
    Test both best and last checkpoints and save results to JSON file.
    
    Args:
        trainer: PyTorch Lightning trainer
        model: Current model instance (last checkpoint)
        test_loader: Test data loader
        checkpoint_callback: ModelCheckpoint callback
        output_dir: Directory to save results
        cfg: Hydra config object
        model_class: Model class for loading checkpoints
        wandb_logger: Optional wandb logger
    
    Returns:
        dict: Results dictionary with test metrics
    """
    
    print("\n" + "=" * 55)
    print("TESTING")
    print("=" * 55)
    
    # Initialize results structure
    results = {
        "experiment_info": {
            "timestamp": datetime.now().isoformat(),
            "dataset": cfg.data.dataset_name,
            "split": cfg.data.split,
            "model_config": OmegaConf.to_container(cfg.model),
            "train_config": OmegaConf.to_container(cfg.train)
        },
        "checkpoints": {}
    }
    
    # Test with LAST checkpoint (current model state)
    print("Testing LAST checkpoint...")
    last_test_results = trainer.test(model, test_loader, verbose=False)
    results["checkpoints"]["last"] = {
        "checkpoint_path": os.path.join(output_dir, "last.ckpt"),
        "test_loss": last_test_results[0]["test_loss"],
        "test_acc": last_test_results[0]["test_acc"]
    }
    print(f"Last checkpoint - Test Loss: {last_test_results[0]['test_loss']:.4f}, Test Acc: {last_test_results[0]['test_acc']:.4f}")
    
    # Test with BEST checkpoint
    print("Testing BEST checkpoint...")
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path and os.path.exists(best_model_path):
        best_model = model_class.load_from_checkpoint(best_model_path)
        best_test_results = trainer.test(best_model, test_loader, verbose=False)
        results["checkpoints"]["best"] = {
            "checkpoint_path": best_model_path,
            "best_epoch": checkpoint_callback.best_model_score.item() if hasattr(checkpoint_callback.best_model_score, 'item') else float(checkpoint_callback.best_model_score),
            "test_loss": best_test_results[0]["test_loss"],
            "test_acc": best_test_results[0]["test_acc"]
        }
        print(f"Best checkpoint - Test Loss: {best_test_results[0]['test_loss']:.4f}, Test Acc: {best_test_results[0]['test_acc']:.4f}")
    else:
        print("Best checkpoint not found!")
        results["checkpoints"]["best"] = {"error": "Best checkpoint not found"}
    
    # Save results to JSON
    results_file = os.path.join(output_dir, 'test_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Print summary
    print("\n" + "=" * 55)
    print("RESULTS SUMMARY")
    print("=" * 55)
    if "error" not in results["checkpoints"]["best"]:
        print(f"BEST checkpoint  - Test Acc: {results['checkpoints']['best']['test_acc']:.4f}")
    print(f"LAST checkpoint  - Test Acc: {results['checkpoints']['last']['test_acc']:.4f}")
    print("=" * 55)
    
    # Handle wandb logging if provided
    if wandb_logger is not None:
        # Use the actual best checkpoint instead of creating a new one
        if best_model_path and os.path.exists(best_model_path):
            wandb_logger.experiment.save(best_model_path)
        # Log summary with test results
        wandb_logger.experiment.log({
            "best_model_path": best_model_path,
            "test_results": results
        })
    
    return results

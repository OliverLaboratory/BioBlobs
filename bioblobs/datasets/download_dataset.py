from pathlib import Path
import argparse

import pandas as pd
from proteinshake.tasks import EnzymeClassTask, StructuralClassTask, ProteinFamilyTask
from tqdm import tqdm
from loguru import logger

from .visualization import protein_to_pdb


def download_data(cfg = None, test_mode=False):
    """Download and process ProteinShake datasets.
    
    Args:
        test_mode (bool): If True, process only 100 files per split type for testing.
    """
    # ============================================================================
    # Dataset Configuration
    # ============================================================================
    DATASET_CONFIGS = {
        "ec_proteinshake": {
            "task_class": EnzymeClassTask,
            "label_field": "EC",
            "label_extractor": lambda x: x.split(".")[0],  
            "description": "Enzyme Commission classification"
        }
    }

    DATASET_NAME = "ec_proteinshake"

    RAW_DATA_DIR = "./data/ps_raw"  # ProteinShake raw data location
    OUTPUT_BASE_DIR = f"./data/{DATASET_NAME}"
    SPLIT_TYPES = ["random", "sequence", "structure"]  # All split types to generate
    SIMILARITY_THRESHOLD = 0.7  # For structure-based split

    # Validate dataset name
    if DATASET_NAME not in DATASET_CONFIGS:
        valid_datasets = list(DATASET_CONFIGS.keys())
        raise ValueError(
            f"Unknown dataset: {DATASET_NAME}. Choose from: {valid_datasets}"
        ) 

    dataset_config = DATASET_CONFIGS[DATASET_NAME]

    # ============================================================================
    # Configure logging
    # ============================================================================
    log_file = Path(OUTPUT_BASE_DIR) / "download_{time}.log"
    logger.add(log_file, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    
    # ============================================================================
    # Test mode notification
    # ============================================================================
    if test_mode:
        logger.warning("TEST MODE: Processing only 100 files per split type")

    # ============================================================================
    # Setup directories
    # ============================================================================
    logger.info("Setting up {} dataset with multiple split types", DATASET_NAME)
    logger.info("Description: {}", dataset_config['description'])
    logger.info("Raw data location: {}", RAW_DATA_DIR)
    logger.info("Output directory: {}", OUTPUT_BASE_DIR)
    logger.info("Split types to generate: {}", SPLIT_TYPES)

    # Create output directories
    pdb_save_dir = Path(OUTPUT_BASE_DIR) / "pdb"
    pdb_save_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for each split type
    for split_type in SPLIT_TYPES:
        split_dir = Path(OUTPUT_BASE_DIR) / split_type
        split_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================================
    # Process both split types
    # ============================================================================
    all_protein_info = {}  # Dictionary to store protein info for each split type
    all_labels = {}  # Dictionary to collect all labels
    token_map = None  # Will be populated from task

    for split_type in SPLIT_TYPES:
        logger.info("=" * 70)
        logger.info("PROCESSING {} SPLIT", split_type.upper())
        logger.info("=" * 70)
        
        # Load ProteinShake task with current split type
        logger.info("Loading {} with {} split...", 
                   dataset_config['task_class'].__name__, split_type)
        task = dataset_config["task_class"](
            split=split_type,
            split_similarity_threshold=SIMILARITY_THRESHOLD,
            root=RAW_DATA_DIR
        )
        dataset = task.dataset
        
        # Get token map from task (all datasets have token_map)
        if token_map is None:
            token_map = task.token_map
            logger.info("Loaded token map with {} classes", len(token_map))
            logger.debug("Sample mappings: {}", dict(list(token_map.items())[:5]))

        # Get split indices
        train_idx = task.train_index
        val_idx = task.val_index
        test_idx = task.test_index

        logger.info("Train set size: {}", len(train_idx))
        logger.info("Validation set size: {}", len(val_idx))
        logger.info("Test set size: {}", len(test_idx))
        total = len(train_idx) + len(val_idx) + len(test_idx)
        logger.info("Total proteins: {}", total)

        # Process proteins
        logger.info("Processing proteins for {} split...", split_type)
        protein_info = []  # List of dicts: {pdb_id, split, label}
        
        protein_generator = dataset.proteins(resolution="atom")
        
        desc = f"Processing {split_type}"
        for idx, protein in enumerate(tqdm(protein_generator, desc=desc)):
            # Test mode: limit to 100 files per split type
            if test_mode and idx >= 100:
                break
            
            # Extract protein information
            protein_id = protein["protein"]["ID"].lower()
            
            # Extract label value based on dataset type
            label_value = protein["protein"][dataset_config["label_field"]]
            
            # Extract the label key that will be used to look up in token_map
            label_key = dataset_config["label_extractor"](label_value)
            
            # Convert to integer label using token_map
            if label_key not in token_map:
                logger.warning("Label key '{}' not found in token_map for protein {}", 
                             label_key, protein_id)
                continue
            label = token_map[label_key]
            
            # Determine which split this protein belongs to
            if idx in train_idx:
                split_name = "train"
            elif idx in val_idx:
                split_name = "val"
            elif idx in test_idx:
                split_name = "test"
            else:
                continue  # Skip if not in any split
            
            # Save PDB file only once (on first split type iteration)
            if split_type == SPLIT_TYPES[0]:
                pdb_filename = f"{protein_id}.pdb"
                pdb_filepath = pdb_save_dir / pdb_filename
                protein_to_pdb(protein, str(pdb_filepath))
            
            # Store protein info for this split
            protein_info.append({
                "pdb_id": protein_id,
                "split": split_name,
                "label": label
            })
            
            # Collect label for shared labels file
            all_labels[protein_id] = label
        
        # Store protein info for this split type
        all_protein_info[split_type] = protein_info
        
        logger.success("Processed {} proteins for {} split", len(protein_info), split_type)

    # ============================================================================
    # Save PDB files summary
    # ============================================================================
    logger.info("=" * 70)
    logger.info("PDB FILES SAVED")
    logger.info("=" * 70)
    logger.success("Saved {} unique PDB files to {}", len(all_labels), pdb_save_dir)

    # ============================================================================
    # Create split CSV files for each split type
    # ============================================================================
    logger.info("=" * 70)
    logger.info("CREATING SPLIT CSV FILES")
    logger.info("=" * 70)

    for split_type in SPLIT_TYPES:
        logger.info("--- {} Split ---", split_type.upper())
        
        df_all = pd.DataFrame(all_protein_info[split_type])
        
        # Create separate DataFrames for each split
        df_train = df_all[df_all["split"] == "train"][["pdb_id"]]
        df_val = df_all[df_all["split"] == "val"][["pdb_id"]]
        df_test = df_all[df_all["split"] == "test"][["pdb_id"]]
        
        # Save split files to split-specific subdirectory
        split_dir = Path(OUTPUT_BASE_DIR) / split_type
        train_csv = split_dir / "train_split.csv"
        val_csv = split_dir / "val_split.csv"
        test_csv = split_dir / "test_split.csv"
        
        df_train.to_csv(train_csv, index=False)
        df_val.to_csv(val_csv, index=False)
        df_test.to_csv(test_csv, index=False)
        
        logger.success("Saved {}/train_split.csv ({} proteins)", split_type, len(df_train))
        logger.success("Saved {}/val_split.csv ({} proteins)", split_type, len(df_val))
        logger.success("Saved {}/test_split.csv ({} proteins)", split_type, len(df_test))

    # ============================================================================
    # Create single shared labels CSV file
    # ============================================================================
    logger.info("=" * 70)
    logger.info("CREATING SHARED LABELS FILE")
    logger.info("=" * 70)

    # Create labels DataFrame (pdb_id -> label) from all collected labels
    df_labels = pd.DataFrame(
        list(all_labels.items()), 
        columns=["pdb_id", "label"]
    )
    df_labels = df_labels.sort_values("pdb_id")  # Sort for easier inspection

    labels_csv = Path(OUTPUT_BASE_DIR) / "labels.csv"
    df_labels.to_csv(labels_csv, index=False)

    logger.success("Saved labels.csv ({} proteins)", len(df_labels))

    # ============================================================================
    # Print summary statistics
    # ============================================================================
    logger.info("=" * 70)
    logger.info("DATASET CREATION SUMMARY")
    logger.info("=" * 70)

    # Directory structure summary
    logger.info("Output directory structure:")
    logger.info("  {}", OUTPUT_BASE_DIR)
    logger.info("    ├── pdb/ ({} PDB files)", len(all_labels))
    
    for split_type in SPLIT_TYPES:
        df_split = pd.DataFrame(all_protein_info[split_type])
        df_train = df_split[df_split["split"] == "train"]
        df_val = df_split[df_split["split"] == "val"]
        df_test = df_split[df_split["split"] == "test"]
        
        logger.info("    ├── {}/", split_type)
        logger.info("    │     ├── train_split.csv ({} proteins)", len(df_train))
        logger.info("    │     ├── val_split.csv ({} proteins)", len(df_val))
        logger.info("    │     └── test_split.csv ({} proteins)", len(df_test))
    
    logger.info("    └── labels.csv ({} total proteins)", len(df_labels))
    
    # Label statistics
    logger.info("")
    logger.info("Label Statistics:")
    logger.info("  Total classes: {}", df_labels['label'].nunique())
    logger.info("  Label range: {} to {}", df_labels['label'].min(), df_labels['label'].max())
    
    logger.info("")
    logger.info("Class Distribution (across all splits):")
    label_counts = df_labels["label"].value_counts().sort_index()
    for label, count in label_counts.items():
        logger.info("    Class {}: {} proteins", label, count)
    
    logger.success("Dataset creation completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and process ProteinShake datasets"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process only 100 files per split type"
    )
    args = parser.parse_args()
    
    download_data(test_mode=args.test)

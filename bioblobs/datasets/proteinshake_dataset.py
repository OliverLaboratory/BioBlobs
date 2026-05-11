from tqdm import tqdm
import torch
from torch_geometric.loader import DataLoader
from loguru import logger
import time

from ..modules.encoders.config import resolve_encoder_name
from .parallel import load_structures_from_pdb
from .proteinshake.adapter import load_prepared_structures
from .task_dataset import TaskDataset
from .bucket_sampler import LengthBucketBatchSampler


def _loader_cfg_value(loader_cfg, key: str, default):
    if loader_cfg is None:
        return default
    if hasattr(loader_cfg, "get"):
        return loader_cfg.get(key, default)
    return getattr(loader_cfg, key, default)


def create_dataloader(
    dataset,
    batch_size=128,
    num_workers=1,
    shuffle=True,
    *,
    loader_cfg=None,
):
    bucket_by_size = bool(_loader_cfg_value(loader_cfg, "bucket_by_size", False))
    bucket_shuffle_batches = bool(
        _loader_cfg_value(loader_cfg, "bucket_shuffle_batches", True)
    )
    bucket_drop_last = bool(_loader_cfg_value(loader_cfg, "bucket_drop_last", False))
    bucket_seed = int(_loader_cfg_value(loader_cfg, "bucket_seed", 0))
    logger.info(
        "Creating dataloader with batch_size={}, num_workers={}, shuffle={}, bucket_by_size={}",
        batch_size,
        num_workers,
        shuffle,
        bucket_by_size,
    )
    common_kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        common_kwargs["prefetch_factor"] = 4

    if bucket_by_size:
        node_counts = getattr(dataset, "node_counts", None)
        if node_counts is not None and len(node_counts) > 0:
            batch_sampler = LengthBucketBatchSampler(
                node_counts,
                batch_size,
                shuffle=shuffle and bucket_shuffle_batches,
                drop_last=bucket_drop_last,
                seed=bucket_seed,
            )
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                **common_kwargs,
            )
        logger.warning(
            "Requested bucket_by_size, but dataset has no node_counts. Falling back to standard batching."
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        **common_kwargs,
    )


class ProteinClassificationDataset(TaskDataset):
    """Backward-compatible alias for the generic task dataset wrapper."""

    pass


def download_and_prepare_ec_dataset(cfg):
    """Integrate download_dataset.py workflow into get_dataset().

    Creates PDB+CSV structure if it doesn't exist:
        data_dir/ec_proteinshake/
        ├── pdb/*.pdb
        ├── {split}/train_split.csv, val_split.csv, test_split.csv
        ├── labels.csv (with full EC strings like "1.2.3.4")
        └── token_maps.json (pre-computed token maps for all 4 levels)

    Args:
        cfg: Configuration object containing:
            - data_dir: Base directory for data storage
            - split: Split method ('random' or 'structure')
            - split_similarity_threshold: Similarity threshold for structure-based split
            - test_mode: If True, limit to 100 proteins for testing

    Returns:
        output_dir: Path to ec_proteinshake directory
        token_maps: Dictionary of token maps for all 4 EC levels
        num_classes: Number of classes at level 1
    """
    from pathlib import Path
    from proteinshake.tasks import EnzymeClassTask
    import pandas as pd
    import json

    # Extract parameters from config
    data_dir = cfg.data_dir
    split = cfg.split
    split_similarity_threshold = cfg.split_similarity_threshold
    test_mode = cfg.get("test_mode", False)

    logger.info(
        f"Downloading and preparing EC dataset (split={split}, threshold={split_similarity_threshold})"
    )

    output_dir = Path(data_dir) / "ec_proteinshake"
    pdb_dir = output_dir / "pdb"
    split_dir = output_dir / split

    # Create directories
    pdb_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    # Load ProteinShake task
    logger.info("Loading EnzymeClassTask from ProteinShake...")
    task = EnzymeClassTask(
        split=split,
        split_similarity_threshold=split_similarity_threshold,
        root=data_dir,
    )

    dataset = task.dataset
    train_index = task.train_index
    val_index = task.val_index
    test_index = task.test_index

    logger.info(
        f"Dataset loaded: {task.size} proteins (train={len(train_index)}, val={len(val_index)}, test={len(test_index)})"
    )

    # Extract num_workers from config
    preprocessing = cfg.get("preprocessing", {})
    if hasattr(preprocessing, "get"):  # DictConfig
        num_workers = preprocessing.get("num_workers", 8)
    else:  # Regular dict or missing
        num_workers = getattr(preprocessing, "num_workers", 8) if preprocessing else 8

    # Process proteins and prepare metadata/tasks
    protein_generator = dataset.proteins(resolution="atom")

    all_labels = {}  # pdb_id -> full EC string (e.g., "1.2.3.4")
    split_assignments = {}  # pdb_id -> split_name
    tasks = []

    logger.info("Loading proteins from ProteinShake and preparing tasks...")
    for idx, protein in enumerate(tqdm(protein_generator, desc="Loading proteins")):
        if test_mode and idx >= 100:
            logger.warning("TEST MODE: Limiting to 100 proteins")
            break

        protein_id = protein["protein"]["ID"].lower()

        # Extract full EC label string
        label_value = protein["protein"]["EC"]
        all_labels[protein_id] = label_value

        # Determine split
        if idx in train_index:
            split_name = "train"
        elif idx in val_index:
            split_name = "val"
        elif idx in test_index:
            split_name = "test"
        else:
            continue

        split_assignments[protein_id] = split_name

        # Prepare task for parallel PDB saving
        pdb_filepath = pdb_dir / f"{protein_id}.pdb"
        tasks.append((protein, pdb_filepath, protein_id))

    logger.info(f"Loaded {len(all_labels)} proteins, prepared {len(tasks)} tasks")

    # Step 3: Save PDB files in parallel
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from .parallel import _save_single_protein

    if num_workers > 1 and len(tasks) > 1:
        logger.info(f"Saving PDB files using {num_workers} workers...")
        failed_saves = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_save_single_protein, task): task[2] for task in tasks
            }

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Saving PDB files"
            ):
                protein_id = futures[future]
                try:
                    pid, success, error = future.result()
                    if not success:
                        failed_saves.append((pid, error))
                except Exception as e:
                    failed_saves.append((protein_id, str(e)))

        if failed_saves:
            logger.warning(f"{len(failed_saves)} proteins failed to save:")
            for pid, error in failed_saves[:5]:  # Show first 5 errors
                logger.warning(f"  {pid}: {error}")
    else:
        # Sequential processing (for num_workers=1 or single file)
        logger.info("Saving PDB files sequentially...")
        for task in tqdm(tasks, desc="Saving PDB files"):
            _save_single_protein(task)

    logger.info(f"Saved {len(all_labels)} PDB files to {pdb_dir}")

    # Pre-compute token maps for all 4 EC levels
    logger.info("Computing token maps for all 4 EC levels...")
    all_ec_labels = set(all_labels.values())
    token_maps = {}

    for level in range(4):
        # Extract labels at this level
        level_labels = set()
        for ec in all_ec_labels:
            ec_parts = ec.split(".")
            if len(ec_parts) > level:
                level_labels.add(ec_parts[level])

        # Create token map: label -> integer
        token_maps[level + 1] = {
            label: idx for idx, label in enumerate(sorted(level_labels))
        }
        logger.info(f"  Level {level + 1}: {len(token_maps[level + 1])} classes")

    # Save token maps to JSON
    token_map_file = output_dir / "token_maps.json"
    with open(token_map_file, "w") as f:
        json.dump(token_maps, f, indent=2)

    # Create labels.csv with full EC strings
    labels_df = pd.DataFrame(
        list(all_labels.items()), columns=["pdb_id", "label"]
    ).sort_values("pdb_id")

    labels_csv = output_dir / "labels.csv"
    labels_df.to_csv(labels_csv, index=False)
    logger.info(
        f"Saved labels.csv ({len(labels_df)} proteins) and token_maps.json ({len(token_maps)} levels)"
    )

    # Create split CSVs
    for split_name in ["train", "val", "test"]:
        pdb_ids = [pid for pid, s in split_assignments.items() if s == split_name]
        split_df = pd.DataFrame({"pdb_id": sorted(pdb_ids)})

        split_csv = split_dir / f"{split_name}_split.csv"
        split_df.to_csv(split_csv, index=False)
        logger.info(f"Saved {split}/{split_name}_split.csv ({len(split_df)} proteins)")

    logger.success(f"EC dataset preparation complete! Output: {output_dir}")

    return output_dir, token_maps, len(token_maps[1])


def load_ec_raw_structures(cfg):
    """Return EC structures dict + num_classes without building task datasets."""
    return _load_from_pdb_csv_structure_impl(cfg)


def load_from_pdb_csv_structure(cfg):
    structures, num_classes = _load_from_pdb_csv_structure_impl(cfg)
    return finalize_task_datasets(cfg, structures, num_classes)


def _load_from_pdb_csv_structure_impl(cfg):
    """Load EC dataset from PDB+CSV structure using Biopandas with parallel processing.

    Expected structure:
        data_dir/ec_proteinshake/
        ├── pdb/*.pdb
        ├── {split}/train_split.csv, val_split.csv, test_split.csv
        ├── labels.csv (with full EC strings)
        └── token_maps.json (pre-computed token maps for all levels)

    Args:
        cfg: Full configuration object containing:
            - datasets.data_dir: Base data directory
            - datasets.dataset_name: Name of dataset (currently 'ec')
            - datasets.split: Split method ('random' or 'structure')
            - datasets.ec_level: EC hierarchy level (1-4), defaults to 1
            - datasets.edge_types: Edge construction method (e.g., 'knn_30')
            - datasets.test_mode: If True, limit to 10 samples per split for testing
            - datasets.preprocessing.min_backbone_completion: Minimum backbone completion rate (default 0.95)
            - datasets.preprocessing.num_workers: Number of parallel workers for PDB parsing (default 4)
            - partitioners.*: Partitioner configuration (optional)

    Returns:
        (train_dataset, val_dataset, test_dataset, num_classes)
    """
    from pathlib import Path
    import pandas as pd
    import json
    from .proteinshake.adapter import _num_workers_from_cfg, _min_completion_from_cfg

    ds_cfg = cfg.datasets
    split = ds_cfg.split
    ec_level = ds_cfg.get("ec_level", 1)
    edge_types = ds_cfg.edge_types
    test_mode = ds_cfg.get("test_mode", False)
    split_similarity_threshold = ds_cfg.get("split_similarity_threshold", 0.7)
    min_completion = _min_completion_from_cfg(ds_cfg)
    num_workers = _num_workers_from_cfg(ds_cfg)

    logger.info(
        f"Loading EC dataset from PDB+CSV structure (split={split}, workers={num_workers}, min_completion={min_completion})"
    )

    base_path = Path(ds_cfg.data_dir) / "ec_proteinshake"
    pdb_dir = base_path / "pdb"
    split_dir = base_path / split

    cache_dir = base_path / "cache"
    cache_dir.mkdir(exist_ok=True)

    # Load token maps
    token_maps_file = base_path / "token_maps.json"
    with open(token_maps_file, "r") as f:
        token_maps = json.load(f)

    token_map = token_maps[str(ec_level)]
    num_classes = len(token_map)
    logger.info(f"Loaded token map for EC level {ec_level}: {num_classes} classes")

    labels_df = pd.read_csv(base_path / "labels.csv")
    ec_string_map = dict(zip(labels_df["pdb_id"], labels_df["label"]))
    logger.info(f"Loaded EC strings for {len(ec_string_map)} proteins")

    structures = {}
    for split_name in ["train", "val", "test"]:
        pdb_ids = pd.read_csv(split_dir / f"{split_name}_split.csv")["pdb_id"].tolist()

        if test_mode:
            pdb_ids = pdb_ids[:10]
            logger.warning(f"TEST MODE: Limiting {split_name} to {len(pdb_ids)} samples")

        cache_file = (
            cache_dir
            / f"{split}_{split_name}_{edge_types}_{split_similarity_threshold}_cache.pt"
        )
        if cache_file.exists() and not test_mode:
            logger.info(f"Loading {split_name} from cache ({cache_file})")
            start_time = time.time()
            structures[split_name] = torch.load(
                cache_file, map_location="cpu", weights_only=False, mmap=True
            )
            logger.info(
                f"  {split_name}: {len(structures[split_name])} structures loaded from cache in {time.time() - start_time:.2f}s"
            )
            continue

        tasks = []
        missing_pdb = 0
        for pdb_id in pdb_ids:
            pdb_path = pdb_dir / f"{pdb_id}.pdb"
            if not pdb_path.exists():
                logger.warning(f"PDB file not found: {pdb_path}")
                missing_pdb += 1
                continue
            tasks.append((pdb_id, pdb_path, ec_string_map[pdb_id], min_completion))

        split_structures, skipped = load_structures_from_pdb(tasks, num_workers, split_name)
        skipped += missing_pdb
        structures[split_name] = split_structures
        logger.info(f"{split_name}: {len(split_structures)} structures loaded ({skipped} skipped)")

        if not test_mode:
            logger.info(f"Saving {split_name} to cache...")
            torch.save(split_structures, cache_file)

    # Filter proteins that lack the required EC level
    if ec_level > 1:
        def _get_ec_string(s):
            return s.get("target_payload") or s.get("ec_string")

        for split_name in ["train", "val", "test"]:
            before = len(structures[split_name])
            structures[split_name] = [
                s for s in structures[split_name]
                if len(_get_ec_string(s).split(".")) >= ec_level
            ]
            filtered = before - len(structures[split_name])
            if filtered > 0:
                logger.info(
                    f"  {split_name}: filtered {filtered}/{before} proteins lacking EC level {ec_level}"
                )

    # Convert EC strings to integer labels based on selected EC level
    logger.info(f"Converting EC strings to integer labels for level {ec_level}...")
    for split_name in ["train", "val", "test"]:
        for structure in structures[split_name]:
            ec_full = structure.get("target_payload", structure.get("ec_string"))
            if ec_full is None:
                raise KeyError(
                    "Expected EC structure to contain 'target_payload' or "
                    f"'ec_string', got keys {sorted(structure.keys())}"
                )
            ec_parts = ec_full.split(".")

            if len(ec_parts) < ec_level:
                raise ValueError(
                    f"EC string '{ec_full}' has fewer than {ec_level} levels for {structure['name']}. "
                    "This should not happen after filtering — check the filtering step."
                )
            ec_level_value = ec_parts[ec_level - 1]
            if ec_level_value not in token_map:
                raise ValueError(
                    f"EC level {ec_level} value '{ec_level_value}' not in token_map for {structure['name']}"
                )
            structure["label"] = token_map[ec_level_value]

    return structures, num_classes


def load_dataset_structures(cfg):
    """Load raw structures (per split) without finalizing into task datasets.

    Used by preprocessing scripts that build PLM mmap caches directly from
    sequences/structures without going through the encoder cache pipeline.
    """
    dataset_name = cfg.datasets.dataset_name

    if dataset_name == "ec":
        return load_ec_raw_structures(cfg)

    if dataset_name in {"go", "go_mf", "go_bp", "go_cc", "pfam", "scop_fam", "scop_sf"}:
        return load_prepared_structures(cfg)

    if dataset_name == "venusx_site_fragment":
        from .venusx_dataset import load_prepared_structures as load_venusx_structures

        return load_venusx_structures(cfg)

    raise ValueError(
        f"Unsupported dataset_name {dataset_name!r}. Supported: 'ec', 'go', "
        f"'go_mf', 'go_bp', 'go_cc', 'pfam', 'scop_fam', 'scop_sf', "
        f"'venusx_site_fragment'."
    )


def finalize_task_datasets(cfg, structures, num_classes):
    """Attach optional preprocessing and wrap structure dicts as datasets."""
    encoder_name = resolve_encoder_name(cfg.encoders)
    _needs_esm_cache = encoder_name == "esm2_static" or encoder_name.endswith("_esm")
    if _needs_esm_cache:
        from .esm2_cache import prepare_esm2_embedding_cache

        logger.info("\nPreparing static ESM2 residue embedding cache...")
        for split_name in ["train", "val", "test"]:
            prepare_esm2_embedding_cache(structures[split_name], cfg, split_name)

    if encoder_name == "saprot_static":
        from .saprot_cache import prepare_saprot_embedding_cache

        logger.info("\nPreparing static SaProt residue embedding cache...")
        for split_name in ["train", "val", "test"]:
            prepare_saprot_embedding_cache(structures[split_name], cfg, split_name)

    from .featurizers import build_featurizer
    featurizer = build_featurizer(cfg)

    # Create datasets
    train_dataset = TaskDataset(
        structures["train"], num_classes=num_classes, featurizer=featurizer
    )
    val_dataset = TaskDataset(
        structures["val"], num_classes=num_classes, featurizer=featurizer
    )
    test_dataset = TaskDataset(
        structures["test"], num_classes=num_classes, featurizer=featurizer
    )

    logger.success(
        f"Datasets created: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}"
    )

    return train_dataset, val_dataset, test_dataset, num_classes


def get_dataset(cfg):
    """
    Get train, validation, and test datasets for the specified protein task.

    EC keeps the existing PDB+CSV preparation path.
    GO and Pfam use the ProteinShake adapter path with task-specific targets.

    Args:
        cfg: Full configuration object (DictConfig or dict) containing:
            - datasets.dataset_name (str): Name of the dataset ('ec', 'go', 'pfam')
            - datasets.split (str): Split method ('random', 'structure')
            - datasets.split_similarity_threshold (float): Similarity threshold for splitting
            - datasets.data_dir (str): Directory to store/load data files
            - datasets.test_mode (bool): If True, limit datasets to small sizes for testing
            - datasets.edge_types (str): Edge construction method and value (e.g., 'knn_30', 'eps_16')
            - datasets.ec_level (int, optional): EC hierarchy level (1-4), defaults to 1
            - datasets.preprocessing.num_workers (int): Number of parallel workers for PDB parsing (default 4)
            - datasets.preprocessing.min_backbone_completion (float): Minimum backbone completion (default 0.95)
            - partitioners.*: Partitioner configuration (optional)

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, num_classes)
    """
    from pathlib import Path

    # Extract parameters from config
    dataset_name = cfg.datasets.dataset_name
    data_dir = cfg.datasets.data_dir
    split = cfg.datasets.split

    if dataset_name == "ec":
        # Check if PDB+CSV structure exists
        base_path = Path(data_dir) / "ec_proteinshake"
        pdb_dir = base_path / "pdb"
        split_dir = base_path / split
        labels_file = base_path / "labels.csv"
        token_maps_file = base_path / "token_maps.json"

        # Check if pdb directory has files (not just exists)
        pdb_has_files = pdb_dir.exists() and any(pdb_dir.iterdir())

        # If not exists or token_maps missing or pdb directory is empty, run integrated download
        if not (
            pdb_has_files
            and split_dir.exists()
            and labels_file.exists()
            and token_maps_file.exists()
        ):
            logger.warning(
                f"PDB+CSV structure not found or incomplete at {base_path}, creating it now..."
            )
            download_and_prepare_ec_dataset(cfg.datasets)
        else:
            logger.info(f"Found existing PDB+CSV structure at {base_path}")

        # Load using Biopandas with parallel processing
        return load_from_pdb_csv_structure(cfg)

    if dataset_name in {"go", "go_mf", "go_bp", "go_cc", "pfam", "scop_fam", "scop_sf"}:
        structures, num_classes = load_prepared_structures(cfg)
        return finalize_task_datasets(cfg, structures, num_classes)

    if dataset_name == "venusx_site_fragment":
        from .venusx_dataset import load_prepared_structures as load_venusx_structures

        structures, num_classes = load_venusx_structures(cfg)
        return finalize_task_datasets(cfg, structures, num_classes)

    raise ValueError(
        "Unknown dataset: "
        f"{dataset_name}. Supported datasets: 'ec', 'go', 'go_mf', 'go_bp', 'go_cc', "
        f"'pfam', 'scop_fam', 'scop_sf', 'venusx_site_fragment'."
    )



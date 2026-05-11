"""Parallel processing utilities for protein dataset preparation (PDB parsing)."""


def _save_single_protein(args):
    """Worker function to save a single protein PDB file in parallel.

    Args:
        args: Tuple of (protein_dict, pdb_filepath, protein_id)

    Returns:
        Tuple of (protein_id, success, error_msg)
    """
    from pathlib import Path
    from bioblobs.datasets.visualization import protein_to_pdb

    protein, pdb_filepath, protein_id = args

    try:
        pdb_filepath = Path(pdb_filepath)
        if not pdb_filepath.exists():
            protein_to_pdb(protein, str(pdb_filepath))
        return (protein_id, True, None)
    except Exception as e:
        return (protein_id, False, str(e))


def _process_single_pdb(args):
    """Worker function to process a single PDB file in parallel.

    Args:
        args: Tuple of (pdb_id, pdb_path, target_payload, min_completion)
              where ``target_payload`` is task-specific serializable metadata.

    Returns:
        Tuple of (structure_dict or None, error_message or None)
    """
    from bioblobs.datasets.atom_representation import (
        parse_pdb_file,
        get_sequence_from_pdb,
        extract_backbone_coords,
        validate_backbone_completion,
    )

    pdb_id, pdb_path, target_payload, min_completion = args

    try:
        df = parse_pdb_file(str(pdb_path))
        sequence = get_sequence_from_pdb(df)

        coords, resnums, adjusted_seq = extract_backbone_coords(df, sequence)

        is_valid, completion_rate = validate_backbone_completion(coords, min_completion)
        if not is_valid:
            return None, f"Low completion rate: {completion_rate:.2%}"

        structure = {
            "name": pdb_id,
            "seq": adjusted_seq,
            "coords": coords,
            "target_payload": target_payload,
            "resnum": resnums,
        }
        return structure, None

    except Exception as e:
        return None, str(e)


def load_structures_from_pdb(tasks, num_workers, split_name):
    """Process PDB files in parallel and return structures sorted by name.

    Args:
        tasks: List of (pdb_id, pdb_path, target_payload, min_completion) tuples.
        num_workers: Number of parallel workers.
        split_name: Split name for the progress bar label.

    Returns:
        Tuple of (structures_list, skipped_count).
        structures_list is sorted by protein name for deterministic ordering.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from loguru import logger
    from tqdm import tqdm

    structures = []
    skipped = 0

    if num_workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_process_single_pdb, task): task[0]
                for task in tasks
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Loading {split_name}",
            ):
                pdb_id = futures[future]
                try:
                    structure, error = future.result()
                    if structure is not None:
                        structures.append(structure)
                    else:
                        skipped += 1
                        if error and "Low completion" not in error:
                            logger.debug("Skipped {}: {}", pdb_id, error)
                except Exception as exc:
                    logger.error("Error processing {}: {}", pdb_id, exc)
                    skipped += 1
    else:
        for task in tqdm(tasks, desc=f"Loading {split_name}"):
            pdb_id = task[0]
            structure, error = _process_single_pdb(task)
            if structure is not None:
                structures.append(structure)
            else:
                skipped += 1
                if error and "Low completion" not in error:
                    logger.debug("Skipped {}: {}", pdb_id, error)

    structures.sort(key=lambda s: s["name"])
    return structures, skipped


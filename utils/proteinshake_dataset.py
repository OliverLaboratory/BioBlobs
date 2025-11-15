import json
import numpy as np
from tqdm import tqdm
import torch
import math
import torch.utils.data as data
import torch.nn.functional as F
import torch_geometric
import torch_cluster
from torch_geometric.loader import DataLoader
import os
from collections import defaultdict
from math import inf

def _get_label_key(pinfo, dataset_name):
    """
    Extract the appropriate label key from protein info based on dataset type.
    
    Args:
        pinfo: Protein information dictionary
        dataset_name: Name of the dataset ('ec', 'proteinfamily', 'scop-fa')
    
    Returns:
        str: The label key for the protein
    """
    if dataset_name == "ec":
        return pinfo["EC"].split(".")[0]
    elif dataset_name == "go":
        return pinfo["molecular_function"]
    elif dataset_name == "scop-fa":
        return pinfo["SCOP-FA"]
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

def _normalize(tensor, dim=-1):
    """
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    """
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True))
    )


def _rbf(D, D_min=0.0, D_max=20.0, D_count=16, device="cpu"):
    """
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    """
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
    return RBF



def create_dataloader(dataset, batch_size=128, num_workers=0, shuffle=True):
    return DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
    )


class ProteinClassificationDataset(data.Dataset):
    """
    A map-style `torch.utils.data.Dataset` for protein classification tasks.
    Transforms JSON/dictionary-style protein structures into featurized protein graphs
    with protein-wise labels for classification.

    Expected data format:
    [
        {
            "name": "protein_id",
            "seq": "SEQUENCE",
            "coords": [[[x,y,z],...], ...],
            "label": class_id  # Integer label for classification
        },
        ...
    ]

    Returns graphs with additional 'y' attribute for classification labels.
    """

    def __init__(
        self,
        data_list,
        num_classes=None,
        num_positional_embeddings=16,
        top_k=30,
        num_rbf=16,
        device="cpu",
    ):
        super(ProteinClassificationDataset, self).__init__()

        self.data_list = data_list
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.device = device
        self.node_counts = [len(e["seq"]) for e in data_list]
        self.num_classes = num_classes or self._infer_num_classes(data_list)

        self.letter_to_num = {
            "C": 4,
            "D": 3,
            "S": 15,
            "Q": 5,
            "K": 11,
            "I": 9,
            "P": 14,
            "T": 16,
            "F": 13,
            "A": 0,
            "G": 7,
            "H": 8,
            "E": 6,
            "L": 10,
            "R": 1,
            "W": 17,
            "V": 19,
            "N": 2,
            "Y": 18,
            "M": 12,
            "X": 20,  # Unknown amino acid
        }
        self.num_amino_acids = 21  # 20 standard amino acids + X for unknown
        self.num_to_letter = {v: k for k, v in self.letter_to_num.items()}

    def _infer_num_classes(self, data_list):
        labels = [item.get("label", 0) for item in data_list if "label" in item]
        return max(labels) + 1 if labels else 1

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        return self._featurize_as_graph(self.data_list[i])

    def _featurize_as_graph(self, protein):
        name = protein["name"]
        with torch.no_grad():
            coords = torch.as_tensor(
                protein["coords"], device=self.device, dtype=torch.float32
            )
            seq = torch.as_tensor(
                [self.letter_to_num.get(a, 20) for a in protein["seq"]],  # Use 20 (X) as default for unknown amino acids
                device=self.device,
                dtype=torch.long,
            )

            # Handle sequence/coordinate length mismatch
            if len(seq) != coords.shape[0]:
                if len(seq) > coords.shape[0]:
                    # Truncate sequence to match coordinates
                    seq = seq[:coords.shape[0]]
                else:
                    # Pad sequence with unknown amino acid token (20 = 'X')
                    pad_length = coords.shape[0] - len(seq)
                    padding = torch.full((pad_length,), 20, device=self.device, dtype=torch.long)
                    seq = torch.cat([seq, padding])
            
            # Validate sequence indices are within bounds (0-20 for the embedding layer)
            if torch.any(seq >= 21) or torch.any(seq < 0):
                # Clamp out-of-bounds values to safe range
                seq = torch.clamp(seq, 0, 20)
                print(f"Warning: Clamped out-of-bounds sequence indices for protein {name}")
            
            assert len(seq) == coords.shape[0], (len(seq), coords.shape[0])
            # Create mask for valid residues (finite coordinates)
            mask = torch.isfinite(coords.sum(dim=(1, 2)))
            coords[~mask] = np.inf
            
            # Filter sequence to match valid coordinates
            # This ensures seq length matches the number of valid residues
            # seq = seq[mask]

            X_ca = coords[:, 1]  # CA coordinates
            edge_index = torch_cluster.knn_graph(X_ca, k=self.top_k)

            pos_embeddings = self._positional_embeddings(edge_index)
            E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
            rbf = _rbf(E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)

            dihedrals = self._dihedrals(coords)
            orientations = self._orientations(X_ca)
            sidechains = self._sidechains(coords)

            node_s = dihedrals
            node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
            edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
            edge_v = _normalize(E_vectors).unsqueeze(-2)

            node_s, node_v, edge_s, edge_v = map(
                torch.nan_to_num, (node_s, node_v, edge_s, edge_v)
            )

            # Add classification label
            label = protein.get("label", 0)
            y = torch.tensor(label, dtype=torch.long, device=self.device)

        data = torch_geometric.data.Data(
            x=X_ca,
            seq=seq,
            name=name,
            resnum=protein['resnum'],
            node_s=node_s,
            node_v=node_v,
            edge_s=edge_s,
            edge_v=edge_v,
            edge_index=edge_index,
            mask=mask,
            y=y,
        )  # Add classification label
        return data

    def _dihedrals(self, X, eps=1e-7):
        # From https://github.com/jingraham/neurips19-graph-protein-design

        X = torch.reshape(X[:, :3], [3 * X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = _normalize(dX, dim=-1)
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]

        # Backbone normals
        n_2 = _normalize(torch.linalg.cross(u_2, u_1), dim=-1)
        n_1 = _normalize(torch.linalg.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2])
        D = torch.reshape(D, [-1, 3])
        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        return D_features

    def _positional_embeddings(
        self, edge_index, num_embeddings=None, period_range=[2, 1000]
    ):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings
        d = edge_index[0] - edge_index[1]

        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=self.device)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def _orientations(self, X):
        forward = _normalize(X[1:] - X[:-1])
        backward = _normalize(X[:-1] - X[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    def _sidechains(self, X):
        n, origin, c = X[:, 0], X[:, 1], X[:, 2]
        c, n = _normalize(c - origin), _normalize(n - origin)
        bisector = _normalize(c + n)
        perp = _normalize(torch.linalg.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec


def generator_to_structures(generator, dataset_name="ec", token_map=None, original_indices=None):
    """
    Convert generator of proteins to list of structures with name, sequence, and coordinates.
    Missing backbone atoms get infinite coordinates and will be filtered by ProteinGraphDataset.

    Args:
        generator: Generator yielding protein data dictionaries
        dataset_name: Name of the dataset for label extraction
        token_map: Pre-computed mapping from labels to integers (optional)
        original_indices: List of original indices corresponding to the generator items (optional)

    Returns:
        tuple: (structures_list, token_map, filtered_indices) where structures_list contains dicts with 'name', 'seq', 'coords', 'label' keys
        and filtered_indices contains the original indices of proteins that passed filtering
    """
    structures = []
    labels_set = set()
    temp_data = []
    filtered_indices = []  

    # Stats
    filtering_stats = {
        "total_processed": 0,
        "successful": 0,
        "partial_residues": 0,
        "zero_length_coords": 0,
    }
    partial_proteins = []

    # Helpers
    BACKBONE = ("N", "CA", "C", "O")
    BACKBONE_SET = set(BACKBONE)
    MISSING = [inf, inf, inf]

    # First pass: collect data and labels (only if token_map not provided)
    print("First pass: collecting data and labels...")
    for i, protein_data in enumerate(tqdm(generator, desc="Collecting data")):
        temp_data.append((protein_data, original_indices[i] if original_indices else i))
        if token_map is None:
            labels_set.add(_get_label_key(protein_data["protein"], dataset_name))

    # Create label mapping if not provided
    if token_map is None:
        sorted_labels = sorted(labels_set)
        token_map = {label: i for i, label in enumerate(sorted_labels)}
        print(f"Found {len(sorted_labels)} unique labels: {sorted_labels}")
    else:
        print(f"Using provided token map with {len(token_map)} labels")

    # Second pass: process the collected data
    print("Second pass: processing structures...")
    for protein_data, original_idx in tqdm(temp_data, desc="Converting proteins"):
        filtering_stats["total_processed"] += 1

        pinfo = protein_data["protein"]
        ainfo = protein_data["atom"]

        name = pinfo["ID"]
        seq = pinfo["sequence"]

        label_key = _get_label_key(pinfo, dataset_name)
        if label_key not in token_map:
            print(f"Warning: Label {label_key} not found in token_map for protein {name}")
            continue
        label = token_map[label_key]

        x = ainfo["x"]; y = ainfo["y"]; z = ainfo["z"]
        atom_types = ainfo["atom_type"]
        residue_numbers = ainfo["residue_number"]

        # Group first-seen backbone atom coords per residue
        residues = defaultdict(dict)
        for res_num, at, xi, yi, zi in zip(residue_numbers, atom_types, x, y, z):
            if at in BACKBONE_SET and at not in residues[res_num]:
                residues[res_num][at] = [xi, yi, zi]

        if not residues:
            filtering_stats["zero_length_coords"] += 1
            continue

        # Build coords in residue order; count completeness
        coords = []
        resnums = []
        complete_residues = 0
        for res_num in sorted(residues):
            atoms = residues[res_num]
            residue_coords = [atoms[a] if a in atoms else MISSING for a in BACKBONE]
            is_complete = all(a in atoms for a in BACKBONE)
            if is_complete:
                complete_residues += 1
            coords.append(residue_coords)
            resnums.append(res_num)

        total_residues = len(residues)
        completion_rate = complete_residues / total_residues if total_residues else 0.0
        
        # Filter out proteins with completion rate < 0.5
        if completion_rate < 0.5:
            continue
            
        if completion_rate < 1.0:
            filtering_stats["partial_residues"] += 1
            partial_proteins.append(
                {
                    "name": name,
                    "total_residues": total_residues,
                    "complete_residues": complete_residues,
                    "completion_rate": completion_rate,
                }
            )
            if completion_rate < 0.1:
                print(
                    f"WARNING: {name} - Very low completion rate: "
                    f"{complete_residues}/{total_residues} ({completion_rate:.2f})"
                )

        filtering_stats["successful"] += 1
        filtered_indices.append(original_idx)  # Track the original index

        # Truncate sequence to match coords length if needed and use the adjusted
        # sequence everywhere to avoid mismatches between seq and coords.
        adjusted_seq = seq[:len(coords)] if len(seq) > len(coords) else seq

        # If sequence and coords lengths still mismatch (coords longer), pad the
        # adjusted sequence with unknown residue symbol 'X' to match coords length.
        if len(adjusted_seq) < len(coords):
            pad_len = len(coords) - len(adjusted_seq)
            adjusted_seq = adjusted_seq + ("X" * pad_len)

        # Now they should match; if not, record a warning and skip the protein.
        if len(adjusted_seq) != len(coords):
            print(
                f"Skipping {name}: final sequence length {len(adjusted_seq)} does not match coords length {len(coords)}"
            )
            continue

        structures.append(
            {
                "name": name,
                "seq": adjusted_seq,
                "coords": coords,
                "label": label,
                "resnum": resnums
            }
        )

    # Summary
    print(f"\n{'=' * 50}")
    print("PROCESSING SUMMARY")
    print(f"{'=' * 50}")
    tp = filtering_stats["total_processed"]
    print(f"Total proteins processed: {tp}")
    print(f"Successfully converted: {filtering_stats['successful']}")
    print(f"With partial residues: {filtering_stats['partial_residues']}")
    if tp:
        print(f"Success rate: {filtering_stats['successful'] / tp:.3f}")
    else:
        print("Success rate: N/A (no proteins processed)")

    if partial_proteins:
        print("\nProteins with incomplete backbone atoms:")
        print(f"{'Protein Name':<15} {'Complete/Total':<15} {'Completion Rate':<15}")
        print("-" * 50)
        for pp in partial_proteins[:10]:
            comp_total = f"{pp['complete_residues']}/{pp['total_residues']}"
            comp_rate = f"{pp['completion_rate']:.3f}"
            print(f"{pp['name']:<15} {comp_total:<15} {comp_rate:<15}")
        if len(partial_proteins) > 10:
            print(f"... and {len(partial_proteins) - 10} more")

    return structures, token_map, filtered_indices

def get_dataset(
    dataset_name, split="structure", split_similarity_threshold=0.7, data_dir="./data", test_mode=False
):
    """
    Get train, validation, and test datasets for the specified protein classification task.
    
    This function splits the data BEFORE converting to structures to preserve correct indices
    and avoid issues with filtering during conversion.

    Args:
        dataset_name (str): Name of the dataset ('ec', 'proteinfamily', 'scop-fa')
        split (str): Split method ('random', 'sequence', 'structure')
        split_similarity_threshold (float): Similarity threshold for splitting
        data_dir (str): Directory to store/load data files
        test_mode (bool): If True, limit datasets to small sizes for testing (100 train, 20 val, 20 test)

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, num_classes)
    """
    # Load the appropriate task
    if dataset_name == "ec":
        from proteinshake.tasks import EnzymeClassTask

        task = EnzymeClassTask(
            split=split, split_similarity_threshold=split_similarity_threshold, root=data_dir
        )
        dataset = task.dataset
        print("Number of proteins:", task.size)
        num_classes = task.num_classes
        train_index, val_index, test_index = (
            task.train_index,
            task.val_index,
            task.test_index,
        )
        token_map = task.token_map

    elif dataset_name == "proteinfamily":
        from proteinshake.tasks import ProteinFamilyTask

        task = ProteinFamilyTask(
            split=split, split_similarity_threshold=split_similarity_threshold, root=data_dir
        )
        dataset = task.dataset
        num_classes = task.num_classes
        train_index, val_index, test_index = (
            task.train_index,
            task.val_index,
            task.test_index,
        )
        token_map = task.token_map

    elif dataset_name == "scop-fa":
        from proteinshake.tasks import StructuralClassTask

        task = StructuralClassTask(
            split=split, split_similarity_threshold=split_similarity_threshold, root=data_dir
        )
        dataset = task.dataset
        num_classes = task.num_classes
        train_index, val_index, test_index = (
            task.train_index,
            task.val_index,
            task.test_index,
        )
        token_map = task.token_map

    elif dataset_name == "go":
        # Import and use the gene ontology dataset function
        from utils.go_dataset import get_gene_ontology_dataset
        return get_gene_ontology_dataset(
            split=split, 
            split_similarity_threshold=split_similarity_threshold, 
            data_dir=data_dir, 
            test_mode=test_mode
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Create data directory if it doesn't exist
    data_dir = os.path.join(data_dir, dataset_name, split)
    os.makedirs(data_dir, exist_ok=True)

    # Check if JSON files already exist for all splits
    train_json_path = os.path.join(data_dir, f"{dataset_name}_train.json")
    val_json_path = os.path.join(data_dir, f"{dataset_name}_val.json")
    test_json_path = os.path.join(data_dir, f"{dataset_name}_test.json")
    token_map_path = os.path.join(data_dir, f"{dataset_name}_token_map.json")
    
    # Paths for filtered indices
    train_indices_path = os.path.join(data_dir, f"{dataset_name}_train_filtered_indices.json")
    val_indices_path = os.path.join(data_dir, f"{dataset_name}_val_filtered_indices.json")
    test_indices_path = os.path.join(data_dir, f"{dataset_name}_test_filtered_indices.json")

    if (os.path.exists(train_json_path) and os.path.exists(val_json_path) and
        os.path.exists(test_json_path) and os.path.exists(token_map_path)):
        
        print("JSON files for all splits already exist. Loading from files...")
        
        # Load token map
        with open(token_map_path, "r") as f:
            token_map = json.load(f)
            
        # Load structures for each split
        with open(train_json_path, "r") as f:
            train_structures = json.load(f)
        with open(val_json_path, "r") as f:
            val_structures = json.load(f)
        with open(test_json_path, "r") as f:
            test_structures = json.load(f)
            
        print(f"Loaded {len(train_structures)} train, {len(val_structures)} val, {len(test_structures)} test structures")
        
    else:
        print("Converting proteins to structures with proper splitting...")
        
        # Get the full protein generator
        protein_generator = dataset.proteins(resolution="atom")
        print("Number of atom level proteins:", len(protein_generator))
        
        # Convert generator to list to enable indexing
        all_proteins = list(protein_generator)
        print(f"Loaded {len(all_proteins)} proteins into memory")
        
        # Create generators for each split using indices
        def create_split_generator(protein_list, indices):
            for idx in indices:
                if idx < len(protein_list):
                    yield protein_list[idx]
        
        # # Build token_map from all proteins to ensure all labels are included
        # print("\nBuilding token map from all proteins...")
        # labels_set = set()
        # for protein_data in tqdm(all_proteins, desc="Collecting all labels"):
        #     labels_set.add(_get_label_key(protein_data["protein"], dataset_name))
        # sorted_labels = sorted(labels_set)
        # token_map = {label: i for i, label in enumerate(sorted_labels)}
        # print(f"Token map created with {len(token_map)} classes: {token_map}")
                
        # Now convert each split using the same token mapping
        print("\nConverting training split...")
        train_generator = create_split_generator(all_proteins, train_index)
        train_structures, _, train_filtered_indices = generator_to_structures(train_generator, dataset_name=dataset_name, token_map=token_map, original_indices=list(train_index))

        print("\nConverting validation split...")
        val_generator = create_split_generator(all_proteins, val_index)
        val_structures, _, val_filtered_indices = generator_to_structures(val_generator, dataset_name=dataset_name, token_map=token_map, original_indices=list(val_index))
        
        print("\nConverting test split...")
        test_generator = create_split_generator(all_proteins, test_index)
        test_structures, _, test_filtered_indices = generator_to_structures(test_generator, dataset_name=dataset_name, token_map=token_map, original_indices=list(test_index))
        print("Finished converting all splits.")
        # Save all data to separate JSON files
        print("Saving processed data to JSON files...")
        
        with open(token_map_path, "w") as f:
            json.dump(token_map, f, indent=2)
            
        with open(train_json_path, "w") as f:
            json.dump(train_structures, f, indent=2)
            
        with open(val_json_path, "w") as f:
            json.dump(val_structures, f, indent=2)
            
        with open(test_json_path, "w") as f:
            json.dump(test_structures, f, indent=2)
            
        # Save filtered indices (convert to native Python integers)
        with open(train_indices_path, "w") as f:
            json.dump([int(idx) for idx in train_filtered_indices], f, indent=2)
            
        with open(val_indices_path, "w") as f:
            json.dump([int(idx) for idx in val_filtered_indices], f, indent=2)
            
        with open(test_indices_path, "w") as f:
            json.dump([int(idx) for idx in test_filtered_indices], f, indent=2)
            
        print(f"Saved {len(train_structures)} train structures to {train_json_path}")
        print(f"Saved {len(val_structures)} val structures to {val_json_path}")
        print(f"Saved {len(test_structures)} test structures to {test_json_path}")
        print(f"Saved token map to {token_map_path}")
        print("Saved filtered indices:")
        print(f"  Train indices ({len(train_filtered_indices)}) to {train_indices_path}")
        print(f"  Val indices ({len(val_filtered_indices)}) to {val_indices_path}")
        print(f"  Test indices ({len(test_filtered_indices)}) to {test_indices_path}")

    # Verify that we have the correct number of classes
    all_labels = set()
    for structures in [train_structures, val_structures, test_structures]:
        all_labels.update(s["label"] for s in structures)
    
    print(f"Found {len(all_labels)} unique labels in processed data")
    assert len(all_labels) <= num_classes, f"More labels found ({len(all_labels)}) than expected ({num_classes})"

    # Apply test mode limiting if enabled
    if test_mode:
        print("\n🧪 TEST MODE ENABLED - Limiting dataset sizes...")
        original_train_size = len(train_structures)
        original_val_size = len(val_structures)
        original_test_size = len(test_structures)
        
        train_structures = train_structures[:100]  # Limit to 100 training samples
        val_structures = val_structures[:20]       # Limit to 20 validation samples  
        test_structures = test_structures[:20]     # Limit to 20 test samples
        
        print(f"  Train: {original_train_size} → {len(train_structures)}")
        print(f"  Val:   {original_val_size} → {len(val_structures)}")
        print(f"  Test:  {original_test_size} → {len(test_structures)}")

    # Create separate datasets for each split
    train_dataset = ProteinClassificationDataset(
        train_structures, num_classes=num_classes
    )
    val_dataset = ProteinClassificationDataset(
        val_structures, num_classes=num_classes
    )
    test_dataset = ProteinClassificationDataset(
        test_structures, num_classes=num_classes
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset, num_classes



from tqdm import tqdm
import json
import os
from proteinshake.tasks import GeneOntologyTask
import torch.utils.data as data
import torch.nn.functional as F
import torch_geometric
import torch_cluster
import torch
from torch_geometric.loader import DataLoader
import numpy as np
from math import inf
import math
from collections import defaultdict


def _get_label_key(pinfo, dataset_name):
    """
    Extract the appropriate label key from protein info based on dataset type.
    
    Args:
        pinfo: Protein information dictionary
        dataset_name: Name of the dataset ('enzymecommission', 'proteinfamily', 'scope')
    
    Returns:
        str: The label key for the protein
    """
    if dataset_name == "enzymecommission":
        return pinfo["EC"].split(".")[0]
    elif dataset_name == "geneontology":
        return pinfo["molecular_function"]
    elif dataset_name == "scope":
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


class ProteinMultiLabelDataset(data.Dataset):
    """
    A map-style `torch.utils.data.Dataset` for protein multi-label classification tasks.
    Transforms JSON/dictionary-style protein structures into featurized protein graphs
    with protein-wise multi-label labels for classification.

    Expected data format:
    [
        {
            "name": "protein_id",
            "seq": "SEQUENCE",
            "coords": [[[x,y,z],...], ...],
            "label": bool_array  # Boolean array for multi-label classification
        },
        ...
    ]

    Returns graphs with additional 'y' attribute for multi-label classification labels.
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
        super(ProteinMultiLabelDataset, self).__init__()

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

            # Add multi-label classification label
            label = protein.get("label", None)
            if label is not None:
                # Convert boolean array to float tensor for multi-label classification
                if isinstance(label, np.ndarray):
                    y = torch.tensor(label, dtype=torch.float32, device=self.device)
                else:
                    # Fallback for single-label case
                    y = torch.tensor(label, dtype=torch.long, device=self.device)
            else:
                # Default empty multi-label (all zeros)
                y = torch.zeros(self.num_classes, dtype=torch.float32, device=self.device)

        data = torch_geometric.data.Data(
            x=X_ca,
            seq=seq,
            name=name,
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





def get_gene_ontology_dataset(
    split="structure", split_similarity_threshold=0.7, data_dir="./data", test_mode=False
):
    """
    Get train, validation, and test datasets for the specified protein classification task.
    
    This function splits the data BEFORE converting to structures to preserve correct indices
    and avoid issues with filtering during conversion.

    Args:
        split (str): Split method ('random', 'sequence', 'structure')
        split_similarity_threshold (float): Similarity threshold for splitting
        data_dir (str): Directory to store/load data files
        test_mode (bool): If True, limit datasets to small sizes for testing (100 train, 20 val, 20 test)

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, num_classes)
    """
    # Load the appropriate task
    task = GeneOntologyTask(
        split=split, split_similarity_threshold=split_similarity_threshold, root=data_dir
    )
    token_map = task.token_map
    num_classes = task.num_classes
    dataset = task.dataset

    train_index, val_index, test_index = (
        task.train_index,
        task.val_index,
        task.test_index,
    )

    print(f"Token map has {len(token_map)}")
    num_classes = len(token_map)

    # Create data directory if it doesn't exist
    data_dir = os.path.join(data_dir, "gene_ontology", split)
    os.makedirs(data_dir, exist_ok=True)

    # Check if JSON files already exist for all splits
    train_json_path = os.path.join(data_dir, "gene_ontology_train.json")
    val_json_path = os.path.join(data_dir, "gene_ontology_val.json")
    test_json_path = os.path.join(data_dir, "gene_ontology_test.json")
    token_map_path = os.path.join(data_dir, "gene_ontology_token_map.json")

    # Paths for filtered indices
    train_indices_path = os.path.join(data_dir, "gene_ontology_train_filtered_indices.json")
    val_indices_path = os.path.join(data_dir, "gene_ontology_val_filtered_indices.json")
    test_indices_path = os.path.join(data_dir, "gene_ontology_test_filtered_indices.json")

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
        
        # Create datasets from loaded structures
        train_dataset = ProteinMultiLabelDataset(train_structures, num_classes=len(token_map), device="cpu")
        val_dataset = ProteinMultiLabelDataset(val_structures, num_classes=len(token_map), device="cpu")
        test_dataset = ProteinMultiLabelDataset(test_structures, num_classes=len(token_map), device="cpu")
        
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
        
        # Build token_map from all proteins to ensure all labels are included
        print("\nBuilding token map from all proteins...")

        train_generator = create_split_generator(all_proteins, train_index)
        val_generator = create_split_generator(all_proteins, val_index)
        test_generator = create_split_generator(all_proteins, test_index)

        # Convert each split to structures
        print("Converting train split...")
        train_structures, _, _ = generator_to_structures(train_generator, "geneontology", token_map, train_index)
        
        print("Converting validation split...")
        val_structures, _, _ = generator_to_structures(val_generator, "geneontology", token_map, val_index)
        
        print("Converting test split...")
        test_structures, _, _ = generator_to_structures(test_generator, "geneontology", token_map, test_index)

        # Save structures to JSON files
        print("Saving structures to JSON files...")
        with open(train_json_path, "w") as f:
            json.dump(train_structures, f)
        with open(val_json_path, "w") as f:
            json.dump(val_structures, f)
        with open(test_json_path, "w") as f:
            json.dump(test_structures, f)
        with open(token_map_path, "w") as f:
            json.dump(token_map, f)

        # Create datasets from structures
        train_dataset = ProteinMultiLabelDataset(train_structures, num_classes=len(token_map), device="cpu")
        val_dataset = ProteinMultiLabelDataset(val_structures, num_classes=len(token_map), device="cpu")
        test_dataset = ProteinMultiLabelDataset(test_structures, num_classes=len(token_map), device="cpu")

    # Apply test mode if requested
    if test_mode:
        print("🔬 TEST MODE: Limiting dataset sizes")
        train_limit = min(100, len(train_dataset))
        val_limit = min(20, len(val_dataset))
        test_limit = min(20, len(test_dataset))
        
        # Create subset datasets
        train_indices = list(range(train_limit))
        val_indices = list(range(val_limit))
        test_indices = list(range(test_limit))
        
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
        test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
        
        print(f"Test mode datasets: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")

    print("\n✅ Dataset loading completed!")
    print(f"📊 Final sizes: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    print(f"🏷️  Number of classes: {len(token_map)}")
    
    return train_dataset, val_dataset, test_dataset, len(token_map)



def generator_to_structures(generator, dataset_name="enzymecommission", token_map=None, original_indices=None):
    """
    Convert generator of proteins to list of structures with name, sequence, and coordinates.
    Missing backbone atoms get infinite coordinates and will be filtered by ProteinGraphDataset.

    Args:
        generator: Generator yielding protein data dictionaries
        dataset_name: Name of the dataset for label extraction
        token_map: Pre-computed mapping from labels to integers (required)
        original_indices: List of original indices corresponding to the generator items (optional)

    Returns:
        tuple: (structures_list, token_map, filtered_indices) where structures_list contains dicts with 'name', 'seq', 'coords', 'label' keys
        and filtered_indices contains the original indices of proteins that passed filtering
    """
    if token_map is None:
        raise ValueError("token_map must be provided.")

    structures = []
    temp_data = []
    filtered_indices = []  # Track which original indices made it through filtering

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

    print("Collecting data...")
    for i, protein_data in enumerate(tqdm(generator, desc="Collecting data")):
        temp_data.append((protein_data, original_indices[i] if original_indices is not None else i))

    print("Processing structures...")
    for protein_data, original_idx in tqdm(temp_data, desc="Converting proteins"):
        filtering_stats["total_processed"] += 1

        pinfo = protein_data["protein"]
        ainfo = protein_data["atom"]

        name = pinfo["ID"]
        seq = pinfo["sequence"]

        # Get label
        tokens = [token_map[i] for i in pinfo['molecular_function']]
        label = np.zeros(len(token_map), dtype=bool)
        label[tokens] = True
        
        # Extract coordinates and atom info
        x = ainfo["x"]
        y = ainfo["y"]
        z = ainfo["z"]

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
        complete_residues = 0
        for res_num in sorted(residues):
            atoms = residues[res_num]
            residue_coords = [atoms[a] if a in atoms else MISSING for a in BACKBONE]
            is_complete = all(a in atoms for a in BACKBONE)
            if is_complete:
                complete_residues += 1
            coords.append(residue_coords)

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
                "label": label.tolist(),  # Convert numpy boolean array to Python list
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



if __name__ == "__main__":
    get_gene_ontology_dataset(
        split="structure", split_similarity_threshold=0.7, data_dir="./data", test_mode=False
    )
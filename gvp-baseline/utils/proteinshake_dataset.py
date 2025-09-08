import json
import numpy as np
import random
from tqdm import tqdm
import torch, math
import torch.utils.data as data
import torch.nn.functional as F
import torch_geometric
import torch_cluster
from torch_geometric.loader import DataLoader
import os
from collections import defaultdict
from math import inf
from typing import Dict, Iterable, List

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


class BatchSampler(data.Sampler):
    """
    From https://github.com/jingraham/neurips19-graph-protein-design.

    A `torch.utils.data.Sampler` which samples batches according to a
    maximum number of graph nodes.

    :param node_counts: array of node counts in the dataset to sample from
    :param max_nodes: the maximum number of nodes in any batch,
                      including batches of a single element
    :param shuffle: if `True`, batches in shuffled order
    """

    def __init__(self, node_counts, max_nodes=3000, shuffle=True):
        self.node_counts = node_counts
        self.idx = [i for i in range(len(node_counts)) if node_counts[i] <= max_nodes]
        self.shuffle = shuffle
        self.max_nodes = max_nodes
        self._form_batches()

    def _form_batches(self):
        self.batches = []
        if self.shuffle:
            random.shuffle(self.idx)
        idx = self.idx
        while idx:
            batch = []
            n_nodes = 0
            while idx and n_nodes + self.node_counts[idx[0]] <= self.max_nodes:
                next_idx, idx = idx[0], idx[1:]
                n_nodes += self.node_counts[next_idx]
                batch.append(next_idx)
            self.batches.append(batch)

    def __len__(self):
        if not self.batches:
            self._form_batches()
        return len(self.batches)

    def __iter__(self):
        if not self.batches:
            self._form_batches()
        for batch in self.batches:
            yield batch


# def create_dataloader(dataset, max_nodes=3000, num_workers=4, shuffle=True):
#     return DataLoader(
#         dataset,
#         num_workers=num_workers,
#         batch_sampler=BatchSampler(
#             dataset.node_counts, max_nodes=max_nodes, shuffle=shuffle
#         ),
#     )
def create_dataloader(dataset, batch_size=128, num_workers=4, shuffle=True):
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
        }
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
                [self.letter_to_num[a] for a in protein["seq"]],
                device=self.device,
                dtype=torch.long,
            )

            # Create mask for valid residues (finite coordinates)
            mask = torch.isfinite(coords.sum(dim=(1, 2)))
            coords[~mask] = np.inf

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


# class ProteinClassificationDataset(data.Dataset):
#     """
#     A map-style `torch.utils.data.Dataset` for protein classification tasks.
#     Transforms JSON/dictionary-style protein structures into featurized protein graphs
#     with protein-wise labels for classification.

#     Expected data format:
#     [
#         {
#             "name": "protein_id",
#             "seq": "SEQUENCE",
#             "coords": [[[x,y,z],...], ...],  # N, CA, C atoms only (3 atoms per residue)
#             "label": class_id  # Integer label for classification
#         },
#         ...
#     ]

#     Returns graphs with additional 'y' attribute for classification labels.
#     """

#     def __init__(
#         self,
#         data_list,
#         num_classes=None,
#         num_positional_embeddings=16,
#         top_k=30,
#         num_rbf=16,
#         device="cpu",
#     ):
#         super(ProteinClassificationDataset, self).__init__()

#         self.data_list = data_list
#         self.top_k = top_k
#         self.num_rbf = num_rbf
#         self.num_positional_embeddings = num_positional_embeddings
#         self.device = device
#         self.node_counts = [len(e["seq"]) for e in data_list]
#         self.num_classes = num_classes or self._infer_num_classes(data_list)

#         self.letter_to_num = {
#             "C": 4,
#             "D": 3,
#             "S": 15,
#             "Q": 5,
#             "K": 11,
#             "I": 9,
#             "P": 14,
#             "T": 16,
#             "F": 13,
#             "A": 0,
#             "G": 7,
#             "H": 8,
#             "E": 6,
#             "L": 10,
#             "R": 1,
#             "W": 17,
#             "V": 19,
#             "N": 2,
#             "Y": 18,
#             "M": 12,
#         }
#         self.num_to_letter = {v: k for k, v in self.letter_to_num.items()}

#     def _infer_num_classes(self, data_list):
#         labels = [item.get("label", 0) for item in data_list if "label" in item]
#         return max(labels) + 1 if labels else 1

#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, i):
#         return self._featurize_as_graph(self.data_list[i])

#     def _featurize_as_graph(self, protein):
#         name = protein["name"]
#         with torch.no_grad():
#             coords = torch.as_tensor(
#                 protein["coords"], device=self.device, dtype=torch.float32
#             )
#             seq = torch.as_tensor(
#                 [self.letter_to_num[a] for a in protein["seq"]],
#                 device=self.device,
#                 dtype=torch.long,
#             )

#             # Create mask for valid residues (finite coordinates)
#             mask = torch.isfinite(coords.sum(dim=(1, 2)))
#             coords[~mask] = np.inf

#             X_ca = coords[:, 1]  # CA coordinates (index 1 in N, CA, C)
#             edge_index = torch_cluster.knn_graph(X_ca, k=self.top_k)

#             pos_embeddings = self._positional_embeddings(edge_index)
#             E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
#             rbf = _rbf(E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)

#             dihedrals = self._dihedrals(coords)
#             orientations = self._orientations(X_ca)
#             sidechains = self._sidechains(coords)

#             node_s = dihedrals
#             node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
#             edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
#             edge_v = _normalize(E_vectors).unsqueeze(-2)

#             node_s, node_v, edge_s, edge_v = map(
#                 torch.nan_to_num, (node_s, node_v, edge_s, edge_v)
#             )

#             # Add classification label
#             label = protein.get("label", 0)
#             y = torch.tensor(label, dtype=torch.long, device=self.device)

#         data = torch_geometric.data.Data(
#             x=X_ca,
#             seq=seq,
#             name=name,
#             node_s=node_s,
#             node_v=node_v,
#             edge_s=edge_s,
#             edge_v=edge_v,
#             edge_index=edge_index,
#             mask=mask,
#             y=y,
#         )  # Add classification label
#         return data

#     def _dihedrals(self, X, eps=1e-7):
#         # From https://github.com/jingraham/neurips19-graph-protein-design
#         # Updated to work with N, CA, C atoms only (3 atoms per residue)

#         X = torch.reshape(X, [3 * X.shape[0], 3])  # Use all 3 atoms: N, CA, C
#         dX = X[1:] - X[:-1]
#         U = _normalize(dX, dim=-1)
#         u_2 = U[:-2]
#         u_1 = U[1:-1]
#         u_0 = U[2:]

#         # Backbone normals
#         n_2 = _normalize(torch.linalg.cross(u_2, u_1), dim=-1)
#         n_1 = _normalize(torch.linalg.cross(u_1, u_0), dim=-1)

#         # Angle between normals
#         cosD = torch.sum(n_2 * n_1, -1)
#         cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
#         D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

#         # This scheme will remove phi[0], psi[-1], omega[-1]
#         D = F.pad(D, [1, 2])
#         D = torch.reshape(D, [-1, 3])
#         # Lift angle representations to the circle
#         D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
#         return D_features

#     def _positional_embeddings(
#         self, edge_index, num_embeddings=None, period_range=[2, 1000]
#     ):
#         # From https://github.com/jingraham/neurips19-graph-protein-design
#         num_embeddings = num_embeddings or self.num_positional_embeddings
#         d = edge_index[0] - edge_index[1]

#         frequency = torch.exp(
#             torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=self.device)
#             * -(np.log(10000.0) / num_embeddings)
#         )
#         angles = d.unsqueeze(-1) * frequency
#         E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
#         return E

#     def _orientations(self, X):
#         forward = _normalize(X[1:] - X[:-1])
#         backward = _normalize(X[:-1] - X[1:])
#         forward = F.pad(forward, [0, 0, 0, 1])
#         backward = F.pad(backward, [0, 0, 1, 0])
#         return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

#     def _sidechains(self, X):
#         # Updated to work with N, CA, C atoms only (3 atoms per residue)
#         n, origin, c = X[:, 0], X[:, 1], X[:, 2]  # N, CA, C atoms
#         c, n = _normalize(c - origin), _normalize(n - origin)
#         bisector = _normalize(c + n)
#         perp = _normalize(torch.linalg.cross(c, n))
#         vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
#         return vec


def print_example_data(data):
    print(f"Protein Name: {data.name}")
    print(f"Sequence: {data.seq}")
    print(f"Number of Nodes: {data.x.shape[0]}")
    print(f"Node Features Shape: {data.node_s.shape}, {data.node_v.shape}")
    print(f"Edge Features Shape: {data.edge_s.shape}, {data.edge_v.shape}")
    print(f"Edge Index Shape: {data.edge_index.shape}")
    print(f"Label (y): {data.y.item() if data.y is not None else 'N/A'}")



def generator_to_structures(generator, dataset_name="enzymecommission", token_map=None, original_indices=None):
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

    # Label key selector
    def _label_key(pinfo):
        if dataset_name == "enzymecommission":
            return pinfo["EC"].split(".")[0]
        elif dataset_name == "proteinfamily":
            return pinfo["Pfam"][0]
        elif dataset_name == "scope":
            return pinfo["SCOP_class"]
        else:
            raise ValueError(f"Unknown dataset_name: {dataset_name}")

    # First pass: collect data and labels (only if token_map not provided)
    print("First pass: collecting data and labels...")
    for i, protein_data in enumerate(tqdm(generator, desc="Collecting data")):
        temp_data.append((protein_data, original_indices[i] if original_indices else i))
        if token_map is None:
            labels_set.add(_label_key(protein_data["protein"]))

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

        label_key = _label_key(pinfo)
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

        # Truncate sequence to match coords length if needed
        adjusted_seq = seq[:len(coords)] if len(seq) > len(coords) else seq

        structures.append(
            {
                "name": name,
                "seq": adjusted_seq,
                "coords": coords,
                "label": label,
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
    dataset_name, split="structure", split_similarity_threshold=0.7, data_dir="./data"
):
    """
    Get train, validation, and test datasets for the specified protein classification task.
    
    This function splits the data BEFORE converting to structures to preserve correct indices
    and avoid issues with filtering during conversion.

    Args:
        dataset_name (str): Name of the dataset ('enzymecommission', 'proteinfamily', 'scope')
        split (str): Split method ('random', 'sequence', 'structure')
        split_similarity_threshold (float): Similarity threshold for splitting
        data_dir (str): Directory to store/load data files

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, num_classes)
    """
    # Load the appropriate task
    if dataset_name == "enzymecommission":
        from proteinshake.tasks import EnzymeClassTask

        task = EnzymeClassTask(
            split=split, split_similarity_threshold=split_similarity_threshold
        )
        dataset = task.dataset
        print("Number of proteins:", task.size)
        num_classes = task.num_classes
        train_index, val_index, test_index = (
            task.train_index,
            task.val_index,
            task.test_index,
        )

    elif dataset_name == "proteinfamily":
        from proteinshake.tasks import ProteinFamilyTask

        task = ProteinFamilyTask(
            split=split, split_similarity_threshold=split_similarity_threshold
        )
        dataset = task.dataset
        num_classes = task.num_classes
        train_index, val_index, test_index = (
            task.train_index,
            task.val_index,
            task.test_index,
        )

    elif dataset_name == "scope":
        from proteinshake.tasks import StructuralClassTask

        task = StructuralClassTask(
            split=split, split_similarity_threshold=split_similarity_threshold
        )
        dataset = task.dataset
        num_classes = task.num_classes
        train_index, val_index, test_index = (
            task.train_index,
            task.val_index,
            task.test_index,
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
        
        # First, get token mapping from training data only to ensure consistency
        print("Creating token mapping from training data...")
        train_generator = create_split_generator(all_proteins, train_index)
        _, token_map, _ = generator_to_structures(train_generator, dataset_name=dataset_name, token_map=None, original_indices=train_index)
        
        print(f"Token map created with {len(token_map)} classes: {token_map}")
        
        # Now convert each split using the same token mapping
        print("\nConverting training split...")
        train_generator = create_split_generator(all_proteins, train_index)
        train_structures, _, train_filtered_indices = generator_to_structures(train_generator, dataset_name=dataset_name, token_map=token_map, original_indices=train_index)
        
        print("\nConverting validation split...")
        val_generator = create_split_generator(all_proteins, val_index)
        val_structures, _, val_filtered_indices = generator_to_structures(val_generator, dataset_name=dataset_name, token_map=token_map, original_indices=val_index)
        
        print("\nConverting test split...")
        test_generator = create_split_generator(all_proteins, test_index)
        test_structures, _, test_filtered_indices = generator_to_structures(test_generator, dataset_name=dataset_name, token_map=token_map, original_indices=test_index)
        
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
            
        # Save filtered indices
        with open(train_indices_path, "w") as f:
            json.dump(train_filtered_indices, f, indent=2)
            
        with open(val_indices_path, "w") as f:
            json.dump(val_filtered_indices, f, indent=2)
            
        with open(test_indices_path, "w") as f:
            json.dump(test_filtered_indices, f, indent=2)
            
        print(f"Saved {len(train_structures)} train structures to {train_json_path}")
        print(f"Saved {len(val_structures)} val structures to {val_json_path}")
        print(f"Saved {len(test_structures)} test structures to {test_json_path}")
        print(f"Saved token map to {token_map_path}")
        print(f"Saved filtered indices:")
        print(f"  Train indices ({len(train_filtered_indices)}) to {train_indices_path}")
        print(f"  Val indices ({len(val_filtered_indices)}) to {val_indices_path}")
        print(f"  Test indices ({len(test_filtered_indices)}) to {test_indices_path}")

    # Verify that we have the correct number of classes
    all_labels = set()
    for structures in [train_structures, val_structures, test_structures]:
        all_labels.update(s["label"] for s in structures)
    
    print(f"Found {len(all_labels)} unique labels in processed data: {sorted(all_labels)}")
    assert len(all_labels) <= num_classes, f"More labels found ({len(all_labels)}) than expected ({num_classes})"

    # Create separate datasets for each split
    train_dataset = ProteinClassificationDataset(
        train_structures, num_classes=num_classes
    )
    val_dataset = ProteinClassificationDataset(val_structures, num_classes=num_classes)
    test_dataset = ProteinClassificationDataset(
        test_structures, num_classes=num_classes
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset, num_classes



# BACKBONE = ("N", "CA", "C")
# INF3 = (float("inf"), float("inf"), float("inf"))

# def generator_to_structures(generator: Iterable[dict], dataset_name: str = "enzymecommission"):
#     """
#     Convert a generator of proteins to a list of structures with name, sequence, coordinates, and label.
#     - Missing backbone atoms get (inf, inf, inf) and can be filtered downstream.
#     - Only uses N, CA, C atoms (O atom is not needed for structural features).
#     - Labels are assigned incrementally (single pass, no pre-scan).
#     """
#     # --- helpers -------------------------------------------------------------
#     def extract_label(info: dict) -> str:
#         if dataset_name == "enzymecommission":
#             return info["EC"].split(".")[0]
#         elif dataset_name == "proteinfamily":
#             return info["Pfam"][0]
#         elif dataset_name == "scope":
#             return info["SCOP-FA"][0]
#         elif dataset_name == "geneontology":
#             return info["molecular_function"][0]
        
#         raise ValueError(f"Unsupported dataset_name: {dataset_name}")

#     # --- state ---------------------------------------------------------------
#     token_map: Dict[str, int] = {}
#     next_token = 0

#     structures: List[dict] = []
#     partial_proteins: List[dict] = []
#     filtering_stats = dict(
#         total_processed=0,
#         successful=0,
#         partial_residues=0,
#         filtered_out=0  # <-- new counter
#     )

#     # --- main loop (single pass) --------------------------------------------
#     for protein_data in generator:
#         filtering_stats["total_processed"] += 1

#         protein_info = protein_data["protein"]
#         atom_info = protein_data["atom"]

#         # label mapping on-the-fly
#         raw_label = extract_label(protein_info)
#         if raw_label not in token_map:
#             token_map[raw_label] = next_token
#             next_token += 1
#         label = token_map[raw_label]

#         name = protein_info["ID"]
#         seq = protein_info["sequence"]

#         x, y, z = atom_info["x"], atom_info["y"], atom_info["z"]
#         atom_types = atom_info["atom_type"]
#         residue_numbers = atom_info["residue_number"]

#         # group atoms by residue -> backbone atom -> coord
#         residues = defaultdict(dict)  # res_num -> {atom_type: (x,y,z)}
#         for ax, ay, az, at, rn in zip(x, y, z, atom_types, residue_numbers):
#             if at in BACKBONE and at not in residues[rn]:
#                 residues[rn][at] = (ax, ay, az)

#         if not residues:
#             filtering_stats["filtered_out"] += 1  # <-- increment counter
#             continue

#         # build coords in residue order, fill missing with INF3
#         coords = []
#         complete = 0
#         for rn in sorted(residues):
#             entry = residues[rn]
#             residue_coords = [entry.get(at, INF3) for at in BACKBONE]  # N, CA, C only
#             if all(at in entry for at in BACKBONE):  # Check if N, CA, C are all present
#                 complete += 1
#             coords.append(residue_coords)

#         total_res = len(residues)
#         completion_rate = complete / total_res if total_res else 0.0

#         if completion_rate < 0.1:
#             print(f"FILTERED OUT: {name} - completion rate {completion_rate:.2f} ({complete}/{total_res}) for N,CA,C atoms")
#             filtering_stats["filtered_out"] += 1  # <-- increment counter
#             continue

#         if completion_rate < 1.0:
#             filtering_stats["partial_residues"] += 1
#             partial_proteins.append(
#                 dict(
#                     name=name,
#                     total_residues=total_res,
#                     complete_residues=complete,
#                     completion_rate=completion_rate,
#                 )
#             )
#             if completion_rate < 0.10:
#                 print(f"WARNING: {name} - very low completion: {complete}/{total_res} ({completion_rate:.2f}) for N,CA,C atoms")

#         filtering_stats["successful"] += 1

#         # align sequence to number of residues we actually built
#         if len(seq) > len(coords):
#             seq = seq[: len(coords)]

#         structures.append(dict(name=name, seq=seq, coords=coords, label=label))

#     # --- summary -------------------------------------------------------------
#     print("\n" + "=" * 50)
#     print("PROCESSING SUMMARY")
#     print("=" * 50)
#     print(f"Total proteins processed: {filtering_stats['total_processed']}")
#     print(f"Successfully converted: {filtering_stats['successful']}")
#     print(f"Filtered out🚫: {filtering_stats['filtered_out']}")
#     print(f"With invalid residues: {filtering_stats['partial_residues']}")
#     if filtering_stats["total_processed"]:
#         print(f"Success rate: {filtering_stats['successful'] / filtering_stats['total_processed']:.3f}")

#     if partial_proteins:
#         print("\nProteins with incomplete backbone atoms (N, CA, C):")
#         print(f"{'Protein Name':<20} {'Complete/Total':<18} {'Completion Rate':<16}")
#         print("-" * 60)
#         for pp in partial_proteins[:5]:
#             ct = f"{pp['complete_residues']}/{pp['total_residues']}"
#             cr = f"{pp['completion_rate']:.3f}"
#             print(f"{pp['name']:<20} {ct:<18} {cr:<16}")
#         if len(partial_proteins) > 10:
#             print(f"... and {len(partial_proteins) - 10} more")

#     return structures




# def get_dataset(
#     dataset_name, split="structure", split_similarity_threshold=0.7, data_dir="./data"
# ):
#     """
#     Get train, validation, and test datasets for the specified protein classification task.

#     Args:
#         dataset_name (str): Name of the dataset ('enzymecommission', 'proteinfamily', 'scope')
#         split (str): Split method ('random', 'sequence', 'structure')
#         split_similarity_threshold (float): Similarity threshold for splitting
#         data_dir (str): Directory to store/load data files

#     Returns:
#         tuple: (train_dataset, val_dataset, test_dataset, num_classes)
#     """
    
#     print(f"Using data directory: {data_dir}")
#     os.makedirs(data_dir, exist_ok=True)

#     # Check if JSON files exist first (before initializing proteinshake)
#     train_json = os.path.join(data_dir, f"{dataset_name}_train.json")
#     val_json = os.path.join(data_dir, f"{dataset_name}_val.json")
#     test_json = os.path.join(data_dir, f"{dataset_name}_test.json")

#     print(f"Checking for cached data files:")
#     print(f"  Train: {train_json} -> exists: {os.path.exists(train_json)}")
#     print(f"  Val: {val_json} -> exists: {os.path.exists(val_json)}")
#     print(f"  Test: {test_json} -> exists: {os.path.exists(test_json)}")

#     if os.path.exists(train_json) and os.path.exists(val_json) and os.path.exists(test_json):
#         print("✅ Loading train/val/test structures from existing JSON files (no proteinshake download needed).")
#         with open(train_json, "r") as f:
#             train_structures = json.load(f)
#         with open(val_json, "r") as f:
#             val_structures = json.load(f)
#         with open(test_json, "r") as f:
#             test_structures = json.load(f)
        
#         # Infer num_classes from the data
#         all_labels = []
#         for structures in [train_structures, val_structures, test_structures]:
#             for structure in structures:
#                 if "label" in structure:
#                     all_labels.append(structure["label"])
        
#         num_classes = max(all_labels) + 1 if all_labels else 2
#         print(f"📊 Loaded cached data: Train={len(train_structures)}, Val={len(val_structures)}, Test={len(test_structures)}")
#         print(f"📊 Inferred num_classes: {num_classes}")
        
#     else:
#         print("❌ JSON files not found. Initializing proteinshake and generating new data...")
        
#         # Only initialize proteinshake when we need to generate new data
#         if dataset_name == "enzymecommission":
#             from proteinshake.tasks import EnzymeClassTask

#             task = EnzymeClassTask(
#                 split=split, split_similarity_threshold=split_similarity_threshold
#             )
#             dataset = task.dataset
#             num_classes = task.num_classes
#             train_index, val_index, test_index = (
#                 task.train_index,
#                 task.val_index,
#                 task.test_index,
#             )

#         elif dataset_name == "proteinfamily":
#             from proteinshake.tasks import ProteinFamilyTask

#             task = ProteinFamilyTask(
#                 split=split, split_similarity_threshold=split_similarity_threshold
#             )
#             dataset = task.dataset
#             num_classes = task.num_classes
#             train_index, val_index, test_index = (
#                 task.train_index,
#                 task.val_index,
#                 task.test_index,
#             )

#         elif dataset_name == "scope":
#             from proteinshake.tasks import StructuralClassTask

#             task = StructuralClassTask(
#                 split=split, split_similarity_threshold=split_similarity_threshold
#             )
#             dataset = task.dataset
#             num_classes = task.num_classes
#             train_index, val_index, test_index = (
#                 task.train_index,
#                 task.val_index,
#                 task.test_index,
#             )

#         elif dataset_name == "geneontology":
#             from proteinshake.tasks import GeneOntologyTask

#             task = GeneOntologyTask(
#                 split=split, split_similarity_threshold=split_similarity_threshold
#             )
#             dataset = task.dataset
#             num_classes = task.num_classes
#             train_index, val_index, test_index = (
#                 task.train_index,
#                 task.val_index,
#                 task.test_index,
#             )

#         else:
#             raise ValueError(f"Unknown dataset: {dataset_name}")

#         print(f"✅ Number of proteins: {task.size} in: {dataset_name} before splitting")

#         # Debug the task split information
#         print(f"\n" + "=" * 50)
#         print("TASK SPLIT INFO")
#         print("=" * 50)
#         print(f"Task dataset size: {task.size}")
#         print(f"Task num_classes: {task.num_classes}")
#         print(f"Split index lengths - Train: {len(train_index)}, Val: {len(val_index)}, Test: {len(test_index)}")
#         print(f"⚠️ Total indices in splits: {len(train_index) + len(val_index) + len(test_index)}")

#         print("Generating train/val/test structures and saving to JSON files.")
        
#         # Convert generator to list for indexing (this is the expensive operation)
#         print("Loading protein structures from dataset...")
#         protein_list = list(dataset.proteins(resolution="atom"))

#         # Analyze split indices coverage
#         all_indices = set(range(len(protein_list)))
#         train_indices_set = set(train_index)
#         val_indices_set = set(val_index)
#         test_indices_set = set(test_index)
#         used_indices = train_indices_set | val_indices_set | test_indices_set
#         missing_indices = all_indices - used_indices
        
#         print(f"\n" + "=" * 50)
#         print("⚠️ SPLIT INDICES CHECK")
#         print("=" * 50)
#         print(f"Total protein indices available: {len(all_indices)}")
#         print(f"Train indices: {len(train_indices_set)}")
#         print(f"Val indices: {len(val_indices_set)}")
#         print(f"Test indices: {len(test_indices_set)}")
#         print(f"Total indices used in splits: {len(used_indices)}")
#         print(f"⚠️ Missing indices (not in any split): {len(missing_indices)}")
        
#         if missing_indices:
#             # Show some protein IDs that are missing
#             missing_proteins = []
#             for idx in sorted(list(missing_indices))[:5]:
#                 try:
#                     protein_data = protein_list[idx]
#                     protein_id = protein_data["protein"]["ID"]
#                     missing_proteins.append(protein_id)
#                 except:
#                     missing_proteins.append(f"idx_{idx}")
#             print(f"Sample missing protein IDs: {missing_proteins}")

#         # Check for overlaps between splits
#         train_val_overlap = train_indices_set & val_indices_set
#         train_test_overlap = train_indices_set & test_indices_set
#         val_test_overlap = val_indices_set & test_indices_set
        
#         if train_val_overlap or train_test_overlap or val_test_overlap:
#             print(f"WARNING: Split overlaps detected!")
#             print(f"Train-Val overlap: {len(train_val_overlap)}")
#             print(f"Train-Test overlap: {len(train_test_overlap)}")
#             print(f"Val-Test overlap: {len(val_test_overlap)}")

#         # Split the protein list using indices
#         train_proteins = [protein_list[i] for i in train_index]
#         val_proteins = [protein_list[i] for i in val_index]
#         test_proteins = [protein_list[i] for i in test_index]
        
#         print(f"Split sizes - Train: {len(train_proteins)}, Val: {len(val_proteins)}, Test: {len(test_proteins)}")
#         print(f"Total split size: {len(train_proteins) + len(val_proteins) + len(test_proteins)}")

#         train_structures = generator_to_structures(train_proteins, dataset_name=dataset_name)
#         val_structures = generator_to_structures(val_proteins, dataset_name=dataset_name)
#         test_structures = generator_to_structures(test_proteins, dataset_name=dataset_name)
        
#         with open(train_json, "w") as f:
#             json.dump(train_structures, f, indent=2)
#         with open(val_json, "w") as f:
#             json.dump(val_structures, f, indent=2)
#         with open(test_json, "w") as f:
#             json.dump(test_structures, f, indent=2)

#     # Create datasets
#     train_dataset = ProteinClassificationDataset(train_structures, num_classes=num_classes)
#     val_dataset = ProteinClassificationDataset(val_structures, num_classes=num_classes)
#     test_dataset = ProteinClassificationDataset(test_structures, num_classes=num_classes)

#     print(f"Train dataset size: {len(train_dataset)}")
#     print(f"Validation dataset size: {len(val_dataset)}")
#     print(f"Test dataset size: {len(test_dataset)}")
    
#     total_final = len(train_dataset) + len(val_dataset) + len(test_dataset)
#     print(f"number of proteins in {dataset_name}: {total_final} after preprocessing")


#     return train_dataset, val_dataset, test_dataset, num_classes

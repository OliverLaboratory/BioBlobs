import json
import numpy as np
import tqdm, random
import torch, math
import torch.utils.data as data
import torch.nn.functional as F
import torch_geometric
import torch_cluster
from torch_geometric.loader import DataLoader
import os

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


def print_example_data(data):
    print(f"Protein Name: {data.name}")
    print(f"Sequence: {data.seq}")
    print(f"Number of Nodes: {data.x.shape[0]}")
    print(f"Node Features Shape: {data.node_s.shape}, {data.node_v.shape}")
    print(f"Edge Features Shape: {data.edge_s.shape}, {data.edge_v.shape}")
    print(f"Edge Index Shape: {data.edge_index.shape}")
    print(f"Label (y): {data.y.item() if data.y is not None else 'N/A'}")


# Data(x=X_ca, seq=seq, name=name,
#     node_s=node_s, node_v=node_v,
#     edge_s=edge_s, edge_v=edge_v,
#     edge_index=edge_index, mask=mask,
#     y=y)



def generator_to_structures(generator, dataset_name="enzymecommission"):
    """
    Convert generator of proteins to list of structures with name, sequence, and coordinates.
    Missing backbone atoms get infinite coordinates and will be filtered by ProteinGraphDataset.

    Args:
        generator: Generator yielding protein data dictionaries

    Returns:
        list: List of dictionaries with 'name', 'seq', 'coords', and 'label' keys
    """
    structures = []
    labels_set = set()
    temp_data = []

    # Track processing statistics
    filtering_stats = {
        "total_processed": 0,
        "successful": 0,
        "partial_residues": 0,
        "zero_length_coords": 0,
    }
    partial_proteins = []

    # First pass: collect all data and extract unique labels
    print("First pass: collecting data and labels...")
    for protein_data in tqdm(generator, desc="Collecting data"):
        temp_data.append(protein_data)

        protein_info = protein_data["protein"]
        if dataset_name == "enzymecommission":
            label = protein_info["EC"].split(".")[0]
        elif dataset_name == "proteinfamily":
            label = protein_info["Pfam"][0]

        labels_set.add(label)

    # Create label mapping
    token_map = {label: i for i, label in enumerate(sorted(list(labels_set)))}
    print(f"Found {len(labels_set)} unique labels: {sorted(list(labels_set))}")

    # Second pass: process the collected data
    print("Second pass: processing structures...")
    for protein_data in tqdm(temp_data, desc="Converting proteins"):
        filtering_stats["total_processed"] += 1

        # Extract protein information
        protein_info = protein_data["protein"]
        atom_info = protein_data["atom"]

        # Get basic information
        name = protein_info["ID"]
        seq = protein_info["sequence"]

        if dataset_name == "enzymecommission":
            label = token_map[protein_info["EC"].split(".")[0]]
        elif dataset_name == "proteinfamily":
            label = token_map[protein_info["Pfam"][0]]

        # Extract atom data
        x_coords = atom_info["x"]
        y_coords = atom_info["y"]
        z_coords = atom_info["z"]
        atom_types = atom_info["atom_type"]
        residue_numbers = atom_info["residue_number"]

        # Group atoms by residue number
        residues = {}
        total_residues = len(set(residue_numbers))

        for i in range(len(x_coords)):
            res_num = residue_numbers[i]
            atom_type = atom_types[i]
            coord = [x_coords[i], y_coords[i], z_coords[i]]

            if res_num not in residues:
                residues[res_num] = {}

            # Take the first occurrence of each backbone atom type per residue
            if (
                atom_type in ["N", "CA", "C", "O"]
                and atom_type not in residues[res_num]
            ):
                residues[res_num][atom_type] = coord

        # Build coords array in residue order
        coords = []
        complete_residues = 0

        for res_num in sorted(residues.keys()):
            backbone_atoms = residues[res_num]
            missing_atoms = [
                atom for atom in ["N", "CA", "C", "O"] if atom not in backbone_atoms
            ]

            # Always create residue coordinates, using inf for missing atoms
            residue_coords = []
            is_complete = True

            for atom_type in ["N", "CA", "C", "O"]:
                if atom_type in backbone_atoms:
                    residue_coords.append(backbone_atoms[atom_type])
                else:
                    residue_coords.append([float("inf"), float("inf"), float("inf")])
                    is_complete = False

            coords.append(residue_coords)

            if is_complete:
                complete_residues += 1

        # Only filter if we have absolutely no coordinates
        if len(coords) == 0:
            filtering_stats["zero_length_coords"] += 1
            continue

        # Track proteins with partial residues for statistics
        completion_rate = (
            complete_residues / total_residues if total_residues > 0 else 0
        )
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

            if completion_rate < 0.1:  # Less than 10% complete - might want to warn
                print(
                    f"WARNING: {name} - Very low completion rate: {complete_residues}/{total_residues} ({completion_rate:.2f})"
                )

        # All proteins are now processed (none filtered)
        filtering_stats["successful"] += 1

        # Truncate sequence to match coords if necessary
        adjusted_seq = seq[: len(coords)] if len(seq) > len(coords) else seq

        structure = {
            "name": name,
            "seq": adjusted_seq,
            "coords": coords,
            "label": label,
        }
        structures.append(structure)

    # Print detailed statistics
    print(f"\n{'=' * 50}")
    print("PROCESSING SUMMARY")
    print(f"{'=' * 50}")
    print(f"Total proteins processed: {filtering_stats['total_processed']}")
    print(f"Successfully converted: {filtering_stats['successful']}")
    print(f"With partial residues: {filtering_stats['partial_residues']}")
    print(
        f"Success rate: {filtering_stats['successful'] / filtering_stats['total_processed']:.3f}"
    )

    if partial_proteins:
        print("\nProteins with incomplete backbone atoms:")
        print(f"{'Protein Name':<15} {'Complete/Total':<15} {'Completion Rate':<15}")
        print("-" * 50)
        for pp in partial_proteins[:10]:  # Show first 10
            comp_total = f"{pp['complete_residues']}/{pp['total_residues']}"
            comp_rate = f"{pp['completion_rate']:.3f}"
            print(f"{pp['name']:<15} {comp_total:<15} {comp_rate:<15}")

        if len(partial_proteins) > 10:
            print(f"... and {len(partial_proteins) - 10} more")

    return structures


def get_dataset(
    dataset_name, split="structure", split_similarity_threshold=0.7, data_dir="./data"
):
    """
    Get train, validation, and test datasets for the specified protein classification task.

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
    os.makedirs(data_dir, exist_ok=True)

    # Check if JSON file already exists
    json_path = os.path.join(data_dir, f"{dataset_name}.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            structures = json.load(f)
        print(f"JSON file {json_path} already exists. Skipping conversion.")
        print(f"First protein: {structures[0]['name']}")
        print(f"Sequence length: {len(structures[0]['seq'])}")
        print(f"Number of residue coordinates: {len(structures[0]['coords'])}")
        print(f"Label: {structures[0]['label']}")
        print(f"Number of proteins: {len(structures)}")

        assert len(set(s["label"] for s in structures)) == num_classes

    else:
        # Convert generator to structures list
        protein_generator = dataset.proteins(resolution="atom")
        print("Number of atom level proteins:", len(protein_generator))

        structures = generator_to_structures(
            protein_generator, dataset_name=dataset_name
        )

        print(f"Processed {len(structures)} proteins")
        if structures:
            print(f"First protein: {structures[0]['name']}")
            print(f"Sequence length: {len(structures[0]['seq'])}")
            print(f"Number of residue coordinates: {len(structures[0]['coords'])}")
            print(f"Label: {structures[0]['label']}")

        # Save structures to JSON file
        with open(json_path, "w") as f:
            json.dump(structures, f, indent=2)

        print(f"Saved {len(structures)} proteins to {json_path}")

    # Create subset data lists using indices
    train_structures = [structures[i] for i in train_index]
    val_structures = [structures[i] for i in val_index]
    test_structures = [structures[i] for i in test_index]

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

import json
import tqdm
import torch
import gvp.data  # assuming gvp is installed as a package (or add its path to sys.path)

# --- Step 1: Load CATH dataset from chain_set.jsonl and chain_set_splits.json ---

class CATHDataset:
    def __init__(self, path, splits_path):
        with open(splits_path) as f:
            dataset_splits = json.load(f)
        train_list, val_list, test_list = dataset_splits['train'], \
            dataset_splits['validation'], dataset_splits['test']
        
        self.train, self.val, self.test = [], [], []
        
        with open(path) as f:
            lines = f.readlines()
        
        for line in tqdm.tqdm(lines):
            entry = json.loads(line)
            name = entry['name']
            coords = entry['coords']
            # Preprocess: Combine backbone atoms for each residue
            entry['coords'] = list(zip(
                coords['N'], coords['CA'], coords['C'], coords['O']
            ))
            if name in train_list:
                self.train.append(entry)
            elif name in val_list:
                self.val.append(entry)
            elif name in test_list:
                self.test.append(entry)

# --- Step 2: Convert to ProteinGraphDataset (the graph format used by GVP models) ---

def print_graph_features(pg_dataset, num_examples=3):
    for i, graph in enumerate(pg_dataset):
        print(f"Protein {i}:")
        print("  Node scalar features shape:", graph.node_s.shape)
        print("  Node vector features shape:", graph.node_v.shape)
        print("  Edge scalar features shape:", graph.edge_s.shape)
        print("  Edge vector features shape:", graph.edge_v.shape)
        print("  Sequence length:", len(graph.seq) if hasattr(graph, 'seq') else "N/A")
        print("  Example node scalar feature:", graph.node_s[0] if graph.node_s.shape[0] > 0 else None)
        print()
        if i >= num_examples - 1:
            break

def main():
    cath = CATHDataset(
        path="data/chain_set.jsonl",
        splits_path="data/chain_set_splits.json"
    )

    # Convert to ProteinGraphDataset for train, val, test splits
    trainset = gvp.data.ProteinGraphDataset(cath.train)
    valset = gvp.data.ProteinGraphDataset(cath.val)
    testset = gvp.data.ProteinGraphDataset(cath.test)

    print("Trainset example features:")
    print_graph_features(trainset)

    print("Valset example features:")
    print_graph_features(valset)

    print("Testset example features:")
    print_graph_features(testset)

if __name__ == "__main__":
    main()
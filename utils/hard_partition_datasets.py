import networkx as nx
from torch_geometric.utils import to_networkx

from proteinshake_dataset import get_dataset, create_dataloader

DATASETS = ['ec']
ALGOS = ['louvain']

def get_partitions(dataset, algo='louvain'):
    all_assignments = []
    for g in dataset:
        g_nx = to_networkx(g)
        if algo == 'louvain': 
            parts = nx.community.louvain_communities(g_nx, seed=123)
            assignments = [None] * len(g_nx.nodes())
            for partition_idx, partition_nodes in enumerate(parts):
                for node in partition_nodes:
                    assignments[node] = partition_idx
        all_assignments.append(assignments)
    return all_assignments

if __name__ == "__main__":


    for dataset in DATASETS:
        train_dataset, val_dataset, test_dataset, num_classes = get_dataset(
                    dataset_name='ec',
                    split='test',
                    test_mode=False,
                )

        splits = [train_dataset, val_dataset, test_dataset]
        for algo in ALGOS:
            for split, split_name in zip(splits, ['train', 'val', 'test']):
                partitions = get_partitions(split, algo=algo)
                with open(f"{dataset}_{algo}_{split_name}.txt", 'w') as out:
                    for part in partitions:
                        out.write(",".join(map(str, part)) + "\n")
                    pass
            
    pass

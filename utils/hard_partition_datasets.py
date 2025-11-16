import sys
import re
from io import StringIO
from collections import defaultdict
import requests

import pandas as pd
import networkx as nx
from Bio.PDB import PDBParser
from torch_geometric.utils import to_networkx

from proteinshake_dataset import get_dataset, create_dataloader

DATASETS = ['ec']
ALGOS = ['louvain', 'scop']
ALGOS = ['scop']



# Path to local SCOP classification file
SCOP_CLA_FILE = "data/dir.cla.scope.2.08-stable.txt"

SCOP_DF = pd.read_csv(
    SCOP_CLA_FILE,
    sep='\t',
    comment='#',
    names=['domain_id', 'pdb_id', 'chain_range', 'scop_class', 'sunid', 'scop_hierarchy'],
    dtype=str
)

def scop(protein):
    all_domains = SCOP_DF.loc[SCOP_DF['pdb_id'] == protein.name.lower()]
    assignments = [None] * len(protein.x)
    for _, domain in all_domains.iterrows():
        chunks = domain['chain_range'].split(",")
        for chunk in chunks:
            chain, res_range = chunk.split(":")
            if res_range:
                try:
                    match = re.match(r'(-?\d+)-(-?\d+)', res_range)
                    if match:
                        range_low = int(match.group(1))
                        range_hi = int(match.group(2))
                    else:
                        continue
                except ValueError:
                    print(res_range)
                    print(protein.resnum)
                    continue
            else:
                range_low = 0
                range_hi = len(protein.x)
            for ind, resnum in enumerate(protein.resnum):
                if resnum in range(range_low, range_hi):
                    assignments[ind] = domain['domain_id']
            pass
    return assignments

def louvain(g):
    g_nx = to_networkx(g)
    parts = nx.community.louvain_communities(g_nx, seed=123)
    assignments = [None] * len(g_nx.nodes())
    for partition_idx, partition_nodes in enumerate(parts):
        for node in partition_nodes:
            assignments[node] = partition_idx
    return assignments

def get_partitions(dataset, algo='louvain'):
    all_assignments = []
    for g in dataset:
        if algo == 'louvain': 
            assignments = louvain(g)
        if algo == 'scop':
            assignments = scop(g)
            pass
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

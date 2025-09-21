
import math
import torch
from torch_geometric.data import Data
import proteinshake.tasks as ps_tasks
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
import numpy as np

def get_gnn_task(name, root='./data'):
    if name == 'enzyme_commission':
        from proteinshake.tasks import EnzymeClassTask
        return EnzymeClassTask(split='structure', split_similarity_threshold=0.7, root=root)
    elif name == 'structural_class':
        from proteinshake.tasks import StructuralClassTask
        return StructuralClassTask(split='structure', split_similarity_threshold=0.7, root=root)
    elif name == 'gene_ontology':
        from proteinshake.tasks import GeneOntologyTask
        return GeneOntologyTask(split='structure', split_similarity_threshold=0.7, root=root)
    else:
        raise ValueError(f"Unknown dataset name: {name}")
    



def get_transformed_graph_dataset(cfg, dataset, task, y_transform=None):
    data_transform = GraphTrainTransform(task, y_transform)
    return dataset.to_graph(eps=cfg.data.graph_eps).pyg(transform=data_transform)



class GraphTrainTransform(object):
    def __init__(self, task, y_transform=None):
        self.task = task
        _,self.task_type = task.task_type
        self.y_transform = y_transform

    def __call__(self, data):
        data, protein_dict = data
        new_data = Data()
        new_data.x = data.x
        new_data.residue_idx = torch.arange(data.num_nodes)
        new_data.edge_index = data.edge_index
        new_data.edge_attr = data.edge_attr
        new_data.y = self.task.target(protein_dict)
        new_data = reshape_data(new_data, self.task_type)
        new_data = add_other_data(new_data, self.task, protein_dict)
        return new_data
    
def reshape_data(data, task_type, y_transform=None):
    if 'binary' in task_type:
        data.y = torch.tensor(data.y).view(-1, 1).float()
    if task_type == 'multi_label':
        data.y = torch.tensor(data.y).view(1, -1).float()
    if task_type == 'regression':
        data.y = torch.tensor(data.y).view(-1, 1).float()
        if y_transform is not None:
            data.y = torch.from_numpy(y_transform.transform(data.y).astype('float32'))
    return data

def add_other_data(data, task, protein_dict):
    if isinstance(task, ps_tasks.LigandAffinityTask):
        fp_maccs = torch.tensor(protein_dict['protein']['fp_maccs'])
        fp_morgan_r2 = torch.tensor(protein_dict['protein']['fp_morgan_r2'])
        other_x = torch.cat((fp_maccs, fp_morgan_r2), dim=-1).float()
        data.other_x = other_x.view(1, -1)
    return data




def get_data_loaders(dataset, task, batch_size, num_workers):

    train_loader = DataLoader(
        Subset(dataset, np.asarray(task.train_index)),
        batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(
        Subset(dataset, np.asarray(task.val_index)),
        batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(
        Subset(dataset, np.asarray(task.test_index)),
        batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
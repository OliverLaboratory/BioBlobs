from proteinshake.tasks import EnzymeClassTask
from proteinshake.tasks import StructuralClassTask
from proteinshake.tasks import GeneOntologyTask
from tqdm import tqdm



datasets = ['enzyme_commission', 'structural_class', 'gene_ontology']

def get_dataset(task_name):
    if task_name == 'enzyme_commission':
        return EnzymeClassTask(split='structure', split_similarity_threshold=0.7, root='./data').dataset
    elif task_name == 'structural_class':
        return StructuralClassTask(split='structure', split_similarity_threshold=0.7, root='./data').dataset
    elif task_name == 'gene_ontology':
        return GeneOntologyTask(split='structure', split_similarity_threshold=0.7, root='./data').dataset
    elif task_name == 'ligand_affinity':
        raise ValueError(f"Unknown task name: {task_name}")

for dataset_name in datasets:
    print(f"Dataset: {dataset_name}")
    dataset = get_dataset(dataset_name)
    protein_generator = dataset.proteins()

    i = 0
    for protein in tqdm(protein_generator, desc="Processing proteins"):
        if len(protein['protein']['sequence']) > 3000:
            print('long sequence:', len(protein['protein']['ID']), len(protein['protein']['sequence']))
            i += 1

    print("total number of proteins with sequence length > 3000:", i)
    




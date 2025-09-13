from proteinshake.tasks import EnzymeClassTask
from utils.visualization import protein_to_pdb
import os
from tqdm import tqdm


task = EnzymeClassTask(split='structure', split_similarity_threshold=0.7, root ='./data')
dataset = task.dataset
protein_generator = dataset.proteins(resolution='atom')
save_dir = './pdb/enzyme_commission_small'

os.makedirs(save_dir, exist_ok=True)

# print(min(task.train_index))

i = 0
for protein in tqdm(protein_generator, desc="Saving PDB files"):
    seq = protein['protein']['sequence']
    if len(seq) < 100:
        print(seq)
        print(type(seq))
        print(f"Number of characters in sequence: {len(seq)}")
        protein_to_pdb(protein, os.path.join(save_dir, f"{i}_{protein['protein']['ID']}.pdb"))
        i += 1
        if i >= 15:
            break

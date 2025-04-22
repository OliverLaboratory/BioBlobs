from proteinshake.datasets import RCSBDataset, Dataset
import torch


backbone_atoms = ['N', 'CA', 'C', 'O']

data_list = []

# coordinates x: L * N_atoms * 3
for i, protein in enumerate(RCSBDataset().proteins(resolution='atom')):
    print(f"Processing protein {i+1}")
    # print(protein['atom'].keys())
    # print('number of atoms:', len(protein['atom']['x']))
    # print(max(protein['atom']['residue_number']))

    x = protein['atom']['x']
    y = protein['atom']['y']
    z = protein['atom']['z']
    atom_type = protein['atom']['atom_type']
    residue_number = protein['atom']['residue_number']

    # Organize atoms by residue number
    residues = {}
    for i in range(len(atom_type)):
        if atom_type[i] in backbone_atoms:
            res_num = residue_number[i]
            if res_num not in residues:
                residues[res_num] = {}
            residues[res_num][atom_type[i]] = [x[i], y[i], z[i]]
    
    # Filter residues that have all backbone atoms
    complete_residues = []
    for res_num, atoms in residues.items():
        if all(atom in atoms for atom in backbone_atoms):
            complete_residues.append(atoms)
    
    if len(complete_residues) == 0:
        print(f"Protein {i+1} has no complete residues, skipping")
        continue
    
    print(f"Protein {i+1} has {len(complete_residues)} complete residues")
    
    # Convert to tensors
    N_coords = torch.tensor([res['N'] for res in complete_residues])
    CA_coords = torch.tensor([res['CA'] for res in complete_residues])
    C_coords = torch.tensor([res['C'] for res in complete_residues])
    O_coords = torch.tensor([res['O'] for res in complete_residues])


    # stack coordinates
    X_protein = torch.stack([N_coords, CA_coords, C_coords, O_coords], dim=1)
    print(X_protein.shape)

    data_list.append(X_protein)

torch.save(data_list, 'protein_data_list.pt')
print(len(data_list))





# proteins = RCSBDataset().to_graph(eps=8).pyg()
# print(len(proteins))
# # print(proteins[0])

# protein = proteins[0]
# print(protein)
# # print(protein)
# protein_pyg = protein[0]
# print(protein_pyg)

# r = protein[1]['residue']['residue_type']
# print(r)
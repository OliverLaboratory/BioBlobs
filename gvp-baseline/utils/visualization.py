import pandas as pd
from pathlib import Path
AA_THREE_TO_ONE = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
AA_ONE_TO_THREE = {v:k for k, v in AA_THREE_TO_ONE.items()}


def protein_to_pdb(protein, path):
    """ Write coordinate list from atom dict to a PDB file.

    Parameters
    ------------
    protein:
        protein data dictionary. must be at the 'atom' resolution.
    path:
        Path to write PDB file.


    """
    path = Path(path)
    # ATOM      1  N   PRO A   1       8.316  21.206  21.530  1.00 17.44           N
    try:
        df = pd.DataFrame(protein['atom'])
        mode = 'atom'
    except KeyError:
        df = pd.DataFrame(protein['residue'])
        mode = 'residue'

    df['residue_name_full'] = df['residue_type'].apply(lambda x : AA_ONE_TO_THREE[x])
    if 'chain_id' not in df.columns:
        df['chain_id'] = 'A'
    df['occupancy'] = [1.00] * len(df)
    df['temp'] = [20.00] * len(df)

    if mode == 'atom':
        df['element'] = df['atom_type'].apply(lambda x: x[:1])
    else:
        df['element'] = 'C'
        df['atom_number'] = df['residue_number']
        df['atom_type'] = 'CA'

    lines = []
    for row in df.itertuples():
        line = "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   " \
               "{:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          " \
               "{:>2s}{:2s}".format(
                                    'ATOM',
                                    row.atom_number,
                                    row.atom_type,
                                    ' ',
                                    row.residue_name_full,
                                    row.chain_id,
                                    row.residue_number,
                                    ' ',
                                    row.x,
                                    row.y,
                                    row.z,
                                    row.occupancy,
                                    row.temp,
                                    row.element,
                                    '  '
                                 )
        lines.append(line)
    with open(path, "w") as p:
        p.write("\n".join(lines))
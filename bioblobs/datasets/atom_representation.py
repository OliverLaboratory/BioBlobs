"""
Biopandas-based utilities for parsing PDB files and extracting backbone coordinates.

This module replaces manual atom dictionary parsing with standardized Biopandas operations.
"""

import pandas as pd
from typing import List, Tuple
from math import inf
from collections import defaultdict


def parse_pdb_file(pdb_path: str) -> pd.DataFrame:
    """Parse PDB file using Biopandas PandasPdb.
    
    Args:
        pdb_path: Path to PDB file
    
    Returns:
        DataFrame with columns: atom_number, atom_name, residue_name, 
                               chain_id, residue_number, x_coord, y_coord, z_coord, etc.
    """
    from biopandas.pdb import PandasPdb
    
    ppdb = PandasPdb().read_pdb(pdb_path)
    return ppdb.df['ATOM']  # Returns ATOM records as DataFrame


def get_sequence_from_pdb(df: pd.DataFrame) -> str:
    """Extract protein sequence from Biopandas DataFrame.
    
    Args:
        df: Biopandas ATOM DataFrame
    
    Returns:
        Protein sequence as string (single letter codes)
    """
    # Three-letter to one-letter amino acid mapping
    AA_THREE_TO_ONE = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    
    # Get unique residues in order
    residue_df = df.drop_duplicates(subset=['residue_number']).sort_values('residue_number')
    
    sequence = ''
    for _, row in residue_df.iterrows():
        residue_name = row['residue_name']
        aa = AA_THREE_TO_ONE.get(residue_name, 'X')  # Use 'X' for unknown
        sequence += aa
    
    return sequence


def extract_backbone_coords(df: pd.DataFrame, sequence: str) -> Tuple[List, List, str]:
    """Extract N, CA, C, O coordinates for all residues in order.
    
    Args:
        df: Biopandas DataFrame from parse_pdb_file()
        sequence: Protein sequence string
    
    Returns:
        coords: List of [[N, CA, C, O], ...] coordinates per residue
                Each atom coordinate is [x, y, z] or [inf, inf, inf] if missing
        resnums: List of residue numbers
        adjusted_seq: Sequence adjusted to match coord length
        
    Missing backbone atoms get [inf, inf, inf]
    """
    BACKBONE = ['N', 'CA', 'C', 'O']
    BACKBONE_SET = set(BACKBONE)
    MISSING = [inf, inf, inf]
    
    # Group by residue_number
    residues = defaultdict(dict)
    
    for _, row in df.iterrows():
        res_num = row['residue_number']
        atom_name = row['atom_name']
        
        # Only process backbone atoms (first occurrence)
        if atom_name in BACKBONE_SET and atom_name not in residues[res_num]:
            residues[res_num][atom_name] = [
                row['x_coord'],
                row['y_coord'],
                row['z_coord']
            ]
    
    # Build coords in residue order
    coords = []
    resnums = []
    
    for res_num in sorted(residues.keys()):
        atoms = residues[res_num]
        # Extract N, CA, C, O in that order (use MISSING if not present)
        residue_coords = [
            atoms.get(atom, MISSING) for atom in BACKBONE
        ]
        coords.append(residue_coords)
        resnums.append(res_num)
    
    # Align sequence length with coord length
    # Truncate if sequence is longer
    if len(sequence) > len(coords):
        adjusted_seq = sequence[:len(coords)]
    # Pad with 'X' if sequence is shorter
    elif len(sequence) < len(coords):
        pad_len = len(coords) - len(sequence)
        adjusted_seq = sequence + ('X' * pad_len)
    else:
        adjusted_seq = sequence
    
    return coords, resnums, adjusted_seq


def validate_backbone_completion(coords: List, min_completion: float = 0.95) -> Tuple[bool, float]:
    """Check if structure has sufficient complete backbone atoms.
    
    Args:
        coords: List of [[N, CA, C, O], ...] per residue
        min_completion: Minimum fraction of complete residues required
    
    Returns:
        is_valid: True if completion >= min_completion
        completion_rate: Fraction of residues with all 4 backbone atoms
    """
    MISSING = [inf, inf, inf]
    
    # Count residues with all 4 backbone atoms present
    complete_count = 0
    for residue in coords:
        if all(coord != MISSING for coord in residue):
            complete_count += 1
    
    completion_rate = complete_count / len(coords) if coords else 0.0
    is_valid = completion_rate >= min_completion
    
    return is_valid, completion_rate

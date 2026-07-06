"""
Configuration for non-standard residue parameterization.
"""

import os
from pathlib import Path

_HERE = Path(__file__).parent
PARAMS_DIR = _HERE / 'params'

# Standard residues that pdb2gmx handles natively
STANDARD_RESIDUES = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    # Protonation variants
    'HID', 'HIE', 'HIP', 'HSD', 'HSE', 'HSP', 'CYX', 'CYM',
    'ASH', 'GLH', 'LYN',
    # DNA/RNA
    'DA', 'DG', 'DC', 'DT', 'RA', 'RG', 'RC', 'RU',
    'DA5', 'DA3', 'DG5', 'DG3', 'DC5', 'DC3', 'DT5', 'DT3',
}

# Common water/ion names to skip
SOLVENT_IONS = {
    'HOH', 'WAT', 'SOL', 'TIP', 'SPC',
    'NA', 'CL', 'K', 'MG', 'CA', 'ZN', 'FE', 'MN', 'CU', 'CO', 'NI',
    'NA+', 'CL-', 'K+', 'MG2', 'CA2', 'ZN2', 'FE2', 'FE3', 'MN2', 'CU2',
    'IOD', 'BR',
}

# Metal ions and their common oxidation states / charges
METAL_INFO = {
    'FE': {'charges': [2, 3], 'default_charge': 2, 'element': 'Fe'},
    'FE2': {'charges': [2], 'default_charge': 2, 'element': 'Fe'},
    'FE3': {'charges': [3], 'default_charge': 3, 'element': 'Fe'},
    'ZN': {'charges': [2], 'default_charge': 2, 'element': 'Zn'},
    'ZN2': {'charges': [2], 'default_charge': 2, 'element': 'Zn'},
    'MN': {'charges': [2, 3], 'default_charge': 2, 'element': 'Mn'},
    'MN2': {'charges': [2], 'default_charge': 2, 'element': 'Mn'},
    'MG': {'charges': [2], 'default_charge': 2, 'element': 'Mg'},
    'MG2': {'charges': [2], 'default_charge': 2, 'element': 'Mg'},
    'CU': {'charges': [1, 2], 'default_charge': 2, 'element': 'Cu'},
    'CU2': {'charges': [2], 'default_charge': 2, 'element': 'Cu'},
    'CO': {'charges': [2, 3], 'default_charge': 2, 'element': 'Co'},
    'CA': {'charges': [2], 'default_charge': 2, 'element': 'Ca'},
    'NI': {'charges': [2], 'default_charge': 2, 'element': 'Ni'},
}

# Cofactors with pre-built parameters
KNOWN_COFACTORS = {
    'HEM': 'heme',       # Heme (generic)
    'HEME': 'heme',
    'HEC': 'heme_c',     # Heme C
    'HEA': 'heme_a',     # Heme A
    'FAD': 'fad',         # Flavin adenine dinucleotide
    'FMN': 'fmn',         # Flavin mononucleotide
    'NAD': 'nad',         # NAD+
    'NAP': 'nadp',        # NADP+
    'NAI': 'nadh',        # NADH
    'PLP': 'plp',         # Pyridoxal phosphate
    'TPP': 'tpp',         # Thiamine pyrophosphate
    'SAM': 'sam',         # S-adenosyl methionine
    'COA': 'coa',         # Coenzyme A
    'ADP': 'adp',
    'ATP': 'atp',
    'GDP': 'gdp',
    'GTP': 'gtp',
}

# Metal-coordinating residues: {residue: {atom_name: coordinating_atom}}
METAL_COORDINATORS = {
    'CYS': {'SG': 'S'},   # Cys thiolate -> Fe, Zn
    'HIS': {'NE2': 'N', 'ND1': 'N'},  # His imidazole -> Fe, Zn, Cu, Mn
    'GLU': {'OE1': 'O', 'OE2': 'O'},  # Glu carboxylate -> Zn, Mn, Mg
    'ASP': {'OD1': 'O', 'OD2': 'O'},  # Asp carboxylate -> Zn, Mn, Mg
    'MET': {'SD': 'S'},   # Met thioether -> Cu
    'TYR': {'OH': 'O'},   # Tyr hydroxyl -> Fe (in some enzymes)
}

# Default coordination distance cutoffs (Angstrom)
COORD_CUTOFFS = {
    'Fe-N': 2.3,   # Fe-N(His), Fe-N(porphyrin)
    'Fe-S': 2.5,   # Fe-S(Cys)
    'Fe-O': 2.3,   # Fe-O(Tyr, water)
    'Zn-N': 2.3,   # Zn-N(His)
    'Zn-S': 2.4,   # Zn-S(Cys)
    'Zn-O': 2.2,   # Zn-O(Asp, Glu, water)
    'Mn-N': 2.3,
    'Mn-O': 2.2,
    'Cu-N': 2.2,
    'Cu-S': 2.3,
    'Mg-O': 2.2,
    'default': 2.8,
}

# Default position restraint force constants (kJ/mol/nm^2)
POSRE_FC_METAL = 10000.0   # Very strong: metal ions
POSRE_FC_COFACTOR = 1000.0  # Standard: cofactor heavy atoms
POSRE_FC_BACKBONE = 1000.0  # Standard: protein backbone (during equil)

# Paths (reuse from gromacs_md config)
try:
    from zhuangzhou.gromacs_md.config import (
        GMX_BIN, SOBTOP_DIR, SOBTOP_EXE, OBABEL_EXE, GMXLIB,
    )
except ImportError:
    GMX_BIN = None
    SOBTOP_DIR = None
    SOBTOP_EXE = None
    OBABEL_EXE = None
    GMXLIB = None

"""
PLIP Analysis Configuration
============================
Single source of truth for PLIP module settings.
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# Docker image name
DOCKER_IMAGE = 'zhuangzhou-plip'

# Path to bundled Dockerfile
DOCKER_DIR = os.path.join(_HERE, 'docker')

# Residue classification (for HETATM auto-fix)
STANDARD_RESIDUES = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
    'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
    'TYR', 'VAL',
    # Protonation variants
    'HID', 'HIE', 'HIP', 'CYX', 'ASH', 'GLH',
    # DNA/RNA
    'DA', 'DT', 'DC', 'DG', 'A', 'U', 'C', 'G',
    # Water / ions
    'HOH', 'WAT', 'SOL', 'NA', 'CL', 'K', 'MG', 'CA', 'ZN', 'FE',
}

WATER_RESNAMES = {'HOH', 'WAT', 'SOL', 'DOD', 'TIP'}

# Cofactors — part of protein machinery, NOT docking ligands
COFACTOR_RESNAMES = {
    'HEM', 'HEC', 'HEA', 'HEB',       # Heme variants
    'FAD', 'FMN',                       # Flavin
    'NAD', 'NAP', 'NDP',               # Nicotinamide
    'ADP', 'ATP', 'AMP',               # Adenine nucleotides
    'SAM', 'COA',                       # S-adenosyl, coenzyme A
    'PLP', 'TPP', 'FES', 'SF4',        # Metal clusters
}

# Common ions and crystallographic additives — not drug ligands
COMMON_IONS = {
    'NA', 'CL', 'MG', 'ZN', 'CA', 'FE', 'MN', 'CO',
    'CU', 'K', 'NI', 'CD', 'SO4', 'PO4',
}

CRYSTALLOGRAPHIC_ADDITIVES = {
    'GOL', 'EDO', 'PEG', 'PGE', 'DMS', 'ACT', 'FMT', 'BME',
    'MPD', 'MES', 'EPE', 'TRS', 'CIT', 'TAR', 'IMD', 'IPA',
    'BOG', 'OGA', 'OLA', 'UNX', 'UNL',
}

# Everything that should NOT be treated as a docking ligand
NON_LIGAND_HETATM = WATER_RESNAMES | COFACTOR_RESNAMES | COMMON_IONS | CRYSTALLOGRAPHIC_ADDITIVES

# Timeouts (seconds)
DOCKER_TIMEOUT = 120
WSL_TIMEOUT = 120
DOCKER_BUILD_TIMEOUT = 600

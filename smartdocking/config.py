"""
SmartDocking Configuration
==========================
Single source of truth for all paths, defaults, and engine settings.

Edit THIS file to change docking behavior. All other modules read from here.

Sections:
  1. PATHS     — external binaries, weights, environments
  2. VINA      — docking engine defaults
  3. UNIMOL    — AI docking engine (requires CUDA + unicore)
  4. PREPARE   — protein/ligand preparation defaults
  5. SITE      — active site definition defaults
"""

import os
import shutil
from typing import Iterable, List

# This module's directory — all local paths are relative to here
_HERE = os.path.dirname(os.path.abspath(__file__))


def _split_paths(value: str) -> List[str]:
    """Split an os.pathsep-delimited environment variable into usable paths."""
    return [p.strip().strip('"') for p in value.split(os.pathsep) if p.strip()]


def _env_paths(name: str, fallback: Iterable[str] = ()) -> List[str]:
    configured = os.environ.get(name, "")
    return _split_paths(configured) or list(fallback)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. PATHS — external dependencies
# ═══════════════════════════════════════════════════════════════════════════════

# Vina executables (searched in order, first found is used)
VINA_EXECUTABLES = _env_paths("ZHUANGZHOU_VINA_EXECUTABLES", fallback=("vina", "vina.exe"))

# ── UniMol Docking V2 (bundled — weights + engines shipped with this module) ──
# Weights (416MB model + dictionaries)
UNIMOL_WEIGHTS_DIR = os.path.join(_HERE, 'weights')
UNIMOL_WEIGHTS     = os.path.join(UNIMOL_WEIGHTS_DIR, 'binding_pose_220908.pt')
UNIMOL_DICT_MOL    = os.path.join(UNIMOL_WEIGHTS_DIR, 'mol.dict.txt')
UNIMOL_DICT_POC    = os.path.join(UNIMOL_WEIGHTS_DIR, 'poc.dict.txt')
UNIMOL_DICT_MOL_INF = os.path.join(UNIMOL_WEIGHTS_DIR, 'dict_mol.txt')  # inference format
UNIMOL_DICT_PKT_INF = os.path.join(UNIMOL_WEIGHTS_DIR, 'dict_pkt.txt')  # inference format

# Engines (bundled source code)
UNIMOL_ENGINES_DIR = os.path.join(_HERE, 'engines')
UNIMOL_DOCKING_V2  = os.path.join(UNIMOL_ENGINES_DIR, 'unimol_docking_v2')
UNIMOL_UNICORE     = os.path.join(UNIMOL_ENGINES_DIR, 'unicore')
UNIMOL_INFER_SCRIPT = os.path.join(UNIMOL_DOCKING_V2, 'unimol', 'infer.py')
UNIMOL_PREDICTOR   = os.path.join(UNIMOL_DOCKING_V2, 'predictor')

# Conda environments with CUDA (for UniMol — only needs torch+CUDA, unicore is bundled)
CUDA_ENVS = {}
for item in os.environ.get("ZHUANGZHOU_CUDA_PYTHONS", "").split(";"):
    if "=" in item:
        name, python_exe = item.split("=", 1)
        if name.strip() and python_exe.strip():
            CUDA_ENVS[name.strip()] = python_exe.strip().strip('"')

# PyMOL (optional, for visualization)
PYMOL_EXE = os.environ.get("ZHUANGZHOU_PYMOL_EXE", "pymol")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. VINA — docking engine defaults
# ═══════════════════════════════════════════════════════════════════════════════

VINA_DEFAULTS = {
    'exhaustiveness': 32,       # search thoroughness (higher = slower + better)
    'num_modes': 9,             # max number of poses to return
    'energy_range': 3,          # max energy difference from best (kcal/mol)
    'cpu': 4,                   # CPU threads (CPU mode)
    'timeout': 600,             # max seconds
    # GPU+ specific
    'gpu_thread': 8000,         # GPU thread count
    'gpu_search_depth': None,   # None = auto
}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. UNIMOL — AI docking engine defaults (when available)
# ═══════════════════════════════════════════════════════════════════════════════

UNIMOL_DEFAULTS = {
    'conf_size': 10,            # number of conformers to generate
    'nthreads': 4,              # preprocessing threads
    'batch_size': 4,            # inference batch size
    'recycling': 4,             # model recycling iterations
    'dist_threshold': 8.0,      # distance threshold (A)
    'steric_clash_fix': False,  # post-process clash fixing
}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PREPARE — protein/ligand preparation
# ═══════════════════════════════════════════════════════════════════════════════

PREPARE_DEFAULTS = {
    'remove_water': True,
    'remove_ions': True,          # remove crystallographic ions/salts for docking
    'keep_cofactors': True,
}

# Crystallographic additives — always remove for docking
# (buffer components, cryoprotectants, detergents, etc.)
CRYSTALLOGRAPHIC_ADDITIVES = {
    'GOL', 'EDO', 'PEG', 'PGE', 'DMS', 'ACT', 'FMT', 'BME',
    'MPD', 'MES', 'EPE', 'TRS', 'CIT', 'TAR', 'IMD', 'IPA',
    'BOG', 'OGA', 'OLA', 'UNX', 'UNL',
}

# Residue classification
STANDARD_RESIDUES = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
    'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
    'TYR', 'VAL',
    # Protonation variants
    'HID', 'HIE', 'HIP', 'CYX', 'ASH', 'GLH',
    # Nucleic acids
    'DA', 'DT', 'DC', 'DG', 'A', 'U', 'C', 'G',
}

WATER_RESNAMES = {'HOH', 'WAT', 'DOD', 'TIP', 'SOL'}

COMMON_IONS = {
    'NA', 'CL', 'MG', 'ZN', 'CA', 'FE', 'MN', 'CO',
    'CU', 'K', 'NI', 'CD', 'SO4', 'PO4',
}

COFACTOR_RESNAMES = {
    'HEM', 'HEC', 'HEA', 'HEB',       # Heme variants
    'FAD', 'FMN',                       # Flavin
    'NAD', 'NAP',                       # Nicotinamide
    'ADP', 'ATP',                       # Adenine nucleotides
    'SAM', 'COA',                       # Others
    'PLP', 'TPP', 'FES', 'SF4',        # Metal clusters
}

# Vina atom type mapping for simple PDB→PDBQT
VINA_ATOM_TYPES = {
    'C': 'C', 'N': 'NA', 'O': 'OA', 'S': 'SA', 'H': 'HD',
    'F': 'F', 'P': 'P', 'CL': 'Cl', 'BR': 'Br', 'I': 'I',
    'FE': 'Fe', 'ZN': 'Zn', 'MG': 'Mg', 'CA': 'Ca', 'MN': 'Mn',
}


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SITE — active site definition
# ═══════════════════════════════════════════════════════════════════════════════

SITE_DEFAULTS = {
    'padding': 10.0,            # extra space around ligand (A)
    'min_box_size': 15.0,       # minimum box dimension (A)
    'max_box_size': 60.0,       # Vina hard limit per dimension
    'heme_fe_offset': 5.5,      # distance above Fe for HEME-aware site (A)
    'fallback_box_size': 30.0,  # box size when using protein center fallback
}


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: check what's available at runtime
# ═══════════════════════════════════════════════════════════════════════════════

def check_availability() -> dict:
    """Check which components are available."""
    status = {}

    # Vina
    for p in VINA_EXECUTABLES:
        resolved = shutil.which(p) or p
        if os.path.exists(resolved):
            is_gpu = 'GPU' in os.path.basename(p).upper()
            status['vina'] = {'path': resolved, 'gpu': is_gpu}
            break
    else:
        status['vina'] = None

    # UniMol (bundled)
    status['unimol_weights'] = os.path.exists(UNIMOL_WEIGHTS)
    status['unimol_infer'] = os.path.exists(UNIMOL_INFER_SCRIPT)
    status['unimol_unicore'] = os.path.isdir(UNIMOL_UNICORE)
    status['unimol_predictor'] = os.path.isdir(UNIMOL_PREDICTOR)

    # CUDA envs
    for name, python_exe in CUDA_ENVS.items():
        status[f'cuda_env_{name}'] = os.path.exists(python_exe)

    # Python packages (current interpreter)
    for pkg in ('rdkit', 'meeko', 'numpy'):
        try:
            __import__(pkg)
            status[pkg] = True
        except ImportError:
            status[pkg] = False

    # unicore — check bundled or installed
    try:
        import unicore
        status['unicore_installed'] = True
    except ImportError:
        status['unicore_installed'] = False

    # PyMOL
    status['pymol'] = bool(shutil.which(PYMOL_EXE) or os.path.exists(PYMOL_EXE))

    return status

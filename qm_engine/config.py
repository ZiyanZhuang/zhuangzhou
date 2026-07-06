"""
QM Engine configuration.

PySCF + gpu4pyscf run in WSL via subprocess bridge.
"""

import os
import shutil

# WSL Python (system Python 3.12 with PySCF installed)
WSL_PYTHON = 'python3'

# Default DFT settings for enzyme active sites
DEFAULT_FUNCTIONAL = 'b3lyp'
DEFAULT_BASIS_OPT = 'def2-svp'        # Geometry optimization
DEFAULT_BASIS_SP = 'def2-tzvp'        # Single-point energy
DEFAULT_DISPERSION = 'd3bj'           # Grimme D3(BJ) dispersion

# Grid quality for DFT (enzyme systems often have metals)
DEFAULT_GRID = (99, 590)  # (radial, angular) points

# Implicit solvation for enzyme environment (Himo: epsilon=4)
DEFAULT_SOLVENT_EPS = 4.0  # Dielectric constant for enzyme interior
USE_SOLVENT = True         # Enable CPCM solvation by default

# Memory limit (MB) - leave room for OS
DEFAULT_MAX_MEMORY = 6000

# QM cluster extraction defaults
DEFAULT_CUTOFF = 5.0     # Angstrom radius from reactive center
DEFAULT_CAP = 'H'        # Capping atom for truncated bonds

# Convergence
DEFAULT_EMTOL = 1e-6     # SCF energy convergence
DEFAULT_GTOL = 4.5e-4    # Gradient convergence (Hartree/Bohr)

# Timeout (seconds) - TS search can be long
TIMEOUT_OPT = 3600       # 1 hour for optimization
TIMEOUT_TS = 7200         # 2 hours for TS search
TIMEOUT_FREQ = 3600       # 1 hour for frequency
TIMEOUT_SP = 1800         # 30 min for single point

# GPU settings
TRY_GPU = True  # Try gpu4pyscf first, fallback to CPU


def has_wsl():
    """Check if WSL is available."""
    return shutil.which('wsl') is not None

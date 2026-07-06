"""
Non-Standard Residue Parameterization for GROMACS MD
=====================================================
Handles HEME, metal cofactors (Fe, Zn, Mn, Cu), and other non-standard
residues that standard pdb2gmx cannot process.

Strategy (escalating complexity):
  1. Pre-built parameter database (HEME, FAD, NAD, common cofactors)
  2. SOBTOP/GAFF automatic parameterization (organic cofactors)
  3. QM-derived parameters via PySCF (metal centers, unusual bonding)

Public API:
  detect_nonstandard()    - Scan PDB for non-standard residues and metals
  parameterize_residue()  - Generate .itp/.gro for a single residue
  build_topology()        - Full system: protein + all non-standard residues
  generate_restraints()   - Position restraints for metals and key atoms
"""

from .detect import detect_nonstandard
from .parameterize import parameterize_residue, parameterize_system
from .restraints import generate_restraints, generate_metal_restraints
from .topology import build_topology, merge_itp_into_topol
from .pdb_prep import prepare_metalloprotein_pdb
from .metal_models import recommend_metal_model, get_itp_for_nonbonded_metal

__all__ = [
    'detect_nonstandard',
    'parameterize_residue',
    'parameterize_system',
    'build_topology',
    'merge_itp_into_topol',
    'generate_restraints',
    'generate_metal_restraints',
    'prepare_metalloprotein_pdb',
    'recommend_metal_model',
    'get_itp_for_nonbonded_metal',
]

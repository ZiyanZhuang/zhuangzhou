"""
AMBER MD Skills — CYP450+Heme and General Protein-Ligand Molecular Dynamics
=============================================================================
Self-contained modules for AMBER-based MD simulations.

Public API:
  run_cyp_heme_md()         — CYP450+heme pipeline (PDB -> tleap -> MD inputs)
  run_cyp_heme_analysis()   — CYP-heme trajectory analysis scripts
  run_ligand_prep()         — Ligand parameterization (RESP + GAFF2)
  run_ligand_analysis()     — Comprehensive 4-module trajectory analysis
  run_protein_ligand_md()   — General protein+ligand pipeline (Vina dock → ff14SB/GAFF2 → MD)
  run_apo_protein_md()      — Apo-protein pipeline (no ligand)
  check_dependencies()      — Verify AMBER tools are found
  check_general_dependencies() — Verify general pipeline tools (WSL-aware)

Protocol engine (for advanced use):
  AmberMDProtocol           — Dataclass with all MD hyperparameters
  default_protocol()        — Factory for system-type-specific defaults
  setup_md()                — Generate MD inputs + run script from protocol
"""

from .pipeline import run_cyp_heme_md
from .analysis import run_cyp_heme_analysis
from .ligand_prep import run_ligand_prep
from .ligand_analysis import run_ligand_analysis
from .general_pipeline import run_protein_ligand_md, run_apo_protein_md
from .config import check_dependencies, check_general_dependencies
from .protocol import AmberMDProtocol, default_protocol, setup_md

__all__ = [
    "run_cyp_heme_md",
    "run_cyp_heme_analysis",
    "run_ligand_prep",
    "run_ligand_analysis",
    "run_protein_ligand_md",
    "run_apo_protein_md",
    "check_dependencies",
    "check_general_dependencies",
    "AmberMDProtocol",
    "default_protocol",
    "setup_md",
]

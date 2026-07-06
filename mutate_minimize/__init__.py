"""
Mutate & Minimize — Structure optimization after point mutations
================================================================
Combines mutagenesis + GROMACS energy minimization in vacuum or solvent.

Public API:
  mutate_and_minimize()  — Mutate PDB + EM optimization → clean PDB
  minimize_structure()   — EM-only optimization on existing PDB
"""

from .engine import mutate_and_minimize, minimize_structure

__all__ = ['mutate_and_minimize', 'minimize_structure']

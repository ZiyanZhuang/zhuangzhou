"""
LigBuilder V3 Integration — De Novo Ligand Design
===================================================
Structure-based de novo drug design using LigBuilder V3 (Peking University).

Backends: WSL (primary) > Docker (portable)

Pipeline:
  1. Cavity detection (pocket definition from coords or auto-detect)
  2. De novo molecular design (Exploring/Growing/Linking modes)
  3. Post-processing (filter, cluster, convert to SDF)

Key: Uses SmartDocking pocket definition to drive LigBuilder area mode.
"""

from .pipeline import run_ligbuilder, run_ligbuilder_cavity

__all__ = ['run_ligbuilder', 'run_ligbuilder_cavity']

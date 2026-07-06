"""
GROMACS MD Skill — Protein-Ligand Molecular Dynamics
=====================================================
Self-contained, portable module for GROMACS 2025.x + GPU.

Setup:
  1. Install GROMACS, SOBTOP, OpenBabel
  2. Copy gromacs_md.ini.example to ~/.zhuangzhou/gromacs_md.ini
  3. Edit paths (or set environment variables: GMX_BIN, SOBTOP_DIR, etc.)

Public API:
  run_protein_ligand_md()  — Full pipeline: split → param → solvate → EM → NVT → NPT → MD
  run_protein_md()         — Pure protein pipeline
  run_full_analysis()      — RMSD, RMSF, Rg, H-bonds, FEL, distances, interaction energy
  render_trajectory()      — MP4/GIF trajectory animation (PyMOL + ffmpeg)
  render_porcupine()       — PCA porcupine plot (PyMOL CGO arrows)
  check_dependencies()     — Verify all tools are found
"""

from .pipeline import (
    run_protein_ligand_md, run_protein_md, run_metalloprotein_md,
    run_metalloprotein_ligand_md,
)
from .analysis import run_full_analysis, compute_coordination, plot_coordination
from .diagnostics import diagnose_failure, check_stage_sanity, format_diagnosis_report
from .visualize import render_trajectory, render_porcupine, process_trajectory
from .config import check_dependencies

__all__ = [
    'run_protein_ligand_md',
    'run_protein_md',
    'run_metalloprotein_md',
    'run_metalloprotein_ligand_md',
    'run_full_analysis',
    'compute_coordination',
    'plot_coordination',
    'diagnose_failure',
    'check_stage_sanity',
    'format_diagnosis_report',
    'render_trajectory',
    'render_porcupine',
    'process_trajectory',
    'check_dependencies',
]

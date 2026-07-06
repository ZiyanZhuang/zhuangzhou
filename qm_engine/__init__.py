"""
QM Engine — DFT / Transition State Calculations for Enzyme Catalysis
=====================================================================
PySCF + gpu4pyscf (GPU) + geomeTRIC (TS search) via WSL.

Public API:
  extract_qm_cluster()    — Extract active site QM cluster from PDB (PLIP-guided)
  run_dft_optimization()  — Geometry optimization (DFT, constrained)
  run_ts_search()         — Transition state search (geomeTRIC RS-P-RFO)
  run_frequency()         — Frequency analysis (verify TS / get thermodynamics)
  run_single_point()      — High-basis single-point energy
  run_ts_pipeline()       — Full pipeline: extract → opt → TS → freq → SP → report
"""

from .pipeline import (
    extract_qm_cluster,
    run_dft_optimization,
    run_ts_search,
    run_frequency,
    run_single_point,
    run_ts_pipeline,
)
from .analysis import (
    plot_energy_profile,
    generate_report,
    analyze_reaction,
)

__all__ = [
    'extract_qm_cluster',
    'run_dft_optimization',
    'run_ts_search',
    'run_frequency',
    'run_single_point',
    'run_ts_pipeline',
    'plot_energy_profile',
    'generate_report',
    'analyze_reaction',
]

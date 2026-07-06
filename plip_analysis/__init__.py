"""
PLIP Analysis Module
====================
Self-contained protein-ligand interaction profiler.

Backends (tried in order):
  1. Native — plip_analysis module (if importable)
  2. Docker — containerized PLIP (portable, no WSL needed)
  3. WSL    — PLIP installed in Windows Subsystem for Linux

Usage:
    from zhuangzhou.plip_analysis import run_plip_analysis

    result = run_plip_analysis('complex.pdb')
    # result['interactions'], result['report_path'], result['backend']

Docker build (one-time):
    docker build -t zhuangzhou-plip zhuangzhou/plip_analysis/docker/
"""

from .analysis import run_plip_analysis

__all__ = ['run_plip_analysis']

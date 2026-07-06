"""
Zhuangzhou SmartDocking
=======================
Self-contained docking pipeline:

  PDB → preprocess → detect cofactors → define active site → Vina docking → report

All code is local — no external script imports required.
"""

from .pipeline import SmartDocking

__all__ = ['SmartDocking']

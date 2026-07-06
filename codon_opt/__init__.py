"""
Codon Optimization Module for Komagataella phaffii (Pichia pastoris)
====================================================================

Backends:
  1. DNAchisel + K. phaffii codon table (primary, classical CAI optimization)
  2. Pichia-CLM deep learning model (optional, when weights available)

Usage:
    from zhuangzhou.codon_opt import optimize_codons, analyze_codons, batch_optimize
"""

from .optimizer import optimize_codons, batch_optimize
from .analysis import analyze_codons, compare_sequences

__all__ = [
    'optimize_codons',
    'batch_optimize',
    'analyze_codons',
    'compare_sequences',
]

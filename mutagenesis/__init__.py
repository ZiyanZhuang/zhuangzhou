"""
Mutagenesis Module — Self-contained protein point mutation toolkit.

Usage:
    from zhuangzhou.mutagenesis import mutate

    # Single mutation
    result = mutate("protein.pdb", "V9L")

    # Multiple mutations
    result = mutate("protein.pdb", ["V9L", "A5G", "K3R"])

    # With options
    result = mutate("protein.pdb", "V9L", chain="A", output="mutant.pdb")
"""

from .mutator import mutate, mutate_single, mutate_multi, parse_mutation

__all__ = ["mutate", "mutate_single", "mutate_multi", "parse_mutation"]

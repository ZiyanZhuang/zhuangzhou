"""
Zhuangzhou MetaKG Module — v0.8.0
==================================
Metabolic Knowledge Graph integration for E. coli (iML1515) and
S. cerevisiae (iMM904) genome-scale models.

Quick start:
    from zhuangzhou.metakg import get_bridge, FluxAnalyzer
    bridge = get_bridge("ecoli")
    paths = bridge.find_path("pep_c", "phe__L_c", max_steps=5)
    analyzer = FluxAnalyzer("ecoli")
    result = analyzer.run_fba("EX_phe__L_e")
"""

from .bridge import MetaKGBridge, get_bridge, clear_cache
from .flux import FluxAnalyzer

__all__ = ["MetaKGBridge", "get_bridge", "clear_cache", "FluxAnalyzer"]

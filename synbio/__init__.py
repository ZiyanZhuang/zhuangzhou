"""
Zhuangzhou SynBio Module — v0.8.0
===================================
Synthetic biology design tools: parts, CRISPR, expression balancing.

Quick start:
    from zhuangzhou.synbio.parts import PartSelector
    from zhuangzhou.synbio.crispr import find_guides, design_crispri

    recs = PartSelector.balance_pathway_expression(
        ["ppsA", "aroG", "tyrB"], organism="ecoli"
    )
    guides = find_guides(gene_seq, cas_system="SpCas9")
"""

from .parts import PartSelector, ECOLI_PROMOTERS, YEAST_PROMOTERS, ECOLI_RBS
from .crispr import find_guides, design_hdr_template, design_crispri

__all__ = [
    "PartSelector", "ECOLI_PROMOTERS", "YEAST_PROMOTERS", "ECOLI_RBS",
    "find_guides", "design_hdr_template", "design_crispri",
]

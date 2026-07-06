"""
Mutation Enricher
=================
Annotates mutation positions with domain context, functional site labels,
literature/natural-variant evidence, and structural context.

Modules:
  domain_annotator    -- InterPro + UniProt REST domain/feature annotations
  literature_validator -- UniProt natural variants + PubMed evidence lookup
  dssp_annotator      -- DSSP per-residue secondary structure + RSA annotation
  structural_context  -- Zone, channel, active site distance enrichment
"""

from .enricher import enrich_mutations
from .dssp_annotator import annotate_dssp, get_ss_label
from .structural_context import (
    load_ca_coords,
    compute_active_site_distances,
    classify_structural_zone,
    identify_channel_residues,
    enrich_structural_context,
)
from .domain_annotator import annotate_domains, get_position_annotation
from .literature_validator import (
    get_uniprot_variants,
    search_pubmed_mutations,
    synthesize_evidence,
    validate_mutations,
)

__all__ = [
    # Main entry point
    'enrich_mutations',
    # dssp_annotator
    'annotate_dssp',
    'get_ss_label',
    # structural_context
    'load_ca_coords',
    'compute_active_site_distances',
    'classify_structural_zone',
    'identify_channel_residues',
    'enrich_structural_context',
    # domain_annotator
    'annotate_domains',
    'get_position_annotation',
    # literature_validator
    'get_uniprot_variants',
    'search_pubmed_mutations',
    'synthesize_evidence',
    'validate_mutations',
]

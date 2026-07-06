"""
Mutation Enricher — Main Entry Point
=====================================
Orchestrates all enrichment sub-modules to annotate a mutation list with:
  - Secondary structure (DSSP: α-helix / β-strand / loop / turn)
  - Real solvent accessibility (RSA, Tien et al. 2013)
  - Structural zone (buried_hydrophobic / active_site / channel / surface / …)
  - Distance to active site (Å, Cα–Cα)
  - Domain annotation (InterPro / Pfam via UniProt API)
  - Functional site labels (active site, binding, metal binding, …)
  - Literature evidence (UniProt natural variants + PubMed, optional)

Usage
-----
from zhuangzhou.mutation_enricher import enrich_mutations

enriched = enrich_mutations(
    mutations   = top200,            # list of dicts with 'position' key
    pdb_path    = 'WT.pdb',
    uniprot_id  = 'Q8ZKF6',
    active_site = {8, 49, 51, 57, 71, 96, 138},
    catalytic   = {49, 57, 71, 138},
    protein_name= 'Nicotinamidase (PncA)',
    pfam_id     = 'PF01872',
)
# Each mutation gains: ss_type, rsa, zone, dist_active_site, in_channel,
#   structural_annotation, domain, functional_sites, site_label,
#   has_natural_variant, evidence_level, evidence_summary
"""

import logging
import os

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = os.path.join(
    os.path.expanduser('~'), '.zhuangzhou', 'structure_cache'
)


def enrich_mutations(
    mutations: list,
    pdb_path: str,
    uniprot_id: str = None,
    active_site: set = None,
    catalytic: set = None,
    protein_name: str = '',
    pfam_id: str = None,
    llm_client=None,
    cache_dir: str = None,
    run_literature: bool = True,
    max_pubmed_queries: int = 3,
    sequence: str = '',
    fasta_header: str = '',
) -> list:
    """Enrich a mutation list with structural and literature annotations.

    Args:
        mutations:           List of mutation dicts (from ESM-2 scan / pipeline).
                             Each dict must have at least 'position' (int).
        pdb_path:            Path to the protein PDB structure.
        uniprot_id:          UniProt accession for API queries (optional but
                             strongly recommended for domain + literature data).
        active_site:         Set of 1-indexed residue numbers forming the active site.
        catalytic:           Subset of active_site with direct catalytic roles.
        protein_name:        Human-readable protein name (used for PubMed queries).
        pfam_id:             Pfam domain ID for broader family literature search.
        llm_client:          Optional ZhuangzhouAgent for evidence synthesis narrative.
        cache_dir:           Cache directory for API results
                             (default: ~/.zhuangzhou/structure_cache/).
        run_literature:      Run UniProt natural-variant + PubMed search (slower,
                             requires network). Set False for offline/fast mode.
        max_pubmed_queries:  Max unique positions to query PubMed (rate-limit guard).

    Returns:
        Enriched copy of the mutations list. Original dicts are not mutated.
        Added fields per mutation:
          ss_type              str   DSSP secondary structure label
                                     ('α-helix', 'β-strand', 'coil/loop', …)
          rsa                  float Relative solvent accessibility [0–1]
          buried               bool  RSA < 0.25
          zone                 str   Structural zone classification
                                     ('catalytic', 'active_site', 'active_site_shell',
                                      'buried_hydrophobic', 'buried_polar', 'surface',
                                      'intermediate')
          dist_active_site     float Minimum Cα–Cα distance to any active site res (Å)
          in_channel           bool  Likely substrate channel residue
          structural_annotation str  One-sentence human-readable annotation
          domain               str   Pfam domain name (e.g. 'NicA (PF01872)')
          functional_sites     list  UniProt functional labels at this position
          site_label           str   Priority functional label ('Active site residue', …)
          has_natural_variant  bool  UniProt natural variant at same position [if lit]
          natural_variants     list  Matching variant records               [if lit]
          n_papers             int   PubMed paper count                     [if lit]
          evidence_level       str   'strong'/'moderate'/'weak'/'none'      [if lit]
          evidence_summary     str   1–2 sentence evidence narrative        [if lit]
    """
    if not mutations:
        return mutations

    cache_dir = cache_dir or _DEFAULT_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    active_site = active_site or set()
    catalytic = catalytic or set()

    # Work on copies so caller's dicts are not mutated.
    # Normalize: accept both plain strings ("E45V") and dicts.
    import re as _re
    _MUT_RE = _re.compile(r'^([A-Za-z*])(\d+)([A-Za-z*-])$')

    def _to_dict(m):
        if isinstance(m, dict):
            return m.copy()
        s = str(m).strip()
        hit = _MUT_RE.match(s)
        if hit:
            return {'mutation': s,
                    'position': int(hit.group(2)),
                    'wt': hit.group(1).upper(),
                    'mut': hit.group(3).upper()}
        return {'mutation': s}

    enriched = [_to_dict(m) for m in mutations]

    # ── Step 1: DSSP annotation ────────────────────────────────────────────────
    dssp_data = {}
    if pdb_path and os.path.isfile(pdb_path):
        try:
            from .dssp_annotator import annotate_dssp
            dssp_data = annotate_dssp(pdb_path, cache_dir=cache_dir)
            logger.info('DSSP: annotated %d residues', len(dssp_data))
        except Exception as exc:
            logger.warning('DSSP annotation failed: %s', exc)

    # ── Step 2: Cα coordinates ─────────────────────────────────────────────────
    ca_coords = {}
    if pdb_path and os.path.isfile(pdb_path):
        try:
            from .structural_context import load_ca_coords
            ca_coords = load_ca_coords(pdb_path)
            logger.info('CA coords: loaded %d residues', len(ca_coords))
        except Exception as exc:
            logger.warning('CA coord loading failed: %s', exc)

    # ── Step 3: Channel residues ───────────────────────────────────────────────
    channel_residues: set = set()
    if ca_coords and dssp_data and active_site:
        try:
            from .structural_context import identify_channel_residues
            channel_residues = identify_channel_residues(
                ca_coords, dssp_data, active_site
            )
            logger.info('Channel residues: %d identified', len(channel_residues))
        except Exception as exc:
            logger.warning('Channel detection failed: %s', exc)

    # ── Step 4: Domain + functional-site annotation (UniProt / InterPro) ───────
    domain_annotation: dict = {
        'uniprot_features': {}, 'domains': [], 'domain_map': {}, 'error': None
    }
    if uniprot_id:
        try:
            from .domain_annotator import annotate_domains
            seq_len = max((m.get('position', 0) for m in mutations), default=0)
            domain_annotation = annotate_domains(
                uniprot_id, seq_len=seq_len, cache_dir=cache_dir
            )
            n_domains = len(domain_annotation.get('domains', []))
            n_sites   = len(domain_annotation.get('uniprot_features', {}))
            logger.info('Domains: %d Pfam entries, %d annotated sites', n_domains, n_sites)
        except Exception as exc:
            logger.warning('Domain annotation failed: %s', exc)

    # Fallback: sequence-based annotation when UniProt ID is missing or returned empty
    if not domain_annotation.get('domains'):
        _seq = sequence
        _header = fasta_header or protein_name
        if _seq:
            try:
                from .domain_annotator import annotate_domains_by_sequence
                seq_annotation = annotate_domains_by_sequence(
                    _seq, protein_name=_header, fasta_header=_header,
                    cache_dir=cache_dir,
                )
                if seq_annotation.get('domains'):
                    domain_annotation = seq_annotation
                    n_domains = len(domain_annotation.get('domains', []))
                    logger.info('Sequence-based domains: %d entries (fusion=%s)',
                                n_domains, seq_annotation.get('fusion_info', {}).get('is_fusion', False))
            except Exception as exc:
                logger.warning('Sequence-based domain annotation failed: %s', exc)

    # ── Step 5: Per-mutation structural + domain enrichment ────────────────────
    try:
        from .structural_context import enrich_structural_context
        from .domain_annotator import get_position_annotation
        for m in enriched:
            pos = m.get('position')
            if pos is None:
                continue
            # Structural context (SS, RSA, zone, distance, channel)
            sc = enrich_structural_context(
                m, ca_coords, dssp_data, active_site,
                catalytic=catalytic, channel_residues=channel_residues,
            )
            m.update(sc)
            # Domain / functional site labels
            pa = get_position_annotation(pos, domain_annotation)
            m['domain']          = pa.get('domain', 'Unknown')
            m['functional_sites'] = pa.get('functional_sites', [])
            m['site_label']      = pa.get('site_label', '')
    except Exception as exc:
        logger.warning('Per-mutation enrichment failed: %s', exc)

    # ── Step 6: Literature validation (optional, requires network) ────────────
    if run_literature and uniprot_id:
        try:
            from .literature_validator import validate_mutations
            enriched = validate_mutations(
                enriched,
                uniprot_id=uniprot_id,
                protein_name=protein_name,
                pfam_id=pfam_id,
                llm_client=llm_client,
                cache_dir=cache_dir,
                max_pubmed_queries=max_pubmed_queries,
            )
            logger.info('Literature validation complete')
        except Exception as exc:
            logger.warning('Literature validation failed: %s', exc)

    return enriched

"""
Zhuangzhou Agent Interface 鈥?LLM agent runtime-native toolkit
=========================================================

Design principle:
  LLM agent runtime IS the agent. Zhuangzhou IS the toolkit.
  Zhuangzhou does NOT try to do its own NLP/intent/routing.
  Instead it exposes clean, robust Python functions that LLM agent runtime
  (or any LLM agent) can call directly via Bash/python.

Usage from LLM agent runtime:
  python -c "
  from zhuangzhou.agent import analyze_mutation
  r = analyze_mutation('V21L', enzyme_class='p450', role='active_site')
  print(r)
  "

All functions:
  analyze_mutation(mutation, ...)    - Chemical intuition analysis
  search_knowledge(query, ...)      - GraphRAG + BeTaThink search
  run_pipeline(pdb_path, ...)       - 5-phase enzyme engineering
  check_environments()              - Sandbox status
  list_skills(category=None)        - Available tools
  execute_skill(name, **params)     - Run any registered skill
  graphrag_query(query, ...)        - Direct KG query
  get_protein_profile(accession)    - One-shot comprehensive protein info
  get_protein_sequence(accession)   - FASTA sequence (UniProt/NCBI)
  search_protein_database(query)    - Search UniProt/NCBI/PDB
"""

import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

BETATHINK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
# Lazy singletons 鈥?only instantiated when first used
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?

_config = {}
_skills_cache = None
_brain_cache = None
_graphrag_cache = None


def _get_skills():
    global _skills_cache
    if _skills_cache is None:
        from zhuangzhou.skills import get_registry
        _skills_cache = get_registry(_config)
    return _skills_cache


def _get_brain():
    global _brain_cache
    if _brain_cache is None:
        from zhuangzhou.brain import BrainConnector
        _brain_cache = BrainConnector(_config)
    return _brain_cache


def _get_graphrag():
    global _graphrag_cache
    if _graphrag_cache is None:
        from zhuangzhou.brain.graphrag import GraphRAGBridge
        _graphrag_cache = GraphRAGBridge()
    return _graphrag_cache


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
# Public API 鈥?clean functions LLM agent runtime can call directly
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?

def analyze_mutation(
    mutation: str,
    enzyme_class: str = '',
    role: str = 'unknown',
    is_conserved: bool = False,
) -> dict:
    """
    Analyze a single amino acid mutation using chemical intuition.

    Args:
        mutation: e.g. 'V21L', 'A123G'
        enzyme_class: e.g. 'p450', 'oxidoreductase', 'hydrolase'
        role: 'active_site', 'second_shell', 'surface', 'buried_core', 'unknown'
        is_conserved: whether position is in a conserved motif

    Returns:
        dict with chemical_score, recommendation, confidence, warnings,
        opportunities, reasoning, graphrag_evidence
    """
    if not mutation or len(mutation) < 3:
        return {'error': f'Invalid mutation "{mutation}". Format: V21L (single letter + position + single letter)'}

    if not re.match(r'^[A-Z]\d+[A-Z]$', mutation):
        return {'error': f'Invalid mutation format "{mutation}". Expected e.g. V21L, A123G'}

    wt = mutation[0]
    mut = mutation[-1]

    if wt.upper() == mut.upper():
        return {'error': f'Invalid mutation {mutation}: wild-type and mutant residues are identical'}

    from zhuangzhou.chemical_intuition import analyze_mutation_chemistry
    context = {
        'role': role,
        'in_conserved_motif': is_conserved,
        'distance_to_site': 5.0 if role == 'active_site' else 15.0,
    }

    analysis = analyze_mutation_chemistry(wt, mut, context)
    analysis['mutation'] = mutation

    # Enrich with GraphRAG evidence
    try:
        analysis['graphrag_evidence'] = _get_graphrag().get_mutation_evidence(
            mutation, enzyme_class)
    except Exception:
        analysis['graphrag_evidence'] = {}

    return analysis


def search_knowledge(
    query: str,
    domain: str = 'all',
    limit: int = 10,
) -> dict:
    """
    Search across BeTaThink KG and AutoGraphRAG domain knowledge graphs.

    Args:
        query: search text
        domain: 'all', 'Cytochrome_P450', 'Computational_Chemistry',
                'Bioinformatics', 'Evolution', 'Synthetic_Biology',
                'Systems_Biology', 'Structural_Biology'
        limit: max results per source

    Returns:
        dict with brain_entities, graphrag_entities, graphrag_community
    """
    results = {'query': query, 'domain': domain}

    try:
        results['brain_entities'] = _get_brain().search(query, limit)
    except Exception:
        results['brain_entities'] = []

    try:
        keywords = query.split()[:5]
        results['graphrag_entities'] = _get_graphrag().search_entities(
            keywords, kg_id=domain, limit=limit)
        results['graphrag_community'] = _get_graphrag().get_community_context(
            query, kg_id=domain, limit=3)
    except Exception:
        results['graphrag_entities'] = {}
        results['graphrag_community'] = {}

    return results


def run_pipeline(
    pdb_path: str,
    binding_site: list = None,
    esm2_dirs: list = None,
    enzyme_class: str = '',
    protein_name: str = '',
    conserved_positions: list = None,
    output_dir: str = None,
) -> dict:
    """
    Run the 5-phase Zhuangzhou enzyme engineering pipeline.

    Args:
        pdb_path: path to .pdb file (REQUIRED)
        binding_site: list of active site residue numbers (e.g. [87, 89, 105])
                      If not provided, will attempt auto-detection via fpocket
        esm2_dirs: list of ESM-2 result directories
                   If not provided, will skip PLM scoring
        enzyme_class: e.g. 'p450', 'oxidoreductase'
        protein_name: human-readable name
        conserved_positions: known conserved positions
        output_dir: where to save results

    Returns:
        dict with status, report_path, top mutations, or error with guidance
    """
    # Validate PDB
    if not pdb_path:
        return {
            'error': 'pdb_path is required',
            'hint': 'Provide the path to a .pdb file, e.g.: run_pipeline("protein.pdb", binding_site=[87,89,105])',
        }
    if not os.path.exists(pdb_path):
        return {
            'error': f'PDB file not found: {pdb_path}',
            'hint': 'Check the file path. Use an absolute path or relative to cwd.',
        }

    # Auto-detect binding site if not provided
    if not binding_site:
        return {
            'error': 'binding_site not provided',
            'hint': (
                'Provide active site residue numbers as a list, e.g. binding_site=[87, 89, 105, 107].\n'
                'You can find these by:\n'
                '  1. Reading the literature for your enzyme\n'
                '  2. Running fpocket: execute_skill("fpocket_detection", pdb_path="...")\n'
                '  3. Checking UniProt active site annotations\n'
                '  4. Using PyMOL to inspect the structure'
            ),
        }

    # ESM-2 is optional 鈥?pipeline can run without PLM scores
    if not esm2_dirs:
        esm2_dirs = []
        logger.info("No ESM-2 dirs provided 鈥?running pipeline without PLM scores")

    from zhuangzhou.core import ZhuangzhouFramework
    fw = ZhuangzhouFramework(_config)

    result = fw.run_enzyme_engineering(
        pdb_path=pdb_path,
        binding_site=set(binding_site),
        esm2_dirs=esm2_dirs,
        enzyme_class=enzyme_class,
        protein_name=protein_name,
        conserved_positions=set(conserved_positions or []),
        output_dir=output_dir,
    )

    p2 = result.get('phases', {}).get('phase2_structure', {})
    p3 = result.get('phases', {}).get('phase3_combinations', {})
    return {
        'status': 'complete',
        'report_path': result.get('report_path', ''),
        'json_path': result.get('json_path', ''),
        'track_a_count': p2.get('track_a_count', 0),
        'track_b_count': p2.get('track_b_count', 0),
        'combinations': len(p3.get('combinations', [])),
        'top_mutations': [
            {'mutation': m['mutation'], 'zz_score': m['zz_score'], 'track': m['track']}
            for m in p2.get('scored_mutations', [])[:10]
        ],
    }


def check_environments() -> dict:
    """Check availability of all compute sandboxes (graphrag, dplm2, WSL, native)."""
    from zhuangzhou.envs.sandbox import check_all_environments
    return check_all_environments()


def list_skills(category: str = None) -> dict:
    """
    List registered computational skills.

    Args:
        category: filter by category 鈥?plm, docking, simulation, chemistry,
                  brain, structure, mutagenesis, visualization, unimol
    """
    reg = _get_skills()
    if category:
        skills = reg.list_by_category(category)
    else:
        skills = reg.list_all()
    return {
        'skills': [
            {'name': s.name, 'category': s.category,
             'description': s.description, 'gpu': s.requires_gpu}
            for s in skills
        ],
        'categories': reg.categories(),
    }


def execute_skill(skill_name: str, **params) -> dict:
    """
    Execute a specific registered skill by name.

    Args:
        skill_name: exact name from list_skills()
        **params: skill-specific parameters
    """
    reg = _get_skills()
    skill = reg.get(skill_name)
    if not skill:
        available = [s.name for s in reg.list_all()]
        return {
            'error': f'Skill "{skill_name}" not found',
            'available_skills': available,
            'hint': f'Use list_skills() to see all {len(available)} available skills',
        }
    params['config'] = _config
    return skill.execute(**params)


def graphrag_query(
    query: str,
    kg_id: str = 'all',
    query_type: str = 'entities',
) -> dict:
    """
    Query AutoGraphRAG knowledge graphs directly.

    Args:
        query: search text or entity name
        kg_id: 'all' or specific domain KG name
        query_type: 'entities', 'relationships', 'community_reports', 'mutation_evidence'
    """
    grag = _get_graphrag()
    if query_type == 'entities':
        return grag.search_entities(query.split(), kg_id=kg_id)
    elif query_type == 'relationships':
        return grag.search_relationships(query, kg_id=kg_id)
    elif query_type == 'community_reports':
        return grag.get_community_context(query, kg_id=kg_id)
    elif query_type == 'mutation_evidence':
        return grag.get_mutation_evidence(query, kg_id=kg_id)
    return {'error': f'Unknown query_type: {query_type}',
            'valid': ['entities', 'relationships', 'community_reports', 'mutation_evidence']}


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
# BioDBAPI 鈥?Online protein/gene/structure retrieval (direct functions)
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?

def get_protein_profile(identifier: str) -> dict:
    """
    One-shot comprehensive protein profile: UniProt + InterPro + PDB + AlphaFold.

    Args:
        identifier: UniProt accession (e.g. P00183 for CYP101/P450cam)

    Returns:
        dict with sequence, function, catalytic activity, domains, cofactors,
        kinetics, GO terms, PDB structures, AlphaFold prediction.
    """
    from zhuangzhou.skills.biodbapi import get_protein_profile as _profile
    return _profile(identifier)


def get_protein_sequence(accession: str) -> dict:
    """
    Get protein FASTA sequence from UniProt or NCBI.

    Args:
        accession: UniProt accession (P00183) or NCBI accession (NP_xxxx)
    """
    if accession.startswith('NP_') or accession.startswith('XP_') or accession.startswith('WP_'):
        from zhuangzhou.skills.biodbapi import get_ncbi_protein_fasta
        return get_ncbi_protein_fasta(accession)
    from zhuangzhou.skills.biodbapi import get_uniprot_sequence
    return get_uniprot_sequence(accession)


def search_protein_database(query: str, database: str = 'uniprot',
                            organism: str = None, max_results: int = 5) -> dict:
    """
    Search protein databases (UniProt, NCBI, PDB).

    Args:
        query: search keywords
        database: 'uniprot', 'ncbi', or 'pdb'
        organism: organism filter (optional)
        max_results: max results to return
    """
    if database == 'ncbi':
        from zhuangzhou.skills.biodbapi import search_ncbi_protein
        return search_ncbi_protein(query, organism=organism, max_results=max_results)
    elif database == 'pdb':
        from zhuangzhou.skills.biodbapi import search_pdb_structure
        return search_pdb_structure(query, max_results=max_results)
    else:
        from zhuangzhou.skills.research import search_uniprot
        return search_uniprot(query, organism=organism, max_results=max_results)


def search_compound(name: str) -> dict:
    """
    Search PubChem for a compound by name, returning authoritative SMILES,
    molecular formula, weight, and 3D SDF download URL.

    Args:
        name: Compound name (e.g. "asiatic acid", "ursolic acid", "camphor")
    """
    from zhuangzhou.skills.biodbapi import search_pubchem_compound
    return search_pubchem_compound(name)


def download_compound_sdf(name: str = None, cid: int = None,
                           output_dir: str = None) -> dict:
    """
    Download compound 3D SDF from PubChem. More reliable than SMILES鈫扲DKit
    for known compounds.

    Args:
        name: Compound name (e.g. "asiatic acid")
        cid: PubChem CID (alternative)
        output_dir: Output directory (default: CWD)
    """
    from zhuangzhou.skills.biodbapi import download_pubchem_sdf
    return download_pubchem_sdf(name=name, cid=cid, output_dir=output_dir)


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
# Legacy compatibility 鈥?ZhuangzhouAgent class wrapping the public functions
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?

class ZhuangzhouAgent:
    """Wrapper class for backwards compatibility with app.py."""

    def __init__(self, config: dict = None):
        global _config
        if config:
            _config.update(config)

    def execute_tool(self, tool_name: str, params: dict) -> dict:
        dispatch = {
            'zz_analyze_mutation': lambda p: analyze_mutation(
                p.get('mutation', ''), p.get('enzyme_class', ''),
                p.get('role', 'unknown'), p.get('is_conserved', False)),
            'zz_search_knowledge': lambda p: search_knowledge(
                p.get('query', ''), p.get('domain', 'all'), p.get('limit', 10)),
            'zz_run_pipeline': lambda p: run_pipeline(
                p.get('pdb_path', ''), p.get('binding_site'),
                p.get('esm2_dirs'), p.get('enzyme_class', ''),
                p.get('protein_name', ''), p.get('conserved_positions'),
                p.get('output_dir')),
            'zz_check_environments': lambda p: check_environments(),
            'zz_list_skills': lambda p: list_skills(p.get('category')),
            'zz_execute_skill': lambda p: execute_skill(
                p.get('skill_name', ''), **p.get('params', {})),
            'zz_graphrag_query': lambda p: graphrag_query(
                p.get('query', ''), p.get('kg_id', 'all'),
                p.get('query_type', 'entities')),
        }
        handler = dispatch.get(tool_name)
        if not handler:
            return {'error': f'Unknown tool: {tool_name}', 'available': list(dispatch.keys())}
        try:
            return handler(params)
        except Exception as e:
            logger.error("Tool %s failed: %s", tool_name, e, exc_info=True)
            return {'error': str(e), 'tool': tool_name}


def get_agent(config: dict = None) -> ZhuangzhouAgent:
    return ZhuangzhouAgent(config)


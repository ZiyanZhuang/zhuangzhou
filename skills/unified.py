"""
Unified Skills — Consolidated wrappers that merge similar skills.

This module reduces cognitive load by providing single-entry-point skills
that dispatch to the correct underlying implementation based on a mode/type
parameter. Old skill names are preserved as aliases for backward compatibility.

Consolidation groups:
  1. uniprot_query — merges 5 UniProt skills into 1
  2. ncbi_query — merges 3 NCBI skills into 1
  3. pubchem_query — merges 3 PubChem skills into 1
  4. generate_conformer — merges 3 conformer skills into 1
  5. esm2_mutation_scan — merges 3 ESM-2 skills into 1
  6. chimerax_visualize — merges 5 ChimeraX skills into 1
  7. generate_report — merges 4 report skills into 1
  8. mutate_protein (enhanced) — adds optimize flag to merge with mutate_minimize
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from .registry import Skill, SkillParam, SkillRegistry

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Unified UniProt Query
# =============================================================================

def uniprot_query(accession: str = None, query: str = None,
                  mode: str = 'full', organism: str = None,
                  max_results: int = 5, config: dict = None, **kw) -> dict:
    """Unified UniProt query: search, entry, sequence, domains, or full annotation.

    Modes:
      - search: Free-text search (requires `query`)
      - entry: Basic entry by accession
      - sequence: FASTA sequence only
      - domains: Domain/region annotations only
      - full: Comprehensive annotation (DEFAULT, recommended)
    """
    # Validate mode first
    valid_modes = ('search', 'entry', 'sequence', 'domains', 'full')
    if mode not in valid_modes:
        return {'error': f'Unknown mode: {mode}',
                'hint': f'Use one of: {", ".join(valid_modes)}'}

    if mode == 'search':
        if not query:
            return {'error': 'query is required for mode=search',
                    'hint': 'Provide a search term like "CYP3A4 human"'}
        from .research import search_uniprot
        return search_uniprot(query=query, organism=organism,
                              max_results=max_results, config=config)

    # All other modes need accession
    if not accession:
        return {'error': 'accession is required for this mode',
                'hint': 'Provide a UniProt accession like P00183'}

    if mode == 'entry':
        from .research import get_uniprot_entry
        return get_uniprot_entry(accession=accession, config=config)
    elif mode == 'sequence':
        from .biodbapi import get_uniprot_sequence
        return get_uniprot_sequence(accession=accession, config=config)
    elif mode == 'domains':
        from .biodbapi import get_uniprot_domains
        return get_uniprot_domains(accession=accession, config=config)
    elif mode == 'full':
        from .biodbapi import get_uniprot_full
        return get_uniprot_full(accession=accession, config=config)
    else:
        return {'error': f'Unknown mode: {mode}',
                'hint': 'Use one of: search, entry, sequence, domains, full'}


# =============================================================================
# 2. Unified NCBI Query
# =============================================================================

def ncbi_query(query: str = None, accession: str = None,
               database: str = 'protein', mode: str = 'search',
               organism: str = None, max_results: int = 5,
               config: dict = None, **kw) -> dict:
    """Unified NCBI query: search protein/gene or fetch FASTA.

    Modes:
      - search: Search by keyword (requires `query`)
      - fasta: Fetch FASTA by accession (requires `accession`)

    Databases (for mode=search):
      - protein: NCBI Protein database
      - gene: NCBI Gene database
    """
    if mode == 'fasta':
        if not accession:
            return {'error': 'accession is required for mode=fasta'}
        from .biodbapi import get_ncbi_protein_fasta
        return get_ncbi_protein_fasta(accession=accession, config=config)

    if mode == 'search':
        if not query:
            return {'error': 'query is required for mode=search'}
        if database == 'gene':
            from .biodbapi import search_ncbi_gene
            return search_ncbi_gene(query=query, organism=organism,
                                    max_results=max_results, config=config)
        else:  # protein
            from .biodbapi import search_ncbi_protein
            return search_ncbi_protein(query=query, organism=organism,
                                       max_results=max_results, config=config)

    return {'error': f'Unknown mode: {mode}', 'hint': 'Use search or fasta'}


# =============================================================================
# 3. Unified PubChem Query
# =============================================================================

def pubchem_query(name: str = None, smiles: str = None, cid: int = None,
                  mode: str = 'search', output_dir: str = None,
                  prefer_3d: bool = True,
                  config: dict = None, **kw) -> dict:
    """Unified PubChem compound operations.

    Modes:
      - search: Look up compound info (canonical SMILES, formula, MW)
      - download: Download 3D SDF file
      - validate: Cross-validate SMILES against PubChem (requires name + smiles)
    """
    if mode == 'search':
        if not name:
            return {'error': 'name is required for mode=search'}
        from .biodbapi import search_pubchem_compound
        return search_pubchem_compound(name=name, config=config)
    elif mode == 'download':
        if not name and not cid and not smiles:
            return {'error': 'At least one of name, cid, or smiles is required'}
        from .biodbapi import download_pubchem_sdf
        return download_pubchem_sdf(name=name, cid=cid, smiles=smiles,
                                     output_dir=output_dir, prefer_3d=prefer_3d,
                                     config=config)
    elif mode == 'validate':
        if not name or not smiles:
            return {'error': 'Both name and smiles are required for validation'}
        from .biodbapi import validate_compound_smiles
        return validate_compound_smiles(name=name, smiles=smiles, config=config)
    else:
        return {'error': f'Unknown mode: {mode}',
                'hint': 'Use search, download, or validate'}


# =============================================================================
# 4. Unified Conformer Generation
# =============================================================================

def generate_conformer(smiles: str = None, smiles_list: list = None,
                       output_path: str = None, output_dir: str = None,
                       num_conformers: int = 1, optimize: str = 'mmff',
                       backend: str = 'auto', compound_name: str = None,
                       remove_hs: bool = False,
                       config: dict = None, **kw) -> dict:
    """Unified 3D conformer generation from SMILES.

    Backends:
      - auto: Try RDKit first, fall back to UniMol (DEFAULT)
      - rdkit: Local RDKit + MMFF/UFF only
      - unimol: UniMol deep learning (requires GPU/dplm2)

    Single molecule: provide `smiles`
    Batch: provide `smiles_list`
    """
    # Batch mode
    if smiles_list:
        if backend == 'unimol':
            # UniMol doesn't have native batch SDF, use individual calls
            results = []
            for i, smi in enumerate(smiles_list):
                r = generate_conformer(smiles=smi, backend='unimol',
                                       num_conformers=num_conformers,
                                       config=config)
                results.append(r)
            return {'results': results, 'count': len(results)}
        else:
            # RDKit batch
            from .unimol import batch_conformer_generate
            return batch_conformer_generate(
                smiles_list=smiles_list, output_dir=output_dir,
                num_conformers=num_conformers, optimize=optimize,
                config=config)

    # Single molecule
    if not smiles:
        return {'error': 'smiles is required (or smiles_list for batch)'}

    if backend == 'unimol':
        from .unimol import unimol_conformer
        return unimol_conformer(smiles=smiles, n_conformers=num_conformers,
                                config=config)
    else:
        # auto or rdkit → use the existing conformer_generate which already
        # handles auto/local/bridge modes
        mode_map = {'auto': 'auto', 'rdkit': 'local', 'unimol': 'bridge'}
        from .unimol import conformer_generate
        # Compute output_path from output_dir if not explicitly provided
        if output_dir and not output_path:
            stem = (compound_name or 'molecule').replace(' ', '_')
            output_path = os.path.join(output_dir, f'{stem}_3d.sdf')
        return conformer_generate(
            smiles=smiles, output_path=output_path,
            num_conformers=num_conformers, optimize=optimize,
            remove_hs=remove_hs, mode=mode_map.get(backend, 'auto'),
            compound_name=compound_name, config=config)


# =============================================================================
# 5. Unified ESM-2 Mutation Scan
# =============================================================================

def esm2_mutation_scan(fasta_path: str = None, output_dir: str = None,
                       model: str = 'esm2_t6_8M_UR50D',
                       backend: str = 'auto',
                       sequence: str = None, uniprot_id: str = None,
                       config: dict = None, **kw) -> dict:
    """Unified ESM-2 zero-shot mutation scanning.

    Input (one of):
      - fasta_path: path to an existing FASTA file
      - sequence:   raw amino-acid sequence string (FASTA written automatically)
      - uniprot_id: UniProt accession — sequence fetched and FASTA written automatically

    Backends:
      - auto: Try GPU (dplm2) first, fall back to WSL
      - gpu: Force dplm2 GPU environment
      - wsl: Force WSL execution

    Default model is the smallest ESM-2 (8M). Pass model='esm2_t33_650M_UR50D'
    for the 650M variant when GPU is available.
    """
    # ── Resolve sequence → FASTA ─────────────────────────────────────────────
    if not fasta_path:
        # Auto-fetch sequence from UniProt if only an accession was given
        if not sequence and uniprot_id:
            try:
                from .biodbapi import get_uniprot_sequence
                res = get_uniprot_sequence(accession=uniprot_id)
                if 'error' in res:
                    return res
                sequence = res.get('sequence', '')
            except Exception as e:
                return {'error': f'UniProt fetch failed: {e}',
                        'hint': 'Check network or provide fasta_path directly.'}

        if not sequence:
            return {
                'error': 'No input: provide fasta_path, sequence, or uniprot_id.',
                'hint': 'Example: esm2_mutation_scan(uniprot_id="P11509") or '
                        'esm2_mutation_scan(fasta_path="/path/to/prot.fasta")',
            }

        # Write FASTA to CWD
        stem = uniprot_id or 'protein'
        fasta_dir = output_dir or os.getcwd()
        os.makedirs(fasta_dir, exist_ok=True)
        fasta_path = os.path.join(fasta_dir, f'{stem}.fasta')
        try:
            with open(fasta_path, 'w', encoding='utf-8') as fh:
                fh.write(f'>{stem}\n{sequence}\n')
            logger.info("esm2_mutation_scan: wrote FASTA to %s", fasta_path)
        except OSError as e:
            return {'error': f'Could not write FASTA: {e}', 'hint': fasta_path}

    if output_dir is None:
        stem = os.path.splitext(os.path.basename(fasta_path))[0]
        output_dir = os.path.join(os.getcwd(), f'{stem}_esm2')

    if backend == 'wsl':
        from .plm import esm2_scan
        return esm2_scan(fasta_path=fasta_path, output_dir=output_dir,
                         model=model, config=config)

    if backend in ('auto', 'gpu'):
        # Pre-check: verify dplm2 bridge is importable before attempting GPU call
        try:
            from zhuangzhou.envs.sandbox import Dplm2Sandbox  # noqa: F401
        except ImportError:
            if backend == 'gpu':
                return {
                    'error': 'dplm2 GPU environment unavailable',
                    'hint': 'Activate py310_DPLM2 conda env or use backend="wsl" instead.',
                }
            logger.info("dplm2 bridge not importable, routing ESM-2 to WSL")
            from .plm import esm2_scan
            return esm2_scan(fasta_path=fasta_path, output_dir=output_dir,
                             model=model, config=config)

        # Try GPU first
        try:
            from .unimol import _call_bridge
            result = _call_bridge('esm2_scan', {
                'fasta_path': fasta_path,
                'output_dir': output_dir,
                'model': model,
            }, timeout=3600)
            if result.get('status') != 'error' and result.get('status') != 'unavailable':
                return result
            # GPU failed, fall back to WSL if auto
            if backend == 'auto':
                logger.info("GPU ESM-2 unavailable, falling back to WSL")
                from .plm import esm2_scan
                return esm2_scan(fasta_path=fasta_path, output_dir=output_dir,
                                 model=model, config=config)
            return result
        except Exception as e:
            if backend == 'auto':
                from .plm import esm2_scan
                return esm2_scan(fasta_path=fasta_path, output_dir=output_dir,
                                 model=model, config=config)
            return {'error': str(e), 'hint': 'GPU environment unavailable'}

    return {'error': f'Unknown backend: {backend}',
            'hint': 'Use auto, gpu, or wsl'}


# =============================================================================
# 6. Unified ChimeraX Visualization
# =============================================================================

def chimerax_visualize(pdb_path: str = None, work_dir: str = None,
                       mode: str = 'render',
                       style: str = 'cartoon', color_by: str = 'chain',
                       highlight_residues: str = '', ligand_resname: str = '',
                       metric: str = 'displacement', n_frames: int = 50,
                       output_dir: str = None,
                       config: dict = None, **kw) -> dict:
    """Unified ChimeraX visualization.

    Modes:
      - render: Static structure rendering (requires pdb_path)
      - trajectory: MD trajectory animation (requires pdb_path)
      - defattr: Per-residue metric coloring (requires work_dir)
      - comparison: First vs last frame overlay (requires work_dir)
      - porcupine: Principal motion vectors (requires work_dir)
      - morph: Morph animation between frames (requires work_dir)
    """
    from .visualization import (
        _chimerax_render, _chimerax_trajectory, _chimerax_defattr,
        _chimerax_comparison, _chimerax_porcupine, _chimerax_morph,
    )

    if mode == 'render':
        if not pdb_path:
            return {'error': 'pdb_path is required for mode=render'}
        return _chimerax_render(pdb_path=pdb_path, output_dir=output_dir,
                                style=style, color_by=color_by,
                                highlight_residues=highlight_residues,
                                ligand_resname=ligand_resname, config=config)
    elif mode == 'trajectory':
        if not pdb_path:
            return {'error': 'pdb_path is required for mode=trajectory'}
        return _chimerax_trajectory(pdb_path=pdb_path, output_dir=output_dir,
                                    color_by=color_by, config=config)
    elif mode == 'defattr':
        if not work_dir:
            return {'error': 'work_dir is required for mode=defattr'}
        return _chimerax_defattr(work_dir=work_dir, metric=metric, config=config)
    elif mode == 'comparison':
        if not work_dir:
            return {'error': 'work_dir is required for mode=comparison'}
        return _chimerax_comparison(work_dir=work_dir, config=config)
    elif mode == 'porcupine':
        if not work_dir:
            return {'error': 'work_dir is required for mode=porcupine'}
        return _chimerax_porcupine(work_dir=work_dir, config=config)
    elif mode == 'morph':
        if not work_dir:
            return {'error': 'work_dir is required for mode=morph'}
        return _chimerax_morph(work_dir=work_dir, n_frames=n_frames, config=config)
    else:
        return {'error': f'Unknown mode: {mode}',
                'hint': 'Use: render, trajectory, defattr, comparison, porcupine, morph'}


# =============================================================================
# 7. Unified Report Generation
# =============================================================================

def generate_report(report_type: str = 'general',
                    results: str = None, results_json: str = None,
                    sections: str = None, sections_json: str = None,
                    output_path: str = None, output_dir: str = None,
                    title: str = None,
                    author: str = 'Zhuangzhou',
                    protein_context: dict = None,
                    config: dict = None, **kw) -> dict:
    """Unified Word report generation.

    Types:
      - mutation: Mutation analysis report
      - docking: Docking analysis report
      - md: MD simulation report
      - general: Free-form report from section definitions

    output_dir: Directory for the output file (overrides the default CWD).
                When both output_path and output_dir are given, output_path wins.
                When only output_dir is given, the filename is auto-generated.
    """
    import os

    # Resolve output_path from output_dir when only a directory is provided
    if output_dir and not output_path:
        os.makedirs(output_dir, exist_ok=True)
        # Determine filename based on report_type (templates will refine if needed)
        _stems = {
            'mutation': 'mutation_report',
            'docking':  'docking_report',
            'md':       'md_report',
            'general':  'report',
        }
        _stem = _stems.get(report_type, 'report')
        output_path = os.path.join(output_dir, f'{_stem}.docx')
    elif output_path and not os.path.isabs(output_path):
        # Make relative output_path absolute relative to CWD (not wherever docx_builder runs)
        output_path = os.path.abspath(output_path)

    if report_type == 'mutation':
        from .report import _generate_mutation_report
        return _generate_mutation_report(results=results, results_json=results_json,
                                          output_path=output_path, title=title,
                                          author=author, config=config,
                                          protein_context=protein_context)
    elif report_type == 'docking':
        from .report import _generate_docking_report
        return _generate_docking_report(results=results, results_json=results_json,
                                         output_path=output_path, title=title,
                                         author=author, config=config,
                                         protein_context=protein_context)
    elif report_type == 'md':
        from .report import _generate_md_report
        return _generate_md_report(results=results, results_json=results_json,
                                    output_path=output_path, title=title,
                                    author=author, config=config,
                                    protein_context=protein_context)
    elif report_type == 'general':
        from .report import _generate_general_report
        return _generate_general_report(sections=sections, sections_json=sections_json,
                                         output_path=output_path, title=title,
                                         author=author, config=config,
                                         results=results)
    elif report_type == 'synbio':
        from .report import _generate_synbio_report
        return _generate_synbio_report(results=results, results_json=results_json,
                                       output_path=output_path, title=title,
                                       author=author, config=config)
    else:
        return {'error': f'Unknown report type: {report_type}',
                'hint': 'Use: mutation, docking, md, general, synbio'}


# =============================================================================
# 8. Enhanced Mutate Protein (with optional optimization)
# =============================================================================

def mutate_protein_enhanced(pdb_path: str, mutations: list,
                            output_path: str = None, optimize: bool = False,
                            mode: str = 'solvated',
                            config: dict = None, **kw) -> dict:
    """Apply mutations with optional GROMACS energy minimization.

    When optimize=False (default): Quick PyMOL rotamer rebuild + BioPython fallback.
    When optimize=True: Apply mutation THEN run GROMACS energy minimization.
    """
    if optimize:
        from .simulation import mutate_minimize
        chain = (config or {}).get('chain', kw.get('chain', 'A'))
        return mutate_minimize(pdb_path=pdb_path, mutations=mutations,
                               output_path=output_path, mode=mode,
                               config=config, chain=chain)
    else:
        from .mutagenesis import _mutate_skill
        return _mutate_skill(pdb_path=pdb_path, mutations=mutations,
                             output_path=output_path, config=config, **kw)


# =============================================================================
# Registration
# =============================================================================

def register_unified_skills(registry: SkillRegistry):
    """Register consolidated unified skills.

    These REPLACE the fine-grained skills in the registry.
    Old skill functions still exist in their original modules for direct Python use.
    """

    # 1. Unified UniProt Query
    registry.register(Skill(
        name='uniprot_query',
        description='Unified UniProt protein query: search by keyword, or get entry/sequence/'
                    'domains/full annotation by accession. Mode=full gives comprehensive data.',
        category='research',
        fn=uniprot_query,
        requires_network=True,
        estimated_time='~3s',
        inputs=[
            SkillParam('accession', 'string', 'UniProt accession (e.g. P00183)', required=False),
            SkillParam('query', 'string', 'Search keywords (for mode=search)', required=False),
            SkillParam('mode', 'string', 'Query mode', required=False, default='full',
                       choices=['search', 'entry', 'sequence', 'domains', 'full']),
            SkillParam('organism', 'string', 'Organism filter (for mode=search)', required=False),
            SkillParam('max_results', 'int', 'Max results (for mode=search)', required=False, default=5),
        ],
        outputs=[SkillParam('result', 'dict', 'UniProt data')],
        source='UniProt REST API',
    ))

    # 2. Unified NCBI Query
    registry.register(Skill(
        name='ncbi_query',
        description='Unified NCBI query: search protein/gene databases or fetch FASTA sequence. '
                    'Use when UniProt accession is unknown.',
        category='research',
        fn=ncbi_query,
        requires_network=True,
        estimated_time='~3s',
        inputs=[
            SkillParam('query', 'string', 'Search keywords', required=False),
            SkillParam('accession', 'string', 'NCBI accession (for mode=fasta)', required=False),
            SkillParam('database', 'string', 'Database to search', required=False, default='protein',
                       choices=['protein', 'gene']),
            SkillParam('mode', 'string', 'Query mode', required=False, default='search',
                       choices=['search', 'fasta']),
            SkillParam('organism', 'string', 'Organism filter', required=False),
            SkillParam('max_results', 'int', 'Max results', required=False, default=5),
        ],
        outputs=[SkillParam('result', 'dict', 'Query results')],
        source='NCBI Entrez E-utilities',
    ))

    # 3. Unified PubChem Query
    registry.register(Skill(
        name='pubchem_query',
        description='Unified PubChem compound operations: search (get SMILES/formula/MW), '
                    'download (3D SDF), or validate (cross-check SMILES against PubChem). '
                    'ALWAYS use mode=search before guessing SMILES.',
        category='chemistry',
        fn=pubchem_query,
        requires_network=True,
        estimated_time='~3s',
        inputs=[
            SkillParam('name', 'string', 'Compound name (e.g. "camphor")', required=False),
            SkillParam('smiles', 'string', 'SMILES string (for validate/download)', required=False),
            SkillParam('cid', 'int', 'PubChem CID', required=False),
            SkillParam('mode', 'string', 'Operation mode', required=False, default='search',
                       choices=['search', 'download', 'validate']),
            SkillParam('output_dir', 'path', 'Output directory (for download)', required=False,
                       is_output_path=True),
            SkillParam('prefer_3d', 'bool', 'Prefer 3D structure (for download)', required=False,
                       default=True),
        ],
        outputs=[SkillParam('result', 'dict', 'Compound data')],
        source='PubChem PUG-REST API',
    ))

    # 4. Unified Conformer Generation
    registry.register(Skill(
        name='generate_conformer',
        description='Generate 3D molecular conformer(s) from SMILES. Auto-validates against '
                    'PubChem. Supports RDKit (fast, local) and UniMol (GPU deep learning) backends. '
                    'Batch mode: provide smiles_list.',
        category='chemistry',
        fn=generate_conformer,
        estimated_time='~5s',
        inputs=[
            SkillParam('smiles', 'string', 'SMILES string (single molecule)', required=False),
            SkillParam('smiles_list', 'list', 'List of SMILES (batch mode)', required=False),
            SkillParam('output_path', 'path', 'Output SDF path (single)', required=False,
                       is_output_path=True),
            SkillParam('output_dir', 'path', 'Output directory (batch)', required=False,
                       is_output_path=True),
            SkillParam('num_conformers', 'int', 'Number of conformers per molecule', required=False,
                       default=1),
            SkillParam('optimize', 'string', 'Force field: mmff/uff', required=False, default='mmff'),
            SkillParam('backend', 'string', 'Computation backend', required=False, default='auto',
                       choices=['auto', 'rdkit', 'unimol']),
            SkillParam('compound_name', 'string', 'Compound name for PubChem validation', required=False),
        ],
        outputs=[SkillParam('result', 'dict', 'SDF path, SMILES, properties')],
        source='RDKit / UniMol',
    ))

    # 5. Unified ESM-2 Mutation Scan
    registry.register(Skill(
        name='esm2_mutation_scan',
        description='ESM-2 zero-shot mutation scanning: predict mutation effects using protein '
                    'language model. Accepts fasta_path, raw sequence string, or UniProt ID. '
                    'Auto-selects GPU (fast) or WSL (fallback) backend. '
                    'Default model: esm2_t6_8M_UR50D (8M, fastest).',
        category='plm',
        fn=esm2_mutation_scan,
        requires_gpu=False,   # GPU optional; WSL fallback works on CPU
        estimated_time='~5-60min',
        inputs=[
            SkillParam('fasta_path', 'path', 'Input FASTA file (optional if sequence or uniprot_id given)',
                       required=False),
            SkillParam('sequence', 'string',
                       'Raw amino-acid sequence (auto-writes FASTA to output_dir)',
                       required=False),
            SkillParam('uniprot_id', 'string',
                       'UniProt accession to fetch sequence automatically (e.g. P11509)',
                       required=False),
            SkillParam('output_dir', 'path', 'Output directory (default: CWD/<stem>_esm2/)',
                       required=False, is_output_path=True),
            SkillParam('model', 'string',
                       'ESM-2 model: esm2_t6_8M_UR50D (8M, default), '
                       'esm2_t12_35M_UR50D (35M), esm2_t30_150M_UR50D (150M), '
                       'esm2_t33_650M_UR50D (650M)',
                       required=False, default='esm2_t6_8M_UR50D'),
            SkillParam('backend', 'string', 'Execution backend', required=False, default='auto',
                       choices=['auto', 'gpu', 'wsl']),
        ],
        outputs=[SkillParam('result', 'dict', 'Mutation scores and rankings')],
        source='ESM-2 (Meta AI)',
    ))

    # 6. Unified ChimeraX Visualization
    registry.register(Skill(
        name='chimerax_visualize',
        description='Unified ChimeraX visualization: structure rendering, MD trajectory animation, '
                    'per-residue metric coloring, structure comparison, porcupine plot, or morph.',
        category='visualization',
        fn=chimerax_visualize,
        estimated_time='~10s',
        inputs=[
            SkillParam('pdb_path', 'path', 'PDB file (for render/trajectory)', required=False),
            SkillParam('work_dir', 'path', 'MD work directory (for defattr/comparison/porcupine/morph)',
                       required=False),
            SkillParam('mode', 'string', 'Visualization mode', required=False, default='render',
                       choices=['render', 'trajectory', 'defattr', 'comparison', 'porcupine', 'morph']),
            SkillParam('style', 'string', 'Rendering style (for render)', required=False, default='cartoon',
                       choices=['cartoon', 'ribbon', 'surface', 'stick']),
            SkillParam('color_by', 'string', 'Color scheme (for render/trajectory)', required=False,
                       default='chain', choices=['chain', 'secondary', 'bfactor', 'element']),
            SkillParam('highlight_residues', 'string', 'Residues to highlight as sticks', required=False),
            SkillParam('ligand_resname', 'string', 'Ligand 3-letter code', required=False),
            SkillParam('metric', 'string', 'Metric type (for defattr)', required=False, default='displacement',
                       choices=['displacement', 'rmsf']),
            SkillParam('n_frames', 'int', 'Interpolation frames (for morph)', required=False, default=50),
            SkillParam('output_dir', 'path', 'Output directory', required=False, is_output_path=True),
        ],
        outputs=[SkillParam('result', 'dict', 'Visualization outputs (cxc, png, defattr)')],
        source='ChimeraX',
        output_type='visualization',
    ))

    # 7. Unified Report Generation
    registry.register(Skill(
        name='generate_report',
        description='Generate Word (.docx) report: mutation analysis, docking, MD simulation, '
                    'or free-form general report. Provide results dict or JSON file. '
                    '调用前先用get_analysis_context获取protein_context以确保蛋白名称正确。',
        category='report',
        fn=generate_report,
        estimated_time='~5s',
        inputs=[
            SkillParam('report_type', 'string', 'Report type', required=False, default='general',
                       choices=['mutation', 'docking', 'md', 'general']),
            SkillParam('results', 'string', 'Results dict or JSON string (for mutation/docking/md)',
                       required=False),
            SkillParam('results_json', 'path', 'JSON file path with results', required=False),
            SkillParam('sections', 'string', 'Section definitions JSON (for general type)', required=False),
            SkillParam('sections_json', 'path', 'Sections JSON file (for general type)', required=False),
            SkillParam('output_path', 'path', 'Full output .docx path (takes precedence over output_dir)',
                       required=False, is_output_path=True),
            SkillParam('output_dir', 'path',
                       'Output directory — filename auto-generated when output_path not given. '
                       'Use this when user says "save to <dir>".',
                       required=False, is_output_path=True),
            SkillParam('title', 'string', 'Report title', required=False),
            SkillParam('author', 'string', 'Author name', required=False, default='Zhuangzhou'),
            SkillParam('protein_context', 'dict', required=False,
                       description='蛋白质上下文（由get_analysis_context技能获取），'
                                   '用于确保报告中蛋白名称、序列、文献正确'),
        ],
        outputs=[SkillParam('result', 'dict', 'Output docx path')],
        source='python-docx',
        workflow_stage='report',
        when_to_use=(
            'ALWAYS call: (1) after esm2_mutation_scan succeeds and beneficial_mutations_json is available — '
            'even without explicit user request; (2) when user wants a Word report or says '
            '"保存所有内容" / "生成报告" / "save all" / "生成报告". '
            'For mutation scan: report_type="mutation", results_json=<esm2_beneficial_mutations.json path>. '
            'For docking: report_type="docking". '
            'Use output_dir when user specified a folder (e.g. "E:\\\\test_zhuangzhou").'
        ),
    ))

    # 8. Enhanced Mutate Protein (with optimize flag)
    registry.register(Skill(
        name='mutate_protein',
        description='Apply point mutations to PDB. PyMOL rotamer rebuild + BioPython fallback. '
                    'Set optimize=True to also run GROMACS energy minimization.',
        category='mutagenesis',
        fn=mutate_protein_enhanced,
        inputs=[
            SkillParam('pdb_path', 'path', 'Input PDB file'),
            SkillParam('mutations', 'list', 'Mutation strings, e.g. ["V9L"] or ["V9L","A5G"]'),
            SkillParam('output_path', 'path', 'Output PDB path (auto-generated if omitted)',
                       required=False),
            SkillParam('optimize', 'bool', 'Run GROMACS energy minimization after mutation',
                       required=False, default=False),
            SkillParam('mode', 'string', 'EM mode when optimize=True: vacuum (fast) or solvated',
                       required=False, default='solvated', choices=['vacuum', 'solvated']),
        ],
        outputs=[SkillParam('result', 'dict', 'Mutation result with output_pdb')],
        source='zhuangzhou/mutagenesis + mutate_minimize',
        output_type='pdb',
    ))

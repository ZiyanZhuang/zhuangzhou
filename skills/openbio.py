"""
Local AI Bioinformatics Skills — v0.2.0
========================================
本地执行版：所有计算均在本地 conda 环境 (dplm2) 或免费公共 API 中运行。
无需任何付费第三方 API。

Tools:
  local_protein_fold    — Boltz-2 蛋白结构预测 (py310_DPLM2 conda env)
  local_protein_design  — ProteinMPNN 序列逆向设计 (py310_DPLM2 conda env)
  local_ligand_design   — LigandMPNN 配体感知序列设计 (py310_DPLM2 conda env)
  local_blast           — NCBI BLAST+ 序列相似性搜索 (免费 E-utilities API)

依赖安装 (py310_DPLM2 env):
  pip install boltz                 # Boltz-2 结构预测
  pip install git+https://github.com/dauparas/ProteinMPNN  # ProteinMPNN
  pip install git+https://github.com/dauparas/LigandMPNN   # LigandMPNN
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

from .registry import Skill, SkillParam, SkillRegistry

logger = logging.getLogger(__name__)

# Bridge scripts live next to this package
_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts')
_BOLTZ2_BRIDGE = os.path.join(_SCRIPTS_DIR, 'boltz2_fold.py')
_PROTEINMPNN_BRIDGE = os.path.join(_SCRIPTS_DIR, 'proteinmpnn_design.py')
_LIGANDMPNN_BRIDGE = os.path.join(_SCRIPTS_DIR, 'ligandmpnn_design.py')

# NCBI BLAST URL API (free, no key needed)
_BLAST_CGI = 'https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi'
_BLAST_POLL_INTERVAL = 10
_BLAST_MAX_WAIT = 300


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_sandbox():
    """Return CondaSandbox for the dplm2 env."""
    from zhuangzhou.envs.sandbox import CondaSandbox
    return CondaSandbox('dplm2', timeout=600)


def _sandbox_result_to_dict(result) -> dict:
    """Convert SandboxResult to plain dict, normalizing errors."""
    if result.success:
        if isinstance(result.data, dict):
            return result.data
        return {'output': result.data}
    return {
        'error': result.error or f'Subprocess failed (rc={result.returncode})',
        'stderr': result.stderr[-500:] if result.stderr else '',
    }


# ---------------------------------------------------------------------------
# local_protein_fold  (Boltz-2 via dplm2 CondaSandbox)
# ---------------------------------------------------------------------------

def local_protein_fold(
    sequence: str = None,
    fasta_path: str = None,
    model: str = 'boltz2',
    output_dir: str = None,
    **kw,
) -> dict:
    """
    Predict protein 3D structure locally using Boltz-2 (or ESMFold fallback).

    Runs in py310_DPLM2 conda environment via CondaSandbox.

    Args:
        sequence:   Amino acid sequence string
        fasta_path: Path to .fasta file (alternative to sequence)
        model:      'boltz2' (default) or 'esmfold'
        output_dir: Where to save the output PDB

    Returns:
        {'pdb_path': str, 'confidence': float, 'model': str}
    """
    if not sequence and not fasta_path:
        return {'error': 'Provide sequence or fasta_path'}

    sandbox = _get_sandbox()
    if not sandbox.available:
        return {
            'error': 'py310_DPLM2 conda environment not found',
            'hint': f'Expected at: {sandbox.python_path}',
        }

    if not os.path.exists(_BOLTZ2_BRIDGE):
        return {
            'error': f'Bridge script not found: {_BOLTZ2_BRIDGE}',
            'hint': 'Run: pip install boltz  (in py310_DPLM2 env)',
        }

    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    _temp_fasta = None
    try:
        if sequence and not fasta_path:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.fasta', mode='w',
                                              delete=False, encoding='utf-8') as tmp:
                tmp.write(f'>query\n{sequence.strip()}\n')
                _temp_fasta = tmp.name
                fasta_path = _temp_fasta

        payload = {
            'fasta_path': fasta_path,
            'model': model,
            'output_dir': output_dir,
        }

        result = sandbox.run_json_bridge(_BOLTZ2_BRIDGE, payload, timeout=600)
        return _sandbox_result_to_dict(result)
    finally:
        if _temp_fasta and os.path.exists(_temp_fasta):
            try:
                os.unlink(_temp_fasta)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# local_protein_design  (ProteinMPNN via dplm2 CondaSandbox)
# ---------------------------------------------------------------------------

def local_protein_design(
    pdb_path: str,
    chain: str = 'A',
    fixed_positions: str = None,
    num_sequences: int = 10,
    temperature: float = 0.1,
    output_dir: str = None,
    **kw,
) -> dict:
    """
    Inverse folding: design new sequences for a given backbone (ProteinMPNN).

    Runs ProteinMPNN locally in py310_DPLM2 conda environment.

    Args:
        pdb_path:        Input PDB backbone structure
        chain:           Chain ID to design (default 'A')
        fixed_positions: Comma-separated positions to keep fixed (e.g. '10,25,50')
        num_sequences:   Number of sequences to generate (default 10)
        temperature:     Sampling temperature 0.0-1.0 (lower = more conservative)
        output_dir:      Where to save output FASTA/scores

    Returns:
        {'sequences': [{'sequence': str, 'score': float}, ...], 'fasta_path': str}
    """
    if not pdb_path or not os.path.exists(pdb_path):
        return {'error': f'PDB file not found: {pdb_path}'}

    sandbox = _get_sandbox()
    if not sandbox.available:
        return {
            'error': 'py310_DPLM2 conda environment not found',
            'hint': f'Expected at: {sandbox.python_path}',
        }

    if not os.path.exists(_PROTEINMPNN_BRIDGE):
        return {
            'error': f'Bridge script not found: {_PROTEINMPNN_BRIDGE}',
            'hint': (
                'Install ProteinMPNN in py310_DPLM2: '
                'pip install git+https://github.com/dauparas/ProteinMPNN'
            ),
        }

    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(pdb_path),
            os.path.splitext(os.path.basename(pdb_path))[0] + '_mpnn',
        )
    os.makedirs(output_dir, exist_ok=True)

    fixed_list = []
    if fixed_positions:
        try:
            fixed_list = [int(x.strip()) for x in fixed_positions.split(',') if x.strip()]
        except ValueError:
            return {'error': f'fixed_positions must be comma-separated integers: {fixed_positions!r}'}

    payload = {
        'pdb_path': pdb_path,
        'chain': chain,
        'fixed_positions': fixed_list,
        'num_sequences': num_sequences,
        'temperature': temperature,
        'output_dir': output_dir,
    }

    result = sandbox.run_json_bridge(_PROTEINMPNN_BRIDGE, payload, timeout=300)
    return _sandbox_result_to_dict(result)


# ---------------------------------------------------------------------------
# local_ligand_design  (LigandMPNN via dplm2 CondaSandbox)
# ---------------------------------------------------------------------------

def local_ligand_design(
    pdb_path: str,
    ligand_resname: str,
    chain: str = 'A',
    num_sequences: int = 10,
    temperature: float = 0.1,
    output_dir: str = None,
    **kw,
) -> dict:
    """
    Ligand-aware protein sequence design (LigandMPNN).

    Redesigns binding pocket residues while considering ligand geometry.
    Significantly better than ProteinMPNN for improving binding affinity.

    Args:
        pdb_path:       Protein-ligand complex PDB
        ligand_resname: Ligand 3-letter residue name (e.g. 'LIG', 'HEM', 'ANP')
        chain:          Protein chain to redesign (default 'A')
        num_sequences:  Number of sequences to generate (default 10)
        temperature:    Sampling temperature 0.0-1.0
        output_dir:     Where to save results

    Returns:
        {'sequences': [{'sequence': str, 'score': float}], 'fasta_path': str}
    """
    if not pdb_path or not os.path.exists(pdb_path):
        return {'error': f'PDB file not found: {pdb_path}'}
    if not ligand_resname:
        return {'error': 'ligand_resname required (e.g. "LIG", "HEM")'}

    sandbox = _get_sandbox()
    if not sandbox.available:
        return {
            'error': 'py310_DPLM2 conda environment not found',
            'hint': f'Expected at: {sandbox.python_path}',
        }

    if not os.path.exists(_LIGANDMPNN_BRIDGE):
        return {
            'error': f'Bridge script not found: {_LIGANDMPNN_BRIDGE}',
            'hint': (
                'Install LigandMPNN in py310_DPLM2: '
                'pip install git+https://github.com/dauparas/LigandMPNN'
            ),
        }

    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(pdb_path),
            os.path.splitext(os.path.basename(pdb_path))[0] + '_ligandmpnn',
        )
    os.makedirs(output_dir, exist_ok=True)

    payload = {
        'pdb_path': pdb_path,
        'ligand_resname': ligand_resname,
        'chain': chain,
        'num_sequences': num_sequences,
        'temperature': temperature,
        'output_dir': output_dir,
    }

    result = sandbox.run_json_bridge(_LIGANDMPNN_BRIDGE, payload, timeout=300)
    return _sandbox_result_to_dict(result)


# ---------------------------------------------------------------------------
# local_blast  (NCBI BLAST+ via free E-utilities — no API key)
# ---------------------------------------------------------------------------

def local_blast(
    sequence: str = None,
    fasta_path: str = None,
    database: str = 'swissprot',
    max_hits: int = 20,
    evalue: float = 1e-5,
    program: str = 'blastp',
    **kw,
) -> dict:
    """
    BLAST sequence similarity search using free NCBI BLAST URL API.

    No API key required. Rate limited to ~1 request per 3s by NCBI policy.

    Args:
        sequence:   Query amino acid (or nucleotide) sequence
        fasta_path: Path to query .fasta file (alternative to sequence)
        database:   'swissprot' (default), 'nr', 'pdb', 'refseq_protein'
        max_hits:   Maximum number of hits to return (default 20)
        evalue:     E-value threshold (default 1e-5)
        program:    'blastp' (protein, default), 'blastn' (nucleotide), 'blastx'

    Returns:
        {'hits': [{'accession', 'title', 'identity', 'evalue', 'score', 'length'}], 'rid': str}
    """
    if not sequence and not fasta_path:
        return {'error': 'Provide sequence or fasta_path'}

    if fasta_path and os.path.exists(fasta_path):
        with open(fasta_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        # Extract sequence from FASTA
        lines = [l for l in content.splitlines() if not l.startswith('>')]
        sequence = ''.join(lines).strip()

    if not sequence:
        return {'error': 'Could not read sequence from fasta_path'}

    # Submit BLAST job
    try:
        rid = _blast_submit(sequence, program=program, database=database,
                             evalue=evalue, max_hits=max_hits)
        if not rid:
            return {'error': 'BLAST submission failed — no RID returned'}
    except Exception as e:
        return {'error': f'BLAST submission error: {e}'}

    # Poll for results
    try:
        hits = _blast_poll_and_parse(rid, max_hits=max_hits)
        return {'hits': hits, 'rid': rid, 'database': database, 'n_hits': len(hits)}
    except Exception as e:
        return {'error': f'BLAST retrieval error: {e}', 'rid': rid}


def _blast_submit(sequence: str, program: str, database: str,
                   evalue: float, max_hits: int) -> Optional[str]:
    """Submit BLAST job, return RID."""
    params = urllib.parse.urlencode({
        'CMD': 'Put',
        'PROGRAM': program,
        'DATABASE': database,
        'QUERY': sequence,
        'FORMAT_TYPE': 'JSON2',
        'HITLIST_SIZE': max_hits,
        'EXPECT': evalue,
        'WORD_SIZE': 3 if 'blastp' in program else 11,
    })
    req = urllib.request.Request(
        _BLAST_CGI,
        data=params.encode('utf-8'),
        headers={
            'Content-Type': 'application/x-www-form-urlencoded',
            'User-Agent': 'Zhuangzhou/0.8.6 (zhuangzhou enzyme engineering)',
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode('utf-8', errors='replace')

    # Extract RID from HTML response
    import re
    m = re.search(r'RID = ([A-Z0-9]+)', body)
    return m.group(1) if m else None


def _blast_poll_and_parse(rid: str, max_hits: int = 20) -> list:
    """Poll BLAST job until complete, parse JSON2 output, return hits list."""
    deadline = time.time() + _BLAST_MAX_WAIT

    while time.time() < deadline:
        time.sleep(_BLAST_POLL_INTERVAL)

        params = urllib.parse.urlencode({
            'CMD': 'Get',
            'RID': rid,
            'FORMAT_TYPE': 'JSON2',
            'HITLIST_SIZE': max_hits,
        })
        url = f'{_BLAST_CGI}?{params}'
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Zhuangzhou/0.8.6'},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode('utf-8', errors='replace')

        # Check if still waiting
        if 'Status=WAITING' in body:
            continue
        if 'Status=FAILED' in body or 'Status=UNKNOWN' in body:
            raise RuntimeError(f'BLAST job {rid} failed')

        # Try JSON2 format parse
        try:
            data = json.loads(body)
            return _parse_blast_json2(data, max_hits)
        except json.JSONDecodeError:
            # May be HTML still, check for results
            if '"BlastOutput2"' in body or '"hits"' in body:
                data = json.loads(body)
                return _parse_blast_json2(data, max_hits)
            # If raw text format, try simple parse
            if 'Query_' in body and 'Score' in body:
                return _parse_blast_text_minimal(body, max_hits)

    raise TimeoutError(f'BLAST job {rid} did not complete within {_BLAST_MAX_WAIT}s')


def _parse_blast_json2(data: dict, max_hits: int) -> list:
    """Parse NCBI BLAST JSON2 format into hits list."""
    hits = []
    try:
        # JSON2 structure: BlastOutput2[0].report.results.search.hits
        blast_out = data.get('BlastOutput2', [])
        if not blast_out:
            return hits
        search = blast_out[0].get('report', {}).get('results', {}).get('search', {})
        raw_hits = search.get('hits', [])

        for h in raw_hits[:max_hits]:
            desc = h.get('description', [{}])[0]
            hsps = h.get('hsps', [{}])
            best_hsp = hsps[0] if hsps else {}

            identity = best_hsp.get('identity', 0)
            align_len = best_hsp.get('align_len', 1)
            pct_identity = round(identity / max(align_len, 1) * 100, 1)

            hits.append({
                'accession': desc.get('accession', ''),
                'title': desc.get('title', '')[:120],
                'identity': pct_identity,
                'evalue': best_hsp.get('evalue', 1.0),
                'score': best_hsp.get('score', 0),
                'bit_score': best_hsp.get('bit_score', 0),
                'length': h.get('len', 0),
            })
    except (KeyError, IndexError, TypeError) as e:
        logger.debug('BLAST JSON2 parse error: %s', e)
    return hits


def _parse_blast_text_minimal(body: str, max_hits: int) -> list:
    """Minimal fallback text parser — returns raw accession lines."""
    import re
    hits = []
    # Match lines like: >sp|P12345|GENE_HUMAN description
    pattern = re.compile(r'^>(\S+)\s+(.+)', re.MULTILINE)
    for m in list(pattern.finditer(body))[:max_hits]:
        hits.append({
            'accession': m.group(1),
            'title': m.group(2)[:120],
            'identity': None,
            'evalue': None,
            'score': None,
            'bit_score': None,
            'length': None,
        })
    return hits


# ---------------------------------------------------------------------------
# Skill registration
# ---------------------------------------------------------------------------

def register(registry: SkillRegistry):
    """Register all local AI bioinformatics skills."""

    registry.register(Skill(
        name='local_protein_fold',
        description=(
            'Predict protein 3D structure from sequence using Boltz-2 (local, no API key). '
            'Runs in py310_DPLM2 conda environment. Better accuracy than ESMFold. '
            'Requires boltz installed: pip install boltz (in dplm2 env). '
            'Returns predicted PDB file and confidence scores.'
        ),
        category='openbio',
        fn=local_protein_fold,
        requires_network=False,
        estimated_time='~3-10min (GPU) / ~30min (CPU)',
        inputs=[
            SkillParam('sequence', 'string', 'Amino acid sequence (ACDEFGHIKLMNPQRSTVWY...)', required=False),
            SkillParam('fasta_path', 'path', 'Path to .fasta file (alternative to sequence)', required=False),
            SkillParam('model', 'string', '"boltz2" (default) or "esmfold"',
                       required=False, default='boltz2', choices=['boltz2', 'esmfold']),
            SkillParam('output_dir', 'path', 'Directory to save predicted PDB', required=False),
        ],
        outputs=[
            SkillParam('pdb_path', 'path', 'Path to predicted PDB file'),
            SkillParam('confidence', 'float', 'Mean pLDDT confidence score (0-100)'),
        ],
        source='zhuangzhou/skills/openbio.py',
        workflow_stage='fetch',
        when_to_use=(
            'When protein has no experimental structure and no AlphaFold DB entry, '
            'or when higher accuracy than ESMFold is needed. '
            'Use after pdb_to_fasta if sequence already available from PDB.'
        ),
        fallback_skills=['fetch_protein_structure'],
        input_from_skills=['pdb_to_fasta'],
    ))

    registry.register(Skill(
        name='local_protein_design',
        description=(
            'Inverse folding: design new amino acid sequences for a given protein backbone (ProteinMPNN). '
            'Given a PDB structure, generates diverse sequences that fold into that backbone. '
            'Runs locally in py310_DPLM2 conda environment. No API key needed. '
            'Useful for: enzyme variant libraries, stability optimization, humanization.'
        ),
        category='openbio',
        fn=local_protein_design,
        requires_network=False,
        estimated_time='~1-5min',
        inputs=[
            SkillParam('pdb_path', 'path', 'Input PDB backbone structure'),
            SkillParam('chain', 'string', 'Chain ID to design', required=False, default='A'),
            SkillParam('fixed_positions', 'string',
                       'Comma-separated residue positions to keep fixed (e.g. "10,25,50")',
                       required=False),
            SkillParam('num_sequences', 'int', 'Number of sequences to generate',
                       required=False, default=10),
            SkillParam('temperature', 'float', 'Sampling temperature 0.0-1.0 (lower=conservative)',
                       required=False, default=0.1),
            SkillParam('output_dir', 'path', 'Directory to save FASTA and scores', required=False),
        ],
        outputs=[
            SkillParam('sequences', 'list', 'List of {sequence, score} dicts'),
            SkillParam('fasta_path', 'path', 'Path to output FASTA with all designed sequences'),
        ],
        source='zhuangzhou/skills/openbio.py',
        workflow_stage='compute',
        when_to_use=(
            'After obtaining backbone structure. For library design, stability engineering, '
            'or generating enzyme variants beyond single-point mutations. '
            'Use local_ligand_design instead when a ligand is present in the active site.'
        ),
        input_from_skills=['smart_docking', 'local_protein_fold', 'pdb_to_fasta'],
    ))

    registry.register(Skill(
        name='local_ligand_design',
        description=(
            'Ligand-aware protein sequence design using LigandMPNN (local, no API key). '
            'Redesigns binding pocket residues while explicitly considering ligand geometry. '
            'Significantly better than ProteinMPNN for improving binding affinity or specificity. '
            'Input: protein-ligand complex PDB. Runs in py310_DPLM2 conda environment.'
        ),
        category='openbio',
        fn=local_ligand_design,
        requires_network=False,
        estimated_time='~1-5min',
        inputs=[
            SkillParam('pdb_path', 'path', 'Protein-ligand complex PDB'),
            SkillParam('ligand_resname', 'string',
                       'Ligand 3-letter residue name in PDB (e.g. "LIG", "HEM", "ANP")'),
            SkillParam('chain', 'string', 'Protein chain to redesign', required=False, default='A'),
            SkillParam('num_sequences', 'int', 'Number of sequences to generate',
                       required=False, default=10),
            SkillParam('temperature', 'float', 'Sampling temperature 0.0-1.0',
                       required=False, default=0.1),
            SkillParam('output_dir', 'path', 'Directory to save results', required=False),
        ],
        outputs=[
            SkillParam('sequences', 'list', 'List of {sequence, score} dicts'),
            SkillParam('fasta_path', 'path', 'Output FASTA path'),
        ],
        source='zhuangzhou/skills/openbio.py',
        workflow_stage='compute',
        when_to_use=(
            'After smart_docking or plip_analysis to improve binding affinity. '
            'Best for: active site remodeling, substrate specificity engineering, '
            'cofactor binding optimization (heme, FAD, NAD, FMN).'
        ),
        input_from_skills=['smart_docking', 'plip_analysis'],
        depends_on=['smart_docking'],
    ))

    registry.register(Skill(
        name='local_blast',
        description=(
            'BLAST sequence similarity search via free NCBI BLAST URL API. '
            'No API key required. Searches nr, swissprot, pdb, or refseq_protein databases. '
            'Find homologous proteins, identify conserved residues, source enzyme candidates.'
        ),
        category='openbio',
        fn=local_blast,
        requires_network=True,
        estimated_time='~30s-3min (depends on NCBI queue)',
        inputs=[
            SkillParam('sequence', 'string', 'Query amino acid sequence', required=False),
            SkillParam('fasta_path', 'path', 'Query .fasta file path', required=False),
            SkillParam('database', 'string', 'Target database: swissprot/nr/pdb/refseq_protein',
                       required=False, default='swissprot',
                       choices=['swissprot', 'nr', 'pdb', 'refseq_protein']),
            SkillParam('max_hits', 'int', 'Max hits to return', required=False, default=20),
            SkillParam('evalue', 'float', 'E-value threshold', required=False, default=1e-5),
            SkillParam('program', 'string', 'blastp/blastn/blastx',
                       required=False, default='blastp', choices=['blastp', 'blastn', 'blastx']),
        ],
        outputs=[
            SkillParam('hits', 'list', 'List of {accession, title, identity, evalue, score, length}'),
            SkillParam('n_hits', 'int', 'Number of hits found'),
            SkillParam('rid', 'string', 'NCBI BLAST Request ID for reference'),
        ],
        source='zhuangzhou/skills/openbio.py',
        workflow_stage='research',
        when_to_use=(
            'Find homologs for enzyme engineering candidates. '
            'Identify functional orthologs across organisms. '
            'Supplement uniprot_fetch for sequence diversity analysis. '
            'Use swissprot for well-annotated hits, nr for comprehensive coverage.'
        ),
        input_from_skills=['pdb_to_fasta'],
    ))

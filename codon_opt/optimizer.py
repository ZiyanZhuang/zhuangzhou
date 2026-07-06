"""
Codon Optimization Engine for Komagataella phaffii.

Backends:
  - 'dnachisel': DNAchisel MaximizeCAI with K. phaffii table (default)
  - 'frequency': Simple highest-frequency codon replacement
  - 'pichia_clm': Pichia-CLM deep learning (when weights available)

All functions return error dicts on failure, never raise.
"""

import logging
import re
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# Standard codon table
CODON_TABLE = {
    'A': ['GCT', 'GCC', 'GCA', 'GCG'],
    'C': ['TGT', 'TGC'],
    'D': ['GAT', 'GAC'],
    'E': ['GAA', 'GAG'],
    'F': ['TTT', 'TTC'],
    'G': ['GGT', 'GGA', 'GGC', 'GGG'],
    'H': ['CAT', 'CAC'],
    'I': ['ATT', 'ATC', 'ATA'],
    'K': ['AAA', 'AAG'],
    'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],
    'M': ['ATG'],
    'N': ['AAT', 'AAC'],
    'P': ['CCT', 'CCC', 'CCA', 'CCG'],
    'Q': ['CAA', 'CAG'],
    'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
    'T': ['ACT', 'ACC', 'ACA', 'ACG'],
    'V': ['GTT', 'GTC', 'GTA', 'GTG'],
    'W': ['TGG'],
    'Y': ['TAT', 'TAC'],
    '*': ['TAA', 'TAG', 'TGA'],
}

VALID_AA = set('ACDEFGHIKLMNPQRSTVWY*')


def _validate_protein(protein_seq: str) -> Optional[Dict]:
    """Validate protein sequence. Returns error dict or None."""
    if not protein_seq:
        return {'error': 'Empty protein sequence', 'hint': 'Provide a non-empty amino acid sequence'}
    cleaned = protein_seq.upper().strip()
    invalid = set(cleaned) - VALID_AA
    if invalid:
        return {
            'error': f'Invalid amino acids: {invalid}',
            'hint': 'Use standard 1-letter amino acid codes (ACDEFGHIKLMNPQRSTVWY*)'
        }
    return None


def _get_kphaffii_table() -> Dict:
    """Load K. phaffii codon usage table from python-codon-tables."""
    try:
        import python_codon_tables as pct
        from .config import KPHAFFII_TAXID
        return pct.get_codons_table(KPHAFFII_TAXID)
    except Exception as e:
        logger.warning("Failed to load codon table from python-codon-tables: %s", e)
        return None


def _optimize_frequency(protein_seq: str, add_stop: bool = True) -> Dict:
    """Simple highest-frequency codon replacement (no dependencies beyond python-codon-tables)."""
    table = _get_kphaffii_table()
    if table is None:
        from .config import KPHAFFII_PREFERRED
        preferred = KPHAFFII_PREFERRED
    else:
        preferred = {}
        for aa, codons in table.items():
            # codons is dict like {'GCT': 0.45, 'GCC': 0.26, ...}
            best = max(codons, key=codons.get)
            # Convert U->T for DNA
            preferred[aa] = best.replace('U', 'T')

    seq = protein_seq.upper().strip()
    if seq.endswith('*'):
        seq = seq[:-1]

    dna = ''
    for aa in seq:
        dna += preferred.get(aa, 'NNN')

    if add_stop:
        dna += preferred.get('*', 'TAA')

    return {
        'dna_sequence': dna,
        'protein_sequence': seq + ('*' if add_stop else ''),
        'method': 'frequency',
        'organism': 'Komagataella phaffii',
        'length_nt': len(dna),
        'length_aa': len(seq),
    }


def _optimize_dnachisel(
    protein_seq: str,
    add_stop: bool = True,
    gc_min: float = None,
    gc_max: float = None,
    avoid_patterns: List[str] = None,
) -> Dict:
    """DNAchisel-based optimization with K. phaffii codon table."""
    try:
        from dnachisel import (
            DnaOptimizationProblem,
            CodonOptimize,
            EnforceTranslation,
            EnforceGCContent,
            AvoidPattern,
        )
    except ImportError:
        return {'error': 'dnachisel not installed', 'hint': 'pip install dnachisel python-codon-tables'}

    from .config import DEFAULT_GC_MIN, DEFAULT_GC_MAX, AVOID_PATTERNS, KPHAFFII_TAXID

    gc_min = gc_min if gc_min is not None else DEFAULT_GC_MIN
    gc_max = gc_max if gc_max is not None else DEFAULT_GC_MAX
    avoid_patterns = avoid_patterns if avoid_patterns is not None else AVOID_PATTERNS

    seq = protein_seq.upper().strip()
    if seq.endswith('*'):
        seq = seq[:-1]

    # Get codon table
    table = _get_kphaffii_table()

    # Build initial DNA from frequency method as seed
    seed_result = _optimize_frequency(seq, add_stop=add_stop)
    initial_dna = seed_result['dna_sequence']

    loc = (0, len(initial_dna))

    # Build constraints and objectives
    constraints = [
        EnforceTranslation(location=loc),
    ]
    objectives = []

    if table:
        objectives.append(CodonOptimize(
            codon_usage_table=table,
            location=loc,
        ))
    else:
        # Fallback to S. cerevisiae (closest available)
        objectives.append(CodonOptimize(species='s_cerevisiae', location=loc))

    # GC content constraint
    constraints.append(EnforceGCContent(mini=gc_min, maxi=gc_max, location=loc))

    # Avoid restriction sites and destabilizing motifs
    for pattern in avoid_patterns:
        constraints.append(AvoidPattern(pattern, location=loc))

    try:
        problem = DnaOptimizationProblem(
            sequence=initial_dna,
            constraints=constraints,
            objectives=objectives,
        )
        problem.resolve_constraints()
        problem.optimize()

        optimized_dna = problem.sequence
        return {
            'dna_sequence': optimized_dna,
            'protein_sequence': seq + ('*' if add_stop else ''),
            'method': 'dnachisel',
            'organism': 'Komagataella phaffii',
            'length_nt': len(optimized_dna),
            'length_aa': len(seq),
            'constraints_passed': not problem.all_constraints_pass() is False,
            'objectives_score': problem.objective_scores_sum(),
        }
    except Exception as e:
        logger.warning("DNAchisel optimization failed: %s, falling back to frequency", e)
        result = _optimize_frequency(seq, add_stop=add_stop)
        result['warning'] = f'DNAchisel failed ({e}), used frequency fallback'
        return result


def _optimize_pichia_clm(protein_seq: str, add_stop: bool = True) -> Dict:
    """Pichia-CLM deep learning backend."""
    try:
        from .pichia_clm import predict_optimized_dna
        return predict_optimized_dna(protein_seq, add_stop=add_stop)
    except Exception as e:
        return {
            'error': f'Pichia-CLM unavailable: {e}',
            'hint': 'Model weights not found. Download .ckpt.data files to E:/Code/PichiaCLM/Model_PichiaCLM/2Target_AllData/'
        }


def optimize_codons(
    protein_seq: str,
    method: str = 'dnachisel',
    add_stop: bool = True,
    gc_min: float = None,
    gc_max: float = None,
    avoid_patterns: List[str] = None,
) -> Dict:
    """
    Optimize codons for heterologous expression in K. phaffii.

    Args:
        protein_seq: Amino acid sequence (1-letter code)
        method: 'dnachisel' (default), 'frequency', or 'pichia_clm'
        add_stop: Append stop codon (default True)
        gc_min/gc_max: GC content bounds (dnachisel only)
        avoid_patterns: Restriction sites to avoid (dnachisel only)

    Returns:
        Dict with dna_sequence, protein_sequence, method, stats, or error dict.
    """
    err = _validate_protein(protein_seq)
    if err:
        return err

    if method == 'dnachisel':
        return _optimize_dnachisel(protein_seq, add_stop, gc_min, gc_max, avoid_patterns)
    elif method == 'frequency':
        return _optimize_frequency(protein_seq, add_stop)
    elif method == 'pichia_clm':
        return _optimize_pichia_clm(protein_seq, add_stop)
    else:
        return {'error': f'Unknown method: {method}', 'hint': "Use 'dnachisel', 'frequency', or 'pichia_clm'"}


def batch_optimize(
    sequences: List[Dict[str, str]],
    method: str = 'dnachisel',
    **kwargs,
) -> List[Dict]:
    """
    Batch codon optimization.

    Args:
        sequences: List of {'name': str, 'protein': str}
        method: Backend to use
        **kwargs: Passed to optimize_codons

    Returns:
        List of result dicts, each including 'name'.
    """
    results = []
    for entry in sequences:
        name = entry.get('name', f'seq_{len(results)+1}')
        protein = entry.get('protein', '')
        result = optimize_codons(protein, method=method, **kwargs)
        result['name'] = name
        results.append(result)
    return results

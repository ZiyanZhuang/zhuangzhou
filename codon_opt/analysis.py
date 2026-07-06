"""
Codon optimization analysis: CAI, GC%, codon frequency distribution.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


REVERSE_CODON_TABLE = {}
_CODON_TO_AA = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}


def _get_kphaffii_table() -> Optional[Dict]:
    try:
        import python_codon_tables as pct
        return pct.get_codons_table(4922)
    except Exception:
        return None


def gc_content(dna_seq: str) -> float:
    """Calculate GC content of a DNA sequence."""
    dna = dna_seq.upper().replace(' ', '').replace('\n', '')
    if not dna:
        return 0.0
    gc = sum(1 for c in dna if c in 'GC')
    return gc / len(dna)


def translate(dna_seq: str) -> str:
    """Translate DNA to protein."""
    dna = dna_seq.upper().replace(' ', '').replace('\n', '')
    protein = ''
    for i in range(0, len(dna) - 2, 3):
        codon = dna[i:i+3]
        aa = _CODON_TO_AA.get(codon, 'X')
        protein += aa
    return protein


def codon_frequency_distribution(dna_seq: str) -> Dict[str, Dict[str, float]]:
    """
    Calculate per-amino-acid codon usage frequencies in a DNA sequence.
    Returns: {AA: {codon: fraction, ...}, ...}
    """
    dna = dna_seq.upper().replace(' ', '').replace('\n', '')
    # Count codons per AA
    aa_counts = {}
    for i in range(0, len(dna) - 2, 3):
        codon = dna[i:i+3]
        aa = _CODON_TO_AA.get(codon)
        if aa is None:
            continue
        if aa not in aa_counts:
            aa_counts[aa] = {}
        aa_counts[aa][codon] = aa_counts[aa].get(codon, 0) + 1

    # Convert to fractions
    result = {}
    for aa, codons in aa_counts.items():
        total = sum(codons.values())
        result[aa] = {c: n / total for c, n in codons.items()}
    return result


def cai_score(dna_seq: str, reference_table: Dict = None) -> float:
    """
    Calculate Codon Adaptation Index (CAI) against K. phaffii reference.

    Uses geometric mean of relative adaptiveness (w) values.
    """
    import math

    if reference_table is None:
        reference_table = _get_kphaffii_table()
    if reference_table is None:
        return -1.0  # Cannot compute

    dna = dna_seq.upper().replace(' ', '').replace('\n', '')

    # Build w values (relative adaptiveness = freq / max_freq_for_that_AA)
    w_table = {}
    for aa, codons in reference_table.items():
        if aa == '*':
            continue
        max_freq = max(codons.values()) if codons else 1.0
        for codon, freq in codons.items():
            codon_dna = codon.replace('U', 'T')
            w_table[codon_dna] = freq / max_freq if max_freq > 0 else 0.0

    log_sum = 0.0
    count = 0
    for i in range(0, len(dna) - 2, 3):
        codon = dna[i:i+3]
        aa = _CODON_TO_AA.get(codon)
        if aa is None or aa == '*':
            continue
        w = w_table.get(codon, 0.001)  # small epsilon for unknown
        if w > 0:
            log_sum += math.log(w)
            count += 1

    if count == 0:
        return 0.0
    return math.exp(log_sum / count)


def analyze_codons(dna_seq: str) -> Dict:
    """
    Comprehensive codon analysis of a DNA sequence.

    Returns dict with: gc_content, cai, length_nt, length_aa,
    protein, codon_distribution.
    """
    if not dna_seq or len(dna_seq) < 3:
        return {'error': 'DNA sequence too short', 'hint': 'Provide at least one codon (3 nt)'}

    dna = dna_seq.upper().replace(' ', '').replace('\n', '')
    protein = translate(dna)
    gc = gc_content(dna)
    cai = cai_score(dna)
    freq_dist = codon_frequency_distribution(dna)

    return {
        'gc_content': round(gc, 4),
        'cai': round(cai, 4),
        'length_nt': len(dna),
        'length_aa': len(protein),
        'protein': protein,
        'codon_distribution': freq_dist,
    }


def compare_sequences(original_dna: str, optimized_dna: str) -> Dict:
    """Compare original vs optimized DNA sequences."""
    orig = analyze_codons(original_dna)
    opt = analyze_codons(optimized_dna)

    if 'error' in orig or 'error' in opt:
        return {'error': 'Analysis failed', 'original': orig, 'optimized': opt}

    # Count changed codons
    orig_dna = original_dna.upper().replace(' ', '').replace('\n', '')
    opt_dna = optimized_dna.upper().replace(' ', '').replace('\n', '')
    n_codons = min(len(orig_dna), len(opt_dna)) // 3
    changed = 0
    for i in range(n_codons):
        if orig_dna[i*3:(i+1)*3] != opt_dna[i*3:(i+1)*3]:
            changed += 1

    return {
        'original': {
            'gc_content': orig['gc_content'],
            'cai': orig['cai'],
            'length_nt': orig['length_nt'],
        },
        'optimized': {
            'gc_content': opt['gc_content'],
            'cai': opt['cai'],
            'length_nt': opt['length_nt'],
        },
        'codons_changed': changed,
        'codons_total': n_codons,
        'change_ratio': round(changed / n_codons, 4) if n_codons > 0 else 0,
        'cai_improvement': round(opt['cai'] - orig['cai'], 4),
        'gc_shift': round(opt['gc_content'] - orig['gc_content'], 4),
        'protein_match': orig['protein'] == opt['protein'],
    }

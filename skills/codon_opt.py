"""
Codon Optimization Skills for Komagataella phaffii (Pichia pastoris).
"""

from .registry import Skill, SkillParam


def _optimize_codons_skill(
    protein_seq: str = '',
    sequence: str = '',           # alias accepted by LLM
    protein_sequence: str = '',   # alias accepted by LLM
    method: str = 'dnachisel',
    add_stop: bool = True,
    output_fasta: str = '',
    name: str = 'optimized',
    **kw,
) -> dict:
    """Optimize codons for K. phaffii expression.

    protein_seq is the canonical param; 'sequence' and 'protein_sequence' are accepted as aliases.
    """
    from zhuangzhou.codon_opt import optimize_codons
    seq = protein_seq or sequence or protein_sequence
    if not seq:
        return {'error': 'No protein sequence provided',
                'hint': 'Pass protein_seq="MKTAY..." (amino acid sequence)'}
    result = optimize_codons(seq, method=method, add_stop=add_stop)

    if 'error' not in result and output_fasta:
        dna = result['dna_sequence']
        header = f">{name} | {result.get('method', method)} | K.phaffii optimized | {len(dna)}nt"
        fasta_lines = [header]
        for i in range(0, len(dna), 60):
            fasta_lines.append(dna[i:i+60])
        fasta_content = '\n'.join(fasta_lines) + '\n'
        with open(output_fasta, 'w', encoding='utf-8') as f:
            f.write(fasta_content)
        result['output_fasta'] = output_fasta

    return result


def _batch_optimize_skill(
    sequences_file: str = '',
    sequences: list = None,
    method: str = 'dnachisel',
    output_fasta: str = '',
) -> dict:
    """Batch codon optimization from FASTA or sequence list."""
    from zhuangzhou.codon_opt import batch_optimize

    seq_list = []
    if sequences_file:
        # Parse FASTA
        current_name = ''
        current_seq = ''
        with open(sequences_file, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_name and current_seq:
                        seq_list.append({'name': current_name, 'protein': current_seq})
                    current_name = line[1:].split()[0]
                    current_seq = ''
                elif line:
                    current_seq += line
            if current_name and current_seq:
                seq_list.append({'name': current_name, 'protein': current_seq})
    elif sequences:
        seq_list = sequences

    if not seq_list:
        return {'error': 'No sequences provided', 'hint': 'Provide sequences_file (FASTA) or sequences list'}

    results = batch_optimize(seq_list, method=method)

    if output_fasta:
        lines = []
        for r in results:
            if 'error' not in r and 'dna_sequence' in r:
                dna = r['dna_sequence']
                name = r.get('name', 'seq')
                lines.append(f">{name} | K.phaffii optimized | {len(dna)}nt")
                for i in range(0, len(dna), 60):
                    lines.append(dna[i:i+60])
        with open(output_fasta, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        for r in results:
            r['output_fasta'] = output_fasta

    return {
        'results': results,
        'total': len(results),
        'succeeded': sum(1 for r in results if 'error' not in r),
    }


def _analyze_codons_skill(dna_seq: str = '', dna_file: str = '',
                          sequence: str = '', **kw) -> dict:
    """Analyze codon usage of a DNA sequence.

    Accepts dna_seq (canonical) or sequence (alias) to avoid LLM param-name errors.
    """
    from zhuangzhou.codon_opt import analyze_codons

    if dna_file:
        with open(dna_file, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        seq = ''.join(l.strip() for l in lines if not l.startswith('>'))
    elif dna_seq:
        seq = dna_seq
    elif sequence:
        seq = sequence
    else:
        return {'error': 'No DNA sequence provided', 'hint': 'Provide dna_seq or dna_file'}

    return analyze_codons(seq)


def _compare_codons_skill(original_dna: str, optimized_dna: str) -> dict:
    """Compare original vs optimized DNA."""
    from zhuangzhou.codon_opt import compare_sequences
    return compare_sequences(original_dna, optimized_dna)


def register_skills(registry):
    """Register codon optimization skills."""
    registry.register(Skill(
        name='codon_optimize',
        description='Optimize codons for K. phaffii (Pichia pastoris) heterologous expression. '
                    'Methods: dnachisel (CAI+constraints), frequency (simple), pichia_clm (deep learning).',
        category='sequence',
        fn=_optimize_codons_skill,
        inputs=[
            SkillParam('protein_seq', 'string', 'Amino acid sequence (1-letter code)', required=True),
            SkillParam('method', 'string', "Backend: 'dnachisel', 'frequency', or 'pichia_clm'", required=False, default='dnachisel'),
            SkillParam('output_fasta', 'path', 'Output FASTA path (optional)', required=False, is_output_path=True),
            SkillParam('name', 'string', 'Sequence name for FASTA header', required=False, default='optimized'),
        ],
        outputs=[
            SkillParam('dna_sequence', 'string', 'Optimized DNA sequence'),
            SkillParam('method', 'string', 'Method used'),
            SkillParam('length_nt', 'int', 'DNA length in nucleotides'),
        ],
    ))

    registry.register(Skill(
        name='batch_codon_optimize',
        description='Batch codon optimization for multiple proteins targeting K. phaffii.',
        category='sequence',
        fn=_batch_optimize_skill,
        inputs=[
            SkillParam('sequences_file', 'path', 'Input FASTA file with protein sequences', required=False),
            SkillParam('sequences', 'list', 'List of {name, protein} dicts', required=False),
            SkillParam('method', 'string', 'Backend method', required=False, default='dnachisel'),
            SkillParam('output_fasta', 'path', 'Output FASTA path', required=False),
        ],
        outputs=[
            SkillParam('results', 'list', 'List of optimization results'),
            SkillParam('total', 'int', 'Total sequences processed'),
        ],
    ))

    registry.register(Skill(
        name='analyze_codon_usage',
        description='Analyze codon usage of a DNA sequence: CAI, GC%, codon frequency distribution.',
        category='sequence',
        fn=_analyze_codons_skill,
        inputs=[
            SkillParam('dna_seq', 'string', 'DNA sequence to analyze', required=False),
            SkillParam('dna_file', 'path', 'DNA FASTA file to analyze', required=False),
        ],
        outputs=[
            SkillParam('cai', 'float', 'Codon Adaptation Index for K. phaffii'),
            SkillParam('gc_content', 'float', 'GC content fraction'),
        ],
    ))

    registry.register(Skill(
        name='compare_codon_usage',
        description='Compare original vs optimized DNA: CAI improvement, GC shift, codons changed.',
        category='sequence',
        fn=_compare_codons_skill,
        inputs=[
            SkillParam('original_dna', 'string', 'Original DNA sequence', required=True),
            SkillParam('optimized_dna', 'string', 'Optimized DNA sequence', required=True),
        ],
        outputs=[
            SkillParam('cai_improvement', 'float', 'CAI score improvement'),
            SkillParam('protein_match', 'bool', 'Whether protein sequences match'),
        ],
    ))

"""
Zhuangzhou Fusion Model
========================
Integrates PLM evolutionary scores with first-principles chemical reasoning.

The key insight: PLMs and chemical intuition have DIFFERENT failure modes.
- PLMs fail at conserved active sites (biased toward consensus)
- Chemical intuition fails at complex epistatic/evolutionary effects

Zhuangzhou uses ADAPTIVE WEIGHTING based on position context:
  - Conserved active site: 80% chemical, 20% PLM
  - Non-conserved distal: 20% chemical, 80% PLM
  - Everything else: 50/50 with confidence modulation
"""

import math
from typing import Dict, List, Optional, Tuple

from .chemical_intuition import (
    analyze_mutation_chemistry,
    classify_conservation_sensitivity,
    suggest_mutations_first_principles,
)
from .structural_context import (
    build_residue_context,
    generate_natural_language_description,
)
from .literature_rag import build_literature_context


def compute_adaptive_weights(context: dict, has_docking: bool = False) -> Tuple[float, ...]:
    """
    Compute adaptive weights for scoring dimensions.

    Returns:
        If has_docking: (w_chem, w_plm, w_dock) summing to 1.0
        Else: (w_chem, w_plm) summing to 1.0
    """
    sensitivity = classify_conservation_sensitivity(
        context.get('position', 0),
        context.get('wt_aa', 'A'),
        context,
    )

    if not has_docking:
        # Original 2D weights
        if sensitivity == 'first_principles':
            return (0.80, 0.20)
        elif sensitivity == 'plm_trusted':
            return (0.20, 0.80)
        else:
            dist = context.get('distance_to_site') or 15
            if dist < 8:
                return (0.65, 0.35)
            elif dist < 15:
                return (0.45, 0.55)
            else:
                return (0.30, 0.70)

    # 3D weights: (chem, plm, dock)
    if sensitivity == 'first_principles':
        return (0.60, 0.10, 0.30)
    elif sensitivity == 'plm_trusted':
        return (0.15, 0.55, 0.30)
    else:
        dist = context.get('distance_to_site') or 15
        if dist < 8:
            return (0.40, 0.20, 0.40)
        elif dist < 15:
            return (0.30, 0.35, 0.35)
        else:
            return (0.15, 0.55, 0.30)


def normalize_score(score: float, min_val: float, max_val: float) -> float:
    """Normalize to 0-1 range."""
    if max_val == min_val:
        return 0.5
    return max(0, min(1, (score - min_val) / (max_val - min_val)))


def fuse_scores(
    mutation: dict,
    context: dict,
    plm_score: float,
    plm_score_range: Tuple[float, float] = (-2, 5),
    chemical_score_range: Tuple[float, float] = (-5, 5),
    docking_score: float = None,
    docking_score_range: Tuple[float, float] = (-12, 0),
    literature_context: dict = None,
) -> dict:
    """
    Fuse PLM, chemical intuition, and optional docking scores.

    Args:
        mutation: {'position': int, 'wt': str, 'mut': str, 'mutation': str}
        context: structural context dict
        plm_score: ESM-2 delta log-likelihood
        plm_score_range: (min, max) of PLM scores for normalization
        chemical_score_range: (min, max) of chemical scores
        docking_score: optional binding affinity (kcal/mol, more negative = better)
        docking_score_range: (min, max) for docking normalization
        literature_context: optional literature RAG results

    Returns:
        dict with fused score, component scores, reasoning
    """
    # Get chemical analysis
    chem_analysis = analyze_mutation_chemistry(
        mutation['wt'], mutation['mut'], context
    )
    chem_score = chem_analysis['chemical_score']

    has_docking = docking_score is not None
    weights = compute_adaptive_weights(context, has_docking=has_docking)

    # Normalize scores to 0-1
    plm_norm = normalize_score(plm_score, *plm_score_range)
    chem_norm = normalize_score(chem_score, *chemical_score_range)
    dock_norm = 0.5  # neutral default

    if has_docking:
        # Docking: more negative = better, so invert
        dock_norm = normalize_score(-docking_score, -docking_score_range[1], -docking_score_range[0])

    # Literature bonus
    lit_bonus = 0.0
    if literature_context:
        reasoning = literature_context.get('literature_reasoning', '')
        if reasoning:
            lit_bonus = 0.1
        known_mutagenesis = literature_context.get('known_mutagenesis')
        if known_mutagenesis:
            desc = known_mutagenesis.get('description', '').lower()
            if 'increase' in desc or 'enhance' in desc or 'improve' in desc:
                lit_bonus = 0.2
            elif 'decrease' in desc or 'abolish' in desc or 'loss' in desc:
                lit_bonus = -0.3

    # Compute fused score
    if has_docking:
        w_chem, w_plm, w_dock = weights
        fused = w_chem * chem_norm + w_plm * plm_norm + w_dock * dock_norm + lit_bonus
    else:
        w_chem, w_plm = weights
        w_dock = 0.0
        fused = w_chem * chem_norm + w_plm * plm_norm + lit_bonus
    fused = max(0, min(1, fused))

    # Build reasoning narrative
    reasoning_parts = []

    # Weight explanation
    sensitivity = classify_conservation_sensitivity(
        context.get('position', 0), mutation['wt'], context)
    if sensitivity == 'first_principles':
        reasoning_parts.append(
            'CONSERVED POSITION: Chemical intuition weighted 80% (PLM biased toward consensus)')
    elif sensitivity == 'plm_trusted':
        reasoning_parts.append(
            'NON-CONSERVED DISTAL: PLM score weighted 80% (evolutionary signal reliable here)')
    else:
        reasoning_parts.append(
            'HYBRID WEIGHTING: Chemical {:.0f}% / PLM {:.0f}%'.format(w_chem * 100, w_plm * 100))

    # Chemical reasoning
    if chem_analysis['opportunities']:
        reasoning_parts.append('Chemical: ' + '; '.join(chem_analysis['opportunities'][:2]))
    if chem_analysis['warnings']:
        reasoning_parts.append('Warning: ' + '; '.join(chem_analysis['warnings'][:2]))

    # PLM interpretation
    if plm_score > 2:
        reasoning_parts.append('PLM: Strong evolutionary signal (DLL={:.2f})'.format(plm_score))
    elif plm_score > 0:
        reasoning_parts.append('PLM: Mild positive signal (DLL={:.2f})'.format(plm_score))
    else:
        reasoning_parts.append('PLM: Negative/neutral (DLL={:.2f})'.format(plm_score))

    # Literature
    if literature_context and literature_context.get('literature_reasoning'):
        reasoning_parts.append('Literature: ' + literature_context['literature_reasoning'][:100])

    return {
        **mutation,
        'zhuangzhou_score': round(fused, 4),
        'chemical_score': chem_score,
        'plm_score': plm_score,
        'docking_score': docking_score,
        'weight_chemical': w_chem,
        'weight_plm': w_plm,
        'weight_docking': w_dock,
        'literature_bonus': lit_bonus,
        'recommendation': chem_analysis['recommendation'],
        'confidence': chem_analysis['confidence'],
        'reasoning': ' | '.join(reasoning_parts),
        'chemical_analysis': chem_analysis,
    }


def run_zhuangzhou_analysis(
    mutations: List[dict],
    contexts: List[dict],
    binding_site: set,
    enzyme_class: str = '',
    uniprot_id: str = '',
    sequence: str = '',
    conserved_positions: set = None,
) -> dict:
    """
    Run the full Zhuangzhou fusion analysis on all mutations.

    This is the main entry point.

    Args:
        mutations: list of {'position', 'wt', 'mut', 'mutation', 'delta_ll', ...}
        contexts: list of structural contexts (from structural_context.py)
        binding_site: set of binding site residue numbers
        enzyme_class: e.g. 'oxidoreductase', 'hydrolase'
        uniprot_id: for literature RAG
        sequence: protein sequence
        conserved_positions: known conserved positions

    Returns:
        dict with 'ranked_mutations', 'active_site_analysis', 'suggestions',
        'report_narrative'
    """
    conserved = conserved_positions or set()

    # Build context map
    context_map = {}
    for ctx in contexts:
        pos = ctx.get('position') or ctx.get('context', {}).get('position')
        if pos:
            context_map[pos] = ctx.get('context', ctx)

    # Score all mutations
    all_plm = [m.get('delta_ll', m.get('consensus_score', 0)) for m in mutations]
    plm_range = (min(all_plm) if all_plm else -2, max(all_plm) if all_plm else 5)

    scored = []
    active_site_analysis = []

    for mut in mutations:
        pos = mut['position']
        ctx = context_map.get(pos, {
            'role': 'unknown',
            'distance_to_site': None,
            'in_conserved_motif': pos in conserved,
        })

        # Ensure conservation flag is set
        if pos in conserved:
            ctx['in_conserved_motif'] = True
        if pos in binding_site:
            ctx['role'] = 'active_site'

        plm_score = mut.get('delta_ll', mut.get('consensus_score', 0))

        # Literature context
        lit_ctx = build_literature_context(
            pos, mut['wt'], mut['mut'],
            enzyme_class=enzyme_class,
            uniprot_id=uniprot_id,
        )

        fused = fuse_scores(mut, ctx, plm_score, plm_range, literature_context=lit_ctx)
        scored.append(fused)

        if ctx.get('role') == 'active_site':
            active_site_analysis.append(fused)

    # Sort by fused score
    scored.sort(key=lambda x: -x['zhuangzhou_score'])

    # Generate first-principles suggestions for active site positions
    suggestions = []
    active_positions = {m['position'] for m in mutations if m['position'] in binding_site}
    for pos in active_positions:
        wt = next((m['wt'] for m in mutations if m['position'] == pos), None)
        if wt:
            ctx = context_map.get(pos, {'role': 'active_site', 'in_conserved_motif': True})
            ctx['role'] = 'active_site'
            ctx['in_conserved_motif'] = True
            first_principles = suggest_mutations_first_principles(pos, wt, ctx)
            suggestions.extend(first_principles[:5])  # Top 5 per position

    # Build narrative
    narrative = _build_narrative(scored, active_site_analysis, suggestions, enzyme_class)

    return {
        'ranked_mutations': scored,
        'active_site_analysis': active_site_analysis,
        'first_principles_suggestions': suggestions,
        'report_narrative': narrative,
        'scoring_method': 'Zhuangzhou adaptive fusion (chemical + PLM)',
        'n_conserved_overrides': sum(
            1 for s in scored if s['weight_chemical'] > 0.6
        ),
    }


def _build_narrative(scored, active_site, suggestions, enzyme_class):
    """Build the Shi Yigong-style analytical narrative."""
    lines = []
    lines.append('# Zhuangzhou Analysis Report')
    lines.append('')
    lines.append('## Methodology')
    lines.append('Adaptive fusion of ESM-2 evolutionary scores and first-principles ')
    lines.append('chemical reasoning. Conserved/active-site positions weighted toward ')
    lines.append('chemical intuition (PLMs encode evolutionary consensus, not maximal activity).')
    lines.append('')

    if enzyme_class:
        lines.append('Enzyme class: **{}**'.format(enzyme_class))
        lines.append('')

    # Active site deep analysis
    if active_site:
        lines.append('## Active Site Mutations (First-Principles Priority)')
        lines.append('')
        for m in active_site:
            lines.append('### {} (Zhuangzhou: {:.3f})'.format(m['mutation'], m['zhuangzhou_score']))
            lines.append('')
            lines.append('- Chemical score: {:+.2f} | PLM score: {:.3f}'.format(
                m['chemical_score'], m['plm_score']))
            lines.append('- Weighting: Chemical {:.0f}% / PLM {:.0f}%'.format(
                m['weight_chemical'] * 100, m['weight_plm'] * 100))
            lines.append('- {}'.format(m['recommendation']))
            lines.append('- Reasoning: {}'.format(m['reasoning']))
            lines.append('')

    # First-principles suggestions
    if suggestions:
        lines.append('## First-Principles Mutation Suggestions')
        lines.append('(Positions where chemical intuition overrides PLM)')
        lines.append('')
        lines.append('| Position | Mutation | Chemical Score | Reasoning |')
        lines.append('|:--------:|----------|:-------------:|-----------|')
        for s in sorted(suggestions, key=lambda x: -x['chemical_score'])[:10]:
            reason = '; '.join(s.get('key_reasoning', []))[:60]
            lines.append('| {} | {} | {:+.2f} | {} |'.format(
                s['position'], s['mutation'], s['chemical_score'], reason))
        lines.append('')

    # Overall ranking
    lines.append('## Overall Ranking (Zhuangzhou Fusion)')
    lines.append('')
    lines.append('| Rank | Mutation | Zhuangzhou | Chem | PLM | Weights | Recommendation |')
    lines.append('|:----:|----------|:----------:|:----:|:---:|:-------:|---------------|')
    for i, m in enumerate(scored[:15], 1):
        lines.append('| {} | {} | {:.3f} | {:+.2f} | {:.3f} | C{:.0f}/P{:.0f} | {} |'.format(
            i, m['mutation'], m['zhuangzhou_score'],
            m['chemical_score'], m['plm_score'],
            m['weight_chemical'] * 100, m['weight_plm'] * 100,
            m['recommendation'][:30]))
    lines.append('')

    return '\n'.join(lines)

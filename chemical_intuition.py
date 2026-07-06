"""
Chemical Intuition Engine
=========================
First-principles reasoning about amino acid mutations.

This module encodes the kind of chemical intuition a structural biologist
like Shi Yigong (施一公) uses when examining a protein structure:
  - Steric compatibility in the local environment
  - Electronic effects (charge, polarity, aromaticity)
  - Hydrogen bonding networks
  - Hydrophobic packing efficiency
  - Backbone conformational preferences (Ramachandran)
  - Catalytic mechanism compatibility

For conserved/active-site positions, this OVERRIDES PLM predictions
because evolutionary conservation reflects organism-level fitness,
not maximal enzymatic activity.
"""

import math
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════
# Amino Acid Property Database (First-Principles)
# ═══════════════════════════════════════════════

AA_PROPERTIES = {
    'G': {'name': 'Glycine', 'mw': 57.05, 'volume': 60.1, 'pI': 5.97,
           'charge_pH7': 0, 'hydrophobicity': -0.4, 'aromaticity': False,
           'hbond_donor': 1, 'hbond_acceptor': 1, 'rotatable_bonds': 0,
           'backbone_flexibility': 'high',  # No sidechain, phi/psi freedom
           'special': 'smallest, allows tight turns, no chirality'},
    'A': {'name': 'Alanine', 'mw': 71.08, 'volume': 88.6, 'pI': 6.00,
           'charge_pH7': 0, 'hydrophobicity': 1.8, 'aromaticity': False,
           'hbond_donor': 1, 'hbond_acceptor': 1, 'rotatable_bonds': 0,
           'backbone_flexibility': 'moderate',
           'special': 'helix-former, hydrophobic reference'},
    'V': {'name': 'Valine', 'mw': 99.13, 'volume': 140.0, 'pI': 5.96,
           'charge_pH7': 0, 'hydrophobicity': 4.2, 'aromaticity': False,
           'hbond_donor': 1, 'hbond_acceptor': 1, 'rotatable_bonds': 1,
           'backbone_flexibility': 'low',  # Beta-branched
           'special': 'beta-branched, restricts backbone, sheet-former'},
    'L': {'name': 'Leucine', 'mw': 113.16, 'volume': 166.7, 'pI': 5.98,
           'charge_pH7': 0, 'hydrophobicity': 3.8, 'aromaticity': False,
           'hbond_donor': 1, 'hbond_acceptor': 1, 'rotatable_bonds': 2,
           'backbone_flexibility': 'moderate',
           'special': 'flexible hydrophobic, helix-former'},
    'I': {'name': 'Isoleucine', 'mw': 113.16, 'volume': 166.7, 'pI': 6.02,
           'charge_pH7': 0, 'hydrophobicity': 4.5, 'aromaticity': False,
           'hbond_donor': 1, 'hbond_acceptor': 1, 'rotatable_bonds': 1,
           'backbone_flexibility': 'low',  # Beta-branched
           'special': 'beta-branched like Val, most hydrophobic'},
    'P': {'name': 'Proline', 'mw': 97.12, 'volume': 112.7, 'pI': 6.30,
           'charge_pH7': 0, 'hydrophobicity': -1.6, 'aromaticity': False,
           'hbond_donor': 0, 'hbond_acceptor': 1, 'rotatable_bonds': 0,
           'backbone_flexibility': 'rigid',  # Cyclic, fixed phi
           'special': 'helix-breaker, cis-trans isomerism, rigidifies loops'},
    'F': {'name': 'Phenylalanine', 'mw': 147.18, 'volume': 189.9, 'pI': 5.48,
           'charge_pH7': 0, 'hydrophobicity': 2.8, 'aromaticity': True,
           'hbond_donor': 1, 'hbond_acceptor': 1, 'rotatable_bonds': 2,
           'backbone_flexibility': 'moderate',
           'special': 'pi-stacking, cation-pi, no H-bond on ring'},
    'W': {'name': 'Tryptophan', 'mw': 186.21, 'volume': 227.8, 'pI': 5.89,
           'charge_pH7': 0, 'hydrophobicity': -0.9, 'aromaticity': True,
           'hbond_donor': 2, 'hbond_acceptor': 1, 'rotatable_bonds': 2,
           'backbone_flexibility': 'low',  # Bulky
           'special': 'largest AA, strong pi-stacking, indole NH H-bond donor'},
    'M': {'name': 'Methionine', 'mw': 131.20, 'volume': 162.9, 'pI': 5.74,
           'charge_pH7': 0, 'hydrophobicity': 1.9, 'aromaticity': False,
           'hbond_donor': 1, 'hbond_acceptor': 1, 'rotatable_bonds': 3,
           'backbone_flexibility': 'moderate',
           'special': 'flexible sulfur, S-pi interactions, oxidation-sensitive'},
    'S': {'name': 'Serine', 'mw': 87.08, 'volume': 89.0, 'pI': 5.68,
           'charge_pH7': 0, 'hydrophobicity': -0.8, 'aromaticity': False,
           'hbond_donor': 2, 'hbond_acceptor': 2, 'rotatable_bonds': 1,
           'backbone_flexibility': 'moderate',
           'special': 'nucleophile (catalytic triad), phosphorylation site'},
    'T': {'name': 'Threonine', 'mw': 101.10, 'volume': 116.1, 'pI': 5.60,
           'charge_pH7': 0, 'hydrophobicity': -0.7, 'aromaticity': False,
           'hbond_donor': 2, 'hbond_acceptor': 2, 'rotatable_bonds': 1,
           'backbone_flexibility': 'low',  # Beta-branched
           'special': 'beta-branched, weaker nucleophile than Ser, sheet-former'},
    'C': {'name': 'Cysteine', 'mw': 103.14, 'volume': 108.5, 'pI': 5.07,
           'charge_pH7': 0, 'hydrophobicity': 2.5, 'aromaticity': False,
           'hbond_donor': 2, 'hbond_acceptor': 2, 'rotatable_bonds': 1,
           'backbone_flexibility': 'moderate',
           'special': 'disulfide bonds, metal coordination, nucleophile, redox-active'},
    'Y': {'name': 'Tyrosine', 'mw': 163.18, 'volume': 193.6, 'pI': 5.66,
           'charge_pH7': 0, 'hydrophobicity': -1.3, 'aromaticity': True,
           'hbond_donor': 2, 'hbond_acceptor': 2, 'rotatable_bonds': 2,
           'backbone_flexibility': 'moderate',
           'special': 'H-bond via OH, pi-stacking, phosphorylation site, pKa~10'},
    'H': {'name': 'Histidine', 'mw': 137.14, 'volume': 153.2, 'pI': 7.59,
           'charge_pH7': 0.1, 'hydrophobicity': -3.2, 'aromaticity': True,
           'hbond_donor': 2, 'hbond_acceptor': 2, 'rotatable_bonds': 2,
           'backbone_flexibility': 'moderate',
           'special': 'proton shuttle (pKa~6), metal coordination, catalytic residue'},
    'D': {'name': 'Aspartate', 'mw': 115.09, 'volume': 111.1, 'pI': 2.77,
           'charge_pH7': -1, 'hydrophobicity': -3.5, 'aromaticity': False,
           'hbond_donor': 1, 'hbond_acceptor': 4, 'rotatable_bonds': 2,
           'backbone_flexibility': 'moderate',
           'special': 'negative charge, metal coordination, catalytic (acid/base)'},
    'E': {'name': 'Glutamate', 'mw': 129.12, 'volume': 138.4, 'pI': 3.22,
           'charge_pH7': -1, 'hydrophobicity': -3.5, 'aromaticity': False,
           'hbond_donor': 1, 'hbond_acceptor': 4, 'rotatable_bonds': 3,
           'backbone_flexibility': 'moderate',
           'special': 'negative charge, metal coordination, catalytic, longer than Asp'},
    'N': {'name': 'Asparagine', 'mw': 114.10, 'volume': 114.1, 'pI': 5.41,
           'charge_pH7': 0, 'hydrophobicity': -3.5, 'aromaticity': False,
           'hbond_donor': 2, 'hbond_acceptor': 3, 'rotatable_bonds': 2,
           'backbone_flexibility': 'moderate',
           'special': 'H-bond rich, glycosylation site, deamidation-prone'},
    'Q': {'name': 'Glutamine', 'mw': 128.13, 'volume': 143.8, 'pI': 5.65,
           'charge_pH7': 0, 'hydrophobicity': -3.5, 'aromaticity': False,
           'hbond_donor': 2, 'hbond_acceptor': 3, 'rotatable_bonds': 3,
           'backbone_flexibility': 'moderate',
           'special': 'H-bond rich, longer than Asn, glutaminase substrate'},
    'K': {'name': 'Lysine', 'mw': 128.17, 'volume': 168.6, 'pI': 9.74,
           'charge_pH7': 1, 'hydrophobicity': -3.9, 'aromaticity': False,
           'hbond_donor': 3, 'hbond_acceptor': 2, 'rotatable_bonds': 4,
           'backbone_flexibility': 'moderate',
           'special': 'positive charge, salt bridges, flexible chain, ubiquitination'},
    'R': {'name': 'Arginine', 'mw': 156.19, 'volume': 173.4, 'pI': 10.76,
           'charge_pH7': 1, 'hydrophobicity': -4.5, 'aromaticity': False,
           'hbond_donor': 5, 'hbond_acceptor': 2, 'rotatable_bonds': 4,
           'backbone_flexibility': 'moderate',
           'special': 'guanidinium: 5 H-bond donors, cation-pi, always charged at pH7'},
}


# ═══════════════════════════════════════════════
# Chemical Reasoning Rules
# ═══════════════════════════════════════════════

def analyze_mutation_chemistry(wt_aa: str, mut_aa: str, context: dict) -> dict:
    """
    First-principles chemical analysis of a single mutation.

    Args:
        wt_aa: Wild-type amino acid (1-letter)
        mut_aa: Mutant amino acid (1-letter)
        context: Dict with structural context:
            - 'role': structural role (active_site, buried_core, etc.)
            - 'secondary_structure': helix/sheet/loop
            - 'interactions': list of PLIP interactions at this position
            - 'neighbors': list of neighboring residue types
            - 'distance_to_site': distance to active site center
            - 'sasa': solvent accessible surface area fraction
            - 'in_conserved_motif': bool

    Returns:
        dict with chemical analysis and reasoning
    """
    wt = AA_PROPERTIES.get(wt_aa, {})
    mt = AA_PROPERTIES.get(mut_aa, {})

    if not wt or not mt:
        return {'error': 'Unknown amino acid', 'score': 0}

    role = context.get('role', 'unknown')
    ss = context.get('secondary_structure', 'loop')
    interactions = context.get('interactions', [])
    is_conserved = context.get('in_conserved_motif', False)

    analysis = {
        'mutation': '{}{}'.format(wt_aa, mut_aa),
        'wt_name': wt['name'],
        'mut_name': mt['name'],
        'reasoning': [],
        'warnings': [],
        'opportunities': [],
        'chemical_score': 0.0,  # -5 to +5 scale
        'confidence': 'low',
    }

    score = 0.0
    reasons = analysis['reasoning']
    warns = analysis['warnings']
    opps = analysis['opportunities']

    # ── Volume Change Analysis ──
    dv = mt['volume'] - wt['volume']
    dv_pct = dv / wt['volume'] * 100

    if role in ('active_site', 'ligand_interacting'):
        # Active site: volume changes reshape the pocket
        if abs(dv_pct) > 30:
            warns.append(
                'Large volume change ({:+.0f}%, {:+.1f} A3) will reshape binding pocket'.format(
                    dv_pct, dv))
            score -= 1.5
        elif abs(dv_pct) > 15:
            reasons.append(
                'Moderate volume change ({:+.0f}%) may fine-tune substrate positioning'.format(dv_pct))
            # Could be good or bad - context-dependent
        else:
            opps.append('Conservative volume change ({:+.0f}%) preserves pocket geometry'.format(dv_pct))
            score += 0.3
    elif role == 'buried_core':
        if dv > 30:
            warns.append('Expanding buried residue by {:.0f} A3 creates cavity → destabilizing'.format(dv))
            score -= 1.0
        elif dv < -30:
            warns.append('Shrinking buried residue by {:.0f} A3 creates void → destabilizing'.format(-dv))
            score -= 0.8

    # ── Charge Change Analysis ──
    dq = mt['charge_pH7'] - wt['charge_pH7']
    if abs(dq) > 0.5:
        if role == 'active_site':
            reasons.append(
                'Charge change {:+.0f} at active site directly affects electrostatic catalysis'.format(dq))
            # This is a strong first-principles signal: changing the electric field
            if dq > 0:
                opps.append('Adding positive charge may stabilize oxyanion/TS if mechanism requires it')
            else:
                opps.append('Adding negative charge may serve as general base or stabilize cation')
            score += 0.5  # Charge changes at active site are high-information
        elif role == 'surface':
            reasons.append('Surface charge change {:+.0f}: affects solubility/pI'.format(dq))
        elif role == 'buried_core':
            warns.append('Burying a charge ({:+.0f}) is highly destabilizing (~3-5 kcal/mol)'.format(dq))
            score -= 3.0

    # ── H-bonding Analysis ──
    d_hbd = mt['hbond_donor'] - wt['hbond_donor']
    d_hba = mt['hbond_acceptor'] - wt['hbond_acceptor']

    if interactions:
        hbond_interactions = [i for i in interactions if 'hbond' in i.get('type', '').lower()]
        if hbond_interactions and (d_hbd < 0 or d_hba < 0):
            warns.append(
                'CRITICAL: Position has {} H-bonds but mutation loses {} donors/{} acceptors'.format(
                    len(hbond_interactions), max(0, -d_hbd), max(0, -d_hba)))
            score -= 2.0  # Breaking existing H-bonds is very costly

    if d_hbd > 0 and role == 'active_site':
        opps.append('Additional H-bond donor may stabilize substrate/TS')
        score += 0.5

    # ── Aromaticity Analysis ──
    if wt['aromaticity'] and not mt['aromaticity']:
        pi_interactions = [i for i in interactions
                         if 'pi' in i.get('type', '').lower() or 'stack' in i.get('type', '').lower()]
        if pi_interactions:
            warns.append(
                'Losing aromatic ring disrupts {} pi-stacking interaction(s)'.format(len(pi_interactions)))
            score -= 1.5
        else:
            reasons.append('Aromatic → non-aromatic: removes pi-electron cloud')

    if not wt['aromaticity'] and mt['aromaticity']:
        if role in ('active_site', 'substrate_channel'):
            opps.append('Introducing aromatic ring may create new pi-stacking with substrate')
            score += 0.5

    # ── Backbone Conformational Analysis ──
    if mt['backbone_flexibility'] == 'rigid' and ss == 'helix':
        warns.append('Proline in helix is a helix-breaker (except at N-cap position)')
        score -= 2.0
    elif mut_aa == 'G' and ss == 'helix':
        warns.append('Glycine in helix: too flexible, entropic penalty')
        score -= 0.5
    elif wt['backbone_flexibility'] == 'high' and mt['backbone_flexibility'] == 'low':
        if role == 'flexible_loop':
            opps.append('Rigidifying a flexible loop may reduce entropic cost of binding')
            score += 0.5

    # ── Hydrophobicity Transfer Analysis ──
    dh = mt['hydrophobicity'] - wt['hydrophobicity']
    if role == 'buried_core' and dh > 0:
        opps.append('Increasing hydrophobicity in core: stabilizing ({:+.1f} Kyte-Doolittle)'.format(dh))
        score += 0.3 * min(dh, 3.0)
    elif role == 'buried_core' and dh < -2:
        warns.append('Decreasing hydrophobicity in core by {:.1f}: destabilizing polarity mismatch'.format(-dh))
        score -= 0.5 * min(-dh / 2.0, 2.0)
    elif role == 'surface' and dh > 2 and mt['hydrophobicity'] > 1.5:
        warns.append('Hydrophobic residue on surface may promote aggregation')
        score -= 0.5
    elif role == 'surface' and dh < -1:
        opps.append('Increasing polarity on surface: may improve solubility ({:+.1f})'.format(dh))
        score += 0.2

    # ── Glycine-specific rules ──
    if wt_aa == 'G':
        # Gly has unique backbone flexibility; replacing it restricts phi/psi
        if role == 'buried_core':
            warns.append('G->X: Glycine in buried core often structurally required for tight packing or turn')
            score -= 0.5
            if mt['volume'] > 150:
                warns.append('G->large AA ({:.0f} A3): massive volume increase in core'.format(mt['volume']))
                score -= 1.0
        elif role == 'surface' and ss == 'loop':
            reasons.append('G->X in surface loop: restricts backbone but may reduce conformational entropy')
    elif mut_aa == 'G':
        # Introducing Gly: gain flexibility, lose sidechain contacts
        if role == 'buried_core':
            warns.append('X->G: cavity-creating mutation, loses sidechain packing contacts')
            score -= 0.8

    # ── Deamidation risk (Asn/Gln) ──
    if wt_aa == 'N':
        if mut_aa in ('A', 'G', 'S', 'D'):
            opps.append('Removing Asn: eliminates deamidation risk (N->D/isoD degradation)')
            score += 0.4
    elif mut_aa == 'N' and role == 'surface':
        warns.append('Introducing Asn on surface: deamidation-prone at elevated temperature')
        score -= 0.3

    # ── Polarity mismatch in buried core ──
    if role == 'buried_core':
        wt_polar = (wt['charge_pH7'] != 0 or wt['hbond_donor'] > 1 or wt['hbond_acceptor'] > 1)
        mt_polar = (mt['charge_pH7'] != 0 or mt['hbond_donor'] > 1 or mt['hbond_acceptor'] > 1)
        wt_hydrophobic = wt['hydrophobicity'] > 1.5
        mt_hydrophobic = mt['hydrophobicity'] > 1.5
        if wt_hydrophobic and mt_polar and not wt_polar:
            warns.append('Hydrophobic->polar in buried core: unsatisfied polar groups destabilize')
            score -= 1.0
        elif wt_polar and mt_hydrophobic and not wt_hydrophobic:
            opps.append('Polar->hydrophobic in buried core: may improve packing if H-bonds not critical')
            score += 0.3

    # ── Proline introduction (non-helix contexts) ──
    if mut_aa == 'P' and ss != 'helix':
        if role == 'buried_core':
            warns.append('Pro in buried core: rigid pyrrolidine ring constrains backbone, possible strain')
            score -= 0.8
        elif role == 'surface' and ss == 'loop':
            opps.append('Pro in surface loop: may rigidify and reduce conformational entropy')
            score += 0.3

    # ── Special Catalytic Residue Rules ──
    if is_conserved:
        # Conserved positions: PLMs are biased, use chemical logic
        analysis['confidence'] = 'high'

        if wt_aa in ('S', 'C', 'H', 'D', 'E', 'K') and role == 'active_site':
            # These are common catalytic residues
            catalytic_functions = {
                'S': 'nucleophile (Ser protease/hydrolase)',
                'C': 'nucleophile/redox center (Cys protease/oxidoreductase)',
                'H': 'proton shuttle/general acid-base (His-dependent catalysis)',
                'D': 'general acid-base/metal coordination',
                'E': 'general acid-base/metal coordination',
                'K': 'Schiff base formation/charge stabilization',
            }
            if wt_aa in catalytic_functions:
                warns.append(
                    'CONSERVED CATALYTIC: {} likely serves as {}. Mutation is HIGH RISK.'.format(
                        wt['name'], catalytic_functions[wt_aa]))
                score -= 3.0

                # BUT: specific substitutions may enhance
                if wt_aa == 'S' and mut_aa == 'C':
                    opps.append(
                        'S→C: thiol is a stronger nucleophile (pKa 8.3 vs 13). '
                        'May enhance catalysis if mechanism uses nucleophilic attack.')
                    score += 2.0
                elif wt_aa == 'H' and mut_aa == 'N':
                    opps.append(
                        'H→N: preserves H-bond capability but removes proton shuttle. '
                        'Useful if pH-independent catalysis desired.')
                elif wt_aa == 'D' and mut_aa == 'E':
                    opps.append(
                        'D→E: longer sidechain, same carboxylate. '
                        'May optimize reach to substrate without changing chemistry.')
                    score += 1.0
                elif wt_aa == 'E' and mut_aa == 'D':
                    opps.append(
                        'E→D: shorter reach, same chemistry. '
                        'Tighter positioning near substrate.')
                    score += 0.8

        elif wt_aa in ('F', 'W', 'Y') and role == 'active_site':
            warns.append(
                'CONSERVED AROMATIC: {} likely involved in substrate positioning/pi-stacking'.format(
                    wt['name']))
            score -= 1.5

            if wt_aa == 'Y' and mut_aa == 'F':
                opps.append(
                    'Y→F: removes H-bond from OH but preserves aromatic ring. '
                    'If OH not involved in catalysis, this reduces desolvation penalty.')
                score += 1.0
            elif wt_aa == 'F' and mut_aa == 'W':
                opps.append(
                    'F→W: larger aromatic surface, additional NH H-bond donor. '
                    'May enhance substrate stacking if space permits.')
                score += 0.5

    else:
        # Non-conserved: medium confidence, less risk
        analysis['confidence'] = 'medium'

    # ── Substrate Channel Specific ──
    if role == 'substrate_channel':
        if dv > 0:
            warns.append('Larger residue in channel narrows passage (may reduce k_on)')
            score -= 0.5
        elif dv < -15:
            opps.append('Smaller residue widens channel: may increase substrate access rate')
            score += 0.8

    # Final score
    analysis['chemical_score'] = round(max(-5, min(5, score)), 2)

    # Determine overall recommendation
    if score > 1.0:
        analysis['recommendation'] = 'FAVORABLE - chemical logic supports this mutation'
    elif score > -0.5:
        analysis['recommendation'] = 'NEUTRAL - no strong chemical argument for/against'
    elif score > -2.0:
        analysis['recommendation'] = 'CAUTION - some chemical concerns, verify experimentally'
    else:
        analysis['recommendation'] = 'UNFAVORABLE - strong chemical reasons against this mutation'

    return analysis


def suggest_mutations_first_principles(
    position: int,
    wt_aa: str,
    context: dict,
) -> List[dict]:
    """
    Suggest mutations based on first-principles chemical reasoning.

    This is the "Shi Yigong looking at the structure" function.
    It considers what a structural biologist would suggest based on:
    - The local chemical environment
    - The catalytic mechanism
    - Known structure-function relationships
    - Organic chemistry principles

    Returns list of suggested mutations ranked by chemical score.
    """
    role = context.get('role', 'unknown')
    suggestions = []

    for mut_aa in 'ACDEFGHIKLMNPQRSTVWY':
        if mut_aa == wt_aa:
            continue

        analysis = analyze_mutation_chemistry(wt_aa, mut_aa, context)
        if analysis.get('chemical_score', 0) > -1.0:  # Don't suggest terrible ones
            suggestions.append({
                'position': position,
                'wt': wt_aa,
                'mut': mut_aa,
                'mutation': '{}{}{}'.format(wt_aa, position, mut_aa),
                'chemical_score': analysis['chemical_score'],
                'recommendation': analysis['recommendation'],
                'key_reasoning': analysis['reasoning'][:2] + analysis['opportunities'][:2],
                'warnings': analysis['warnings'][:2],
                'confidence': analysis['confidence'],
            })

    suggestions.sort(key=lambda x: -x['chemical_score'])
    return suggestions


def classify_conservation_sensitivity(position: int, wt_aa: str, context: dict) -> str:
    """
    Determine whether a position should be reasoned about with
    first-principles (conserved) or PLM scores (non-conserved).

    Returns: 'first_principles', 'plm_trusted', or 'hybrid'
    """
    role = context.get('role', 'unknown')
    is_conserved = context.get('in_conserved_motif', False)
    distance = context.get('distance_to_site', 999)

    if is_conserved and role in ('active_site', 'cofactor_coordinating'):
        return 'first_principles'
    elif role == 'active_site' and distance < 5:
        return 'first_principles'
    elif role in ('surface', 'buried_core') and distance > 15:
        return 'plm_trusted'
    elif role == 'second_shell':
        return 'hybrid'
    else:
        return 'hybrid'

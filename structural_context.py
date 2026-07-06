"""
Structural Context Builder
==========================
Integrates structural analysis data (PLIP, fpocket, CAVER, B-factors)
into a rich context for each mutation position.

This feeds the Chemical Intuition Engine with the spatial/chemical
environment that a structural biologist would "see" when examining
the structure in PyMOL.
"""

import json
import math
import os
from typing import Dict, List, Optional, Set, Tuple


def parse_plip_interactions(plip_json_path: str) -> Dict[int, List[dict]]:
    """
    Parse PLIP interaction analysis into per-residue interaction maps.

    Returns: {residue_number: [{'type': 'hbond', 'partner': 'LIG', ...}, ...]}
    """
    if not plip_json_path or not os.path.exists(plip_json_path):
        return {}

    with open(plip_json_path, encoding='utf-8', errors='replace') as f:
        data = json.load(f)

    interactions_by_residue = {}

    for site in data.get('binding_sites', data.get('interactions', [])):
        for itype in ['hbonds', 'hydrophobic', 'pi_stacking', 'salt_bridges',
                      'pi_cation', 'halogen_bonds', 'metal_coordination']:
            for interaction in site.get(itype, []):
                resnum = interaction.get('resnr', interaction.get('resnum'))
                if resnum is None:
                    continue

                if resnum not in interactions_by_residue:
                    interactions_by_residue[resnum] = []

                interactions_by_residue[resnum].append({
                    'type': itype.rstrip('s'),  # hbonds -> hbond
                    'partner': interaction.get('restype_lig', interaction.get('partner', 'UNK')),
                    'distance': interaction.get('dist', interaction.get('distance')),
                    'resname': interaction.get('restype', ''),
                })

    return interactions_by_residue


def parse_fpocket_data(fpocket_json_path: str) -> Dict[str, Set[int]]:
    """
    Parse fpocket results into pocket residue sets.

    Returns: {'pocket_1': {res1, res2, ...}, 'pocket_2': {...}, ...}
    """
    if not fpocket_json_path or not os.path.exists(fpocket_json_path):
        return {}

    with open(fpocket_json_path, encoding='utf-8', errors='replace') as f:
        data = json.load(f)

    pockets = {}
    for pocket in data.get('pockets', []):
        pid = pocket.get('id', pocket.get('pocket_id', 'unknown'))
        residues = set(pocket.get('residues', []))
        pockets['pocket_{}'.format(pid)] = residues

    return pockets


def build_residue_context(
    pdb_path: str,
    position: int,
    wt_aa: str,
    binding_site: Set[int],
    plip_interactions: Dict[int, List[dict]] = None,
    fpocket_pockets: Dict[str, Set[int]] = None,
    channel_residues: Set[int] = None,
    conserved_positions: Set[int] = None,
    sequence: str = None,
) -> dict:
    """
    Build comprehensive structural context for a single residue position.

    This is what the "Shi Yigong Agent" sees when examining the structure.
    """
    context = {
        'position': position,
        'wt_aa': wt_aa,
        'role': 'unknown',
        'secondary_structure': 'loop',
        'interactions': [],
        'neighbors': [],
        'distance_to_site': None,
        'sasa': None,
        'in_conserved_motif': position in (conserved_positions or set()),
        'in_pocket': None,
        'in_channel': position in (channel_residues or set()),
        'local_environment': '',
    }

    # Role assignment
    if position in binding_site:
        context['role'] = 'active_site'
    elif position in (channel_residues or set()):
        context['role'] = 'substrate_channel'
    # Additional roles determined by caller

    # PLIP interactions
    if plip_interactions and position in plip_interactions:
        context['interactions'] = plip_interactions[position]

    # Pocket membership
    if fpocket_pockets:
        for pocket_name, residues in fpocket_pockets.items():
            if position in residues:
                context['in_pocket'] = pocket_name
                break

    # Sequence context (local neighbors)
    if sequence and 0 < position <= len(sequence):
        idx = position - 1  # 0-based
        start = max(0, idx - 3)
        end = min(len(sequence), idx + 4)
        local_seq = sequence[start:end]
        context['local_sequence'] = local_seq
        context['neighbors'] = list(local_seq)

    return context


def build_all_contexts(
    pdb_path: str,
    mutations: List[dict],
    binding_site: Set[int],
    plip_json: str = None,
    fpocket_json: str = None,
    channel_residues: Set[int] = None,
    conserved_positions: Set[int] = None,
    sequence: str = None,
) -> List[dict]:
    """
    Build structural contexts for all mutation positions.
    """
    plip_interactions = parse_plip_interactions(plip_json) if plip_json else {}
    fpocket_pockets = parse_fpocket_data(fpocket_json) if fpocket_json else {}

    contexts = []
    for mut in mutations:
        pos = mut['position']
        wt = mut.get('wt', '')

        ctx = build_residue_context(
            pdb_path=pdb_path,
            position=pos,
            wt_aa=wt,
            binding_site=binding_site,
            plip_interactions=plip_interactions,
            fpocket_pockets=fpocket_pockets,
            channel_residues=channel_residues,
            conserved_positions=conserved_positions,
            sequence=sequence,
        )

        contexts.append({**mut, 'context': ctx})

    return contexts


def generate_natural_language_description(context: dict) -> str:
    """
    Generate a natural language description of the structural environment.

    This is the "what Shi Yigong would say" when looking at this position.
    """
    pos = context.get('position', '?')
    wt = context.get('wt_aa', '?')
    role = context.get('role', 'unknown')
    interactions = context.get('interactions', [])
    ss = context.get('secondary_structure', 'loop')
    dist = context.get('distance_to_site')
    in_pocket = context.get('in_pocket')
    in_channel = context.get('in_channel', False)
    conserved = context.get('in_conserved_motif', False)
    local_seq = context.get('local_sequence', '')

    parts = []

    # Location
    if role == 'active_site':
        parts.append('Position {} ({}) is in the active site'.format(pos, wt))
    elif role == 'substrate_channel':
        parts.append('Position {} ({}) lines the substrate channel'.format(pos, wt))
    elif dist and dist < 10:
        parts.append('Position {} ({}) is near the active site ({:.1f}A)'.format(pos, wt, dist))
    else:
        parts.append('Position {} ({}) is distal from the active site'.format(pos, wt))
        if dist:
            parts[-1] += ' ({:.1f}A)'.format(dist)

    # Secondary structure
    if ss != 'loop':
        parts.append('located in a {}'.format(ss))

    # Conservation
    if conserved:
        parts.append('CONSERVED across homologs (high-confidence structural role)')

    # Interactions
    if interactions:
        itypes = set(i['type'] for i in interactions)
        parts.append('forming {} interaction(s): {}'.format(
            len(interactions), ', '.join(itypes)))

    # Pocket/channel
    if in_pocket:
        parts.append('within {}'.format(in_pocket))
    if in_channel:
        parts.append('lining a substrate access tunnel')

    # Local sequence
    if local_seq:
        parts.append('local sequence context: ...{}...'.format(local_seq))

    return '. '.join(parts) + '.'

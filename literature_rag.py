"""
Literature RAG for Enzyme Mutation Design
==========================================
Retrieves relevant mutation studies and catalytic mechanism information
from online databases and local knowledge bases.

Sources:
  - UniProt: active site annotations, known mutagenesis experiments
  - PDB: catalytic site atlas (CSA) annotations
  - Literature: mechanism-specific mutation knowledge
  - Local BeTaThink KG: if available
"""

import json
import os
import re
import urllib.request
import urllib.error
from typing import Dict, List, Optional


# ═══════════════════════════════════════════════
# Built-in Catalytic Mechanism Knowledge Base
# ═══════════════════════════════════════════════

CATALYTIC_MECHANISMS = {
    'serine_protease': {
        'catalytic_triad': ['S', 'H', 'D'],
        'key_interactions': ['oxyanion_hole', 'acyl-enzyme_intermediate'],
        'mutation_rules': {
            'S→C': 'Converts to thiol protease mechanism. Cys is stronger nucleophile (pKa 8.3 vs 13).',
            'S→T': 'Threonine is weaker nucleophile. Usually reduces activity.',
            'H→N': 'Removes proton shuttle. Kills acid-base catalysis.',
            'H→A': 'Complete loss of catalytic machinery.',
            'D→E': 'Longer reach, same chemistry. May optimize geometry.',
            'D→N': 'Removes charge. Disrupts His activation.',
        },
    },
    'metalloenzyme': {
        'coordinating_residues': ['H', 'D', 'E', 'C'],
        'key_interactions': ['metal_coordination', 'substrate_activation'],
        'mutation_rules': {
            'H→N': 'Loses metal coordination. Usually fatal for activity.',
            'H→C': 'Alternative metal coordination. May change metal preference.',
            'D→E': 'Longer reach for metal coordination. Can work if geometry permits.',
            'E→D': 'Shorter reach. May tighten metal binding.',
            'C→S': 'Weaker metal binding (O vs S). Changes redox properties.',
        },
    },
    'oxidoreductase': {
        'key_residues': ['C', 'H', 'Y', 'W'],
        'cofactors': ['NAD+', 'FAD', 'FMN', 'heme'],
        'mutation_rules': {
            'C→S': 'Removes thiol redox chemistry. May still H-bond.',
            'Y→F': 'Removes OH radical capability. Preserves pi-stacking.',
            'W→F': 'Smaller aromatic, removes indole NH. Weakens electron transfer.',
            'H→Q': 'Removes metal coordination. Preserves H-bond ability.',
        },
    },
    'transferase': {
        'key_residues': ['D', 'E', 'K', 'R', 'H'],
        'mutation_rules': {
            'D→E': 'Common beneficial mutation. Optimizes reach for group transfer.',
            'K→R': 'Both positive. Arg has more H-bonds. May stabilize TS.',
            'R→K': 'Fewer H-bonds but more flexible. May improve dynamics.',
        },
    },
    'hydrolase': {
        'key_residues': ['D', 'E', 'S', 'H', 'C'],
        'mutation_rules': {
            'E→Q': 'Removes catalytic acid. Useful for substrate trap mutants.',
            'D→N': 'Removes catalytic base. Useful for mechanistic studies.',
            'S→A': 'Removes nucleophile. Complete loss of hydrolysis.',
        },
    },
}

# Known beneficial mutations from literature (enzyme class → mutations)
KNOWN_BENEFICIAL_MUTATIONS = {
    'lipase': {
        'thermostability': ['surface salt bridges (K→R, E→D)', 'disulfide engineering (G→C pairs)'],
        'activity': ['lid opening mutations', 'tunnel widening'],
    },
    'protease': {
        'specificity': ['S1/S2 pocket reshaping', 'loop grafting'],
        'stability': ['Pro insertion in loops', 'Gly→Ala in helices'],
    },
    'oxidoreductase': {
        'cofactor_switch': ['Rossmann fold mutations for NAD↔NADP preference'],
        'activity': ['active site Phe→Trp for electron transfer'],
    },
    'general': {
        'thermostability': [
            'Surface Lys→Arg (more H-bonds)',
            'Asn/Gln→other (prevent deamidation)',
            'Gly→Ala in helices (reduce entropy)',
            'Add disulfide bridges',
            'Fill cavities with larger hydrophobics',
        ],
        'solubility': [
            'Surface hydrophobic→charged',
            'Remove aggregation-prone patches',
        ],
    },
}


def query_uniprot_active_site(uniprot_id: str) -> dict:
    """
    Query UniProt for active site annotations and known mutagenesis.

    Returns:
        dict with 'active_sites', 'metal_binding', 'mutagenesis', 'catalytic_activity'
    """
    url = 'https://rest.uniprot.org/uniprotkb/{}.json'.format(uniprot_id)

    try:
        req = urllib.request.Request(url, headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, Exception) as e:
        return {'error': str(e), 'active_sites': [], 'mutagenesis': []}

    result = {
        'active_sites': [],
        'metal_binding': [],
        'binding_sites': [],
        'mutagenesis': [],
        'catalytic_activity': [],
        'domains': [],
    }

    # Parse features
    for feature in data.get('features', []):
        ftype = feature.get('type', '')
        location = feature.get('location', {})
        start = location.get('start', {}).get('value')
        end = location.get('end', {}).get('value')
        desc = feature.get('description', '')

        if ftype == 'Active site':
            result['active_sites'].append({
                'position': start, 'description': desc
            })
        elif ftype == 'Metal binding':
            result['metal_binding'].append({
                'position': start, 'description': desc
            })
        elif ftype == 'Binding site':
            result['binding_sites'].append({
                'start': start, 'end': end, 'description': desc
            })
        elif ftype == 'Mutagenesis':
            result['mutagenesis'].append({
                'start': start, 'end': end, 'description': desc,
                'original': feature.get('alternativeSequence', {}).get('originalSequence', ''),
                'alternativeSequences': feature.get('alternativeSequence', {}).get('alternativeSequences', []),
            })
        elif ftype == 'Domain':
            result['domains'].append({
                'start': start, 'end': end, 'description': desc
            })

    # Parse catalytic activity
    for comment in data.get('comments', []):
        if comment.get('commentType') == 'CATALYTIC ACTIVITY':
            reaction = comment.get('reaction', {})
            result['catalytic_activity'].append({
                'name': reaction.get('name', ''),
                'ec': reaction.get('ecNumber', ''),
            })

    return result


def search_mutation_literature(enzyme_name: str, mutation: str) -> List[dict]:
    """
    Search for relevant mutation studies.

    In production this would query PubMed/Google Scholar via API.
    Currently returns structured knowledge from built-in database.
    """
    results = []

    # Check built-in knowledge
    for ec_class, strategies in KNOWN_BENEFICIAL_MUTATIONS.items():
        for goal, mutations in strategies.items():
            for m in mutations:
                if mutation.lower() in m.lower() or enzyme_name.lower() in ec_class:
                    results.append({
                        'source': 'built-in_knowledge',
                        'enzyme_class': ec_class,
                        'goal': goal,
                        'strategy': m,
                        'relevance': 'high' if mutation.lower() in m.lower() else 'moderate',
                    })

    return results


def get_mechanism_rules(enzyme_class: str) -> dict:
    """Get catalytic mechanism rules for an enzyme class."""
    for mech_name, mech in CATALYTIC_MECHANISMS.items():
        if enzyme_class.lower() in mech_name or mech_name in enzyme_class.lower():
            return {'mechanism': mech_name, **mech}

    return {'mechanism': 'unknown', 'mutation_rules': {}}


def build_literature_context(
    position: int,
    wt_aa: str,
    mut_aa: str,
    enzyme_class: str = '',
    uniprot_id: str = '',
) -> dict:
    """
    Build literature-informed context for a mutation.

    Combines UniProt annotations, catalytic mechanism rules,
    and known beneficial mutation patterns.
    """
    context = {
        'position': position,
        'mutation': '{}{}{}'.format(wt_aa, position, mut_aa),
        'mechanism_rules': {},
        'uniprot_annotations': {},
        'known_patterns': [],
        'literature_reasoning': '',
    }

    # Get mechanism-specific rules
    if enzyme_class:
        mech = get_mechanism_rules(enzyme_class)
        context['mechanism_rules'] = mech

        # Check if this specific mutation has known effects
        mut_key = '{}→{}'.format(wt_aa, mut_aa)
        if mut_key in mech.get('mutation_rules', {}):
            context['literature_reasoning'] = mech['mutation_rules'][mut_key]

    # Query UniProt if ID available
    if uniprot_id:
        uniprot_data = query_uniprot_active_site(uniprot_id)
        context['uniprot_annotations'] = uniprot_data

        # Check if position has known mutagenesis data
        for mutagenesis in uniprot_data.get('mutagenesis', []):
            if mutagenesis.get('start') == position:
                context['known_mutagenesis'] = mutagenesis
                context['literature_reasoning'] += ' UniProt: {}'.format(
                    mutagenesis.get('description', ''))

    # Search known patterns
    context['known_patterns'] = search_mutation_literature(
        enzyme_class, '{}{}{}'.format(wt_aa, position, mut_aa))

    return context

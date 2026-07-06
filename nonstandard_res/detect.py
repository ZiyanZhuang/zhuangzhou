"""
Detect non-standard residues, metal ions, and coordination bonds in PDB files.
"""

import logging
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .config import (
    STANDARD_RESIDUES, SOLVENT_IONS, METAL_INFO,
    METAL_COORDINATORS, COORD_CUTOFFS, KNOWN_COFACTORS,
)

logger = logging.getLogger(__name__)


def detect_nonstandard(pdb_path: str, *, chain: str = None) -> dict:
    """
    Scan PDB file for non-standard residues, metal ions, and coordination.

    Returns:
        {
            'nonstandard_residues': [
                {'resname': 'HEM', 'chain': 'A', 'resid': 400,
                 'n_atoms': 43, 'type': 'cofactor', 'known': True,
                 'param_key': 'heme'},
            ],
            'metal_ions': [
                {'resname': 'FE', 'chain': 'A', 'resid': 401,
                 'element': 'Fe', 'charge': 2,
                 'coord': (x, y, z)},
            ],
            'coordination_bonds': [
                {'metal': 'FE_401_A', 'protein_res': 'CYS_350_A',
                 'metal_atom': 'FE', 'protein_atom': 'SG',
                 'distance': 2.35, 'bond_type': 'Fe-S'},
            ],
            'cofactor_metal_bonds': [
                {'metal': 'FE_401_A', 'cofactor_res': 'HEM_400_A',
                 'metal_atom': 'FE', 'cofactor_atom': 'NA',
                 'distance': 2.01},
            ],
            'summary': {
                'n_nonstandard': 1,
                'n_metals': 1,
                'n_coord_bonds': 5,
                'needs_special_params': True,
            },
        }
    """
    if not os.path.isfile(pdb_path):
        return {'error': f'PDB file not found: {pdb_path}'}

    # Parse PDB
    atoms = _parse_pdb(pdb_path, chain)
    if not atoms:
        return {'error': 'No atoms parsed from PDB'}

    # Group by residue
    residues = defaultdict(list)
    for atom in atoms:
        rk = (atom['resname'], atom['resid'], atom['chain'])
        residues[rk].append(atom)

    # Classify residues
    nonstandard = []
    metals = []
    standard_atoms = []

    for (resname, resid, ch), res_atoms in residues.items():
        if resname in STANDARD_RESIDUES:
            standard_atoms.extend(res_atoms)
            continue
        if resname in SOLVENT_IONS:
            # Check if it's a metal ion (single atom)
            if resname.rstrip('0123456789+') in METAL_INFO or resname in METAL_INFO:
                metal_key = resname.rstrip('0123456789+')
                if metal_key not in METAL_INFO:
                    metal_key = resname
                info = METAL_INFO.get(metal_key, {})
                metals.append({
                    'resname': resname,
                    'chain': ch,
                    'resid': resid,
                    'element': info.get('element', resname),
                    'charge': info.get('default_charge', 2),
                    'coord': (res_atoms[0]['x'], res_atoms[0]['y'], res_atoms[0]['z']),
                    'key': f'{resname}_{resid}_{ch}',
                })
            continue

        # Non-standard residue
        param_key = KNOWN_COFACTORS.get(resname)
        # Check if it contains a metal atom
        has_metal = any(
            a['element'] in ('Fe', 'Zn', 'Mn', 'Cu', 'Co', 'Ni', 'Mg')
            for a in res_atoms
        )

        res_info = {
            'resname': resname,
            'chain': ch,
            'resid': resid,
            'n_atoms': len(res_atoms),
            'type': 'cofactor' if param_key else ('metalloenzyme' if has_metal else 'ligand'),
            'known': param_key is not None,
            'param_key': param_key,
            'has_metal': has_metal,
            'key': f'{resname}_{resid}_{ch}',
        }

        # Extract metal from cofactor (e.g., Fe from HEME)
        if has_metal:
            for a in res_atoms:
                if a['element'] in ('Fe', 'Zn', 'Mn', 'Cu', 'Co', 'Ni', 'Mg'):
                    metals.append({
                        'resname': resname,
                        'chain': ch,
                        'resid': resid,
                        'element': a['element'],
                        'charge': METAL_INFO.get(a['element'].upper(), {}).get('default_charge', 2),
                        'coord': (a['x'], a['y'], a['z']),
                        'atom_name': a['name'],
                        'in_cofactor': True,
                        'key': f'{a["name"]}_{resid}_{ch}',
                    })

        nonstandard.append(res_info)

    # Detect coordination bonds
    coord_bonds = []
    cofactor_metal_bonds = []

    for metal in metals:
        mx, my, mz = metal['coord']
        mel = metal['element']

        # Check protein residue coordination
        for (resname, resid, ch), res_atoms in residues.items():
            if resname not in METAL_COORDINATORS:
                continue
            coord_atoms = METAL_COORDINATORS[resname]
            for a in res_atoms:
                if a['name'] not in coord_atoms:
                    continue
                d = _dist(mx, my, mz, a['x'], a['y'], a['z'])
                bond_type = f'{mel}-{coord_atoms[a["name"]]}'
                cutoff = COORD_CUTOFFS.get(bond_type, COORD_CUTOFFS['default'])
                if d <= cutoff:
                    coord_bonds.append({
                        'metal': metal['key'],
                        'protein_res': f'{resname}_{resid}_{ch}',
                        'metal_atom': metal.get('atom_name', metal['resname']),
                        'protein_atom': a['name'],
                        'distance': round(d, 3),
                        'bond_type': bond_type,
                    })

        # Check cofactor coordination (Fe-N in porphyrin ring)
        for (resname, resid, ch), res_atoms in residues.items():
            if resname in STANDARD_RESIDUES or resname in SOLVENT_IONS:
                continue
            # Same residue as metal (e.g., Fe within HEME)
            if metal.get('in_cofactor') and resid == metal['resid'] and ch == metal['chain']:
                for a in res_atoms:
                    if a['element'] in ('Fe', 'Zn', 'Mn', 'Cu'):
                        continue  # Skip metal itself
                    d = _dist(mx, my, mz, a['x'], a['y'], a['z'])
                    if d <= 2.5 and a['element'] in ('N', 'O', 'S'):
                        cofactor_metal_bonds.append({
                            'metal': metal['key'],
                            'cofactor_res': f'{resname}_{resid}_{ch}',
                            'metal_atom': metal.get('atom_name', mel),
                            'cofactor_atom': a['name'],
                            'distance': round(d, 3),
                        })

    return {
        'nonstandard_residues': nonstandard,
        'metal_ions': metals,
        'coordination_bonds': coord_bonds,
        'cofactor_metal_bonds': cofactor_metal_bonds,
        'summary': {
            'n_nonstandard': len(nonstandard),
            'n_metals': len(metals),
            'n_coord_bonds': len(coord_bonds),
            'n_cofactor_metal_bonds': len(cofactor_metal_bonds),
            'needs_special_params': len(nonstandard) > 0 or len(metals) > 0,
            'known_cofactors': [r['resname'] for r in nonstandard if r['known']],
            'unknown_residues': [r['resname'] for r in nonstandard if not r['known']],
        },
    }


def _parse_pdb(pdb_path, chain=None):
    """Parse PDB file into list of atom dicts."""
    atoms = []
    with open(pdb_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            ch = line[21].strip()
            if chain and ch != chain:
                continue

            name = line[12:16].strip()
            resname = line[17:20].strip()
            resid = int(line[22:26].strip())
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            # Element from columns 76-78 or infer from atom name
            element = line[76:78].strip() if len(line) > 77 else ''
            if not element:
                element = name.lstrip('0123456789')[0:2].strip()
                if len(element) > 1 and element[1].islower():
                    pass  # Two-letter element like Fe, Zn
                else:
                    element = element[0]
            # Normalize to title case: FE->Fe, ZN->Zn, MN->Mn, etc.
            if len(element) == 2:
                element = element[0].upper() + element[1].lower()
            elif len(element) == 1:
                element = element.upper()

            atoms.append({
                'name': name, 'resname': resname, 'resid': resid,
                'chain': ch, 'x': x, 'y': y, 'z': z,
                'element': element,
                'record': line[:6].strip(),
            })
    return atoms


def _dist(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

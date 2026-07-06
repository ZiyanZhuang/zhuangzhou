"""
Active Site Definition
======================
Defines the docking box (center + size) for Vina.

Strategies (priority order):
  1. From existing ligand coordinates in PDB
  2. HEME-aware: above Fe atom (CYP enzymes)
  3. Geometric center of protein (fallback)

All defaults (padding, box limits, HEME offset) come from config.py.
"""

import logging

import numpy as np

from .config import SITE_DEFAULTS

logger = logging.getLogger(__name__)


def define_site_from_ligand(ligand_info: dict, padding: float = None) -> dict:
    """Define docking box from ligand coordinates in PDB."""
    padding = padding if padding is not None else SITE_DEFAULTS['padding']
    min_box = SITE_DEFAULTS['min_box_size']
    max_box = SITE_DEFAULTS['max_box_size']

    coords = np.array([(a[1], a[2], a[3]) for a in ligand_info['atoms']])
    center = coords.mean(axis=0)
    span = coords.max(axis=0) - coords.min(axis=0)
    size = np.clip(span + padding, min_box, max_box)

    grid = {
        'center_x': float(center[0]), 'center_y': float(center[1]),
        'center_z': float(center[2]),
        'size_x': float(size[0]), 'size_y': float(size[1]),
        'size_z': float(size[2]),
    }
    logger.info("Site from ligand %s: center=(%.1f, %.1f, %.1f) size=(%.0f, %.0f, %.0f)",
                ligand_info.get('resname', '?'),
                grid['center_x'], grid['center_y'], grid['center_z'],
                grid['size_x'], grid['size_y'], grid['size_z'])
    return grid


def define_site_from_heme(heme_info: dict, padding: float = None) -> dict:
    """Define docking box above HEME iron (CYP-specific)."""
    padding = padding if padding is not None else SITE_DEFAULTS['padding']
    fe_offset = SITE_DEFAULTS['heme_fe_offset']

    fe = np.array(heme_info['fe_coord'])
    normal = np.array(heme_info.get('heme_normal', [0, 0, 1]))
    center = fe + normal * fe_offset

    grid = {
        'center_x': float(center[0]), 'center_y': float(center[1]),
        'center_z': float(center[2]),
        'size_x': 20.0 + padding, 'size_y': 20.0 + padding,
        'size_z': 15.0 + padding,
    }
    logger.info("Site from HEME: center=(%.1f, %.1f, %.1f), FE dist=%.1f A",
                center[0], center[1], center[2], np.linalg.norm(center - fe))
    return grid


def define_site_from_protein(pdb_file: str, padding: float = None) -> dict:
    """Fallback: docking box at protein geometric center."""
    fallback_box = SITE_DEFAULTS['fallback_box_size']
    padding = padding if padding is not None else SITE_DEFAULTS['padding']
    max_box = SITE_DEFAULTS['max_box_size']

    coords = []
    with open(pdb_file, encoding='utf-8') as f:
        for line in f:
            if line.startswith('ATOM'):
                try:
                    coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                except ValueError:
                    pass
    center = np.array(coords).mean(axis=0)
    box = min(fallback_box + padding, max_box)
    grid = {
        'center_x': float(center[0]), 'center_y': float(center[1]),
        'center_z': float(center[2]),
        'size_x': box, 'size_y': box, 'size_z': box,
    }
    logger.info("Site from protein center: (%.1f, %.1f, %.1f)", center[0], center[1], center[2])
    return grid


def define_active_site(pdb_file: str, ligands: list = None,
                       cofactors: list = None, padding: float = None) -> dict:
    """
    Auto-choose best strategy for active site definition.
    Priority: ligand coords > HEME > protein center.

    Returns grid dict with 'site_strategy' field indicating which method was used.
    """
    if ligands:
        best = max(ligands, key=lambda l: l['n_atoms'])
        grid = define_site_from_ligand(best, padding)
        grid['site_strategy'] = 'ligand'
        grid['site_detail'] = f"Based on ligand {best.get('resname', '?')} ({best.get('n_atoms', '?')} atoms)"
        return grid

    if cofactors:
        heme = next((c for c in cofactors if c.get('type') == 'HEME' and 'fe_coord' in c), None)
        if heme:
            grid = define_site_from_heme(heme, padding)
            grid['site_strategy'] = 'heme'
            grid['site_detail'] = f"Above HEME Fe atom ({heme['resname']} chain {heme.get('chain', '?')})"
            return grid

    logger.warning("No ligand or HEME cofactor found — using protein geometric center as docking site. "
                   "Results may be unreliable. Consider providing a reference ligand or binding site coordinates.")
    grid = define_site_from_protein(pdb_file, padding)
    grid['site_strategy'] = 'protein_center'
    grid['site_detail'] = ("WARNING: No co-crystallized ligand or HEME detected. "
                           "Docking box placed at protein geometric center — "
                           "binding poses may be non-specific.")
    return grid

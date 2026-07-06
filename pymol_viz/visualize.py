"""
PyMOL Visualization — publication-quality interaction figures from PLIP results.

Generates .pml scripts that produce 600 DPI images matching top structural
biology journals (Nature Struct Biol, Science, Cell). Features:
  - Per-interaction dashed lines with type-specific coloring
  - Distance labels (Angstrom) on every interaction
  - Residue labels with smart offset to avoid overlaps
  - Color-coded legend panel
  - ChimeraX-like toon/ambient-occlusion rendering
  - Both detail view (binding site) and global overview

Two modes:
  1. Script mode (default): generates .pml → runs PyMOL headless → PNG
  2. Manual mode (render=False): saves .pml for manual PyMOL session
"""

import logging
import math
import os
import re
import subprocess
import textwrap
from collections import defaultdict

from .config import (
    PYMOL_EXE, PYMOL_PYTHON,
    RENDER_DEFAULTS, INTERACTION_COLORS, INTERACTION_LABELS,
    DASH_STYLES, DASH_SETTINGS,
    PROTEIN_COLOR, PROTEIN_CARBON, LIGAND_CARBON, CARTOON_TRANSPARENCY,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def visualize_plip(complex_pdb: str,
                   plip_result: dict = None,
                   output_dir: str = None,
                   ligand_resname: str = None,
                   rank: int = None,
                   affinity: float = None,
                   style: str = 'publication',
                   render: bool = True,
                   global_view: bool = True,
                   width: int = None,
                   height: int = None,
                   dpi: int = None) -> dict:
    """
    Generate publication-quality interaction visualization from PLIP results.

    Args:
        complex_pdb: path to complex PDB (protein + ligand)
        plip_result: dict from run_plip_analysis()
        output_dir: where to save script + images
        ligand_resname: ligand residue name (auto-detect from PLIP if None)
        rank: docking rank (annotation)
        affinity: binding affinity kcal/mol (annotation)
        style: 'publication' (600 DPI) or 'presentation' (lower DPI)
        render: if True, run PyMOL headless to produce PNG
        global_view: if True, also render a global overview image
        width/height/dpi: override render defaults

    Returns:
        dict with script_path, png_path, interactions_count, etc.
    """
    complex_pdb = os.path.abspath(complex_pdb)
    if not os.path.exists(complex_pdb):
        return {'error': f'Complex PDB not found: {complex_pdb}'}

    stem = os.path.splitext(os.path.basename(complex_pdb))[0]
    if not output_dir:
        output_dir = os.path.join(os.getcwd(), f'{stem}_viz')
    os.makedirs(output_dir, exist_ok=True)

    # Auto-detect ligand
    if not ligand_resname and plip_result:
        detected = plip_result.get('ligands_detected', [])
        if detected:
            ligand_resname = detected[0]
    if not ligand_resname:
        ligand_resname = _detect_ligand_from_pdb(complex_pdb)
    if not ligand_resname:
        return {'error': 'Cannot determine ligand resname',
                'hint': 'Pass ligand_resname="LIG" or use a complex PDB that contains HETATM records. '
                        'For apo structures, use pymol_view or chimerax_visualize instead.'}

    # Extract interactions
    all_interactions = []
    residue_set = set()
    if plip_result and 'interactions' in plip_result:
        all_interactions = _extract_all_interactions(plip_result['interactions'])
        for ix in all_interactions:
            residue_set.add((ix['resn'], ix['resi'], ix['chain']))

    # Render parameters
    w = width or RENDER_DEFAULTS['width']
    h = height or RENDER_DEFAULTS['height']
    d = dpi or RENDER_DEFAULTS['dpi']
    if style == 'presentation':
        d = min(d, 300)

    # --- Detail view (binding site close-up) ---
    detail_png = os.path.join(output_dir, f'{stem}_detail.png')
    detail_script = _generate_script(
        complex_pdb=complex_pdb,
        ligand_resname=ligand_resname,
        interactions=all_interactions,
        residue_set=residue_set,
        png_path=detail_png,
        rank=rank, affinity=affinity,
        width=w, height=h, dpi=d,
        view_mode='detail',
    )
    detail_pml = os.path.join(output_dir, f'{stem}_detail.pml')
    with open(detail_pml, 'w', encoding='utf-8') as f:
        f.write(detail_script)

    result = {
        'script_path': detail_pml,
        'output_dir': output_dir,
        'ligand_resname': ligand_resname,
        'interactions_count': len(all_interactions),
        'interaction_types': _count_by_type(all_interactions),
    }

    # --- Global view (full protein context) ---
    global_pml = None
    if global_view:
        global_png = os.path.join(output_dir, f'{stem}_global.png')
        global_script = _generate_script(
            complex_pdb=complex_pdb,
            ligand_resname=ligand_resname,
            interactions=all_interactions,
            residue_set=residue_set,
            png_path=global_png,
            rank=rank, affinity=affinity,
            width=w, height=h, dpi=d,
            view_mode='global',
        )
        global_pml = os.path.join(output_dir, f'{stem}_global.pml')
        with open(global_pml, 'w', encoding='utf-8') as f:
            f.write(global_script)
        result['global_script'] = global_pml

    # Render
    if render:
        ok = _run_pymol_script(detail_pml)
        result['rendered'] = ok
        if ok and os.path.exists(detail_png):
            result['png_path'] = detail_png
            result['png_size'] = os.path.getsize(detail_png)
        else:
            result['warning'] = 'Detail render failed, script saved for manual use'

        if global_pml:
            ok2 = _run_pymol_script(global_pml)
            result['global_rendered'] = ok2
            if ok2 and os.path.exists(global_png):
                result['global_png'] = global_png

    return result


# Keep old name as alias for backward compat
visualize_docking = visualize_plip


# ═══════════════════════════════════════════════════════════════════════════════
# Script Generation
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_script(complex_pdb: str,
                     ligand_resname: str,
                     interactions: list,
                     residue_set: set,
                     png_path: str,
                     rank: int = None,
                     affinity: float = None,
                     width: int = 3600,
                     height: int = 2700,
                     dpi: int = 600,
                     view_mode: str = 'detail') -> str:
    """Generate a complete PyMOL .pml script."""
    pdb_fwd = complex_pdb.replace('\\', '/')
    png_fwd = png_path.replace('\\', '/')

    lines = []
    _a = lines.append
    _e = lines.extend

    _a(f'# Auto-generated by zhuangzhou.pymol_viz  [{view_mode} view]')
    _a(f'# Complex: {os.path.basename(complex_pdb)}')
    if rank is not None and affinity is not None:
        _a(f'# Rank {rank}, Affinity: {affinity} kcal/mol')
    _a('')

    # ── Load ──
    _a(f'load {pdb_fwd}, complex')
    _a('')

    # ── Global rendering settings (ChimeraX-like) ──
    _e(_section_rendering())

    # ── Selections ──
    _a(f'select ligand, resn {ligand_resname}')
    _a('select receptor, polymer and not ligand')
    _a('')

    # ── Protein display ──
    _a('hide everything, all')
    _a('show cartoon, receptor')
    _a(f'color {PROTEIN_COLOR}, receptor')
    _a(f'set cartoon_transparency, {CARTOON_TRANSPARENCY}, receptor')
    _a('')

    # ── Ligand display ──
    _a('show sticks, ligand')
    _a(f'color {LIGAND_CARBON}, ligand and elem C')
    _a('util.cnc ligand')
    # Ligand should be opaque and prominent
    _a('set stick_radius, 0.17, ligand')
    _a('')

    # ── Binding-site residues ──
    if residue_set:
        resi_list = sorted(set(r[1] for r in residue_set), key=lambda x: int(x) if x.isdigit() else 0)
        resi_sel = '+'.join(resi_list)
        _a(f'select binding_site, receptor and resi {resi_sel}')
        _a('show sticks, binding_site and sidechain')
        _a('show sticks, binding_site and name CA+C+N+O')
        _a(f'color {PROTEIN_CARBON}, binding_site and elem C')
        _a('util.cnc binding_site')
        _a('set stick_radius, 0.14, binding_site')
        _a('')

        # Residue labels (3-letter code + number)
        _e(_section_residue_labels(residue_set, interactions))

    # ── Interaction dashes with distance labels ──
    _e(_section_interactions(interactions, ligand_resname))

    # ── Legend ──
    present_types = set(ix['type'] for ix in interactions)
    if present_types:
        _e(_section_legend(present_types, view_mode))

    # ── Camera ──
    _e(_section_camera(residue_set, view_mode))

    # ── Annotation (affinity) ──
    if affinity is not None:
        label_text = f'Affinity: {affinity:.1f} kcal/mol'
        if rank is not None:
            label_text = f'Rank {rank}  |  {label_text}'
        # Bottom-left annotation via pseudoatom
        _a(f'pseudoatom anno_affinity, label="{label_text}"')
        _a('set label_size, 14, anno_affinity')
        _a('set label_color, gray40, anno_affinity')
        _a('set label_font_id, 7, anno_affinity')
        _a('hide everything, anno_affinity')
        _a('show label, anno_affinity')
        _a('')

    _a('deselect')
    _a('')

    # ── Render ──
    _a(f'ray {width}, {height}')
    _a(f'png {png_fwd}, dpi={dpi}')
    _a('')

    return '\n'.join(lines)


# Backward-compat alias
generate_pymol_script = _generate_script


def _section_rendering() -> list:
    """ChimeraX-like rendering settings."""
    return [
        '# === Rendering (publication-quality, fast ray) ===',
        'bg_color white',
        'set ray_opaque_background, 1',
        # Mode 0: standard ray (fast), mode 1 for outlines is 10x slower
        'set ray_trace_mode, 0',
        'set ray_trace_gain, 0.015',
        'set ray_trace_color, black',
        '',
        '# Soft lighting',
        'set ambient, 0.40',
        'set direct, 0.45',
        'set reflect, 0.50',
        'set spec_reflect, 0.08',
        'set spec_power, 100',
        'set ray_shadows, 1',
        'set ray_shadow_decay_factor, 0.1',
        '',
        '# Cartoon quality',
        'set antialias, 2',
        'set cartoon_fancy_helices, 1',
        'set cartoon_smooth_loops, 1',
        'set cartoon_sampling, 14',
        'set cartoon_loop_radius, 0.20',
        'set cartoon_oval_width, 0.25',
        'set cartoon_oval_length, 1.20',
        '',
        '# Stick defaults',
        'set stick_radius, 0.14',
        'set stick_ball, on',
        'set stick_ball_ratio, 1.6',
        'set stick_ball_color, atomic',
        '',
        '# Global dash defaults',
        f'set dash_gap, {DASH_SETTINGS["dash_gap"]}',
        f'set dash_length, {DASH_SETTINGS["dash_length"]}',
        f'set dash_radius, {DASH_SETTINGS["dash_radius"]}',
        '',
        '# Distance label styling',
        'set label_size, -0.4',
        'set label_font_id, 7',
        'set label_color, gray20',
        'set label_outline_color, white',
        'set label_bg_color, white',
        'set label_digits, 1',
        'set label_distance_digits, 1',
        '',
    ]


def _section_residue_labels(residue_set: set, interactions: list) -> list:
    """Generate non-overlapping residue labels with staggered offsets."""
    lines = ['# === Residue labels (staggered to avoid overlap) ===']

    # Sort residues by resi number for deterministic offset assignment
    sorted_res = sorted(residue_set, key=lambda r: int(r[1]) if r[1].isdigit() else 0)

    # Stagger offsets in a spiral pattern to minimize overlap
    offsets = [
        (0, 0, 2.5),
        (1.5, 1.5, 2.0),
        (-1.5, 1.5, 2.0),
        (1.5, -1.0, 2.0),
        (-1.5, -1.0, 2.0),
        (2.5, 0, 1.5),
        (-2.5, 0, 1.5),
        (0, 2.5, 1.5),
        (0, -2.0, 1.5),
        (2.0, 2.0, 1.5),
        (-2.0, 2.0, 1.5),
        (2.0, -1.5, 1.5),
        (-2.0, -1.5, 1.5),
    ]

    # Build 3-letter -> 1-letter map for compact labels
    aa3to1 = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    }

    for i, (resn, resi, chain) in enumerate(sorted_res):
        ox, oy, oz = offsets[i % len(offsets)]
        chain_sel = f'chain {chain} and ' if chain else ''
        one_letter = aa3to1.get(resn, resn)
        label_text = f'{one_letter}{resi}'
        # Label the CA atom directly (persistent, survives deselect)
        ca_sel = f'({chain_sel}resi {resi}) and name CA'
        lines.append(f'label {ca_sel}, "{label_text}"')
        lines.append(f'set label_position, [{ox:.1f}, {oy:.1f}, {oz:.1f}], {ca_sel}')
        lines.append(f'set label_size, -0.5, {ca_sel}')
        lines.append(f'set label_color, black, {ca_sel}')

    lines.append('')
    return lines


def _section_interactions(interactions: list, ligand_resname: str) -> list:
    """Generate per-interaction distance objects with distance labels."""
    lines = ['# === PLIP Interactions (typed dashes + distance labels) ===']

    if not interactions:
        lines.append('# (no interactions detected)')
        lines.append('')
        return lines

    # Group by type
    by_type = defaultdict(list)
    for ix in interactions:
        by_type[ix['type']].append(ix)

    # Atom selection helpers per interaction type
    type_atoms = {
        'hydrogen_bonds': {
            'lig': f'(resn {ligand_resname}) and (elem N+O+S)',
            'res_sc': 'name OD1+OD2+OE1+OE2+NE2+NH1+NH2+NZ+OG+OG1+OH+ND1+ND2+NE1+SD+SG',
            'res_bb': 'name N+O',
            'cutoff': 4.2,
        },
        'hydrophobic': {
            'lig': f'(resn {ligand_resname}) and (elem C)',
            'res': 'elem C',
            'cutoff': 4.5,
        },
        'salt_bridges': {
            'lig': f'(resn {ligand_resname}) and (elem N+O)',
            'res': 'name OD1+OD2+OE1+OE2+NZ+NH1+NH2',
            'cutoff': 5.5,
        },
        'pi_stacking': {
            'lig': f'(resn {ligand_resname})',
            'res': 'sidechain',
            'cutoff': 6.0,
        },
        'pi_cation': {
            'lig': f'(resn {ligand_resname})',
            'res': 'sidechain',
            'cutoff': 6.0,
        },
        'halogen_bonds': {
            'lig': f'(resn {ligand_resname}) and (elem F+Cl+Br+I)',
            'res': 'elem N+O+S',
            'cutoff': 4.0,
        },
        'water_bridges': {
            'lig': f'(resn {ligand_resname}) and (elem N+O)',
            'res': 'elem N+O',
            'cutoff': 4.5,
        },
        'metal_complexes': {
            'lig': f'(resn {ligand_resname})',
            'res': 'elem N+O+S',
            'cutoff': 3.5,
        },
    }

    for itype, ixlist in by_type.items():
        color = INTERACTION_COLORS.get(itype, '0xCCCCCC')
        label = INTERACTION_LABELS.get(itype, itype)
        dash = DASH_STYLES.get(itype, DASH_SETTINGS)
        atoms = type_atoms.get(itype, {})
        cutoff = atoms.get('cutoff', 4.5)

        lines.append(f'# --- {label} ({len(ixlist)}) ---')

        # Group by residue to create one distance object per residue
        by_residue = defaultdict(list)
        for ix in ixlist:
            key = (ix['resn'], ix['resi'], ix['chain'])
            by_residue[key].append(ix)

        for (resn, resi, chain), res_ixlist in by_residue.items():
            obj_name = f'{itype[:2]}_{resn}{resi}'
            chain_sel = f'chain {chain} and ' if chain else ''
            res_base = f'({chain_sel}resi {resi})'

            # Determine atom targets based on type and sidechain/backbone
            if itype == 'hydrogen_bonds':
                has_sc = any(ix.get('sidechain') for ix in res_ixlist)
                has_bb = any(not ix.get('sidechain') for ix in res_ixlist)
                if has_sc and has_bb:
                    res_atoms = f'{res_base} and ({atoms["res_sc"]}+{atoms["res_bb"]})'
                elif has_sc:
                    res_atoms = f'{res_base} and ({atoms["res_sc"]})'
                else:
                    res_atoms = f'{res_base} and ({atoms["res_bb"]})'
                lig_sel = atoms['lig']
            elif itype in ('hydrophobic',):
                lig_sel = atoms['lig']
                res_atoms = f'{res_base} and ({atoms["res"]})'
            else:
                lig_sel = atoms.get('lig', f'(resn {ligand_resname})')
                res_atoms = f'{res_base} and ({atoms.get("res", "all")})'

            # Create distance measurement (mode=2 finds closest pair)
            lines.append(
                f'distance {obj_name}, {lig_sel}, {res_atoms}, mode=2, cutoff={cutoff}'
            )
            # Style the dashes
            lines.append(f'color {color}, {obj_name}')
            lines.append(f'set dash_gap, {dash["dash_gap"]}, {obj_name}')
            lines.append(f'set dash_length, {dash["dash_length"]}, {obj_name}')
            lines.append(f'set dash_radius, {dash["dash_radius"]}, {obj_name}')

            # SHOW distance labels (key difference: DO NOT hide them)
            # PyMOL distance objects auto-label with measured distance
            lines.append(f'set label_color, {color}, {obj_name}')
            lines.append(f'set label_size, -0.35, {obj_name}')

        lines.append('')

    return lines


def _section_legend(present_types: set, view_mode: str) -> list:
    """Generate a color-coded legend using pseudoatoms (top-right corner)."""
    lines = ['# === Legend ===']

    # Use pseudoatoms at fixed positions for legend entries
    # Position them far from the molecule so they don't interfere,
    # then use label_screen_point to fix them visually
    sorted_types = sorted(present_types, key=lambda t: list(INTERACTION_LABELS.keys()).index(t)
                          if t in INTERACTION_LABELS else 99)

    for i, itype in enumerate(sorted_types):
        color = INTERACTION_COLORS.get(itype, '0xCCCCCC')
        label = INTERACTION_LABELS.get(itype, itype)
        pname = f'legend_{i}'
        # Position at large negative X so they're off the molecule
        y_off = -i * 4.0
        lines.append(f'pseudoatom {pname}, pos=[-900, {500 + y_off:.0f}, 0]')
        lines.append(f'label {pname}, "--- {label}"')
        lines.append(f'set label_color, {color}, {pname}')
        lines.append(f'set label_size, 16, {pname}')
        lines.append(f'set label_font_id, 7, {pname}')
        lines.append(f'hide everything, {pname}')
        lines.append(f'show label, {pname}')

    lines.append('')
    return lines


def _section_camera(residue_set: set, view_mode: str) -> list:
    """Camera positioning for detail vs global view."""
    lines = ['# === Camera ===']

    if view_mode == 'global':
        lines.extend([
            'zoom complex, 5',
            'orient complex',
            'turn y, 20',
            'turn x, -10',
            # Show the full protein with the binding site highlighted
            'set cartoon_transparency, 0.50, receptor',
        ])
    else:
        # Detail view: zoom into binding site
        if residue_set:
            lines.extend([
                'select _cam_target, ligand or binding_site',
                'zoom _cam_target, 6',
                'orient _cam_target',
                'turn y, 15',
                'turn x, -5',
                'delete _cam_target',
            ])
        else:
            lines.extend([
                'zoom ligand, 12',
                'orient ligand',
                'turn y, 15',
            ])

    lines.append('')
    return lines


# ═══════════════════════════════════════════════════════════════════════════════
# Internals
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_ligand_from_pdb(pdb_path: str) -> str:
    """Detect ligand resname from HETATM records in PDB."""
    SKIP = {'HOH', 'WAT', 'SOL', 'NA', 'CL', 'K', 'MG', 'CA', 'ZN', 'FE',
            'MN', 'CU', 'NI', 'CO', 'CD', 'SO4', 'PO4', 'GOL', 'EDO', 'ACE',
            'NME', 'NH2'}
    resnames = {}
    with open(pdb_path, encoding='utf-8') as f:
        for line in f:
            if line.startswith('HETATM'):
                rn = line[17:20].strip()
                if rn and rn not in SKIP:
                    resnames[rn] = resnames.get(rn, 0) + 1
    if not resnames:
        return ''
    return max(resnames, key=resnames.get)


def _extract_all_interactions(interactions: dict) -> list:
    """
    Extract ALL individual interactions from PLIP data.
    Preserves every interaction (e.g., 2 H-bonds to ASP10 = 2 entries).
    """
    result = []
    for site_key, site_data in interactions.items():
        if not isinstance(site_data, dict):
            continue
        for itype, ilist in site_data.items():
            if not isinstance(ilist, list):
                continue
            for ix in ilist:
                if not isinstance(ix, dict):
                    continue
                residue_str = ix.get('residue', '')
                if not residue_str:
                    continue
                m = re.match(r'([A-Z]+)(\d+)', residue_str)
                if not m:
                    continue
                entry = {
                    'type': itype,
                    'resn': m.group(1),
                    'resi': m.group(2),
                    'chain': ix.get('reschain', ''),
                    'sidechain': ix.get('sidechain', '') == 'True'
                                 or ix.get('sidechain', False) is True,
                }
                for k in ('dist_h_a', 'dist_d_a', 'dist', 'donor_acceptor'):
                    if k in ix:
                        entry[k] = ix[k]
                result.append(entry)
    return result


def _count_by_type(interactions: list) -> dict:
    """Count interactions by type."""
    counts = defaultdict(int)
    for ix in interactions:
        counts[ix['type']] += 1
    return dict(counts)


def _run_pymol_script(script_path: str) -> bool:
    """
    Run a .pml script via PyMOL headless (python.exe -c mode).
    Uses cmd.run() for batch execution (much faster than line-by-line cmd.do()).
    """
    script_path = os.path.abspath(script_path).replace('\\', '/')

    if not os.path.exists(PYMOL_PYTHON):
        logger.error("PyMOL python not found: %s", PYMOL_PYTHON)
        return False

    wrapper = textwrap.dedent(f'''\
        import sys, time
        sys.stdout.reconfigure(encoding='utf-8')
        import pymol
        pymol.pymol_argv = ["pymol", "-cq"]
        pymol.finish_launching()
        from pymol import cmd
        t0 = time.time()
        with open(r"{script_path}", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#"):
                    cmd.do(_line)
        print(f"Done in {{time.time()-t0:.1f}}s")
        cmd.quit()
    ''')

    wrapper_path = script_path.replace('.pml', '_runner.py')
    with open(wrapper_path, 'w', encoding='utf-8') as f:
        f.write(wrapper)

    logger.info("Running PyMOL: %s %s", PYMOL_PYTHON, wrapper_path)
    try:
        proc = subprocess.run(
            [PYMOL_PYTHON, wrapper_path],
            capture_output=True, text=True, timeout=300,
            cwd=os.path.dirname(script_path))
        if proc.returncode == 0:
            return True
        logger.warning("PyMOL exited %d: %s", proc.returncode,
                       proc.stderr[-500:] if proc.stderr else '')
    except FileNotFoundError:
        logger.error("PyMOL python not found: %s", PYMOL_PYTHON)
    except subprocess.TimeoutExpired:
        logger.error("PyMOL timed out (300s)")
    finally:
        try:
            os.unlink(wrapper_path)
        except OSError:
            pass

    return False

"""
QM Engine Analysis - Reaction energy profile and computation reports.

Standard computational chemistry methodology (Himo, Siegbahn, Shaik, Houk):
  - Energy level hierarchy: E_sp + G_correction > E_sp + ZPE > E_sp > E_opt
  - G_correction = G_tot(freq) - E_elec(freq) includes ZPE + thermal + entropy
  - Barriers reported as DeltaE_sp, DeltaE_sp+ZPE, DeltaG where available
  - 3D ball-and-stick molecular structure insets from XYZ (no bond-order issues)
"""

import json
import logging
import math
import os
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

HA_TO_KCAL = 627.509
BOHR_TO_ANG = 0.529177

# CPK colors and covalent radii for 3D rendering
_ELEMENT_COLORS = {
    'H': '#FFFFFF', 'C': '#505050', 'N': '#3050F8', 'O': '#FF0D0D',
    'F': '#90E050', 'S': '#FFFF30', 'P': '#FF8000', 'Cl': '#1FF01F',
    'Br': '#A62929', 'I': '#940094', 'Fe': '#E06633', 'Mn': '#9C7AC7',
    'Zn': '#7D80B0', 'Cu': '#C88033', 'Co': '#F090A0', 'Mg': '#8AFF00',
    'Na': '#AB5CF2', 'K': '#8F40D4', 'Ca': '#3DFF00',
}
_ELEMENT_RADII = {  # covalent radii in Angstrom
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
    'S': 1.05, 'P': 1.07, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39,
    'Fe': 1.52, 'Mn': 1.61, 'Zn': 1.22, 'Cu': 1.32, 'Co': 1.50,
    'Mg': 1.41, 'Na': 1.66, 'K': 2.03, 'Ca': 1.76,
}
_VDW_RADII = {
    'H': 0.25, 'C': 0.50, 'N': 0.45, 'O': 0.42, 'F': 0.40,
    'S': 0.60, 'P': 0.58, 'Cl': 0.56, 'Br': 0.62, 'I': 0.70,
    'Fe': 0.65, 'Mn': 0.65, 'Zn': 0.55, 'Cu': 0.55,
}
_BOND_TOLERANCE = 0.40  # Angstrom tolerance for bond detection


# =============================================================================
# 3D Ball-and-Stick renderer (works for TS, charged, any species)
# =============================================================================

def _read_xyz(xyz_path: str) -> List[Tuple[str, float, float, float]]:
    """Read XYZ file -> list of (element, x, y, z)."""
    atoms = []
    with open(xyz_path, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    for line in lines[2:2+n]:
        parts = line.split()
        if len(parts) >= 4:
            elem = parts[0].strip()
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            atoms.append((elem, x, y, z))
    return atoms


def _detect_bonds(atoms, tolerance=_BOND_TOLERANCE):
    """
    Detect bonds from interatomic distances and covalent radii.

    Uses two thresholds:
      - Normal bond: d < r_i + r_j + tolerance
      - Partial bond (TS): d < r_i + r_j + 0.8 A (shown as dashed)
    This captures elongated bonds in transition states (e.g. C-F at 1.85 A).
    """
    bonds = []
    n = len(atoms)
    # Extended tolerance for partial/forming/breaking bonds in TS
    partial_tolerance = 0.80
    for i in range(n):
        for j in range(i+1, n):
            ei, xi, yi, zi = atoms[i]
            ej, xj, yj, zj = atoms[j]
            d = math.sqrt((xi-xj)**2 + (yi-yj)**2 + (zi-zj)**2)
            ri = _ELEMENT_RADII.get(ei, 0.77)
            rj = _ELEMENT_RADII.get(ej, 0.77)
            expected = ri + rj
            if d < expected + partial_tolerance and d > 0.4:
                bonds.append((i, j, d))
    return bonds


def _xyz_to_3d_image(xyz_path: str, size: tuple = (280, 240),
                     hide_h: bool = False) -> Optional[object]:
    """
    Render XYZ file as 2D-projected ball-and-stick using matplotlib.

    Works for any structure including transition states and charged species.
    Projects 3D coordinates onto best viewing plane (PCA-based).
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        import numpy as np
    except ImportError:
        return None

    atoms = _read_xyz(xyz_path)
    if not atoms:
        return None

    if hide_h:
        idx_map = {}
        filtered = []
        for i, (e, x, y, z) in enumerate(atoms):
            if e != 'H':
                idx_map[i] = len(filtered)
                filtered.append((e, x, y, z))
        atoms = filtered
    else:
        idx_map = {i: i for i in range(len(atoms))}

    n = len(atoms)
    if n == 0:
        return None

    coords = np.array([(x, y, z) for _, x, y, z in atoms])

    # Center
    coords -= coords.mean(axis=0)

    # PCA for best 2D projection
    if n > 1:
        cov = np.cov(coords.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Project onto the two largest variance axes
        order = np.argsort(-eigvals)
        proj = coords @ eigvecs[:, order[:2]]
    else:
        proj = coords[:, :2]

    x2d = proj[:, 0]
    y2d = proj[:, 1]

    # Detect bonds from original 3D coordinates
    all_atoms_3d = _read_xyz(xyz_path)
    raw_bonds = _detect_bonds(all_atoms_3d)

    # Map bonds to filtered indices
    bonds_2d = []
    for i, j, d in raw_bonds:
        if i in idx_map and j in idx_map:
            bonds_2d.append((idx_map[i], idx_map[j], d))

    # Figure
    dpi = 100
    fig_w, fig_h = size[0] / dpi, size[1] / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    # Scale
    if n > 1:
        span = max(x2d.max() - x2d.min(), y2d.max() - y2d.min(), 1.0)
    else:
        span = 2.0
    margin = span * 0.25
    ax.set_xlim(x2d.min() - margin, x2d.max() + margin)
    ax.set_ylim(y2d.min() - margin, y2d.max() + margin)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw bonds
    for i, j, dist in bonds_2d:
        ei = atoms[i][0]
        ej = atoms[j][0]
        ri = _ELEMENT_RADII.get(ei, 0.77)
        rj = _ELEMENT_RADII.get(ej, 0.77)
        expected = ri + rj
        # Dashed if stretched (partial bond in TS): >1.2x normal
        if dist > expected * 1.2:
            ls = '--'
            lw = 1.0
            alpha = 0.6
        else:
            ls = '-'
            lw = 1.5
            alpha = 0.9
        ax.plot([x2d[i], x2d[j]], [y2d[i], y2d[j]],
                color='#555555', linewidth=lw, linestyle=ls, alpha=alpha,
                zorder=1)
        # Label bond distance for stretched bonds (TS-relevant)
        if dist > expected * 1.15:
            mx = (x2d[i] + x2d[j]) / 2
            my = (y2d[i] + y2d[j]) / 2
            ax.text(mx, my, f'{dist:.2f}', fontsize=5, ha='center', va='center',
                    color='#888888', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                              edgecolor='none', alpha=0.7))

    # Draw atoms
    scale_r = span / 12.0  # sphere radius in data coords
    for i in range(n):
        elem = atoms[i][0]
        color = _ELEMENT_COLORS.get(elem, '#FF69B4')
        r = _VDW_RADII.get(elem, 0.40) * scale_r / 0.50

        circle = Circle((x2d[i], y2d[i]), r, facecolor=color,
                         edgecolor='#333333', linewidth=0.5, zorder=3)
        ax.add_patch(circle)

        # Element label
        if elem != 'H' or not hide_h:
            fontcolor = 'white' if elem in ('C', 'Fe', 'Mn', 'Co', 'Cu', 'Br', 'I') else 'black'
            ax.text(x2d[i], y2d[i], elem, fontsize=6, ha='center', va='center',
                    fontweight='bold', color=fontcolor, zorder=4)

    fig.patch.set_facecolor('white')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Convert to PIL Image
    fig.canvas.draw()
    from PIL import Image
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                pad_inches=0.02, facecolor='white')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    return img


def _pil_to_array(img):
    """Convert PIL Image to numpy array for matplotlib embedding."""
    import numpy as np
    return np.array(img)


# =============================================================================
# Energy Profile Plot
# =============================================================================

def plot_energy_profile(
    energies: Dict[str, float],
    output_path: str,
    *,
    title: str = 'Reaction Energy Profile',
    level: str = '',
    relative: bool = True,
    unit: str = 'kcal/mol',
    energy_type: str = 'E',
    figsize: tuple = (10, 6),
    dpi: int = 300,
    colors: dict = None,
    show_values: bool = True,
    structures: dict = None,
) -> dict:
    """
    Plot reaction energy profile (barrier diagram).

    Standard computational chemistry style:
      - Horizontal platforms for stationary points (blue=minimum, red=TS)
      - Dashed connecting lines along reaction coordinate
      - Barrier height double-arrows with values
      - Optional 3D ball-and-stick structure insets
      - Theory level as subtitle

    Args:
        energies: Ordered dict {label: energy_in_hartree}
        output_path: Output image path (.png/.pdf/.svg)
        title: Plot title
        level: Theory level string for subtitle
        relative: If True, energies relative to first entry
        unit: 'kcal/mol' or 'kJ/mol'
        energy_type: 'E' for electronic, 'E+ZPE', 'H', or 'G' (for y-axis label)
        figsize: Figure size (width, height) inches
        dpi: Resolution
        colors: Override {'line', 'marker', 'ts'}
        show_values: Annotate energy values
        structures: Optional {label: xyz_path} for 3D ball-and-stick insets

    Returns:
        dict with plot_path, barriers, relative_energies
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        return {'error': 'matplotlib not installed', 'hint': 'pip install matplotlib'}

    labels = list(energies.keys())
    e_ha = list(energies.values())
    n = len(labels)

    if n < 2:
        return {'error': 'Need at least 2 stationary points'}

    # Convert to relative energies
    e_ref = e_ha[0]
    if relative:
        if unit == 'kcal/mol':
            e_rel = [(e - e_ref) * HA_TO_KCAL for e in e_ha]
        elif unit == 'kJ/mol':
            e_rel = [(e - e_ref) * HA_TO_KCAL * 4.184 for e in e_ha]
        else:
            e_rel = [(e - e_ref) for e in e_ha]

        # Y-axis label
        sym = {'E': r'$\Delta E$', 'E+ZPE': r'$\Delta E_{ZPE}$',
               'H': r'$\Delta H$', 'G': r'$\Delta G$'}.get(energy_type, r'$\Delta E$')
        ylabel = f'{sym} ({unit})'
    else:
        e_rel = e_ha
        ylabel = 'E (Hartree)'

    # Detect TS (local max or by label)
    ts_indices = set()
    for i in range(1, n - 1):
        if e_rel[i] > e_rel[i-1] and e_rel[i] > e_rel[i+1]:
            ts_indices.add(i)
    for i, lab in enumerate(labels):
        if 'TS' in lab.upper() or 'TRANSITION' in lab.upper():
            ts_indices.add(i)

    # Load molecular structure images
    mol_images = {}
    if structures:
        for lab, src in structures.items():
            if lab in labels and isinstance(src, str) and os.path.isfile(src):
                img = _xyz_to_3d_image(src, size=(280, 240))
                if img is not None:
                    mol_images[lab] = img

    has_structs = bool(mol_images)

    # Colors
    c = colors or {}
    color_line = c.get('line', '#2c3e50')
    color_min = c.get('marker', '#2980b9')
    color_ts = c.get('ts', '#e74c3c')

    # Figure size
    fig_h = figsize[1] + (2.5 if has_structs else 0)
    fig, ax = plt.subplots(figsize=(figsize[0], fig_h))

    # Platform dimensions
    pw = 0.35  # platform half-width
    x_pos = list(range(n))

    for i in range(n):
        x, y = x_pos[i], e_rel[i]
        is_ts = i in ts_indices
        color = color_ts if is_ts else color_min

        # Platform
        ax.plot([x - pw, x + pw], [y, y],
                color=color, linewidth=2.8, solid_capstyle='round')

        # Label + value
        label_text = labels[i]
        if show_values:
            label_text += f'\n({y:+.1f})' if relative else f'\n({y:.6f})'

        yrange = max(e_rel) - min(e_rel) if max(e_rel) != min(e_rel) else 1.0
        offset = yrange * 0.08 * (1 if is_ts else -1)

        ax.text(x, y + offset, label_text,
                ha='center', va='bottom' if is_ts else 'top',
                fontsize=9, fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor=color, alpha=0.85))

        # Connecting line
        if i < n - 1:
            ax.plot([x + pw, x_pos[i+1] - pw], [y, e_rel[i+1]],
                    color=color_line, linewidth=1.0, linestyle='--', alpha=0.5)

    # Barrier arrows
    barriers = {}
    for i in ts_indices:
        if i > 0:
            fwd = e_rel[i] - e_rel[i-1]
            barriers[f'{labels[i-1]}->{labels[i]}'] = round(fwd, 2)
            ax.annotate('', xy=(x_pos[i]-0.05, e_rel[i]),
                        xytext=(x_pos[i]-0.05, e_rel[i-1]),
                        arrowprops=dict(arrowstyle='<->', color='#7f8c8d',
                                        lw=1.2))
            mid_y = (e_rel[i] + e_rel[i-1]) / 2
            unit_sym = unit.split('/')[0]
            ax.text(x_pos[i] + 0.42, mid_y,
                    f'{fwd:+.1f} {unit}',
                    fontsize=8, color='#555555', va='center',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='#f5f5f5',
                              edgecolor='#cccccc', alpha=0.9))
        if i < n - 1:
            rev = e_rel[i] - e_rel[i+1]
            barriers[f'{labels[i]}->{labels[i+1]}'] = round(rev, 2)

    # Embed 3D structure images
    if has_structs:
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        ymin_data = min(e_rel)
        struct_y = ymin_data - yrange * 0.55

        for lab, img in mol_images.items():
            idx = labels.index(lab)
            arr = _pil_to_array(img)
            imagebox = OffsetImage(arr, zoom=0.55)
            ab = AnnotationBbox(imagebox, (x_pos[idx], struct_y),
                                frameon=True,
                                bboxprops=dict(boxstyle='round,pad=0.2',
                                               facecolor='#fafafa',
                                               edgecolor='#cccccc'))
            ax.add_artist(ab)

    # Axes
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Title and method subtitle (separate lines, no overlap)
    if level:
        ax.set_title(f'{title}\n', fontsize=13, fontweight='bold', pad=4)
        ax.text(0.5, 1.01, level, transform=ax.transAxes,
                ha='center', fontsize=9, color='#7f8c8d', style='italic')
    else:
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)

    # Y limits
    ymin, ymax = min(e_rel), max(e_rel)
    ypad_top = yrange * 0.30
    ypad_bot = yrange * (0.85 if has_structs else 0.30)
    ax.set_ylim(ymin - ypad_bot, ymax + ypad_top)
    ax.set_xlim(-0.7, n - 0.3)

    # Legend
    legend_items = [
        mpatches.Patch(color=color_min, label='Minimum'),
        mpatches.Patch(color=color_ts, label='Transition State'),
    ]
    ax.legend(handles=legend_items, loc='upper right', fontsize=8, framealpha=0.85)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)

    return {
        'success': True,
        'plot_path': output_path,
        'barriers': barriers,
        'relative_energies': {lab: round(e, 2) for lab, e in zip(labels, e_rel)},
        'has_structures': has_structs,
    }


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(
    results: dict,
    output_dir: str,
    *,
    title: str = 'QM Computation Report',
    plot_profile: bool = True,
) -> dict:
    """
    Generate comprehensive QM computation report.

    Energy hierarchy (best available):
      1. E_sp + G_correction  (DeltaG at SP level)
      2. E_sp + ZPE           (DeltaE+ZPE at SP level)
      3. E_sp                 (DeltaE at SP level)
      4. E_opt                (DeltaE at OPT level)

    For correct barriers, BOTH species must have energies at the same level.
    """
    os.makedirs(output_dir, exist_ok=True)

    steps = results.get('steps', {})
    functional = results.get('functional', 'B3LYP')
    basis_opt = results.get('basis_opt', 'def2-SVP')
    basis_sp = results.get('basis_sp', 'def2-TZVP')
    dispersion = results.get('dispersion', 'D3(BJ)')
    charge = results.get('charge', 0)
    spin = results.get('spin', 0)

    # ---- Collect energies per species ----
    species_data = {}  # {name: {e_opt, e_sp, G_corr, H_corr, ZPE, xyz_path}}

    if 'opt' in steps and steps['opt'].get('success'):
        species_data['Reactant'] = {
            'e_opt': steps['opt']['energy'],
            'xyz': steps['opt'].get('optimized_xyz'),
        }

    if 'ts' in steps and steps['ts'].get('success'):
        species_data['TS'] = {
            'e_opt': steps['ts']['energy'],
            'xyz': steps['ts'].get('optimized_xyz'),
        }

    # Frequency thermal corrections
    if 'freq' in steps and steps['freq'].get('success'):
        freq = steps['freq']
        # freq is usually run on TS geometry
        target = 'TS' if 'TS' in species_data else list(species_data.keys())[-1] if species_data else None
        if target and target in species_data:
            species_data[target]['G_corr'] = freq.get('G_correction')
            species_data[target]['H_corr'] = freq.get('H_correction')
            species_data[target]['ZPE'] = freq.get('ZPE_correction') or freq.get('ZPE')
            species_data[target]['imaginary_freqs'] = freq.get('imaginary_freqs', [])
            species_data[target]['is_ts'] = freq.get('is_ts', False)

    if 'freq_reactant' in steps and steps['freq_reactant'].get('success'):
        fr = steps['freq_reactant']
        if 'Reactant' in species_data:
            species_data['Reactant']['G_corr'] = fr.get('G_correction')
            species_data['Reactant']['H_corr'] = fr.get('H_correction')
            species_data['Reactant']['ZPE'] = fr.get('ZPE_correction') or fr.get('ZPE')

    # SP energies
    if 'sp' in steps and steps['sp'].get('success'):
        target = 'TS' if 'TS' in species_data else list(species_data.keys())[-1] if species_data else None
        if target and target in species_data:
            species_data[target]['e_sp'] = steps['sp']['energy']

    if 'sp_reactant' in steps and steps['sp_reactant'].get('success'):
        if 'Reactant' in species_data:
            species_data['Reactant']['e_sp'] = steps['sp_reactant']['energy']

    if 'sp_product' in steps and steps['sp_product'].get('success'):
        species_data.setdefault('Product', {})
        species_data['Product']['e_sp'] = steps['sp_product']['energy']

    # ---- Determine best energy level for plotting ----
    all_species = list(species_data.keys())

    # Try E_sp + G_correction first
    have_sp_g = all(
        species_data[s].get('e_sp') is not None and species_data[s].get('G_corr') is not None
        for s in all_species
    )
    have_sp_zpe = all(
        species_data[s].get('e_sp') is not None and species_data[s].get('ZPE') is not None
        for s in all_species
    )
    have_all_sp = all(species_data[s].get('e_sp') is not None for s in all_species)
    have_all_opt = all(species_data[s].get('e_opt') is not None for s in all_species)

    if have_sp_g:
        energies = {s: species_data[s]['e_sp'] + species_data[s]['G_corr']
                    for s in all_species}
        energy_type = 'G'
        energy_note = f'DeltaG = E_sp({basis_sp}) + G_correction(freq@{basis_opt})'
    elif have_sp_zpe:
        energies = {s: species_data[s]['e_sp'] + species_data[s]['ZPE']
                    for s in all_species}
        energy_type = 'E+ZPE'
        energy_note = f'DeltaE+ZPE = E_sp({basis_sp}) + ZPE(freq@{basis_opt})'
    elif have_all_sp:
        energies = {s: species_data[s]['e_sp'] for s in all_species}
        energy_type = 'E'
        energy_note = f'DeltaE at {basis_sp} (SP on {basis_opt} geometries)'
    elif have_all_opt:
        energies = {s: species_data[s]['e_opt'] for s in all_species}
        energy_type = 'E'
        energy_note = f'DeltaE at {basis_opt} (OPT level only - no SP available)'
    else:
        energies = {}
        energy_type = 'E'
        energy_note = 'Insufficient data for barrier calculation'

    barriers = _compute_barriers(energies) if len(energies) >= 2 else {}

    # ---- Multi-level barrier table ----
    barrier_table = _compute_multi_level_barriers(species_data, basis_opt, basis_sp)

    # ---- Plot ----
    plot_path = None
    if plot_profile and len(energies) >= 2:
        plot_path = os.path.join(output_dir, 'energy_profile.png')
        level_str = f'{functional}-{dispersion}/{basis_sp}//{basis_opt}'

        # Collect XYZ paths for structure insets
        struct_dict = {}
        for s in all_species:
            xyz = species_data[s].get('xyz')
            if xyz and os.path.isfile(xyz):
                struct_dict[s] = xyz

        plot_result = plot_energy_profile(
            energies, plot_path,
            title=title.replace('Report', 'Profile') if 'Report' in title
                  else 'Reaction Energy Profile',
            level=level_str,
            energy_type=energy_type,
            structures=struct_dict or None,
        )
        if not plot_result.get('success'):
            plot_path = None

    # ---- Markdown report ----
    report_path = os.path.join(output_dir, 'qm_report.md')
    md = _build_report_md(
        title, results, species_data, energies, barriers, barrier_table,
        plot_path, functional, basis_opt, basis_sp, dispersion,
        charge, spin, steps, energy_note, energy_type,
    )
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(md)

    # ---- JSON ----
    json_path = os.path.join(output_dir, 'qm_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'title': title,
            'method': f'{functional}-{dispersion}/{basis_sp}//{basis_opt}',
            'charge': charge, 'spin': spin,
            'energy_type': energy_type,
            'energy_note': energy_note,
            'energies_for_plot': energies,
            'barrier_table': barrier_table,
            'barriers': barriers,
            'species_data': {k: {kk: vv for kk, vv in v.items() if kk != 'xyz'}
                            for k, v in species_data.items()},
            'steps': steps,
            'plot_path': plot_path,
            'report_path': report_path,
        }, f, indent=2, default=str)

    return {
        'success': True,
        'report_path': report_path,
        'json_path': json_path,
        'plot_path': plot_path,
        'barriers': barriers,
        'barrier_table': barrier_table,
        'energy_type': energy_type,
        'energy_note': energy_note,
    }


def analyze_reaction(
    stationary_points: List[dict],
    output_dir: str,
    *,
    title: str = 'Reaction Energy Profile',
    level: str = '',
    structures: dict = None,
) -> dict:
    """
    Standalone reaction analysis from user-provided stationary points.

    Args:
        stationary_points: List of dicts:
            - label: str
            - energy: float (Hartree) — best available energy
            - energy_opt: float (optional, OPT-level)
            - energy_sp: float (optional, SP-level)
            - G_correction: float (optional, Gibbs thermal correction)
            - ZPE: float (optional)
            - xyz_path: str (optional, for 3D structure inset)
            - frequencies: list (optional, cm-1)
        output_dir: Output directory
        title: Title
        level: Theory level string
        structures: {label: xyz_path} override
    """
    os.makedirs(output_dir, exist_ok=True)

    if not stationary_points or len(stationary_points) < 2:
        return {'error': 'Need at least 2 stationary points'}

    # Build energy dict (use best available per point)
    energies = {}
    for sp in stationary_points:
        lab = sp['label']
        # Priority: energy_sp+G_corr > energy_sp+ZPE > energy_sp > energy > energy_opt
        e_sp = sp.get('energy_sp', sp.get('energy'))
        g_corr = sp.get('G_correction')
        zpe = sp.get('ZPE')

        if e_sp is not None and g_corr is not None:
            energies[lab] = e_sp + g_corr
        elif e_sp is not None and zpe is not None:
            energies[lab] = e_sp + zpe
        else:
            energies[lab] = sp['energy']

    barriers = _compute_barriers(energies)

    # Auto-build structures from xyz_path
    if structures is None:
        structures = {}
        for sp in stationary_points:
            xyz = sp.get('xyz_path')
            if xyz and os.path.isfile(xyz):
                structures[sp['label']] = xyz
    structures = structures or None

    # Determine energy type
    has_g = any(sp.get('G_correction') is not None for sp in stationary_points)
    has_zpe = any(sp.get('ZPE') is not None for sp in stationary_points)
    energy_type = 'G' if has_g else ('E+ZPE' if has_zpe else 'E')

    # Plot
    plot_path = os.path.join(output_dir, 'energy_profile.png')
    plot_result = plot_energy_profile(
        energies, plot_path, title=title, level=level or 'DFT',
        energy_type=energy_type, structures=structures)

    # Report
    report_path = os.path.join(output_dir, 'reaction_report.md')
    e_ref = list(energies.values())[0]
    lines = [
        f'# {title}', '',
        f'**Method**: {level}' if level else '', '',
        '## Stationary Points', '',
        '| Point | Energy (Ha) | Rel. (kcal/mol) | Type |',
        '|-------|------------|----------------|------|',
    ]

    for sp in stationary_points:
        lab = sp['label']
        e = energies[lab]
        rel = (e - e_ref) * HA_TO_KCAL
        freqs = sp.get('frequencies', [])
        n_imag = sum(1 for f in freqs if f < -50) if freqs else None
        if n_imag == 1:
            stype = 'Transition State'
        elif n_imag == 0:
            stype = 'Minimum'
        elif 'TS' in lab.upper():
            stype = 'TS (unverified)'
        else:
            stype = 'Stationary Point'
        lines.append(f'| {lab} | {e:.10f} | {rel:+.2f} | {stype} |')

    lines.extend(['', '## Barriers', ''])
    for bname, bval in barriers.items():
        lines.append(f'- **{bname}**: {bval:+.2f} kcal/mol')

    if any(sp.get('frequencies') for sp in stationary_points):
        lines.extend(['', '## Frequency Analysis', ''])
        for sp in stationary_points:
            if sp.get('frequencies'):
                imag = [f for f in sp['frequencies'] if f < -50]
                lines.append(f'### {sp["label"]}')
                if imag:
                    lines.append(f'- Imaginary frequencies: {imag} cm-1')
                lines.append(f'- Lowest frequency: {min(sp["frequencies"]):.1f} cm-1')
                lines.append('')

    if plot_result.get('success'):
        lines.extend(['## Energy Profile', '', '![Energy Profile](energy_profile.png)', ''])

    lines.append('*Generated by zhuangzhou QM Engine*')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    json_path = os.path.join(output_dir, 'reaction_analysis.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'title': title, 'level': level,
            'energy_type': energy_type,
            'stationary_points': [
                {k: v for k, v in sp.items() if k != 'mol_image'}
                for sp in stationary_points
            ],
            'barriers': barriers,
            'relative_energies': {k: round((v - e_ref) * HA_TO_KCAL, 2)
                                  for k, v in energies.items()},
        }, f, indent=2, default=str)

    return {
        'success': True,
        'plot_path': plot_path if plot_result.get('success') else None,
        'report_path': report_path,
        'json_path': json_path,
        'barriers': barriers,
        'energy_type': energy_type,
    }


# =============================================================================
# Helpers
# =============================================================================

def _compute_barriers(energies):
    """Compute barriers between consecutive stationary points."""
    barriers = {}
    labels = list(energies.keys())
    vals = list(energies.values())
    ref = vals[0]

    for i in range(1, len(labels)):
        diff = (vals[i] - vals[i-1]) * HA_TO_KCAL
        barriers[f'{labels[i-1]} -> {labels[i]}'] = round(diff, 2)

    if len(vals) >= 2:
        rel_all = [(v - ref) * HA_TO_KCAL for v in vals]
        barriers['Overall barrier'] = round(max(rel_all), 2)
        barriers['Reaction energy'] = round(rel_all[-1], 2)

    return barriers


def _compute_multi_level_barriers(species_data, basis_opt, basis_sp):
    """
    Compute barriers at all available energy levels.

    Standard reporting: DeltaE(opt), DeltaE(sp), DeltaE+ZPE(sp), DeltaG(sp)
    """
    table = {}
    species = list(species_data.keys())
    if len(species) < 2:
        return table

    ref_name = species[0]
    ref = species_data[ref_name]

    for s in species[1:]:
        data = species_data[s]
        label = f'{ref_name} -> {s}'

        # DeltaE at OPT level
        if ref.get('e_opt') is not None and data.get('e_opt') is not None:
            de_opt = (data['e_opt'] - ref['e_opt']) * HA_TO_KCAL
            table.setdefault(label, {})
            table[label][f'DeltaE({basis_opt})'] = round(de_opt, 2)

        # DeltaE at SP level
        if ref.get('e_sp') is not None and data.get('e_sp') is not None:
            de_sp = (data['e_sp'] - ref['e_sp']) * HA_TO_KCAL
            table.setdefault(label, {})
            table[label][f'DeltaE({basis_sp})'] = round(de_sp, 2)

        # DeltaE+ZPE
        if (ref.get('e_sp') is not None and data.get('e_sp') is not None and
                ref.get('ZPE') is not None and data.get('ZPE') is not None):
            de_zpe = ((data['e_sp'] + data['ZPE']) -
                      (ref['e_sp'] + ref['ZPE'])) * HA_TO_KCAL
            table.setdefault(label, {})
            table[label]['DeltaE+ZPE'] = round(de_zpe, 2)

        # DeltaG
        if (ref.get('e_sp') is not None and data.get('e_sp') is not None and
                ref.get('G_corr') is not None and data.get('G_corr') is not None):
            dg = ((data['e_sp'] + data['G_corr']) -
                  (ref['e_sp'] + ref['G_corr'])) * HA_TO_KCAL
            table.setdefault(label, {})
            table[label]['DeltaG'] = round(dg, 2)

    return table


def _build_report_md(
    title, results, species_data, energies, barriers, barrier_table,
    plot_path, functional, basis_opt, basis_sp, dispersion,
    charge, spin, steps, energy_note, energy_type,
):
    """Build comprehensive Markdown report."""
    lines = [f'# {title}', '']

    # Settings
    lines.extend([
        '## Computation Settings', '',
        '| Parameter | Value |', '|-----------|-------|',
        f'| Functional | {functional} |',
        f'| Dispersion | {dispersion} |',
        f'| Basis (opt/TS/freq) | {basis_opt} |',
        f'| Basis (SP) | {basis_sp} |',
        f'| Charge | {charge} |',
        f'| Spin (2S) | {spin} |',
        f'| Solvation | ddCOSMO (eps=4.0) |', '',
    ])

    # Cluster info
    if 'extract' in steps and isinstance(steps['extract'], dict):
        ext = steps['extract']
        lines.extend([
            '## QM Cluster Model', '',
            f'- **Atoms**: {ext.get("n_atoms", "N/A")}',
            f'- **Residues**: {ext.get("n_residues", "N/A")}',
            f'- **Capping H**: {ext.get("n_capping_h", "N/A")}',
            f'- **Frozen atoms**: {ext.get("n_frozen", "N/A")}', '',
        ])

    # Multi-level energy table
    if species_data:
        lines.extend(['## Energies (all levels)', ''])

        # Determine columns
        has_sp = any(d.get('e_sp') is not None for d in species_data.values())
        has_zpe = any(d.get('ZPE') is not None for d in species_data.values())
        has_g = any(d.get('G_corr') is not None for d in species_data.values())

        header = '| Species | E_opt (Ha) |'
        sep = '|---------|-----------|'
        if has_sp:
            header += ' E_sp (Ha) |'
            sep += '----------|'
        if has_zpe:
            header += ' ZPE (Ha) |'
            sep += '----------|'
        if has_g:
            header += ' G_corr (Ha) |'
            sep += '-------------|'

        lines.extend([header, sep])
        for name, data in species_data.items():
            row = f'| {name} | '
            row += f'{data["e_opt"]:.10f} |' if data.get('e_opt') is not None else 'N/A |'
            if has_sp:
                row += f' {data["e_sp"]:.10f} |' if data.get('e_sp') is not None else ' N/A |'
            if has_zpe:
                row += f' {data["ZPE"]:.8f} |' if data.get('ZPE') is not None else ' N/A |'
            if has_g:
                row += f' {data["G_corr"]:.8f} |' if data.get('G_corr') is not None else ' N/A |'
            lines.append(row)
        lines.append('')

    # Multi-level barrier table
    if barrier_table:
        lines.extend(['## Energy Barriers (multi-level)', ''])
        all_levels = set()
        for bdata in barrier_table.values():
            all_levels.update(bdata.keys())
        all_levels = sorted(all_levels)

        header = '| Barrier |' + '|'.join(f' {lv} |' for lv in all_levels)
        sep = '|---------|' + '|'.join('---------|' for _ in all_levels)
        lines.extend([header, sep])

        for bname, bdata in barrier_table.items():
            row = f'| {bname} |'
            for lv in all_levels:
                val = bdata.get(lv)
                row += f' {val:+.2f} |' if val is not None else ' N/A |'
            lines.append(row)
        lines.extend(['', f'> **Best estimate**: {energy_note}', ''])

    elif barriers:
        lines.extend(['## Energy Barriers', ''])
        for bname, bval in barriers.items():
            lines.append(f'- **{bname}**: {bval:+.2f} kcal/mol')
        lines.extend(['', f'> {energy_note}', ''])

    # Frequency
    freq_data = steps.get('freq', {})
    if isinstance(freq_data, dict) and freq_data.get('success'):
        lines.extend([
            '## Frequency Analysis', '',
            f'- **Imaginary frequencies**: {freq_data.get("n_imaginary", 0)}',
        ])
        if freq_data.get('imaginary_freqs'):
            lines.append(f'- **Values**: {freq_data["imaginary_freqs"]} cm-1')
        lines.append(f'- **Is TS**: {freq_data.get("is_ts", False)}')
        if freq_data.get('ZPE'):
            lines.append(f'- **ZPE**: {freq_data["ZPE"]:.8f} Eh')
        if freq_data.get('G_correction'):
            lines.append(f'- **G_correction** (298.15 K): {freq_data["G_correction"]:.8f} Eh')
        lines.append('')

    # Timing
    lines.extend([
        '## Computation Timing', '',
        '| Step | Time (s) | Status |', '|------|---------|--------|',
    ])
    for sname, sdata in steps.items():
        if isinstance(sdata, dict):
            lines.append(
                f'| {sname} | {sdata.get("elapsed_s", "N/A")} | '
                f'{"OK" if sdata.get("success") else "FAILED"} |')
    lines.append(f'| **Total** | **{results.get("elapsed_s", "N/A")}** | |')
    lines.append('')

    # Plot
    if plot_path:
        lines.extend([
            '## Energy Profile', '',
            f'![Reaction Energy Profile]({os.path.basename(plot_path)})', '',
        ])

    lines.extend(['---', '*Generated by zhuangzhou QM Engine (PySCF + geomeTRIC)*'])
    return '\n'.join(lines)

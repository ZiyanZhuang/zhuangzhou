"""
ChimeraX MD Trajectory Analysis
=================================
Trajectory visualization, attribute mapping, morph animation,
and BILD porcupine arrows for ChimeraX.
"""

import logging
import os
import numpy as np

from .config import CHIMERAX_EXE, RENDER_DEFAULTS, TIMEOUT_RENDER

logger = logging.getLogger(__name__)


def export_defattr(values, attr_name='displacement', output_path=None,
                   recipient='residues'):
    """Export per-residue (or per-atom) values as ChimeraX .defattr file.

    Args:
        values: dict {resid: float} or list of dicts with 'resid' and attr_name keys
        attr_name: attribute name (alphanumeric, no spaces)
        output_path: where to write .defattr file
        recipient: 'residues' or 'atoms'

    Returns:
        dict with defattr_path
    """
    if not output_path:
        return {'error': 'output_path required'}

    # Normalize input
    if isinstance(values, list):
        mapping = {}
        for item in values:
            rid = item.get('resid')
            val = item.get(attr_name, item.get('displacement', item.get('value')))
            if rid is not None and val is not None:
                mapping[int(rid)] = float(val)
    elif isinstance(values, dict):
        mapping = {int(k): float(v) for k, v in values.items()}
    else:
        return {'error': f'values must be dict or list, got {type(values).__name__}'}

    if not mapping:
        return {'error': 'No values to export'}

    lines = [
        f'attribute: {attr_name}',
        f'recipient: {recipient}',
    ]
    for resid in sorted(mapping):
        lines.append(f'\t:{resid}\t{mapping[resid]:.4f}')

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    logger.info("Exported .defattr: %s (%d values)", output_path, len(mapping))
    return {'defattr_path': output_path, 'n_values': len(mapping), 'attr_name': attr_name}


def export_bild_arrows(origins, vectors, output_path,
                       color='1 0 0', scale=3.0, radius=0.15):
    """Export porcupine arrows as ChimeraX BILD format.

    Args:
        origins: Nx3 array of arrow start points
        vectors: Nx3 array of displacement vectors
        output_path: .bild file path
        color: RGB as "R G B" string (0-1 range)
        scale: vector scaling factor
        radius: arrow cylinder radius (A)

    Returns:
        dict with bild_path, n_arrows
    """
    origins = np.asarray(origins)
    vectors = np.asarray(vectors)

    if origins.shape[0] != vectors.shape[0]:
        return {'error': 'origins and vectors must have same length'}

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    n = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f'.color {color}\n')
        for i in range(len(origins)):
            mag = np.linalg.norm(vectors[i])
            if mag < 0.3:
                continue
            o = origins[i]
            tip = o + vectors[i] * scale
            # Cylinder body (80% of arrow)
            mid = o + vectors[i] * scale * 0.8
            f.write(f'.cylinder {o[0]:.3f} {o[1]:.3f} {o[2]:.3f} '
                    f'{mid[0]:.3f} {mid[1]:.3f} {mid[2]:.3f} {radius:.3f}\n')
            # Cone head (20% of arrow)
            f.write(f'.cone {mid[0]:.3f} {mid[1]:.3f} {mid[2]:.3f} '
                    f'{tip[0]:.3f} {tip[1]:.3f} {tip[2]:.3f} {radius * 2.5:.3f}\n')
            n += 1

    logger.info("Exported BILD arrows: %s (%d arrows)", output_path, n)
    return {'bild_path': output_path, 'n_arrows': n}


def generate_trajectory_cxc(pdb_path, output_dir=None, color_by='secondary',
                            heme=True, save_mp4=True, save_png=False):
    """Generate .cxc script for trajectory animation in ChimeraX.

    ChimeraX handles multi-model PDB as coordsets natively.

    Args:
        pdb_path: multi-model PDB trajectory file
        output_dir: where to save .cxc (and mp4 if rendered)
        color_by: secondary / chain / bfactor
        heme: show HEME cofactor
        save_mp4: include movie recording commands
        save_png: save first-frame PNG

    Returns:
        dict with cxc_path
    """
    pdb_path = os.path.abspath(pdb_path)
    if not os.path.exists(pdb_path):
        return {'error': f'PDB not found: {pdb_path}'}

    if not output_dir:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(pdb_path))[0]
    pdb_fwd = pdb_path.replace('\\', '/')
    out_fwd = output_dir.replace('\\', '/')

    lines = [
        f'open "{pdb_fwd}"',
        'set bgColor white',
        'cartoon',
        'hide atoms',
    ]

    if color_by == 'secondary':
        lines += [
            'color helix #4477AA',
            'color strand #DDAA33',
            'color coil #BB5566',
        ]
    elif color_by == 'chain':
        lines.append('color bychain')

    if heme:
        lines += [
            'show :HEM atoms',
            'style :HEM stick',
            'color :HEM #CC7777',
            'color :HEM@FE #FF8800',
            'size :HEM@FE atomRadius 0.6',
        ]

    lines += [
        'lighting full',
        'graphics silhouettes true',
    ]

    if save_png:
        lines.append(
            f'save "{out_fwd}/{stem}_frame1.png" '
            f'width {RENDER_DEFAULTS["width"]} '
            f'height {RENDER_DEFAULTS["height"]} '
            f'supersample {RENDER_DEFAULTS["supersample"]}'
        )

    if save_mp4:
        lines += [
            f'movie record',
            f'coordset #1',
            f'wait',
            f'movie encode "{out_fwd}/{stem}_chimerax.mp4" quality high',
        ]

    lines.append('# Remove "exit" line below to keep ChimeraX open for interaction')
    lines.append('exit')

    cxc_path = os.path.join(output_dir, f'{stem}_trajectory.cxc')
    with open(cxc_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    logger.info("Trajectory .cxc script: %s", cxc_path)
    return {
        'cxc_path': cxc_path,
        'hint': 'Open in ChimeraX: File > Open, or run from CLI. '
                'Remove the "exit" line to keep ChimeraX open.',
    }


def generate_morph_cxc(pdb_first, pdb_last, output_dir=None,
                       n_frames=50, save_mp4=True):
    """Generate .cxc script for morphing between two structures.

    Args:
        pdb_first: starting PDB
        pdb_last: ending PDB
        output_dir: output directory
        n_frames: interpolation frames
        save_mp4: include movie encoding

    Returns:
        dict with cxc_path
    """
    pdb_first = os.path.abspath(pdb_first)
    pdb_last = os.path.abspath(pdb_last)
    for p in [pdb_first, pdb_last]:
        if not os.path.exists(p):
            return {'error': f'PDB not found: {p}'}

    if not output_dir:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    first_fwd = pdb_first.replace('\\', '/')
    last_fwd = pdb_last.replace('\\', '/')
    out_fwd = output_dir.replace('\\', '/')

    lines = [
        f'open "{first_fwd}"',
        f'open "{last_fwd}"',
        'set bgColor white',
        'cartoon',
        'hide atoms',
        'color helix #4477AA',
        'color strand #DDAA33',
        'color coil #BB5566',
        'show :HEM atoms',
        'style :HEM stick',
        'color :HEM #CC7777',
        f'morph #1 #2 frames {n_frames}',
        'lighting full',
        'graphics silhouettes true',
    ]

    if save_mp4:
        lines += [
            'movie record',
            'coordset #3',  # morph model
            'wait',
            f'movie encode "{out_fwd}/morph_animation.mp4" quality high',
        ]

    lines.append('exit')

    cxc_path = os.path.join(output_dir, 'morph_animation.cxc')
    with open(cxc_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    return {'cxc_path': cxc_path}


def generate_comparison_cxc(work_dir, comparison_result=None):
    """Generate .cxc script for structure comparison visualization.

    Colors protein by per-residue displacement from compare_structures(),
    shows HEME, highlights mobile regions.

    Args:
        work_dir: MD simulation directory (with structure_comparison/)
        comparison_result: output of compare_structures() (auto-loaded if None)

    Returns:
        dict with cxc_path, defattr_path
    """
    work_dir = os.path.abspath(work_dir)
    compare_dir = os.path.join(work_dir, 'structure_comparison')
    out_dir = os.path.join(work_dir, 'chimerax_viz')
    os.makedirs(out_dir, exist_ok=True)

    # Auto-load comparison result
    if comparison_result is None:
        frame_first = os.path.join(compare_dir, 'frame_0ps.pdb')
        if not os.path.exists(frame_first):
            return {'error': 'No structure comparison data. Run compare_structures() first.'}
        # Find last frame
        import glob
        frames = sorted(glob.glob(os.path.join(compare_dir, 'frame_*ps.pdb')))
        if len(frames) < 2:
            return {'error': 'Need at least 2 frames for comparison'}
        frame_last = frames[-1]
    else:
        pdbs = comparison_result.get('frame_pdbs', [])
        if len(pdbs) < 2:
            return {'error': 'comparison_result has <2 frames'}
        frame_first = pdbs[0]
        frame_last = pdbs[-1]

    if not os.path.exists(frame_first) or not os.path.exists(frame_last):
        return {'error': 'Frame PDBs not found'}

    # Export displacement as .defattr
    result = {}
    if comparison_result and 'per_residue' in comparison_result:
        defattr_path = os.path.join(out_dir, 'displacement.defattr')
        dr = export_defattr(
            comparison_result['per_residue'],
            attr_name='displacement',
            output_path=defattr_path,
        )
        result.update(dr)

    first_fwd = frame_first.replace('\\', '/')
    last_fwd = frame_last.replace('\\', '/')
    out_fwd = out_dir.replace('\\', '/')
    defattr_fwd = os.path.join(out_dir, 'displacement.defattr').replace('\\', '/')

    # Identify mobile residues for labeling
    mobile_specs = ''
    if comparison_result and 'mobile_regions' in comparison_result:
        mobile_ids = [str(r['resid']) for r in comparison_result['mobile_regions'][:10]]
        mobile_specs = ','.join(mobile_ids)

    lines = [
        '# Structure Comparison Visualization',
        '# Load first (blue) and last (red) frames',
        f'open "{first_fwd}" name first',
        f'open "{last_fwd}" name last',
        '',
        '# Style',
        'set bgColor white',
        'cartoon',
        'hide atoms',
        '',
        '# Color first=blue, last=red',
        'color #1 #377EB8',
        'color #2 #E41A1C70',  # semi-transparent red
        '',
        '# Superimpose',
        'matchmaker #2 to #1',
        '',
        '# HEME',
        'show :HEM atoms',
        'style :HEM stick',
        'color #1:HEM #CC7777',
        'color #2:HEM #FF666670',
        '',
    ]

    # Color by displacement attribute
    if os.path.exists(os.path.join(out_dir, 'displacement.defattr')):
        lines += [
            f'# Color by displacement',
            f'open "{defattr_fwd}"',
            f'color byattribute displacement #1 palette #377EB8:#FFFFFF:#E41A1C',
            '',
        ]

    # Label mobile residues
    if mobile_specs:
        lines += [
            f'# Mobile residues as sticks',
            f'show #1:{mobile_specs} atoms',
            f'style #1:{mobile_specs} stick',
            f'label #1:{mobile_specs} residues text "{{0.name}}{{0.number}}"',
            '',
        ]

    lines += [
        'lighting full',
        'graphics silhouettes true',
        f'save "{out_fwd}/comparison_overlay.png" '
        f'width {RENDER_DEFAULTS["width"]} height {RENDER_DEFAULTS["height"]} '
        f'supersample {RENDER_DEFAULTS["supersample"]}',
        'exit',
    ]

    cxc_path = os.path.join(out_dir, 'comparison.cxc')
    with open(cxc_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    result['cxc_path'] = cxc_path
    result['png_path'] = os.path.join(out_dir, 'comparison_overlay.png')
    logger.info("Comparison .cxc: %s", cxc_path)
    return result


def generate_porcupine_cxc(work_dir, pca_data=None):
    """Generate porcupine plot as BILD arrows for ChimeraX.

    Uses PCA mode 1 displacement vectors from gmx covar/anaeig output,
    or falls back to first-vs-last CA displacement.

    Args:
        work_dir: MD directory
        pca_data: dict with 'origins' (Nx3) and 'vectors' (Nx3) arrays.
            If None, computes from structure_comparison frame PDBs.

    Returns:
        dict with bild_path, cxc_path
    """
    work_dir = os.path.abspath(work_dir)
    out_dir = os.path.join(work_dir, 'chimerax_viz')
    os.makedirs(out_dir, exist_ok=True)

    if pca_data and 'origins' in pca_data and 'vectors' in pca_data:
        origins = np.asarray(pca_data['origins'])
        vectors = np.asarray(pca_data['vectors'])
    else:
        # Compute from first/last frame CA displacement
        compare_dir = os.path.join(work_dir, 'structure_comparison')
        import glob
        frames = sorted(glob.glob(os.path.join(compare_dir, 'frame_*ps.pdb')))
        if len(frames) < 2:
            return {'error': 'No frame PDBs found. Run compare_structures() first.'}

        ca_first = _parse_ca(frames[0])
        ca_last = _parse_ca(frames[-1])
        common = sorted(set(ca_first) & set(ca_last))
        if not common:
            return {'error': 'No common CA atoms between frames'}

        origins = np.array([ca_first[r] for r in common])
        vectors = np.array([ca_last[r] - ca_first[r] for r in common])

    bild_path = os.path.join(out_dir, 'porcupine.bild')
    bild_result = export_bild_arrows(origins, vectors, bild_path,
                                     color='0.9 0.2 0.1', scale=2.0, radius=0.12)

    # Generate accompanying .cxc to load structure + arrows
    ref_pdb = None
    compare_dir = os.path.join(work_dir, 'structure_comparison')
    import glob
    frames = sorted(glob.glob(os.path.join(compare_dir, 'frame_*ps.pdb')))
    if frames:
        ref_pdb = frames[0]

    cxc_lines = [
        '# Porcupine Plot — Principal Motion Vectors',
        'set bgColor white',
    ]
    if ref_pdb:
        cxc_lines.append(f'open "{ref_pdb.replace(chr(92), "/")}"')
        cxc_lines += ['cartoon', 'hide atoms', 'color #B0B0B0']
        cxc_lines += [
            'show :HEM atoms', 'style :HEM stick',
            'color :HEM #CC7777',
        ]
    cxc_lines += [
        f'open "{bild_path.replace(chr(92), "/")}"',
        'lighting full',
        'graphics silhouettes true',
        f'save "{out_dir.replace(chr(92), "/")}/porcupine_chimerax.png" '
        f'width {RENDER_DEFAULTS["width"]} height {RENDER_DEFAULTS["height"]} '
        f'supersample {RENDER_DEFAULTS["supersample"]}',
        'exit',
    ]

    cxc_path = os.path.join(out_dir, 'porcupine.cxc')
    with open(cxc_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(cxc_lines) + '\n')

    result = {**bild_result, 'cxc_path': cxc_path,
              'png_path': os.path.join(out_dir, 'porcupine_chimerax.png')}
    return result


def _parse_ca(pdb_path):
    """Parse CA atom coordinates from PDB. Returns {resid: xyz_array}."""
    ca = {}
    with open(pdb_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('ENDMDL'):
                break
            if (line.startswith('ATOM') or line.startswith('HETATM')):
                atom = line[12:16].strip()
                resname = line[17:20].strip()
                if atom == 'CA' and resname not in {'HEM', 'HEME', 'HEC'}:
                    resid = int(line[22:26].strip())
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    ca[resid] = np.array([x, y, z])
    return ca

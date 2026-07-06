"""
ChimeraX Structure Renderer
=============================
Publication-quality structure rendering via ChimeraX command scripts.

Generates .cxc scripts and optionally executes them for PNG output.
Works on Windows (GUI + exit) and Linux (--nogui + OSMesa).
"""

import logging
import os
import subprocess
import tempfile

from .config import CHIMERAX_EXE, RENDER_DEFAULTS, TIMEOUT_RENDER, FORCE_NOGUI

logger = logging.getLogger(__name__)


def _run_chimerax(script_path, timeout=TIMEOUT_RENDER):
    """Execute a .cxc script via ChimeraX CLI. Returns (stdout, success)."""
    if not CHIMERAX_EXE or not os.path.isfile(CHIMERAX_EXE):
        return 'ChimeraX not found', False

    script_path_fwd = script_path.replace('\\', '/')

    if FORCE_NOGUI:
        cmd = [CHIMERAX_EXE, '--nogui', '--offscreen', '--script', script_path]
    else:
        # Windows: GUI mode — open the .cxc script file (ChimeraX executes all
        # commands including the trailing 'exit', then closes the window).
        # Path MUST be quoted inside the command string to handle spaces.
        cmd = [CHIMERAX_EXE, '--cmd', f'open "{script_path_fwd}"']

    # Save log path alongside the script for post-mortem debugging
    log_path = script_path.replace('.cxc', '_chimerax.log')

    try:
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            env=env, errors='replace',
        )
        output = (result.stdout or '') + (result.stderr or '')
        # Always save the full log — invaluable for debugging
        try:
            with open(log_path, 'w', encoding='utf-8') as lf:
                lf.write(f'CMD: {" ".join(cmd)}\n')
                lf.write(f'RETURNCODE: {result.returncode}\n\n')
                lf.write('=== STDOUT ===\n')
                lf.write(result.stdout or '')
                lf.write('\n=== STDERR ===\n')
                lf.write(result.stderr or '')
        except OSError:
            pass
        return output, result.returncode == 0
    except subprocess.TimeoutExpired:
        return f'ChimeraX timed out after {timeout}s', False
    except Exception as e:
        return str(e), False


def render_structure(pdb_path, output_png=None, output_dir=None,
                     style='cartoon', color_by='chain',
                     selections=None, highlight_residues=None,
                     heme=True, ligand_resname=None,
                     width=None, height=None, execute=True):
    """Render a PDB structure with ChimeraX.

    Args:
        pdb_path: input PDB file
        output_png: explicit output PNG path
        output_dir: output directory (auto-names PNG)
        style: cartoon / ribbon / surface / stick
        color_by: chain / secondary / bfactor / element
        selections: list of ChimeraX selection strings to highlight
        highlight_residues: list of residue specs like ['10', '151', '307']
            to show as sticks
        heme: auto-detect and render HEME cofactor
        ligand_resname: 3-letter ligand residue name to highlight
        width/height: image dimensions (pixels)
        execute: if True, run ChimeraX to produce PNG

    Returns:
        dict with cxc_path, png_path (if executed), status
    """
    pdb_path = os.path.abspath(pdb_path)
    if not os.path.exists(pdb_path):
        return {'error': f'PDB not found: {pdb_path}'}

    stem = os.path.splitext(os.path.basename(pdb_path))[0]
    if not output_dir:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    if not output_png:
        output_png = os.path.join(output_dir, f'{stem}_chimerax.png')
    output_png = os.path.abspath(output_png)

    w = width or RENDER_DEFAULTS['width']
    h = height or RENDER_DEFAULTS['height']
    ss = RENDER_DEFAULTS['supersample']
    pdb_fwd = pdb_path.replace('\\', '/')
    png_fwd = output_png.replace('\\', '/')

    lines = [
        f'open "{pdb_fwd}"',
        f'set bgColor {RENDER_DEFAULTS["bg_color"]}',
    ]

    # Style
    if style == 'cartoon':
        lines.append('cartoon')
        lines.append('hide atoms')
    elif style == 'ribbon':
        lines.append('ribbon')
        lines.append('hide atoms')
    elif style == 'surface':
        lines.append('surface')
    elif style == 'stick':
        lines.append('show atoms')
        lines.append('style stick')

    # Coloring
    if color_by == 'chain':
        lines.append('color bychain')
    elif color_by == 'secondary':
        lines.append('color helix #4477AA')
        lines.append('color strand #DDAA33')
        lines.append('color coil #BB5566')
    elif color_by == 'bfactor':
        lines.append('color bfactor palette blue:white:red')
    elif color_by == 'element':
        lines.append('color byhetero')

    # HEME detection
    if heme:
        lines.append('# HEME cofactor')
        lines.append('show :HEM atoms')
        lines.append('style :HEM stick')
        lines.append('color :HEM #CC7777')
        lines.append('color :HEM@FE #FF8800')
        lines.append('size :HEM@FE atomRadius 0.6')

    # Ligand
    if ligand_resname:
        rn = ligand_resname
        lines.append(f'# Ligand {rn}')
        lines.append(f'show :{rn} atoms')
        lines.append(f'style :{rn} stick')
        lines.append(f'color :{rn} #44BB66')

    # Highlight specific residues as sticks
    if highlight_residues:
        specs = ','.join(str(r) for r in highlight_residues)
        lines.append(f'# Highlighted residues')
        lines.append(f'show :{specs} atoms')
        lines.append(f'style :{specs} stick')

    # Custom selections
    if selections:
        for sel in selections:
            lines.append(sel)

    # Rendering
    lines.append('lighting full')
    lines.append('graphics silhouettes true')
    lines.append(f'save "{png_fwd}" width {w} height {h} supersample {ss}')
    lines.append('exit')

    # Write script
    cxc_path = os.path.join(output_dir, f'{stem}_render.cxc')
    with open(cxc_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    result = {'cxc_path': cxc_path, 'png_path': output_png}

    if execute:
        output, ok = _run_chimerax(cxc_path)
        if os.path.exists(output_png):
            sz = os.path.getsize(output_png)
            result['status'] = 'rendered'
            result['file_size'] = sz
            logger.info("ChimeraX render: %s (%d KB)", output_png, sz // 1024)
        else:
            result['status'] = 'script_only'
            result['warning'] = ('ChimeraX could not render image '
                                 '(nogui without OSMesa?). '
                                 'Open the .cxc script manually.')
            if 'error' in output.lower() or 'limitation' in output.lower():
                result['chimerax_log'] = output[-500:]

    return result

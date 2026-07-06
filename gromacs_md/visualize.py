"""
GROMACS MD Visualization — trajectory animation + porcupine plot.

Uses PyMOL headless for rendering, ffmpeg for video assembly.
"""

import glob
import logging
import os
import subprocess
import tempfile

import numpy as np

from .config import GMX_BIN, PYMOL_PYTHON, FFMPEG, TIMEOUT_SETUP

logger = logging.getLogger(__name__)


def _run_gmx(cmd, work_dir, input_text=None, timeout=TIMEOUT_SETUP):
    full_cmd = f'"{GMX_BIN}" {cmd}'
    try:
        r = subprocess.run(full_cmd, shell=True, capture_output=True, text=True,
                           input=input_text, timeout=timeout, cwd=work_dir)
        return (r.stdout or '') + (r.stderr or ''), r.returncode == 0
    except Exception as e:
        return str(e), False


def _find_group_number(work_dir, tpr, group_name='non-Water'):
    """Find index group number by name from gmx make_ndx output."""
    import re
    out, _ = _run_gmx(f'make_ndx -f {tpr}', work_dir, input_text='q\n')
    for line in out.splitlines():
        m = re.match(r'\s*(\d+)\s+' + re.escape(group_name) + r'\s', line)
        if m:
            return m.group(1)
    # Fallback: try common group numbers
    if group_name == 'non-Water':
        return '18'
    return '0'


def render_trajectory(work_dir, tpr='md.tpr', xtc='md.xtc',
                      dt_ps=100, fps=15, width=1280, height=960):
    """Render MD trajectory as MP4 + GIF + fitted PDB.

    Args:
        work_dir: MD simulation directory
        dt_ps: time between frames in ps (default 100 -> ~100 frames for 10ns)
        fps: playback frames per second

    Returns:
        dict with mp4_path, gif_path, fitted_pdb, n_frames
    """
    if not PYMOL_PYTHON:
        return {'error': 'PyMOL not found', 'hint': 'Set PYMOL_PYTHON in config'}
    if not FFMPEG:
        return {'error': 'ffmpeg not found', 'hint': 'Set FFMPEG in config'}

    anim_dir = os.path.join(work_dir, 'animation')
    render_dir = os.path.join(anim_dir, 'rendered')
    os.makedirs(render_dir, exist_ok=True)

    # 1. Fix PBC: center on protein, whole molecules
    pbc_xtc = os.path.join(anim_dir, 'traj_pbc.xtc')
    _run_gmx(f'trjconv -s {tpr} -f {xtc} -o "{pbc_xtc}" -center -pbc mol -ur compact',
             work_dir, input_text='1\n0\n')

    # 2. Remove translation + rotation: least-squares fit backbone to reference
    fit_xtc = os.path.join(anim_dir, 'traj_fit.xtc')
    _run_gmx(f'trjconv -s {tpr} -f "{pbc_xtc}" -o "{fit_xtc}" -fit rot+trans',
             work_dir, input_text='4\n0\n')  # fit=Backbone, output=System

    # Determine non-Water group number (varies by system)
    nonwater_grp = _find_group_number(work_dir, tpr, 'non-Water')

    # 3. Export fitted non-water trajectory as single multi-model PDB (with SS records)
    traj_pdb = os.path.join(work_dir, 'trajectory_nowater.pdb')
    _run_gmx(f'trjconv -s {tpr} -f "{fit_xtc}" -o "{traj_pdb}" -dt {dt_ps}',
             work_dir, input_text=f'{nonwater_grp}\n')
    if os.path.exists(traj_pdb):
        _inject_ss_records(traj_pdb, work_dir=work_dir)

    # 4. Extract non-water frames as separate PDB files for rendering
    _run_gmx(f'trjconv -s {tpr} -f "{fit_xtc}" -o "{anim_dir}/frames.pdb" -dt {dt_ps} -sep',
             work_dir, input_text=f'{nonwater_grp}\n')

    frames = sorted(glob.glob(os.path.join(anim_dir, 'frames*.pdb')),
                    key=lambda x: int(os.path.basename(x).replace('frames', '').replace('.pdb', '')))
    if not frames:
        return {'error': 'No frames extracted'}

    # Auto-detect cofactors/ligands from first frame
    cofactor_resnames = set()
    ligand_resnames = set()
    standard_aa = {
        'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
        'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL',
        'HID','HIE','HIP','HSD','HSE','HSP','CYX','CYM','ASH','GLH','LYN'}
    known_cofactors = {'HEM','HEME','HEC','HEA','HEO2','FAD','FMN','NAD',
                       'NAP','PLP','TPP','COA','SF4','FES','F3S'}
    skip_names = {'NA','CL','K','SOL','HOH','WAT','TIP','SPC','NA+','CL-','K+',
                  'IOD','BR','DOD','FE','ZN','MN','MG','CU','CO','NI','CA',
                  'FE2','FE3','ZN2','MN2','MG2','CU2','CA2'}
    with open(frames[0], encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith(('ATOM','HETATM')):
                # GROMACS trjconv writes ALL atoms as ATOM (never HETATM)
                # 4-char resnames (e.g. HEME) span columns 17-20, so check both
                rn3 = line[17:20].strip()
                rn4 = line[17:21].strip()
                # Use the longer match if it's a known cofactor
                if rn4 in known_cofactors:
                    cofactor_resnames.add(rn4)
                elif rn3 in known_cofactors:
                    cofactor_resnames.add(rn3)
                elif rn4 not in standard_aa and rn4 not in skip_names and rn4:
                    # No HETATM gate — GROMACS writes ATOM for everything
                    ligand_resnames.add(rn4)

    # Build PyMOL cofactor/ligand display commands
    cofactor_cmds = []
    for rn in cofactor_resnames:
        cofactor_cmds.append(f'cmd.show("sticks", "resn {rn}")')
        cofactor_cmds.append(f'cmd.color("salmon", "resn {rn} and elem C")')
        cofactor_cmds.append(f'cmd.set("stick_radius", 0.25, "resn {rn}")')
        # Show Fe as sphere
        cofactor_cmds.append(f'cmd.show("spheres", "resn {rn} and elem FE")')
        cofactor_cmds.append(f'cmd.set("sphere_scale", 0.5, "resn {rn} and elem FE")')
        cofactor_cmds.append(f'cmd.color("orange", "resn {rn} and elem FE")')
    for rn in ligand_resnames:
        cofactor_cmds.append(f'cmd.show("sticks", "resn {rn}")')
        cofactor_cmds.append(f'cmd.color("cyan", "resn {rn} and elem C")')
        cofactor_cmds.append(f'cmd.set("stick_radius", 0.2, "resn {rn}")')
    cofactor_block = '\n'.join(cofactor_cmds)

    # 3. PyMOL rendering script
    frames_escaped = anim_dir.replace('\\', '/')
    render_escaped = render_dir.replace('\\', '/')
    pymol_script = f'''
import pymol
from pymol import cmd
import glob, os

frames = sorted(glob.glob(os.path.join(r"{frames_escaped}", "frames*.pdb")),
                key=lambda x: int(os.path.basename(x).replace("frames","").replace(".pdb","")))
for f in frames:
    cmd.load(f, "traj")

cmd.bg_color("white")
cmd.hide("everything")
cmd.show("cartoon", "polymer")
cmd.color("slate", "polymer")

# Cofactors and ligands (auto-detected)
{cofactor_block}

# Metal ions as spheres
cmd.show("spheres", "elem FE+ZN+MN+MG+CU+CO+NI and not polymer")
cmd.set("sphere_scale", 0.5, "elem FE+ZN+MN+MG+CU+CO+NI")
cmd.color("orange", "elem FE")

cmd.orient("polymer")
cmd.zoom("polymer", 5)
cmd.set("orthoscopic", 1)
cmd.set("ambient", 0.4)
cmd.set("spec_reflect", 0.3)
cmd.set("ray_shadows", 0)

nframes = cmd.count_states("traj")
for i in range(1, nframes + 1):
    cmd.set("state", i)
    cmd.ray({width}, {height})
    cmd.png(os.path.join(r"{render_escaped}", f"frame_{{i:04d}}.png"), dpi=150)
cmd.quit()
'''
    script_path = os.path.join(anim_dir, 'render_traj.py')
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(pymol_script)

    subprocess.run([PYMOL_PYTHON, script_path], capture_output=True,
                   text=True, timeout=600, cwd=anim_dir)

    rendered = sorted(glob.glob(os.path.join(render_dir, 'frame_*.png')))
    if len(rendered) < 2:
        return {'error': f'Only {len(rendered)} frames rendered'}

    result = {'n_frames': len(rendered)}

    # Trajectory PDB (non-water, fitted)
    if os.path.exists(traj_pdb):
        result['trajectory_pdb'] = traj_pdb

    # 4. Assemble MP4
    mp4_path = os.path.join(work_dir, 'trajectory_animation.mp4')
    subprocess.run([
        FFMPEG, '-y', '-framerate', str(fps),
        '-i', os.path.join(render_dir, 'frame_%04d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
        mp4_path,
    ], capture_output=True, text=True, timeout=120)
    if os.path.exists(mp4_path):
        result['mp4'] = mp4_path

    # 5. Assemble GIF
    gif_path = os.path.join(work_dir, 'trajectory_animation.gif')
    subprocess.run([
        FFMPEG, '-y', '-framerate', str(fps - 2),
        '-i', os.path.join(render_dir, 'frame_%04d.png'),
        '-vf', 'scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',
        gif_path,
    ], capture_output=True, text=True, timeout=120)
    if os.path.exists(gif_path):
        result['gif'] = gif_path

    logger.info("Trajectory animation: %d frames, mp4=%s, gif=%s",
                len(rendered), result.get('mp4'), result.get('gif'))
    return result


def render_porcupine(work_dir, tpr='md.tpr', xtc='md.xtc',
                     scale=3.0, threshold=0.3):
    """Generate porcupine plot showing principal motion (PC1).

    Uses gmx covar + anaeig for PCA, then PyMOL CGO arrows.

    Args:
        scale: arrow amplification factor
        threshold: minimum displacement (A) to show arrow

    Returns:
        dict with png_path, pse_path, top_residues
    """
    if not PYMOL_PYTHON:
        return {'error': 'PyMOL not found'}

    pca_dir = os.path.join(work_dir, 'pca')
    os.makedirs(pca_dir, exist_ok=True)

    # 1. Covariance analysis on C-alpha
    avg_pdb = os.path.join(pca_dir, 'average.pdb')
    eigvec = os.path.join(pca_dir, 'eigenvec.trr')
    eigval = os.path.join(pca_dir, 'eigenval.xvg')
    extreme = os.path.join(pca_dir, 'ev1_extreme.pdb')

    if not os.path.exists(eigvec):
        _run_gmx(f'covar -s {tpr} -f {xtc} -o "{eigval}" -v "{eigvec}" -av "{avg_pdb}"',
                 work_dir, input_text='3\n3\n', timeout=300)

    if not os.path.exists(extreme):
        _run_gmx(f'anaeig -s {tpr} -f {xtc} -v "{eigvec}" -eig "{eigval}" '
                 f'-extr "{extreme}" -first 1 -last 1 -nframes 2',
                 work_dir, input_text='3\n3\n')

    if not os.path.exists(extreme):
        return {'error': 'PCA extreme structures not generated'}

    # 2. Parse CA coordinates from extreme structures
    models = _parse_ca_coords(extreme)
    if len(models) < 2:
        return {'error': f'Expected 2 models, got {len(models)}'}

    displacement = models[1] - models[0]
    magnitudes = np.linalg.norm(displacement, axis=1)
    res_info = _parse_ca_info(extreme)

    avg_models = _parse_ca_coords(avg_pdb)
    origins = avg_models[0] if avg_models else (models[0] + models[1]) / 2

    # 3. Generate PyMOL CGO script
    cgo_lines = []
    arrow_count = 0
    for i in range(len(origins)):
        if magnitudes[i] < threshold:
            continue
        ox, oy, oz = origins[i]
        dx, dy, dz = displacement[i] * scale
        total_len = np.sqrt(dx**2 + dy**2 + dz**2)
        if total_len < 0.1:
            continue

        norm_mag = min(magnitudes[i] / max(magnitudes.max(), 1e-6), 1.0)
        r, g, b = norm_mag, 0.1, 1.0 - norm_mag
        shaft_r, head_r, head_len = 0.12, 0.25, 0.4
        frac = max(0, 1 - head_len / total_len)
        sx, sy, sz = ox + dx * frac, oy + dy * frac, oz + dz * frac
        ex, ey, ez = ox + dx, oy + dy, oz + dz

        cgo_lines.append(
            f'CYLINDER,{ox:.3f},{oy:.3f},{oz:.3f},{sx:.3f},{sy:.3f},{sz:.3f},'
            f'{shaft_r},{r:.2f},{g:.2f},{b:.2f},{r:.2f},{g:.2f},{b:.2f},'
        )
        cgo_lines.append(
            f'CONE,{sx:.3f},{sy:.3f},{sz:.3f},{ex:.3f},{ey:.3f},{ez:.3f},'
            f'{head_r},0.0,{r:.2f},{g:.2f},{b:.2f},{r:.2f},{g:.2f},{b:.2f},1.0,1.0,'
        )
        arrow_count += 1

    # Top displaced residues
    top_idx = np.argsort(magnitudes)[-10:]
    top_residues = [(res_info[i], float(magnitudes[i]))
                    for i in reversed(top_idx) if i < len(res_info)]

    # Label commands
    label_cmds = []
    for idx in top_idx:
        if idx < len(res_info) and magnitudes[idx] > threshold:
            resid_num = ''.join(c for c in res_info[idx] if c.isdigit())
            if resid_num:
                label_cmds.append(
                    f'cmd.label("resi {resid_num} and name CA", '
                    f'"\\"{res_info[idx]}\\\"")')

    avg_escaped = avg_pdb.replace('\\', '/')
    png_path = os.path.join(work_dir, 'porcupine_plot.png').replace('\\', '/')
    pse_path = os.path.join(work_dir, 'porcupine_plot.pse').replace('\\', '/')

    pymol_script = f'''
import pymol
from pymol import cmd
from pymol.cgo import *

cmd.load(r"{avg_escaped}", "average")
cmd.bg_color("white")
cmd.hide("everything")
cmd.show("cartoon", "average")
cmd.spectrum("b", "blue_white_red", "average")
cmd.set("cartoon_transparency", 0.3)

obj = [
{chr(10).join('    ' + l for l in cgo_lines)}
]
cmd.load_cgo(obj, "porcupine")

{chr(10).join(label_cmds)}

cmd.set("label_size", 14)
cmd.set("label_color", "black")
cmd.set("label_font_id", 7)
cmd.orient("average")
cmd.zoom("average", 5)
cmd.set("orthoscopic", 1)
cmd.set("ambient", 0.4)
cmd.set("spec_reflect", 0.3)
cmd.set("ray_shadows", 0)

cmd.ray(2400, 1800)
cmd.png(r"{png_path}", dpi=300)
cmd.save(r"{pse_path}")
cmd.quit()
'''

    # Set B-factors for spectrum coloring
    bfactor_cmds = []
    for i, mag in enumerate(magnitudes):
        if i < len(res_info):
            resid_num = ''.join(c for c in res_info[i] if c.isdigit())
            if resid_num:
                bfactor_cmds.append(
                    f'cmd.alter("resi {resid_num} and name CA", "b={mag:.3f}")')

    # Insert b-factor commands after load
    bf_block = '\n'.join(bfactor_cmds)
    pymol_script = pymol_script.replace(
        'cmd.spectrum("b"',
        bf_block + '\ncmd.spectrum("b"'
    )

    script_path = os.path.join(pca_dir, 'porcupine.py')
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(pymol_script)

    subprocess.run([PYMOL_PYTHON, script_path], capture_output=True,
                   text=True, timeout=120, cwd=pca_dir)

    result = {
        'n_arrows': arrow_count,
        'top_residues': top_residues,
        'max_displacement': float(magnitudes.max()),
        'mean_displacement': float(magnitudes.mean()),
    }

    real_png = os.path.join(work_dir, 'porcupine_plot.png')
    real_pse = os.path.join(work_dir, 'porcupine_plot.pse')
    if os.path.exists(real_png):
        result['png'] = real_png
    if os.path.exists(real_pse):
        result['pse'] = real_pse

    return result


def _parse_ca_coords(pdb_path):
    """Parse CA coordinates grouped by MODEL from PDB."""
    models = []
    current = []
    with open(pdb_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('ENDMDL') and current:
                models.append(np.array(current))
                current = []
            elif line.startswith('ATOM') and ' CA ' in line:
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except (ValueError, IndexError):
                    continue
                current.append([x, y, z])
    if current:
        models.append(np.array(current))
    return models


def _parse_ca_info(pdb_path):
    """Get residue labels for each CA atom."""
    info = []
    with open(pdb_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('ATOM') and ' CA ' in line:
                info.append(f'{line[17:20].strip()}{line[22:26].strip()}')
            if line.startswith('ENDMDL'):
                break
    return info


def _inject_ss_records(pdb_path, reference_pdb=None, work_dir=None):
    """Inject HELIX/SHEET records into trajectory PDB for proper visualization.

    Strategy: copy from reference PDB if available, otherwise run gmx do_dssp.
    """
    ss_lines = []

    # Try reference PDB first (original crystal structure)
    if reference_pdb and os.path.exists(reference_pdb):
        with open(reference_pdb, encoding='utf-8', errors='replace') as f:
            for line in f:
                if line.startswith(('HELIX', 'SHEET')):
                    ss_lines.append(line)
                elif line.startswith(('ATOM', 'HETATM')):
                    break
    else:
        # Search for any PDB in work_dir that might have SS records
        if work_dir:
            for name in ['2CPP.pdb', 'clean.pdb', 'input.pdb']:
                candidate = os.path.join(work_dir, name)
                if os.path.exists(candidate):
                    with open(candidate, encoding='utf-8', errors='replace') as f:
                        for line in f:
                            if line.startswith(('HELIX', 'SHEET')):
                                ss_lines.append(line)
                            elif line.startswith(('ATOM', 'HETATM')):
                                break
                    if ss_lines:
                        break

    if not ss_lines:
        return False

    # Read original, inject SS after REMARK/CRYST1 but before first MODEL/ATOM
    with open(pdb_path, encoding='utf-8', errors='replace') as f:
        content = f.readlines()

    new_content = []
    ss_injected = False
    for line in content:
        if not ss_injected and line.startswith(('MODEL', 'ATOM', 'HETATM')):
            for ss in ss_lines:
                new_content.append(ss if ss.endswith('\n') else ss + '\n')
            ss_injected = True
        new_content.append(line)

    with open(pdb_path, 'w', encoding='utf-8') as f:
        f.writelines(new_content)

    logger.info("Injected %d SS records into %s", len(ss_lines), pdb_path)
    return True


def process_trajectory(work_dir, tpr='md.tpr', xtc='md.xtc',
                       dt_ps=None, output_format='xtc'):
    """Process trajectory: fix PBC + remove translation/rotation.

    Produces analysis-ready trajectory with protein centered and
    least-squares fitted (backbone) to remove overall motion.
    PDB output includes HELIX/SHEET records from reference structure.

    Args:
        work_dir: MD simulation directory
        dt_ps: time step for output (None = keep all frames)
        output_format: 'xtc', 'pdb', or 'both'

    Returns:
        dict with fitted_xtc, fitted_pdb paths
    """
    result = {}
    proc_dir = os.path.join(work_dir, 'processed')
    os.makedirs(proc_dir, exist_ok=True)

    # Step 1: Fix PBC
    pbc_xtc = os.path.join(proc_dir, 'traj_pbc.xtc')
    out, ok = _run_gmx(
        f'trjconv -s {tpr} -f {xtc} -o "{pbc_xtc}" -center -pbc mol -ur compact',
        work_dir, input_text='1\n0\n')
    if not ok:
        return {'error': 'PBC fix failed', 'detail': out[-500:]}

    # Step 2: Least-squares fit
    dt_flag = f'-dt {dt_ps}' if dt_ps else ''

    if output_format in ('xtc', 'both'):
        fitted_xtc = os.path.join(work_dir, 'trajectory_fitted.xtc')
        out, ok = _run_gmx(
            f'trjconv -s {tpr} -f "{pbc_xtc}" -o "{fitted_xtc}" -fit rot+trans {dt_flag}',
            work_dir, input_text='4\n0\n')
        if ok and os.path.exists(fitted_xtc):
            result['fitted_xtc'] = fitted_xtc

    if output_format in ('pdb', 'both'):
        fitted_pdb = os.path.join(work_dir, 'trajectory_fitted.pdb')
        nowater_dt = f'-dt {dt_ps}' if dt_ps else '-dt 100'
        out, ok = _run_gmx(
            f'trjconv -s {tpr} -f "{pbc_xtc}" -o "{fitted_pdb}" -fit rot+trans {nowater_dt}',
            work_dir, input_text='4\n17\n')
        if ok and os.path.exists(fitted_pdb):
            _inject_ss_records(fitted_pdb, work_dir=work_dir)
            result['fitted_pdb'] = fitted_pdb

    logger.info("Trajectory processed: %s", result)
    return result

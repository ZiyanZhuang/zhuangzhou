"""
Vina Docking Engine
===================
Calls AutoDock Vina (CPU or GPU+) for molecular docking.
Parses results from log and PDBQT output.

Executable paths and defaults come from config.py.
"""

import logging
import os
import re
import subprocess

from .config import VINA_EXECUTABLES, VINA_DEFAULTS

logger = logging.getLogger(__name__)


def _find_vina() -> tuple:
    """Find available Vina executable. Returns (path, is_gpu)."""
    for p in VINA_EXECUTABLES:
        if os.path.exists(p):
            is_gpu = 'GPU' in os.path.basename(p).upper()
            return p, is_gpu
    return None, False


def run_vina(receptor_pdbqt: str, ligand_pdbqt: str, grid: dict,
             output_dir: str, exhaustiveness: int = None,
             num_modes: int = None, flex_pdbqt: str = None,
             search_depth: int = None, timeout: int = None) -> dict:
    """
    Run Vina docking.

    All parameters default to VINA_DEFAULTS from config.py if not specified.

    Returns:
        dict with output_pdbqt, log_file, affinities, best_affinity
        or dict with 'error' on failure
    """
    # Apply defaults from config
    exhaustiveness = exhaustiveness or VINA_DEFAULTS['exhaustiveness']
    num_modes = num_modes or VINA_DEFAULTS['num_modes']
    timeout = timeout or VINA_DEFAULTS['timeout']
    gpu_thread = VINA_DEFAULTS['gpu_thread']
    gpu_search_depth = search_depth or VINA_DEFAULTS['gpu_search_depth']
    cpu_count = VINA_DEFAULTS['cpu']

    vina_exe, is_gpu = _find_vina()
    if vina_exe is None:
        return {'error': 'Vina executable not found',
                'hint': f'Searched: {VINA_EXECUTABLES}'}

    os.makedirs(output_dir, exist_ok=True)
    out_pdbqt = os.path.abspath(os.path.join(output_dir, 'vina_docked.pdbqt'))
    log_file = os.path.abspath(os.path.join(output_dir, 'vina_log.txt'))

    cmd = [
        vina_exe,
        '--receptor', os.path.abspath(receptor_pdbqt),
        '--ligand', os.path.abspath(ligand_pdbqt),
        '--center_x', str(grid['center_x']),
        '--center_y', str(grid['center_y']),
        '--center_z', str(grid['center_z']),
        '--size_x', str(min(grid['size_x'], 60)),
        '--size_y', str(min(grid['size_y'], 60)),
        '--size_z', str(min(grid['size_z'], 60)),
        '--out', out_pdbqt,
        '--log', log_file,
        '--num_modes', str(num_modes),
    ]

    if flex_pdbqt:
        cmd += ['--flex', os.path.abspath(flex_pdbqt)]

    if is_gpu:
        cmd += ['--thread', str(gpu_thread)]
        if gpu_search_depth:
            cmd += ['--search_depth', str(gpu_search_depth)]
    else:
        cmd += ['--exhaustiveness', str(exhaustiveness), '--cpu', str(cpu_count)]

    label = 'Vina-GPU+' if is_gpu else 'Vina CPU'
    logger.info("%s: center=(%.1f, %.1f, %.1f) box=(%.0f, %.0f, %.0f) exhaust=%d",
                label,
                grid['center_x'], grid['center_y'], grid['center_z'],
                grid['size_x'], grid['size_y'], grid['size_z'],
                exhaustiveness)

    try:
        cwd = os.path.dirname(vina_exe) if is_gpu else None
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        if result.returncode != 0:
            return {'error': f'{label} failed (rc={result.returncode})',
                    'stderr': result.stderr[-500:] if result.stderr else ''}
    except subprocess.TimeoutExpired:
        return {'error': f'{label} timed out ({timeout}s)'}
    except Exception as e:
        return {'error': str(e)}

    # Parse results
    affinities = _parse_vina_log(log_file)
    if not affinities and os.path.exists(out_pdbqt):
        affinities = _parse_vina_pdbqt(out_pdbqt)

    best = affinities[0][1] if affinities else None
    logger.info("%s done: best = %s kcal/mol, %d poses", label, best, len(affinities))

    return {
        'output_pdbqt': out_pdbqt,
        'log_file': log_file,
        'affinities': affinities,
        'best_affinity': best,
        'vina_exe': label,
        'exhaustiveness': exhaustiveness,
    }


def _parse_vina_log(log_file: str) -> list:
    """Parse Vina log. Returns [(mode, affinity, rmsd_lb, rmsd_ub)]."""
    affinities = []
    if not os.path.exists(log_file):
        return affinities
    with open(log_file, encoding='utf-8') as f:
        in_table = False
        for line in f:
            if '-----+---' in line:
                in_table = True
                continue
            if in_table:
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        affinities.append((
                            int(parts[0]), float(parts[1]),
                            float(parts[2]), float(parts[3])))
                    except ValueError:
                        if affinities:
                            break
    return affinities


def _parse_vina_pdbqt(pdbqt_file: str) -> list:
    """Parse affinities from PDBQT REMARK lines."""
    affinities = []
    mode = 0
    with open(pdbqt_file, encoding='utf-8') as f:
        for line in f:
            m = re.search(r'REMARK VINA RESULT:\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', line)
            if m:
                mode += 1
                affinities.append((mode, float(m.group(1)), float(m.group(2)), float(m.group(3))))
    return affinities

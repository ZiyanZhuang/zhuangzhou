"""
Mutate & Minimize Engine
========================
Lightweight pipeline: mutate protein → pdb2gmx → [solvate] → EM → extract PDB.

Unlike full MD, this is fast (~30s) and focused on structural relaxation only.
Supports vacuum EM (fastest) and solvated EM (more physical).
"""

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


def mutate_and_minimize(
    pdb_path: str,
    mutations: Union[str, List[str]],
    *,
    output: str = None,
    output_dir: str = None,
    chain: str = "A",
    rebuild: bool = True,
    mode: str = "solvated",
    ff: str = "amber99sb-ildn",
    water: str = "tip3p",
    emtol: float = 500.0,
    nsteps: int = 50000,
    gpu: bool = True,
    keep_workdir: bool = False,
) -> dict:
    """
    Apply point mutation(s) then energy-minimize the structure.

    Args:
        pdb_path: Input PDB file path
        mutations: Single mutation string or list, e.g. "V9L" or ["V9L", "A5G"]
        output: Explicit output PDB path (auto-generated if None)
        output_dir: Directory for output files
        chain: Chain ID for mutation (default "A")
        rebuild: Try PyMOL side-chain rebuild before EM
        mode: "vacuum" (fastest, no solvent) or "solvated" (physical, with water+ions)
        ff: GROMACS force field
        water: Water model
        emtol: Energy minimization force tolerance (kJ/mol/nm)
        nsteps: Max EM steps
        gpu: Use GPU for EM
        keep_workdir: Keep intermediate GROMACS files

    Returns:
        dict with:
          success: bool
          output_pdb: path to minimized PDB
          mutations: list of applied mutations
          mutation_method: 'pymol' or 'backbone_only'
          em_converged: bool
          potential_energy: float (kJ/mol)
          max_force: float (kJ/mol/nm)
          em_steps: int
          elapsed_s: float
    """
    if not os.path.exists(pdb_path):
        return {'error': f'PDB not found: {pdb_path}', 'hint': 'Check file path'}

    if isinstance(mutations, str):
        mutations = [mutations]
    if not mutations:
        return {'error': 'No mutations provided', 'hint': 'e.g. "V9L" or ["V9L","A5G"]'}

    t0 = time.time()

    # Step 1: Mutate
    from zhuangzhou.mutagenesis import mutate
    mut_result = mutate(pdb_path, mutations, chain=chain, rebuild=rebuild)
    if not mut_result.get('success'):
        return mut_result  # pass through error

    mutant_pdb = mut_result['output_pdb']
    mutation_method = mut_result.get('method', 'unknown')
    mut_labels = mut_result.get('mutations', mutations)

    # Step 2: Minimize
    em_result = minimize_structure(
        mutant_pdb,
        output=output,
        output_dir=output_dir,
        mode=mode,
        ff=ff,
        water=water,
        emtol=emtol,
        nsteps=nsteps,
        gpu=gpu,
        keep_workdir=keep_workdir,
    )

    if 'error' in em_result:
        em_result['mutations'] = mut_labels
        em_result['mutation_method'] = mutation_method
        em_result['mutant_pdb_before_em'] = mutant_pdb
        return em_result

    elapsed = time.time() - t0
    em_result['mutations'] = mut_labels
    em_result['mutation_method'] = mutation_method
    em_result['elapsed_s'] = round(elapsed, 1)

    # Clean up intermediate mutant PDB if EM succeeded and output differs
    if em_result.get('output_pdb') != mutant_pdb and not keep_workdir:
        try:
            os.remove(mutant_pdb)
        except OSError:
            pass

    em_result['next_steps'] = [
        'smart_docking: 对突变体进行分子对接评估结合能力变化',
        'analyze_mutation: 用化学直觉分析突变对酶功能的影响',
        'gromacs_md: 运行MD模拟评估突变体动力学稳定性',
    ]

    return em_result


def minimize_structure(
    pdb_path: str,
    *,
    output: str = None,
    output_dir: str = None,
    mode: str = "solvated",
    ff: str = "amber99sb-ildn",
    water: str = "tip3p",
    emtol: float = 500.0,
    nsteps: int = 50000,
    gpu: bool = True,
    keep_workdir: bool = False,
) -> dict:
    """
    Energy-minimize a protein PDB structure using GROMACS.

    Args:
        pdb_path: Input PDB file
        output: Output PDB path (auto-named if None)
        output_dir: Output directory
        mode: "vacuum" or "solvated"
        ff: Force field
        water: Water model
        emtol: Force convergence tolerance
        nsteps: Max minimization steps
        gpu: Use GPU acceleration
        keep_workdir: Retain GROMACS work files

    Returns:
        dict with success, output_pdb, em_converged, potential_energy, etc.
    """
    from zhuangzhou.gromacs_md.config import GMX_BIN, GPU_FLAGS_EM, TIMEOUT_EM

    if not GMX_BIN or not os.path.isfile(GMX_BIN):
        return {'error': 'GROMACS not found',
                'hint': 'Set GMX_BIN env var or edit ~/.zhuangzhou/gromacs_md.ini'}

    if not os.path.exists(pdb_path):
        return {'error': f'PDB not found: {pdb_path}'}

    t0 = time.time()

    # Determine output path
    stem = Path(pdb_path).stem
    if output:
        out_pdb = output
    elif output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_pdb = os.path.join(output_dir, f'{stem}_em.pdb')
    else:
        # Convention: output defaults to CWD (project workspace), not input file dir
        out_pdb = os.path.join(os.getcwd(), f'{stem}_em.pdb')

    # Create temp work directory
    work_dir = out_pdb.replace('.pdb', '_gmx_em')
    os.makedirs(work_dir, exist_ok=True)

    try:
        # Copy input PDB to work dir
        work_pdb = os.path.join(work_dir, 'input.pdb')
        shutil.copy2(pdb_path, work_pdb)

        # Step 1: pdb2gmx — generate topology
        cmd = (f'pdb2gmx -f input.pdb -o protein.gro '
               f'-water {water} -ff {ff} -ignh')
        out, ok = _run_gmx(GMX_BIN, cmd, work_dir)
        if not ok:
            return {'error': 'pdb2gmx failed', 'detail': out[-500:],
                    'hint': 'Check for non-standard residues or missing atoms'}

        if mode == 'solvated':
            # Step 2a: Box + solvate + ions
            result = _solvate_for_em(GMX_BIN, work_dir)
            if 'error' in result:
                return result
            em_input_gro = 'system.gro'
        else:
            # Step 2b: Vacuum — large box with PBC but no solvent
            cmd = 'editconf -f protein.gro -o box.gro -c -d 3.0 -bt cubic'
            out, ok = _run_gmx(GMX_BIN, cmd, work_dir)
            if not ok:
                return {'error': 'editconf (vacuum box) failed', 'detail': out[-300:]}
            em_input_gro = 'box.gro'

        # Step 3: Write EM MDP
        em_mdp = os.path.join(work_dir, 'em.mdp')
        _write_em_mdp(em_mdp, emtol=emtol, nsteps=nsteps, vacuum=(mode == 'vacuum'))

        # Step 4: grompp
        cmd = (f'grompp -f em.mdp -c {em_input_gro} -p topol.top '
               f'-o em.tpr -maxwarn 5')
        out, ok = _run_gmx(GMX_BIN, cmd, work_dir)
        if not ok:
            return {'error': 'grompp failed', 'detail': out[-500:]}

        # Step 5: mdrun EM
        gpu_flags = GPU_FLAGS_EM if gpu else ''
        cmd = f'mdrun -v -deffnm em {gpu_flags}'
        out, ok = _run_gmx(GMX_BIN, cmd, work_dir, timeout=TIMEOUT_EM)
        if not ok:
            return {'error': 'EM mdrun failed', 'detail': out[-500:],
                    'hint': 'Try mode="vacuum" or increase nsteps'}

        # Step 6: Extract protein from minimized structure → PDB
        if mode == 'solvated':
            # trjconv to extract protein only (group 1 = Protein)
            cmd = 'trjconv -s em.tpr -f em.gro -o em_protein.pdb -pbc mol -ur compact'
            out, ok = _run_gmx(GMX_BIN, cmd, work_dir, input_text='Protein\n')
            if not ok:
                # Fallback: editconf
                cmd = 'editconf -f em.gro -o em_protein.pdb'
                out, ok = _run_gmx(GMX_BIN, cmd, work_dir)
            source_pdb = os.path.join(work_dir, 'em_protein.pdb')
        else:
            cmd = 'editconf -f em.gro -o em_protein.pdb'
            out, ok = _run_gmx(GMX_BIN, cmd, work_dir)
            source_pdb = os.path.join(work_dir, 'em_protein.pdb')

        if not os.path.exists(source_pdb):
            return {'error': 'Failed to extract minimized PDB'}

        # Copy to final output
        os.makedirs(os.path.dirname(out_pdb) or '.', exist_ok=True)
        shutil.copy2(source_pdb, out_pdb)

        # Parse EM results from log
        em_info = _parse_em_log(os.path.join(work_dir, 'em.log'))
        elapsed = time.time() - t0

        result = {
            'success': True,
            'output_pdb': out_pdb,
            'mode': mode,
            'ff': ff,
            'em_converged': em_info.get('converged', False),
            'potential_energy': em_info.get('potential_energy'),
            'max_force': em_info.get('max_force'),
            'em_steps': em_info.get('steps', 0),
            'elapsed_s': round(elapsed, 1),
        }

        return result

    finally:
        if not keep_workdir and os.path.exists(work_dir):
            try:
                shutil.rmtree(work_dir)
            except OSError:
                pass


def _run_gmx(gmx_bin, cmd, work_dir, input_text=None, timeout=120):
    """Run a GROMACS command."""
    full_cmd = f'"{gmx_bin}" {cmd}'
    try:
        result = subprocess.run(
            full_cmd, shell=True, capture_output=True, text=True,
            input=input_text, timeout=timeout, cwd=work_dir,
        )
        output = (result.stdout or '') + (result.stderr or '')
        return output, result.returncode == 0
    except subprocess.TimeoutExpired:
        return f'Timeout after {timeout}s', False
    except Exception as e:
        return str(e), False


def _write_em_mdp(path, emtol=500.0, nsteps=50000, vacuum=False):
    """Write energy minimization MDP file."""
    # GROMACS 2025 removed group scheme; vacuum uses large box + PBC
    lines = [
        f'; Energy Minimization {"(vacuum-like)" if vacuum else "(solvated)"}',
        'integrator  = steep',
        f'emtol       = {emtol}',
        'emstep      = 0.01',
        f'nsteps      = {nsteps}',
        'nstlist     = 10',
        'cutoff-scheme = Verlet',
        'coulombtype = PME',
        'rcoulomb    = 1.0',
        'rvdw        = 1.0',
        'pbc         = xyz',
    ]

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def _solvate_for_em(gmx_bin, work_dir):
    """Box + solvate + ions for EM-only (no index groups needed)."""
    from zhuangzhou.gromacs_md.config import DEFAULT_BOX_DIST, DEFAULT_BOX_TYPE

    # editconf
    cmd = (f'editconf -f protein.gro -o box.gro -c '
           f'-d {DEFAULT_BOX_DIST} -bt {DEFAULT_BOX_TYPE}')
    out, ok = _run_gmx(gmx_bin, cmd, work_dir)
    if not ok:
        return {'error': 'editconf failed', 'detail': out[-300:]}

    # solvate
    cmd = 'solvate -cp box.gro -cs spc216.gro -o solv.gro -p topol.top'
    out, ok = _run_gmx(gmx_bin, cmd, work_dir)
    if not ok:
        return {'error': 'solvate failed', 'detail': out[-300:]}

    # ions
    ions_mdp = os.path.join(work_dir, 'ions.mdp')
    with open(ions_mdp, 'w', encoding='utf-8') as f:
        f.write('; Ions\nintegrator = steep\nnsteps = 0\n')

    cmd = 'grompp -f ions.mdp -c solv.gro -p topol.top -o ions.tpr -maxwarn 5'
    out, ok = _run_gmx(gmx_bin, cmd, work_dir)
    if not ok:
        return {'error': 'grompp (ions) failed', 'detail': out[-300:]}

    # Detect water group name dynamically (SPC/TIP3P → SOL, OPC → OPC, etc.)
    water_group = 'SOL'
    solv_gro = os.path.join(work_dir, 'solv.gro')
    if os.path.exists(solv_gro):
        with open(solv_gro, encoding='utf-8', errors='replace') as _f:
            for _line in _f:
                _res = _line[5:10].strip() if len(_line) > 10 else ''
                if _res in ('SOL', 'HOH', 'WAT', 'OPC', 'TIP3', 'SPC'):
                    water_group = _res
                    break

    cmd = 'genion -s ions.tpr -o system.gro -p topol.top -pname NA -nname CL -neutral'
    out, ok = _run_gmx(gmx_bin, cmd, work_dir, input_text=f'{water_group}\n')
    if not ok:
        return {'error': 'genion failed', 'detail': out[-300:]}

    return {'success': True}


def _parse_em_log(log_path):
    """Extract EM convergence info from GROMACS log."""
    info = {'converged': False, 'steps': 0}
    if not os.path.exists(log_path):
        return info

    try:
        with open(log_path, encoding='utf-8', errors='replace') as f:
            content = f.read()

        import re

        # Check convergence: "Steepest Descents converged to Fmax < 500 in 1981 steps"
        conv_match = re.search(
            r'Steepest Descents converged to Fmax < [\d.]+ in (\d+) steps', content)
        if conv_match:
            info['converged'] = True
            info['steps'] = int(conv_match.group(1))

        # Extract potential energy and max force from last step
        for line in content.split('\n'):
            if 'Potential Energy' in line and '=' in line:
                try:
                    val = line.split('=')[-1].strip().split()[0]
                    info['potential_energy'] = float(val)
                except (ValueError, IndexError):
                    pass
            if 'Maximum force' in line and '=' in line:
                try:
                    val = line.split('=')[-1].strip().split()[0]
                    info['max_force'] = float(val)
                except (ValueError, IndexError):
                    pass

        # Fallback step count from "Step=N" patterns
        if info['steps'] == 0:
            steps = re.findall(r'Step=\s*(\d+)', content)
            if steps:
                info['steps'] = int(steps[-1])

    except Exception:
        pass

    return info

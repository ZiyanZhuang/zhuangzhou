"""
GROMACS MD Pipeline — Full protein-ligand MD workflow.

Pipeline: Split → Parameterize → Merge → Solvate → EM → NVT → NPT → MD → Analysis
"""

import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import (
    GPU_FLAGS_EM, GPU_FLAGS_FULL,
    TIMEOUT_EM, TIMEOUT_EQUIL, TIMEOUT_MD,
    DEFAULT_FF, DEFAULT_WATER, DEFAULT_BOX_DIST, DEFAULT_BOX_TYPE,
    DEFAULT_TEMP, DEFAULT_PRESSURE, TEMPLATE_DIR,
    OBABEL_EXE,
)
from .diagnostics import check_stage_sanity, diagnose_failure, format_diagnosis_report
from .prepare import (
    run_gmx, split_complex, parameterize_protein, parameterize_ligand,
    merge_system, solvate_system, create_index, _write_template,
    _extract_atomtypes, _remove_atomtypes_from_itp,
)
from .restraint_protocol import (
    get_equilibration_schedule, generate_soft_posre,
    generate_metal_posre, generate_cofactor_posre,
    inject_restraints_into_topology,
)
from .heme_topology import setup_heme_system, finalize_heme_topology, validate_heme_topology
from .heme_params import deploy_heme_files, build_topol, dedup_ligand_atomtypes
from .config import GMX_BIN, METALLOPROTEIN_FF

logger = logging.getLogger(__name__)


def _fix_nan_coordinates(gro_path):
    """Replace NaN coordinates in GRO file with 0.000 (pdb2gmx -missing artifact).

    GRO uses fixed-width format: residue(5) name(5) atomname(5) atomnr(5) x(8.3f) y(8.3f) z(8.3f).
    Coordinates start at column 20 in groups of 8 chars each.
    """
    try:
        with open(gro_path, encoding='utf-8') as f:
            lines = f.readlines()
        fixed = False
        for i, line in enumerate(lines):
            if i < 2 or i >= len(lines) - 1:  # skip header, count, box
                continue
            if 'nan' in line.lower():
                # Parse and rewrite coordinate fields (cols 20-44 for x,y,z)
                prefix = line[:20]
                coords_str = line[20:44]
                vel_str = line[44:] if len(line) > 44 else '\n'
                new_coords = []
                for j in range(3):
                    field = coords_str[j*8:(j+1)*8].strip()
                    try:
                        val = float(field)
                    except (ValueError, OverflowError):
                        val = 0.000
                    new_coords.append(f'{val:8.3f}')
                lines[i] = prefix + ''.join(new_coords) + vel_str
                fixed = True
        if fixed:
            with open(gro_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            logger.warning("Fixed NaN coordinates in %s", os.path.basename(gro_path))
    except Exception as e:
        logger.warning("Could not check/fix NaN in %s: %s", gro_path, e)


def _preflight_check(pdb_path, work_dir):
    """Validate inputs before starting any pipeline.

    Returns: (ok: bool, error_msg: str or None)
    """
    # Check PDB exists
    if not pdb_path or not os.path.exists(pdb_path):
        return False, f'Input PDB not found: {pdb_path}'

    # Check PDB is not empty
    try:
        size = os.path.getsize(pdb_path)
        if size < 100:
            return False, f'PDB file too small ({size} bytes): {pdb_path}'
    except OSError as e:
        return False, f'Cannot read PDB: {e}'

    # Check PDB has ATOM records
    has_atoms = False
    with open(pdb_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('ATOM'):
                has_atoms = True
                break
    if not has_atoms:
        return False, 'PDB has no ATOM records (is this a valid protein structure?)'

    # Check GROMACS is available
    if not GMX_BIN or not os.path.exists(str(GMX_BIN)):
        return False, f'GROMACS not found at: {GMX_BIN}. Set GMX_BIN env or update gromacs_md.ini'

    # Check work_dir is writable
    try:
        os.makedirs(work_dir, exist_ok=True)
        test_file = os.path.join(work_dir, '.write_test')
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write('ok')
        os.remove(test_file)
    except OSError as e:
        return False, f'Cannot write to work_dir: {e}'

    # Check MDP templates exist
    if not TEMPLATE_DIR or not os.path.isdir(str(TEMPLATE_DIR)):
        return False, f'MDP template directory not found: {TEMPLATE_DIR}'
    for mdp in ['em.mdp', 'nvt.mdp', 'npt.mdp', 'md.mdp', 'ions.mdp']:
        if not os.path.exists(str(TEMPLATE_DIR / mdp)):
            return False, f'Missing MDP template: {TEMPLATE_DIR / mdp}'

    return True, None


def _check_fe_coordination(work_dir, stage_name, coordination):
    """
    Check Fe-coordination distances after a simulation stage.

    Measures Fe-SG/Fe-NE2 distances in the output .gro file and warns
    if coordination is lost (distance > 0.35 nm).

    Returns:
        dict with check results and distances
    """
    import subprocess
    gro_file = os.path.join(work_dir, f'{stage_name}.gro')
    if not os.path.exists(gro_file):
        return {'checked': False, 'reason': f'{stage_name}.gro not found'}

    result = {'checked': True, 'distances': [], 'ok': True}

    # Parse GRO to measure distances directly
    with open(gro_file, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    if len(lines) < 3:
        return {'checked': False, 'reason': 'GRO file too short'}

    natoms = int(lines[1].strip())
    atoms = {}
    for line in lines[2:2 + natoms]:
        if len(line) < 44:
            continue
        try:
            resid = int(line[0:5].strip())
            resname = line[5:10].strip()
            atom_name = line[10:15].strip()
            x = float(line[20:28])
            y = float(line[28:36])
            z = float(line[36:44])
            atoms[(resname, resid, atom_name)] = (x, y, z)
        except (ValueError, IndexError):
            continue

    for c in coordination:
        metal_resid = c['metal_resid']
        metal_atom = c['metal_atom']
        prot_resid = c['protein_resid']
        prot_atom = c['protein_atom']

        # Find Fe position
        fe_pos = None
        for rn in ('HEME', 'HEM', 'HEO2'):
            key = (rn, metal_resid, metal_atom)
            if key in atoms:
                fe_pos = atoms[key]
                break

        # Find coordinating atom position
        prot_pos = None
        for rn in (c['protein_res'], 'CYS', 'CYS2', 'CYM', 'HSD', 'HSE', 'HIS'):
            key = (rn, prot_resid, prot_atom)
            if key in atoms:
                prot_pos = atoms[key]
                break

        if fe_pos and prot_pos:
            dx = fe_pos[0] - prot_pos[0]
            dy = fe_pos[1] - prot_pos[1]
            dz = fe_pos[2] - prot_pos[2]
            dist_nm = (dx**2 + dy**2 + dz**2) ** 0.5
            dist_a = dist_nm * 10

            status = 'OK' if dist_nm < 0.35 else 'DRIFTED'
            if dist_nm > 0.35:
                result['ok'] = False

            result['distances'].append({
                'bond_type': c['bond_type'],
                'atom1': f"{c['protein_res']}{prot_resid}-{prot_atom}",
                'atom2': f"{metal_atom}(HEME{metal_resid})",
                'distance_nm': round(dist_nm, 4),
                'distance_A': round(dist_a, 2),
                'crystal_A': round(c['distance'], 2),
                'status': status,
            })
            logger.info("  [%s] %s: %s-%s = %.3f nm (crystal: %.3f nm) [%s]",
                        stage_name, c['bond_type'],
                        f"{c['protein_res']}{prot_resid}", prot_atom,
                        dist_nm, c['distance'] / 10, status)

    return result


def _build_tau_t_ref_t(tc_grps, temp, tau=0.1):
    """Build tau_t and ref_t strings matching the number of tc_grps."""
    n_groups = len(tc_grps.split())
    tau_t = '     '.join([str(tau)] * n_groups)
    ref_t = '     '.join([str(temp)] * n_groups)
    return tau_t, ref_t


def _write_mdp_files(work_dir, sim_time_ns=10.0, temp=DEFAULT_TEMP,
                     tc_grps='Protein_LIG Water_and_ions',
                     has_ligand=True,
                     equil_define='-DPOSRES',
                     prod_define=''):
    """Write all MDP files with correct parameters (legacy single-phase).

    Args:
        equil_define: Preprocessor defines for NVT/NPT equilibration.
            Standard: '-DPOSRES' for backbone restraints.
            Metalloprotein: '-DPOSRES -DPOSRES_HEM' etc.
        prod_define: Preprocessor defines for production MD.
            Usually empty (no restraints) or '-DPOSRES_METAL' for metals.
    """
    nsteps_md = int(sim_time_ns * 1e6 / 2)  # dt=0.002 ps
    tau_t, ref_t = _build_tau_t_ref_t(tc_grps, temp)

    for name in ['em.mdp', 'ions.mdp', 'nvt.mdp', 'npt.mdp', 'md.mdp']:
        src = TEMPLATE_DIR / name
        dst = os.path.join(work_dir, name)
        with open(src, encoding='utf-8', errors='replace') as f:
            content = f.read()

        # Determine define for this stage
        if name in ('nvt.mdp', 'npt.mdp'):
            define = equil_define
        elif name == 'md.mdp':
            define = prod_define
        else:
            define = ''

        # Substitute placeholders
        content = content.replace('__DEFINE__', define)
        content = content.replace('__NSTEPS_MD__', str(nsteps_md))
        content = content.replace('__TEMP__', str(temp))
        content = content.replace('__TC_GRPS__', tc_grps)
        content = content.replace('__TAU_T__', tau_t)
        content = content.replace('__REF_T__', ref_t)
        content = content.replace('__PRESSURE__', str(DEFAULT_PRESSURE))

        with open(dst, 'w', encoding='utf-8') as f:
            f.write(content)


def _write_mdp_for_phase(work_dir, phase, temp=DEFAULT_TEMP,
                          tc_grps='Protein Non-Protein'):
    """Write a single MDP file for a stepdown equilibration phase.

    Args:
        phase: dict from get_equilibration_schedule()
        temp: temperature (K)
        tc_grps: thermostat coupling groups
    """
    name = phase['name']
    defines = phase.get('defines', '')
    nsteps = phase.get('nsteps', 50000)
    ensemble = phase.get('ensemble', 'NVT')
    barostat = phase.get('barostat', 'C-rescale')

    # Choose template based on ensemble type
    if ensemble == 'minimize':
        template_name = 'em.mdp'
    elif ensemble == 'NVT':
        template_name = 'nvt.mdp'
    elif name == 'md':
        template_name = 'md.mdp'
    else:
        template_name = 'npt.mdp'

    src = TEMPLATE_DIR / template_name
    dst = os.path.join(work_dir, f'{name}.mdp')
    with open(src, encoding='utf-8', errors='replace') as f:
        content = f.read()

    # Substitute all placeholders (tau_t/ref_t count must match tc_grps)
    tau_t, ref_t = _build_tau_t_ref_t(tc_grps, temp)
    content = content.replace('__DEFINE__', defines)
    content = content.replace('__NSTEPS_MD__', str(nsteps))
    content = content.replace('__TEMP__', str(temp))
    content = content.replace('__TC_GRPS__', tc_grps)
    content = content.replace('__TAU_T__', tau_t)
    content = content.replace('__REF_T__', ref_t)
    content = content.replace('__PRESSURE__', str(DEFAULT_PRESSURE))

    # Override nsteps for equilibration phases
    if ensemble != 'minimize' and name != 'md':
        content = _override_nsteps(content, nsteps)

    # Override barostat if specified and this is an NPT phase
    if ensemble == 'NPT' and barostat:
        content = _override_barostat(content, barostat)

    # For multi-step NPT (npt2, npt3), set continuation=yes
    if name in ('npt2', 'npt3'):
        content = content.replace('continuation    = no', 'continuation    = yes')

    with open(dst, 'w', encoding='utf-8') as f:
        f.write(content)

    logger.info("Wrote MDP: %s (defines=%s, nsteps=%d)", dst, defines, nsteps)


def _override_nsteps(mdp_content, nsteps):
    """Override nsteps in MDP content."""
    import re
    return re.sub(r'nsteps\s*=\s*\d+', f'nsteps      = {nsteps}', mdp_content)


def _override_barostat(mdp_content, barostat):
    """Override barostat in MDP content."""
    import re
    mdp_content = re.sub(r'pcoupl\s*=\s*\S+', f'pcoupl          = {barostat}',
                         mdp_content, count=1)
    # Adjust tau_p for stability
    if barostat == 'C-rescale':
        mdp_content = re.sub(r'tau_p\s*=\s*\S+', 'tau_p           = 5.0',
                             mdp_content)
    elif barostat == 'Parrinello-Rahman':
        mdp_content = re.sub(r'tau_p\s*=\s*\S+', 'tau_p           = 2.0',
                             mdp_content)
    return mdp_content


def _run_stage(work_dir, stage, grompp_cmd, mdrun_extra='', timeout=TIMEOUT_EQUIL,
               sanity_check=True, sandbox=None):
    """Run a simulation stage (grompp + mdrun) with post-stage sanity checks."""
    t0 = time.time()

    # grompp
    output, ok = run_gmx(grompp_cmd, work_dir, sandbox=sandbox)
    if not ok:
        return {'error': f'{stage} grompp failed', 'detail': output[-500:]}

    # mdrun
    cmd = f'mdrun -v -deffnm {stage} {mdrun_extra}'
    output, ok = run_gmx(cmd, work_dir, timeout=timeout, sandbox=sandbox)
    elapsed = time.time() - t0

    if not ok:
        # Run diagnostics on failure (local only for now)
        detail = output[-500:]
        diag = {}
        if not sandbox:
            diag = diagnose_failure(work_dir, stage)
            if diag.get('root_cause'):
                detail += f'\n\nDiagnosis: {diag["root_cause"]}'
        return {'error': f'{stage} mdrun failed', 'detail': detail,
                'elapsed_s': elapsed, 'diagnosis': diag}

    result = {'success': True, 'elapsed_s': round(elapsed, 1)}

    # Extract performance (can be from file or output text)
    perf = _get_performance(output if sandbox else os.path.join(work_dir, f'{stage}.log'),
                            is_text=bool(sandbox))
    if perf:
        result['ns_per_day'] = perf

    # Post-stage sanity check (skip for remote for now)
    if sanity_check and not sandbox:
        sanity = check_stage_sanity(work_dir, stage)
        if not sanity['ok']:
            logger.error("[%s] SANITY CHECK FAILED: %s", stage, sanity['diagnosis'])
            result['success'] = False
            result['error'] = f'{stage} sanity check failed: {sanity["diagnosis"]}'
            result['sanity'] = sanity
            # Run full diagnosis
            diag = diagnose_failure(work_dir, stage)
            result['diagnosis'] = diag
            logger.error("Root cause: %s", diag.get('root_cause', 'unknown'))
            for rec in diag.get('recommendations', []):
                logger.info("  Recommendation: %s", rec)
        elif sanity['warnings']:
            result['sanity_warnings'] = sanity['warnings']

    return result


def _fix_pbc_after_em(work_dir, stage='em'):
    """Fix PBC-wrapped atoms after energy minimization to prevent LINCS blowup."""
    import re as _re
    gro_file = os.path.join(work_dir, f'{stage}.gro')
    tpr_file = os.path.join(work_dir, f'{stage}.tpr')
    if not os.path.exists(gro_file) or not os.path.exists(tpr_file):
        return
    # Detect 'System' group number (usually 0 but may differ)
    sys_grp = '0'
    ndx_out, _ = run_gmx(f'make_ndx -f {stage}.tpr', work_dir, input_text='q\n')
    if ndx_out:
        for line in ndx_out.splitlines():
            m = _re.match(r'\s*(\d+)\s+System\s', line)
            if m:
                sys_grp = m.group(1)
                break
    fixed_gro = os.path.join(work_dir, f'{stage}_pbc.gro')
    pbc_cmd = f'trjconv -s {stage}.tpr -f {stage}.gro -o {stage}_pbc.gro -pbc whole'
    pbc_out, pbc_ok = run_gmx(pbc_cmd, work_dir, input_text=f'{sys_grp}\n')
    if pbc_ok and os.path.exists(fixed_gro):
        shutil.move(fixed_gro, gro_file)
        logger.info("Fixed PBC-wrapped atoms in %s.gro", stage)


def _get_performance(source, is_text=False):
    """Extract ns/day from GROMACS log file or output text."""
    try:
        if is_text:
            lines = source.splitlines()
        else:
            if not os.path.exists(source):
                return None
            with open(source, encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
                
        for line in lines:
            if 'Performance:' in line:
                parts = line.split()
                if 'Performance:' in parts:
                    idx = parts.index('Performance:')
                    return float(parts[idx + 1])
    except Exception:
        pass
    return None


def run_protein_ligand_md(
    complex_pdb: str,
    work_dir: str,
    sim_time_ns: float = 10.0,
    ff: str = DEFAULT_FF,
    water: str = DEFAULT_WATER,
    box_dist: float = DEFAULT_BOX_DIST,
    box_type: str = DEFAULT_BOX_TYPE,
    temp: float = DEFAULT_TEMP,
    gpu: bool = True,
    sandbox: bool = None,
) -> dict:
    """
    Run complete protein-ligand MD pipeline.

    Args:
        sandbox: Optional Sandbox instance for remote execution.
    """
    work_dir = os.path.abspath(work_dir)

    # Pre-flight validation (skip local check if remote)
    if not sandbox:
        ok, err = _preflight_check(complex_pdb, work_dir)
        if not ok:
            return {'error': err, 'hint': 'Fix input before running pipeline'}

    os.makedirs(work_dir, exist_ok=True)
    report = {
        'input': complex_pdb,
        'work_dir': work_dir,
        'ff': ff,
        'sim_time_ns': sim_time_ns,
        'temp': temp,
        'gpu': gpu,
        'start_time': datetime.now().isoformat(),
        'stages': {},
    }
    
    # Remote setup
    if sandbox and hasattr(sandbox, 'upload'):
        logger.info("Setting up remote workspace on %s", getattr(sandbox, 'host_name', 'remote'))
        sandbox.run(f'mkdir -p "{work_dir}"', use_conda=False)
        sandbox.upload(complex_pdb, work_dir)

    # ── Stage 1: Split complex ──────────────────────────────────────────
    logger.info("=== Stage 1: Split complex ===")
    protein_pdb, ligand_pdb, resname = split_complex(complex_pdb, work_dir)

    if ligand_pdb is None:
        # Pure protein — redirect to simpler pipeline
        return run_protein_md(complex_pdb, work_dir, sim_time_ns, ff, water,
                              box_dist, box_type, temp, gpu, sandbox=sandbox)

    report['ligand_resname'] = resname
    report['stages']['split'] = {'success': True}

    # ── Stage 2: Parameterize protein ───────────────────────────────────
    logger.info("=== Stage 2: Parameterize protein ===")
    protein_gro, err = parameterize_protein(work_dir, protein_pdb, ff, water)
    if err:
        report['stages']['protein_param'] = {'error': err[-300:]}
        report['error'] = 'Protein parameterization failed'
        return report
    report['stages']['protein_param'] = {'success': True}
    
    if sandbox:
        sandbox.upload(protein_gro, work_dir)
        sandbox.upload(os.path.join(work_dir, 'topol.top'), work_dir)

    # ── Stage 3: Parameterize ligand ────────────────────────────────────
    logger.info("=== Stage 3: Parameterize ligand (SOBTOP/GAFF) ===")
    ligand_itp, ligand_gro, err = parameterize_ligand(work_dir, ligand_pdb, resname)
    if err:
        report['stages']['ligand_param'] = {'error': err}
        report['error'] = 'Ligand parameterization failed'
        return report
    report['stages']['ligand_param'] = {'success': True}
    
    if sandbox:
        sandbox.upload(ligand_itp, work_dir)
        sandbox.upload(ligand_gro, work_dir)

    # ── Stage 4: Merge system ───────────────────────────────────────────
    logger.info("=== Stage 4: Merge protein + ligand ===")
    complex_gro, ok = merge_system(work_dir, protein_gro, ligand_gro,
                                   ligand_itp, resname)
    if not ok:
        report['stages']['merge'] = {'error': 'Merge failed'}
        report['error'] = 'System merge failed'
        return report
    report['stages']['merge'] = {'success': True}
    
    if sandbox:
        sandbox.upload(os.path.join(work_dir, 'topol.top'), work_dir)
        sandbox.upload(os.path.join(work_dir, 'ligand_atomtypes.itp'), work_dir)

    # ── Stage 5: Solvate + ions ─────────────────────────────────────────
    logger.info("=== Stage 5: Solvate + add ions ===")
    system_gro, ok, err = solvate_system(work_dir, complex_gro, box_dist, box_type)
    if not ok:
        report['stages']['solvate'] = {'error': err}
        report['error'] = 'Solvation failed'
        return report
    report['stages']['solvate'] = {'success': True}
    
    if sandbox:
        sandbox.upload(system_gro, work_dir)
        sandbox.upload(os.path.join(work_dir, 'topol.top'), work_dir)

    # ── Stage 6: Create index ───────────────────────────────────────────
    logger.info("=== Stage 6: Create index groups ===")
    ndx_path, tc_grps = create_index(work_dir, system_gro, resname)
    logger.info("tc-grps: %s", tc_grps)

    # Write MDP files (with backbone restraints during equilibration)
    _write_mdp_files(work_dir, sim_time_ns, temp, tc_grps,
                     has_ligand=True, equil_define='-DPOSRES')
                     
    if sandbox:
        sandbox.upload(ndx_path, work_dir)
        for mdp in ['em.mdp', 'nvt.mdp', 'npt.mdp', 'md.mdp', 'ions.mdp']:
            sandbox.upload(os.path.join(work_dir, mdp), work_dir)

    # ── Stage 7+: EM, NVT, NPT, MD ─────────────────────────────────────
    gpu_em = GPU_FLAGS_EM if gpu else ''
    gpu_full = GPU_FLAGS_FULL if gpu else ''
    
    stages = [
        ('em', 'grompp -f em.mdp -c system.gro -p topol.top -o em.tpr -maxwarn 10', gpu_em, TIMEOUT_EM),
        ('nvt', 'grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -n index.ndx -o nvt.tpr -maxwarn 10', gpu_full, TIMEOUT_EQUIL),
        ('npt', 'grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -n index.ndx -o npt.tpr -maxwarn 10', gpu_full, TIMEOUT_EQUIL),
        ('md', 'grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -n index.ndx -o md.tpr -maxwarn 10', gpu_full, int(sim_time_ns * 3600 * 2))
    ]

    for name, grompp, flags, timeout in stages:
        logger.info(f"=== Stage: {name.upper()} ===")
        res = _run_stage(work_dir, name, grompp, flags, timeout, sandbox=sandbox)
        report['stages'][name] = res
        if 'error' in res:
            report['error'] = f'Stage {name} failed'
            return report
        if name == 'em':
            _fix_pbc_after_em(work_dir)

    # ── Summary ─────────────────────────────────────────────────────────
    if sandbox:
        logger.info("Collecting results from remote...")
        sandbox.sync_from_remote(work_dir, work_dir)
        
    report['end_time'] = datetime.now().isoformat()
    report['success'] = True
    _write_report(report, work_dir)
    return report


def run_protein_md(
    protein_pdb: str,
    work_dir: str,
    sim_time_ns: float = 10.0,
    ff: str = DEFAULT_FF,
    water: str = DEFAULT_WATER,
    box_dist: float = DEFAULT_BOX_DIST,
    box_type: str = DEFAULT_BOX_TYPE,
    temp: float = DEFAULT_TEMP,
    gpu: bool = True,
    sandbox: Any = None,
) -> dict:
    """Run MD for a pure protein system (no ligand)."""
    work_dir = os.path.abspath(work_dir)

    if not sandbox:
        ok, err = _preflight_check(protein_pdb, work_dir)
        if not ok:
            return {'error': err, 'hint': 'Fix input before running pipeline'}

    os.makedirs(work_dir, exist_ok=True)
    report = {
        'input': protein_pdb,
        'work_dir': work_dir,
        'ff': ff,
        'sim_time_ns': sim_time_ns,
        'start_time': datetime.now().isoformat(),
        'stages': {},
    }
    
    if sandbox and hasattr(sandbox, 'upload'):
        sandbox.run(f'mkdir -p "{work_dir}"', use_conda=False)
        sandbox.upload(protein_pdb, work_dir)

    # Parameterize
    protein_gro, err = parameterize_protein(work_dir, protein_pdb, ff, water)
    if err:
        report['error'] = f'pdb2gmx failed: {err[-300:]}'
        return report
    
    if sandbox:
        sandbox.upload(protein_gro, work_dir)
        sandbox.upload(os.path.join(work_dir, 'topol.top'), work_dir)

    # Solvate
    system_gro, ok, err = solvate_system(work_dir, protein_gro, box_dist, box_type)
    if not ok:
        report['error'] = err
        return report
    
    if sandbox:
        sandbox.upload(system_gro, work_dir)
        sandbox.upload(os.path.join(work_dir, 'topol.top'), work_dir)

    tc_grps = 'Protein Non-Protein'
    _write_mdp_files(work_dir, sim_time_ns, temp, tc_grps,
                     has_ligand=False, equil_define='-DPOSRES')
                     
    if sandbox:
        for mdp in ['em.mdp', 'nvt.mdp', 'npt.mdp', 'md.mdp', 'ions.mdp']:
            sandbox.upload(os.path.join(work_dir, mdp), work_dir)

    # EM → NVT → NPT → MD
    gpu_em = GPU_FLAGS_EM if gpu else ''
    gpu_full = GPU_FLAGS_FULL if gpu else ''

    for stage, grompp, flags, timeout in [
        ('em', 'grompp -f em.mdp -c system.gro -p topol.top -o em.tpr -maxwarn 10',
         gpu_em, TIMEOUT_EM),
        ('nvt', 'grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr -maxwarn 10',
         gpu_full, TIMEOUT_EQUIL),
        ('npt', 'grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr -maxwarn 10',
         gpu_full, TIMEOUT_EQUIL),
        ('md', 'grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o md.tpr -maxwarn 10',
         gpu_full, max(TIMEOUT_MD, int(sim_time_ns * 3600))),
    ]:
        result = _run_stage(work_dir, stage, grompp, flags, timeout, sandbox=sandbox)
        report['stages'][stage] = result
        if 'error' in result:
            report['error'] = f'{stage} failed'
            return report
        if stage == 'em':
            _fix_pbc_after_em(work_dir)

    if sandbox:
        sandbox.sync_from_remote(work_dir, work_dir)

    report['success'] = True
    report['end_time'] = datetime.now().isoformat()
    _write_report(report, work_dir)
    return report


def _add_coordination_restraints(work_dir, gro_path, coordination, pdb2gmx_output=''):
    """
    Add distance restraints for metal-ligand coordination bonds.

    If specbond.dat didn't create bonded terms (common issue in GROMACS),
    we add harmonic distance restraints to maintain coordination geometry.
    This is the recommended approach for non-bonded metal models in P450
    simulations (see Shahrokh et al. 2012, JCTC).

    Writes:
      - posre_coord.itp: intermolecular distance restraints
      - Modifies topol.top to include it
    """
    if not coordination:
        return

    topol_path = os.path.join(work_dir, 'topol.top')
    if not os.path.exists(topol_path) or not os.path.exists(gro_path):
        return

    # Parse GRO to find atom indices for metal and coordinating atoms
    atom_indices = {}  # (resname, resid, atom_name) -> index
    with open(gro_path, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    natoms = int(lines[1].strip())
    for i, line in enumerate(lines[2:2 + natoms], start=1):
        # GRO format: residue number (5), residue name (5), atom name (5), atom number (5)
        if len(line) < 25:
            continue
        try:
            resid = int(line[0:5].strip())
            resname = line[5:10].strip()
            atom_name = line[10:15].strip()
        except (ValueError, IndexError):
            continue
        key = (resname, resid, atom_name)
        atom_indices[key] = i

    # Build distance restraints
    restraint_lines = []
    restraint_lines.append('; Distance restraints for metal-ligand coordination\n')
    restraint_lines.append('; Generated by zhuangzhou metalloprotein pipeline\n')
    restraint_lines.append('; Based on crystal structure distances\n')
    restraint_lines.append('[ distance_restraints ]\n')
    restraint_lines.append('; i  j  type  index  type\'  low    up1    up2    fac\n')

    for idx, c in enumerate(coordination):
        # Find metal atom index
        # The metal is part of HEME — try both HEM and HEME naming
        metal_resid = c['metal_resid']
        metal_atom = c['metal_atom']
        metal_idx = None
        for rn in ('HEME', 'HEM', 'HEO2'):
            key = (rn, metal_resid, metal_atom)
            if key in atom_indices:
                metal_idx = atom_indices[key]
                break

        # Find protein atom index
        prot_resid = c['protein_resid']
        prot_atom = c['protein_atom']
        prot_resname = c['protein_res']
        prot_idx = None
        # Try various naming conventions (pdb2gmx may have renamed)
        for rn in (prot_resname, 'CYS2', 'CYS', 'HSD', 'HSE', 'HIS', 'MET'):
            key = (rn, prot_resid, prot_atom)
            if key in atom_indices:
                prot_idx = atom_indices[key]
                break

        if metal_idx is None or prot_idx is None:
            logger.warning("Cannot find atoms for coordination: %s", c)
            continue

        # Distance restraint parameters
        d_nm = c['distance'] / 10.0  # Angstrom -> nm
        low = max(0.0, d_nm - 0.02)   # -0.2 A tolerance
        up1 = d_nm + 0.02             # +0.2 A tolerance
        up2 = d_nm + 0.05             # +0.5 A beyond = penalty

        restraint_lines.append(
            f'; {c["bond_type"]}: {prot_resname}{prot_resid}-{prot_atom} '
            f'-- {metal_atom} (d={c["distance"]:.3f} A)\n')
        restraint_lines.append(
            f'  {prot_idx:6d}  {metal_idx:6d}  1  {idx:4d}  1'
            f'  {low:.4f}  {up1:.4f}  {up2:.4f}  1.0\n')

    if len(restraint_lines) <= 5:
        return  # No valid restraints found

    # Write restraint file
    coord_itp = os.path.join(work_dir, 'posre_coord.itp')
    with open(coord_itp, 'w', encoding='utf-8') as f:
        f.writelines(restraint_lines)

    # Add to topology INSIDE the Protein_chain molecule section
    # Distance restraints must be within the moleculetype they reference
    with open(topol_path, encoding='utf-8', errors='replace') as f:
        content = f.read()

    insert = (
        '\n; Fe coordination distance restraints\n'
        '#ifdef POSRES_COORD\n'
        '#include "posre_coord.itp"\n'
        '#endif\n'
    )

    # Find insertion point: after #ifdef POSRES ... #endif block in protein
    # Or before [ moleculetype ] of next molecule, or before [ system ]
    import re
    # Look for the existing position restraints block end
    posres_match = re.search(r'#ifdef\s+POSRES\b.*?#endif', content, re.DOTALL)
    if posres_match:
        # Insert after the existing POSRES block
        insert_pos = posres_match.end()
        content = content[:insert_pos] + insert + content[insert_pos:]
    else:
        # Fallback: insert before [ system ]
        sys_pos = content.find('[ system ]')
        if sys_pos >= 0:
            content = content[:sys_pos] + insert + '\n' + content[sys_pos:]
        else:
            content += insert

    with open(topol_path, 'w', encoding='utf-8') as f:
        f.write(content)

    logger.info("Written coordination restraints: %s", coord_itp)


def _run_stepdown_equilibration(work_dir, schedule, gpu=True, use_ndx=False, sandbox=None):
    """
    Run a multi-phase equilibration + production schedule.

    Args:
        schedule: list of phase dicts from get_equilibration_schedule()
        gpu: use GPU acceleration
        use_ndx: whether to include -n index.ndx in grompp
        sandbox: optional Sandbox for remote execution
    """
    gpu_em = GPU_FLAGS_EM if gpu else ''
    gpu_full = GPU_FLAGS_FULL if gpu else ''
    ndx_flag = ' -n index.ndx' if use_ndx else ''

    results = {}
    prev_gro = 'system.gro'
    prev_cpt = None

    for i, phase in enumerate(schedule):
        name = phase['name']
        ensemble = phase.get('ensemble', 'NVT')

        logger.info("=== Phase %d/%d: %s — %s ===",
                     i + 1, len(schedule), name.upper(), phase['description'])

        # Build grompp command
        if ensemble == 'minimize':
            grompp = (f'grompp -f {name}.mdp -c {prev_gro} '
                      f'-p topol.top -o {name}.tpr -maxwarn 10')
            flags = gpu_em
            timeout = TIMEOUT_EM
        elif name == 'nvt':
            grompp = (f'grompp -f {name}.mdp -c {prev_gro} -r {prev_gro} '
                      f'-p topol.top{ndx_flag} -o {name}.tpr -maxwarn 10')
            flags = gpu_full
            timeout = TIMEOUT_EQUIL
        elif name == 'md':
            cpt_flag = f' -t {prev_cpt}' if prev_cpt else ''
            grompp = (f'grompp -f {name}.mdp -c {prev_gro}{cpt_flag} '
                      f'-p topol.top{ndx_flag} -o {name}.tpr -maxwarn 10')
            flags = gpu_full
            timeout = max(TIMEOUT_MD, int(phase.get('nsteps', 5000000) * 2 / 1e6 * 3600))
        else:
            # NPT phases (npt, npt1, npt2, npt3)
            cpt_flag = f' -t {prev_cpt}' if prev_cpt else ''
            grompp = (f'grompp -f {name}.mdp -c {prev_gro} -r {prev_gro}{cpt_flag} '
                      f'-p topol.top{ndx_flag} -o {name}.tpr -maxwarn 10')
            flags = gpu_full
            timeout = TIMEOUT_EQUIL

        result = _run_stage(work_dir, name, grompp, flags, timeout, sandbox=sandbox)
        results[name] = result

        if 'error' in result:
            return results

        # After EM: fix PBC-wrapped atoms (prevents LINCS blowup in NVT)
        if ensemble == 'minimize' and not sandbox:
            _fix_pbc_after_em(work_dir, name)
        elif ensemble == 'minimize' and sandbox:
            # Remote PBC fix
            pbc_cmd = f'gmx trjconv -s {name}.tpr -f {name}.gro -o {name}_pbc.gro -pbc whole'
            pbc_r = sandbox.run(f'echo 0 | {pbc_cmd}', work_dir=work_dir, timeout=60)
            if pbc_r.success:
                sandbox.run(f'mv {name}_pbc.gro {name}.gro', work_dir=work_dir,
                            timeout=15, use_conda=False)

        # Update chain: next phase starts from this phase's output
        # For remote execution, assume output exists if stage succeeded
        if sandbox:
            prev_gro = f'{name}.gro'
            if ensemble != 'minimize':
                prev_cpt = f'{name}.cpt'
        else:
            gro_out = os.path.join(work_dir, f'{name}.gro')
            cpt_out = os.path.join(work_dir, f'{name}.cpt')
            if os.path.exists(gro_out):
                prev_gro = f'{name}.gro'
            if os.path.exists(cpt_out):
                prev_cpt = f'{name}.cpt'

    return results


def run_metalloprotein_md(
    pdb_path: str,
    work_dir: str,
    sim_time_ns: float = 10.0,
    water: str = 'tip3p',
    box_dist: float = DEFAULT_BOX_DIST,
    box_type: str = DEFAULT_BOX_TYPE,
    temp: float = DEFAULT_TEMP,
    gpu: bool = True,
    stepdown: bool = True,
    sandbox: Any = None,
) -> dict:
    """
    Run MD for a metalloprotein system (HEME/Fe centers).

    Uses AMBER14SB + GAFF force field:
      - Protein: pdb2gmx with CYP residue (Shahrokh 2012 charges)
      - HEM: separate moleculetype (hem.itp, pre-parameterized GAFF)
      - Fe-S: [intermolecular_interactions] bond in topol.top
      - Force field params: hem_ff.itp (Fe + GAFF atomtypes + bonded)

    Workflow:
      1. Split PDB: protein (CYS→CYP) + HEM
      2. Deploy heme parameter files (hem_ff.itp, hem.itp)
      3. pdb2gmx on protein-only (CYP auto-recognized)
      4. Merge protein.gro + hem.gro, build topol.top
      5. Inject Fe-S intermolecular bond
      6. Solvate + ions
      7. Stepdown equilibration → Production MD

    Args:
        pdb_path: PDB file with protein + HEM
        work_dir: output directory
        sim_time_ns: production MD length (ns)
        water: water model
        box_dist: min distance to box edge (nm)
        box_type: box geometry
        temp: simulation temperature (K)
        gpu: use GPU acceleration
        stepdown: gradual restraint release protocol

    Returns:
        dict with stage results, coordination info, output paths
    """
    work_dir = os.path.abspath(work_dir)
    ff = METALLOPROTEIN_FF

    ok, err = _preflight_check(pdb_path, work_dir)
    if not ok:
        return {'error': err, 'hint': 'Fix input before running pipeline'}

    os.makedirs(work_dir, exist_ok=True)

    report = {
        'input': pdb_path,
        'work_dir': work_dir,
        'ff': ff,
        'sim_time_ns': sim_time_ns,
        'temp': temp,
        'gpu': gpu,
        'system_type': 'metalloprotein',
        'start_time': datetime.now().isoformat(),
        'stages': {},
    }

    # ── Stage 1: Split PDB + deploy heme params ───────────────────────
    logger.info("=== Stage 1: Metalloprotein PDB Preparation (AMBER14SB + GAFF) ===")
    heme_setup = setup_heme_system(work_dir, pdb_path, ff=ff, water=water)
    if 'error' in heme_setup:
        report['error'] = f'Heme setup failed: {heme_setup["error"]}'
        return report

    protein_pdb = heme_setup['protein_pdb']
    hem_pdb = heme_setup['hem_pdb']
    report['stages']['pdb_prep'] = {
        'success': True,
        'proximal_cys_resid': heme_setup['proximal_resid'],
        'has_ligand': heme_setup['ligand_pdb'] is not None,
        'warnings': heme_setup['warnings'],
    }
    logger.info("Proximal CYS → CYF at resid %s", heme_setup['proximal_resid'])

    # ── Stage 2: pdb2gmx on protein-only (CYF auto-recognized) ───────
    logger.info("=== Stage 2: pdb2gmx (%s, protein with CYF) ===", ff)
    protein_gro, err = parameterize_protein(work_dir, protein_pdb, ff, water)
    if err:
        report['stages']['pdb2gmx'] = {'error': err[-500:]}
        report['error'] = 'pdb2gmx failed'
        report['hint'] = (
            'Ensure CYF residue is installed in the force field. '
            'Run setup_p450_ff.sh to patch amber14sb_parmbsc1.ff.'
        )
        return report

    _fix_nan_coordinates(protein_gro)
    report['stages']['pdb2gmx'] = {'success': True}

    # ── Stage 3: Merge protein + HEM coordinates ─────────────────────
    logger.info("=== Stage 3: Merge protein + HEM ===")
    complex_gro = _merge_protein_hem(work_dir, protein_gro, hem_pdb)
    if not complex_gro:
        report['error'] = 'Failed to merge protein + HEM coordinates'
        return report
    report['stages']['merge'] = {'success': True}

    # ── Stage 4: Build topol.top (without Fe-S yet) ──────────────────
    logger.info("=== Stage 4: Build topology ===")
    topol_path = build_topol(work_dir, ff=ff, water=water,
                             has_ligand=False,
                             proximal_resid=heme_setup['proximal_resid'])
    if not topol_path:
        report['error'] = 'Failed to build topol.top'
        return report

    # ── Stage 5: Solvate + ions (before Fe-S injection) ────────────
    # Fe-S bond must be injected AFTER solvation because gmx solvate
    # appends SOL count to the end of topol.top. If [intermolecular_interactions]
    # is already at the end, the SOL line ends up in the wrong section.
    logger.info("=== Stage 5: Solvate + add ions ===")
    system_gro, ok, err = solvate_system(work_dir, complex_gro, box_dist, box_type)
    if not ok:
        report['stages']['solvate'] = {'error': err}
        report['error'] = 'Solvation failed'
        return report
    report['stages']['solvate'] = {'success': True}

    # ── Stage 5b: Inject Fe-S bond (after solvation) ───────────────
    logger.info("=== Stage 5b: Fe-S bond injection ===")
    fe_s_result = finalize_heme_topology(work_dir, complex_gro, topol_path)
    if 'error' in fe_s_result:
        report['error'] = f'Fe-S bond: {fe_s_result["error"]}'
        return report

    report['stages']['heme_topology'] = {
        'success': True,
        'fe_index': fe_s_result['fe_index'],
        'sg_index': fe_s_result['sg_index'],
        'bond_b0_nm': fe_s_result['bond_b0_nm'],
        'bond_kb': fe_s_result['bond_kb'],
    }
    logger.info("Fe-S bond: atom %d -- atom %d (b0=%.3f nm, Kb=%.1f)",
                fe_s_result['fe_index'], fe_s_result['sg_index'],
                fe_s_result['bond_b0_nm'], fe_s_result['bond_kb'])

    # Validate
    validation = validate_heme_topology(work_dir)
    report['stages']['heme_validation'] = validation
    for check in validation.get('checks', []):
        logger.info("  %s", check)

    # ── Stage 6: Create index groups ─────────────────────────────────
    logger.info("=== Stage 6: Create index groups ===")
    tc_grps = 'Protein Non-Protein'

    # Try gmx make_ndx to verify groups exist
    ndx_path = os.path.join(work_dir, 'index.ndx')
    ndx_cmd = f'make_ndx -f "{system_gro}" -o "{ndx_path}"'
    ndx_out, ndx_ok = run_gmx(ndx_cmd, work_dir, input_text='q\n')

    use_ndx = False
    if ndx_ok and os.path.exists(ndx_path):
        # Check if Protein group exists and includes HEME
        use_ndx = True
        logger.info("Index file created: %s", ndx_path)

    # ── Stage 5: Generate professional restraints ───────────────────────
    logger.info("=== Stage 5: Generate restraint files ===")

    has_soft_posre = False
    has_cofactor_posre = False

    # Generate soft posre (for stepdown)
    if stepdown:
        soft_path = generate_soft_posre(
            os.path.join(work_dir, 'topol.top'), work_dir)
        has_soft_posre = soft_path is not None

    # Generate cofactor posre (HEME heavy atoms)
    cofac_path = generate_cofactor_posre(
        system_gro, work_dir,
        cofactor_resnames={'HEM'})
    has_cofactor_posre = cofac_path is not None

    # Inject all restraint blocks into topology
    inject_restraints_into_topology(
        os.path.join(work_dir, 'topol.top'), work_dir,
        has_soft=has_soft_posre,
        has_metal=False,
        has_cofactor=has_cofactor_posre,
        has_coord=False,
    )

    report['stages']['restraints'] = {
        'success': True,
        'soft_posre': has_soft_posre,
        'cofactor_posre': has_cofactor_posre,
        'stepdown': stepdown,
    }

    # ── Stage 7+: Stepdown equilibration + production ────────────────
    system_type = 'metalloprotein'
    schedule = get_equilibration_schedule(
        system_type=system_type,
        sim_time_ns=sim_time_ns,
        stepdown=stepdown,
    )

    # Write MDP files for each phase
    for phase in schedule:
        _write_mdp_for_phase(work_dir, phase, temp=temp, tc_grps=tc_grps)

    # Also write ions.mdp (needed for solvation step)
    ions_mdp = os.path.join(work_dir, 'ions.mdp')
    if not os.path.exists(ions_mdp):
        _write_template('ions.mdp', ions_mdp)

    # Remote execution: upload work_dir to server, run there, download results
    run_dir = work_dir  # local by default
    if sandbox and hasattr(sandbox, 'sync_to_remote'):
        hp = sandbox.host_profile
        remote_dir = f'{hp.remote_work_dir}/{os.path.basename(work_dir)}'
        sandbox.run(f'mkdir -p {remote_dir}', timeout=15, use_conda=False)
        logger.info("Uploading work_dir to %s:%s ...", sandbox.host_name, remote_dir)
        r = sandbox.sync_to_remote(work_dir, remote_dir)
        if not r.success:
            report['error'] = f'Upload to remote failed: {r.error}'
            return report
        run_dir = remote_dir
        logger.info("Upload complete, running stages on remote GPU")

    # Run all phases
    stage_results = _run_stepdown_equilibration(
        run_dir, schedule, gpu=gpu, use_ndx=use_ndx, sandbox=sandbox)

    # Download results if remote
    if sandbox and run_dir != work_dir:
        logger.info("Downloading results from %s:%s ...", sandbox.host_name, run_dir)
        sandbox.sync_from_remote(run_dir, work_dir)

    report['stages'].update(stage_results)

    # Check for failures
    for stage_name, result in stage_results.items():
        if 'error' in result:
            report['error'] = f'{stage_name} failed'
            _write_report(report, work_dir)
            return report

    # ── Summary ────────────────────────────────────────────────────────
    report['success'] = True
    report['end_time'] = datetime.now().isoformat()
    report['outputs'] = {
        'trajectory': os.path.join(work_dir, 'md.xtc'),
        'structure': os.path.join(work_dir, 'md.gro'),
        'topology': os.path.join(work_dir, 'topol.top'),
        'tpr': os.path.join(work_dir, 'md.tpr'),
        'log': os.path.join(work_dir, 'md.log'),
        'checkpoint': os.path.join(work_dir, 'md.cpt'),
        'energy': os.path.join(work_dir, 'md.edr'),
    }
    report['next_steps'] = [
        'gromacs_process_trajectory: 处理轨迹 (PBC修复 + 旋转平移拟合)',
        'gromacs_analysis: 运行RMSD/RMSF/Rg/氢键/Fe配位分析',
        'gromacs_visualize: 生成轨迹动画和PCA豪猪图',
    ]
    _write_report(report, work_dir)

    return report


def run_metalloprotein_ligand_md(
    complex_pdb: str,
    work_dir: str,
    sim_time_ns: float = 10.0,
    water: str = 'tip3p',
    box_dist: float = DEFAULT_BOX_DIST,
    box_type: str = DEFAULT_BOX_TYPE,
    temp: float = DEFAULT_TEMP,
    gpu: bool = True,
    stepdown: bool = True,
    ligand_str_files: dict = None,
    sandbox: Any = None,
) -> dict:
    """
    Run MD for a metalloprotein + small molecule ligand system.

    Uses AMBER14SB + GAFF:
      - Protein: pdb2gmx with CYF residue (Shahrokh 2012 charges)
      - HEM: separate moleculetype (hem.itp, pre-parameterized GAFF)
      - Ligand: SOBTOP/GAFF parameterization
      - Fe-S: [intermolecular_interactions] bond
      - Force field params: hem_ff.itp (Fe + GAFF atomtypes + bonded)

    Workflow:
      1. Split PDB: protein (CYS→CYF) + HEM + ligand
      2. Deploy heme parameter files
      3. pdb2gmx on protein-only (CYF auto-recognized)
      4. Parameterize ligand (SOBTOP/GAFF)
      5. Merge protein + HEM + ligand, build topol.top
      6. Inject Fe-S bond, deduplicate ligand atomtypes
      7. Solvate + ions
      8. Stepdown equilibration → Production MD

    Args:
        complex_pdb: PDB with protein+HEME+ligand (HETATM)
        work_dir: output directory
        sim_time_ns: production MD length (ns)
        water: water model
        box_dist: box edge distance (nm)
        box_type: box geometry
        temp: temperature (K)
        gpu: use GPU
        stepdown: gradual restraint release protocol
        ligand_str_files: unused (kept for API compatibility)

    Returns:
        dict with stage results, paths, performance metrics
    """
    work_dir = os.path.abspath(work_dir)
    ff = METALLOPROTEIN_FF

    ok, err = _preflight_check(complex_pdb, work_dir)
    if not ok:
        return {'error': err, 'hint': 'Fix input before running pipeline'}

    os.makedirs(work_dir, exist_ok=True)

    report = {
        'input': complex_pdb,
        'work_dir': work_dir,
        'ff': ff,
        'sim_time_ns': sim_time_ns,
        'temp': temp,
        'gpu': gpu,
        'system_type': 'metalloprotein_ligand',
        'start_time': datetime.now().isoformat(),
        'stages': {},
    }

    # ── Stage 1: Split PDB + deploy heme params ─────────────────────
    logger.info("=== Stage 1: Metalloprotein+Ligand PDB Preparation (AMBER14SB + GAFF) ===")
    heme_setup = setup_heme_system(work_dir, complex_pdb, ff=ff, water=water)
    if 'error' in heme_setup:
        report['error'] = f'Heme setup failed: {heme_setup["error"]}'
        return report

    protein_pdb = heme_setup['protein_pdb']
    hem_pdb = heme_setup['hem_pdb']
    ligand_pdb = heme_setup['ligand_pdb']
    ligand_resname = heme_setup['ligand_resname']

    if not ligand_pdb:
        logger.warning("No ligand found, falling back to metalloprotein-only pipeline")
        return run_metalloprotein_md(
            complex_pdb, work_dir, sim_time_ns, water,
            box_dist, box_type, temp, gpu, stepdown, sandbox=sandbox)

    report['stages']['pdb_prep'] = {
        'success': True,
        'proximal_cys_resid': heme_setup['proximal_resid'],
        'ligand_resname': ligand_resname,
        'warnings': heme_setup['warnings'],
    }
    logger.info("Proximal CYS → CYF at resid %s, ligand: %s",
                heme_setup['proximal_resid'], ligand_resname)

    # ── Stage 2: pdb2gmx on protein-only ─────────────────────────────
    logger.info("=== Stage 2: pdb2gmx (%s, protein with CYF) ===", ff)
    protein_gro, err = parameterize_protein(work_dir, protein_pdb, ff, water)
    if err:
        report['stages']['pdb2gmx'] = {'error': err[-500:]}
        report['error'] = 'pdb2gmx failed'
        report['hint'] = (
            'Ensure CYF residue is installed in the force field. '
            'Run setup_p450_ff.sh to patch amber14sb_parmbsc1.ff.'
        )
        return report

    _fix_nan_coordinates(protein_gro)
    report['stages']['pdb2gmx'] = {'success': True}

    # ── Stage 3: Parameterize ligand (SOBTOP/GAFF) ──────────────────
    logger.info("=== Stage 3: Parameterize ligand (SOBTOP/GAFF) ===")
    ligand_itp, ligand_gro, err = parameterize_ligand(work_dir, ligand_pdb, ligand_resname)
    if err:
        report['stages']['ligand_param'] = {'error': err}
        report['error'] = f'Ligand {ligand_resname} parameterization failed'
        report['hint'] = 'Ensure SOBTOP and OpenBabel are installed.'
        return report
    report['stages']['ligand_param'] = {
        'success': True,
        'ligand_resname': ligand_resname,
        'method': 'gaff',
    }

    # Extract and deduplicate ligand atomtypes against hem_ff.itp
    ligand_at_path = os.path.join(work_dir, 'ligand_atomtypes.itp')
    _extract_atomtypes(ligand_itp, ligand_at_path)
    _remove_atomtypes_from_itp(ligand_itp)
    n_dedup = dedup_ligand_atomtypes(ligand_at_path)
    logger.info("Ligand atomtype deduplication: removed %d types shared with hem_ff.itp", n_dedup)

    # ── Stage 4: Merge protein + HEM + ligand ────────────────────────
    logger.info("=== Stage 4: Merge protein + HEM + ligand ===")
    complex_gro = _merge_protein_hem_ligand(work_dir, protein_gro, hem_pdb, ligand_gro)
    if not complex_gro:
        report['error'] = 'Failed to merge protein + HEM + ligand coordinates'
        return report
    report['stages']['merge'] = {'success': True}

    # ── Stage 5: Build topol.top + Fe-S bond ─────────────────────────
    logger.info("=== Stage 5: Build topology + Fe-S bond ===")
    topol_path = build_topol(work_dir, ff=ff, water=water,
                             has_ligand=True, ligand_resname=ligand_resname,
                             proximal_resid=heme_setup['proximal_resid'])
    if not topol_path:
        report['error'] = 'Failed to build topol.top'
        return report

    fe_s_result = finalize_heme_topology(work_dir, complex_gro, topol_path)
    if 'error' in fe_s_result:
        report['error'] = f'Fe-S bond: {fe_s_result["error"]}'
        return report

    report['stages']['heme_topology'] = {
        'success': True,
        'fe_index': fe_s_result['fe_index'],
        'sg_index': fe_s_result['sg_index'],
    }
    logger.info("Fe-S bond: atom %d -- atom %d", fe_s_result['fe_index'], fe_s_result['sg_index'])

    # Validate
    validation = validate_heme_topology(work_dir)
    report['stages']['heme_validation'] = validation

    # ── Stage 6: Solvate + ions ──────────────────────────────────────
    logger.info("=== Stage 6: Solvate + ions ===")
    system_gro, ok, err = solvate_system(work_dir, complex_gro, box_dist, box_type)
    if not ok:
        report['stages']['solvate'] = {'error': err}
        report['error'] = 'Solvation failed'
        return report
    report['stages']['solvate'] = {'success': True}

    # ── Stage 7: Create index groups ─────────────────────────────────
    logger.info("=== Stage 7: Create index groups ===")
    ndx_path, tc_grps = _create_metalloprotein_ligand_index(
        work_dir, system_gro, [ligand_resname])
    use_ndx = os.path.exists(ndx_path) if ndx_path else False
    logger.info("tc-grps: %s", tc_grps)

    # ── Stage 8: Generate restraints ─────────────────────────────────
    logger.info("=== Stage 8: Generate restraint files ===")
    has_soft = False
    has_cofactor = False

    if stepdown:
        soft_path = generate_soft_posre(
            os.path.join(work_dir, 'topol.top'), work_dir)
        has_soft = soft_path is not None

    cofac_path = generate_cofactor_posre(
        system_gro, work_dir, cofactor_resnames={'HEM'})
    has_cofactor = cofac_path is not None

    inject_restraints_into_topology(
        os.path.join(work_dir, 'topol.top'), work_dir,
        has_soft=has_soft, has_metal=False,
        has_cofactor=has_cofactor, has_coord=False,
    )

    # ── Stage 9+: Stepdown equilibration + production ────────────────
    schedule = get_equilibration_schedule(
        system_type='metalloprotein_ligand',
        sim_time_ns=sim_time_ns,
        stepdown=stepdown,
    )

    for phase in schedule:
        _write_mdp_for_phase(work_dir, phase, temp=temp, tc_grps=tc_grps)

    ions_mdp = os.path.join(work_dir, 'ions.mdp')
    if not os.path.exists(ions_mdp):
        _write_template('ions.mdp', ions_mdp)

    # Remote execution
    run_dir = work_dir
    if sandbox and hasattr(sandbox, 'sync_to_remote'):
        hp = sandbox.host_profile
        remote_dir = f'{hp.remote_work_dir}/{os.path.basename(work_dir)}'
        sandbox.run(f'mkdir -p {remote_dir}', timeout=15, use_conda=False)
        logger.info("Uploading work_dir to %s:%s ...", sandbox.host_name, remote_dir)
        r = sandbox.sync_to_remote(work_dir, remote_dir)
        if not r.success:
            report['error'] = f'Upload to remote failed: {r.error}'
            return report
        run_dir = remote_dir

    stage_results = _run_stepdown_equilibration(
        run_dir, schedule, gpu=gpu, use_ndx=use_ndx, sandbox=sandbox)

    if sandbox and run_dir != work_dir:
        sandbox.sync_from_remote(run_dir, work_dir)

    report['stages'].update(stage_results)

    for stage_name, result in stage_results.items():
        if 'error' in result:
            report['error'] = f'{stage_name} failed'
            _write_report(report, work_dir)
            return report

    # ── Summary ──────────────────────────────────────────────────────
    report['success'] = True
    report['end_time'] = datetime.now().isoformat()
    report['outputs'] = {
        'trajectory': os.path.join(work_dir, 'md.xtc'),
        'structure': os.path.join(work_dir, 'md.gro'),
        'topology': os.path.join(work_dir, 'topol.top'),
        'tpr': os.path.join(work_dir, 'md.tpr'),
        'log': os.path.join(work_dir, 'md.log'),
        'checkpoint': os.path.join(work_dir, 'md.cpt'),
        'energy': os.path.join(work_dir, 'md.edr'),
    }
    report['next_steps'] = [
        'gromacs_process_trajectory: 处理轨迹 (PBC修复 + 旋转平移拟合)',
        'gromacs_analysis: 运行RMSD/RMSF/Rg/氢键/Fe配位/配体RMSD分析',
        'gromacs_visualize: 生成轨迹动画和PCA豪猪图',
    ]
    _write_report(report, work_dir)
    return report


def _add_hydrogens_to_hem(hem_pdb, hem_itp_path, work_dir):
    """Add hydrogen atoms to HEM PDB to match hem.itp atom count.

    PDB crystal structures typically lack H atoms but hem.itp defines
    73 atoms (43 heavy + 30 H). Uses OpenBabel to add H, then validates
    the atom count matches hem.itp.

    Returns path to hydrogenated HEM PDB, or original if already complete.
    """
    # Count atoms in hem.itp
    itp_natoms = 0
    if os.path.exists(hem_itp_path):
        with open(hem_itp_path, encoding='utf-8', errors='replace') as f:
            in_atoms = False
            for line in f:
                s = line.strip()
                if '[ atoms ]' in s:
                    in_atoms = True
                    continue
                if in_atoms and s.startswith('['):
                    break
                if in_atoms and s and not s.startswith(';'):
                    itp_natoms += 1

    # Count atoms in input PDB
    pdb_natoms = 0
    with open(hem_pdb, encoding='utf-8', errors='replace') as f:
        for line in f:
            if line[:6].strip() in ('ATOM', 'HETATM'):
                pdb_natoms += 1

    if pdb_natoms >= itp_natoms:
        logger.info("HEM already has %d atoms (itp expects %d), no H addition needed",
                     pdb_natoms, itp_natoms)
        return hem_pdb

    logger.info("HEM has %d atoms but itp expects %d — adding hydrogens with OpenBabel",
                pdb_natoms, itp_natoms)

    if not OBABEL_EXE or not os.path.isfile(OBABEL_EXE):
        logger.warning("OpenBabel not found, proceeding without H addition")
        return hem_pdb

    import subprocess
    hem_h_pdb = os.path.join(work_dir, 'hem_h.pdb')
    try:
        subprocess.run(
            [OBABEL_EXE, hem_pdb, '-O', hem_h_pdb, '-h'],
            capture_output=True, text=True, timeout=30
        )
    except Exception as e:
        logger.warning("OpenBabel failed to add H: %s", e)
        return hem_pdb

    if os.path.exists(hem_h_pdb):
        # Count output atoms
        h_natoms = 0
        with open(hem_h_pdb, encoding='utf-8', errors='replace') as f:
            for line in f:
                if line[:6].strip() in ('ATOM', 'HETATM'):
                    h_natoms += 1
        logger.info("HEM after H addition: %d atoms (itp expects %d)", h_natoms, itp_natoms)
        if h_natoms >= itp_natoms:
            return hem_h_pdb

    return hem_pdb


from .heme_params import build_perfect_heme_gro

def _merge_protein_hem(work_dir, protein_gro, hem_pdb):
    """Merge protein.gro + hem.pdb into complex.gro using deterministic perfect heme."""
    complex_gro = os.path.join(work_dir, 'complex.gro')
    hem_itp = os.path.join(work_dir, 'hem.itp')

    # Read protein .gro
    with open(protein_gro, encoding='utf-8', errors='replace') as f:
        prot_lines = f.readlines()
    prot_natoms = int(prot_lines[1].strip())
    prot_atoms = prot_lines[2:2 + prot_natoms]
    prot_box = prot_lines[2 + prot_natoms] if len(prot_lines) > 2 + prot_natoms else '  10.0  10.0  10.0\n'

    # Generate perfect HEM gro lines matching hem.itp exactly
    hem_atoms = build_perfect_heme_gro(hem_itp, hem_pdb, os.path.join(work_dir, 'hem_perfect.gro'), start_atomnr=prot_natoms + 1)
    
    if not hem_atoms:
        logger.error("Failed to generate perfect HEM atoms")
        return None

    # Write combined .gro
    total = prot_natoms + len(hem_atoms)
    with open(complex_gro, 'w', encoding='utf-8') as f:
        f.write('Protein + HEM system\n')
        f.write(f'{total}\n')
        f.writelines(prot_atoms)
        f.writelines(hem_atoms)
        f.write(prot_box)

    logger.info("Merged: %d protein + %d HEM = %d atoms", prot_natoms, len(hem_atoms), total)
    return complex_gro


def _merge_protein_hem_ligand(work_dir, protein_gro, hem_pdb, ligand_gro):
    """Merge protein.gro + hem.pdb + ligand.gro into complex.gro."""
    complex_gro = os.path.join(work_dir, 'complex.gro')

    # Read protein .gro
    with open(protein_gro, encoding='utf-8', errors='replace') as f:
        prot_lines = f.readlines()
    prot_natoms = int(prot_lines[1].strip())
    prot_atoms = prot_lines[2:2 + prot_natoms]
    prot_box = prot_lines[2 + prot_natoms] if len(prot_lines) > 2 + prot_natoms else '  10.0  10.0  10.0\n'

    # Convert HEM PDB to GRO atom lines
    hem_atoms = _pdb_to_gro_atoms(hem_pdb, start_atomnr=prot_natoms + 1, max_atoms=73)
    if not hem_atoms:
        logger.error("No HEM atoms extracted from %s", hem_pdb)
        return None

    # Read ligand .gro
    with open(ligand_gro, encoding='utf-8', errors='replace') as f:
        lig_lines = f.readlines()
    lig_natoms = int(lig_lines[1].strip())
    lig_atoms = lig_lines[2:2 + lig_natoms]

    # Renumber ligand atoms
    start_num = prot_natoms + len(hem_atoms) + 1
    renumbered_lig = []
    for i, line in enumerate(lig_atoms):
        new_num = start_num + i
        new_line = line[:15] + f'{new_num:5d}' + line[20:]
        renumbered_lig.append(new_line)

    # Write combined .gro
    total = prot_natoms + len(hem_atoms) + lig_natoms
    with open(complex_gro, 'w', encoding='utf-8') as f:
        f.write('Protein + HEM + Ligand system\n')
        f.write(f'{total}\n')
        f.writelines(prot_atoms)
        f.writelines(hem_atoms)
        f.writelines(renumbered_lig)
        f.write(prot_box)

    logger.info("Merged: %d protein + %d HEM + %d ligand = %d atoms",
                prot_natoms, len(hem_atoms), lig_natoms, total)
    return complex_gro


def _pdb_to_gro_atoms(pdb_path, start_atomnr=1, max_atoms=None):
    """Convert PDB ATOM/HETATM lines to GRO format lines.

    GRO format: resid(5) resname(5) atomname(5) atomnr(5) x(8.3f) y(8.3f) z(8.3f)
    PDB coordinates are in Angstroms, GRO in nm (divide by 10).
    """
    gro_lines = []
    atomnr = start_atomnr

    with open(pdb_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            record = line[:6].strip()
            if record not in ('ATOM', 'HETATM'):
                continue
            if max_atoms and len(gro_lines) >= max_atoms:
                break
            try:
                resid = int(line[22:26])
                resname = line[17:20].strip()
                atomname = line[12:16].strip()
                x = float(line[30:38]) / 10.0  # A -> nm
                y = float(line[38:46]) / 10.0
                z = float(line[46:54]) / 10.0
            except (ValueError, IndexError):
                continue

            # GRO format: %5d%-5s%5s%5d%8.3f%8.3f%8.3f
            gro_line = f'{resid:5d}{resname:<5s}{atomname:>5s}{atomnr:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n'
            gro_lines.append(gro_line)
            atomnr += 1

    return gro_lines


def _create_metalloprotein_ligand_index(work_dir, system_gro, lig_resnames):
    """
    Create index groups for metalloprotein + ligand system.

    Groups:
      - Protein_HEM_LIG: protein + HEM + ligand(s)
      - Water_Ion: solvent + counter ions
    """
    ndx_path = os.path.join(work_dir, 'index.ndx')

    # HEM is classified as "Other" by GROMACS, not "Protein"
    lig_sel = ' | '.join(f'r {rn}' for rn in lig_resnames)
    stdin = f'"Protein" | "Other" | {lig_sel}\n"Water" | "Ion"\nq\n'
    cmd = f'make_ndx -f "{system_gro}" -o index.ndx'
    output, ok = run_gmx(cmd, work_dir, input_text=stdin)

    if ok and os.path.exists(ndx_path):
        with open(ndx_path, encoding='utf-8', errors='replace') as f:
            groups = [l.strip('[] \n') for l in f if l.startswith('[')]

        prot_lig = None
        wat_ion = None
        for g in groups:
            # Look for the merged group: Protein_Other_LIG or Protein_LIG
            if 'Protein' in g and (
                any(rn in g for rn in lig_resnames) or 'Other' in g):
                prot_lig = g
            elif 'Water' in g and 'Ion' in g:
                wat_ion = g
        prot_lig = prot_lig or f'Protein_{lig_resnames[0]}'
        wat_ion = wat_ion or 'Water_Ion'
        return ndx_path, f'{prot_lig} {wat_ion}'

    # Python fallback
    return _python_metalloprotein_ligand_ndx(system_gro, lig_resnames, ndx_path)


def _python_metalloprotein_ligand_ndx(system_gro, lig_resnames, ndx_path):
    """Generate index.ndx for metalloprotein+ligand by parsing .gro."""
    lig_set = set(lig_resnames)
    SOLVENT = {'SOL', 'HOH', 'WAT', 'NA', 'CL', 'Na+', 'Cl-', 'K', 'K+'}

    prot_lig_atoms = []
    solvent_atoms = []

    with open(system_gro, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    natoms = int(lines[1].strip())
    for i, line in enumerate(lines[2:2 + natoms], start=1):
        res = line[5:10].strip()
        if res in SOLVENT:
            solvent_atoms.append(i)
        else:
            # Everything not solvent goes into protein+ligand group
            # (HEME is already merged into protein by CHARMM27 pdb2gmx)
            prot_lig_atoms.append(i)

    def write_group(f, name, atoms):
        f.write(f'[ {name} ]\n')
        for j in range(0, len(atoms), 15):
            chunk = atoms[j:j + 15]
            f.write(' '.join(str(a) for a in chunk) + '\n')

    group_name = f'Protein_{lig_resnames[0]}' if lig_resnames else 'Protein'
    with open(ndx_path, 'w', encoding='utf-8') as f:
        write_group(f, group_name, prot_lig_atoms)
        write_group(f, 'Water_Ion', solvent_atoms)

    return ndx_path, f'{group_name} Water_Ion'


def _write_report(report, work_dir):
    """Write a human-readable MD report."""
    report_path = os.path.join(work_dir, 'md_report.md')
    lines = [
        f'# MD Simulation Report',
        f'',
        f'- **Input:** {report["input"]}',
        f'- **Force Field:** {report.get("ff", "N/A")}',
        f'- **Simulation Time:** {report.get("sim_time_ns", "N/A")} ns',
        f'- **Temperature:** {report.get("temp", 300)} K',
        f'- **GPU:** {report.get("gpu", True)}',
        f'- **Start:** {report.get("start_time", "N/A")}',
        f'- **End:** {report.get("end_time", "N/A")}',
        f'',
        f'## Stage Results',
        f'',
    ]
    for stage, result in report.get('stages', {}).items():
        status = 'OK' if result.get('success') else 'FAILED'
        perf = result.get('ns_per_day', '')
        elapsed = result.get('elapsed_s', '')
        line = f'| {stage:12s} | {status:6s} |'
        if elapsed:
            line += f' {elapsed:>8.1f}s |'
        if perf:
            line += f' {perf:.1f} ns/day |'
        lines.append(line)

    lines.append('')
    if report.get('success'):
        lines.append('**Status: COMPLETED SUCCESSFULLY**')
    else:
        lines.append(f'**Status: FAILED** — {report.get("error", "unknown")}')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    logger.info("Report written to %s", report_path)

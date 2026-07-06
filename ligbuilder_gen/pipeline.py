"""
LigBuilder V3 Pipeline
=======================
WSL-first execution with Docker fallback.

Pipeline stages:
  1. Generate cavity.input (area mode from pocket coordinates)
  2. Run cavity → pocket_*.txt + grid_*.txt
  3. Generate build.input (de novo design config)
  4. Run build -Automatic → ligand.lig, lead.lig, mol2 files
  5. Post-process: convert mol2 → SDF, generate summary
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .config import (
    LIGBUILDER_ROOT, LIGBUILDER_WSL_ROOT, DESIGN_DEFAULTS,
    CAVITY_DEFAULTS, DOCKER_IMAGE, DOCKER_DIR, DOCKER_TIMEOUT,
    DOCKER_BUILD_TIMEOUT, WSL_TIMEOUT, DATABASE_DIRS,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Public API
# ============================================================================

def run_ligbuilder_cavity(protein_pdb: str, output_dir: str = None,
                          pocket: dict = None, config: dict = None) -> dict:
    """
    Run LigBuilder cavity detection on a protein.

    Args:
        protein_pdb: Path to protein PDB file
        output_dir: Output directory (default: <pdb_dir>/<stem>_ligbuilder/)
        pocket: Optional pocket definition {'center_x','center_y','center_z',
                'size_x','size_y','size_z'} — if provided, uses area mode
        config: Override config dict

    Returns:
        dict with pocket_file, grid_file paths or error
    """
    pdb_path = os.path.abspath(protein_pdb)
    if not os.path.isfile(pdb_path):
        return {'error': f'PDB not found: {protein_pdb}', 'hint': 'Check file path'}

    stem = Path(pdb_path).stem
    if not output_dir:
        output_dir = os.path.join(os.getcwd(), f'{stem}_ligbuilder')
    os.makedirs(output_dir, exist_ok=True)

    # Try WSL first, then Docker
    for backend_name, backend_fn in [('WSL', _run_cavity_wsl), ('Docker', _run_cavity_docker)]:
        try:
            result = backend_fn(pdb_path, output_dir, pocket, config)
            if result and 'error' not in result:
                result['backend'] = backend_name
                logger.info("Cavity detection via %s succeeded", backend_name)
                return result
            elif result:
                logger.warning("Cavity via %s failed: %s", backend_name, result.get('error'))
        except Exception as e:
            logger.warning("Cavity via %s exception: %s", backend_name, e)

    return {
        'error': 'Cavity detection failed (WSL + Docker)',
        'hint': 'Check WSL/Docker installation and LigBuilder path',
    }


def run_ligbuilder(protein_pdb: str, output_dir: str = None,
                   pocket: dict = None, design_params: dict = None,
                   config: dict = None) -> dict:
    """
    Full LigBuilder pipeline: cavity → design → post-process.

    Args:
        protein_pdb: Path to protein PDB
        output_dir: Output directory
        pocket: Pocket definition from SmartDocking
        design_params: Override design parameters (mode, sessions, molecules, etc.)
        config: Override config

    Returns:
        dict with generated ligands, summary, or error
    """
    pdb_path = os.path.abspath(protein_pdb)
    if not os.path.isfile(pdb_path):
        return {'error': f'PDB not found: {protein_pdb}', 'hint': 'Check file path'}

    stem = Path(pdb_path).stem
    if not output_dir:
        output_dir = os.path.join(os.getcwd(), f'{stem}_ligbuilder')
    os.makedirs(output_dir, exist_ok=True)

    params = {**DESIGN_DEFAULTS, **(design_params or {})}

    # Try WSL first, then Docker
    for backend_name, backend_fn in [('WSL', _run_full_wsl), ('Docker', _run_full_docker)]:
        try:
            result = backend_fn(pdb_path, output_dir, pocket, params, config)
            if result and 'error' not in result:
                result['backend'] = backend_name
                logger.info("LigBuilder design via %s succeeded", backend_name)
                return result
            elif result:
                logger.warning("Design via %s failed: %s", backend_name, result.get('error'))
        except Exception as e:
            logger.warning("Design via %s exception: %s", backend_name, e)

    return {
        'error': 'LigBuilder design failed (WSL + Docker)',
        'hint': 'Check WSL/Docker installation and LigBuilder path',
    }


# ============================================================================
# Config Generation
# ============================================================================

def _generate_cavity_input(pdb_wsl_path: str, lb_root: str,
                           pocket: dict = None) -> str:
    """Generate cavity.input content for LigBuilder."""
    cfg = dict(CAVITY_DEFAULTS)

    lines = [
        f'PARAMETER_DIRECTORY\t\t{lb_root}/parameter/',
        f'RECEPTOR_FILE\t\t\t{pdb_wsl_path}',
    ]

    if pocket:
        # Area mode (mode 2) — use SmartDocking pocket coordinates
        cx, cy, cz = pocket['center_x'], pocket['center_y'], pocket['center_z']
        sx, sy, sz = pocket.get('size_x', 20), pocket.get('size_y', 20), pocket.get('size_z', 20)
        pad = cfg['padding']

        lines.append('DETECT_MODE\t\t\t2')
        lines.append('ONLY_GENERATE_GRID\t\t0')
        lines.append(f'MIN_X\t\t\t\t{cx - sx/2 - pad:.2f}')
        lines.append(f'MAX_X\t\t\t\t{cx + sx/2 + pad:.2f}')
        lines.append(f'MIN_Y\t\t\t\t{cy - sy/2 - pad:.2f}')
        lines.append(f'MAX_Y\t\t\t\t{cy + sy/2 + pad:.2f}')
        lines.append(f'MIN_Z\t\t\t\t{cz - sz/2 - pad:.2f}')
        lines.append(f'MAX_Z\t\t\t\t{cz + sz/2 + pad:.2f}')
    else:
        # Whole protein mode
        lines.append('DETECT_MODE\t\t\t0')
        lines.append('ONLY_GENERATE_GRID\t\t0')

    lines.extend([
        f'ATOM_RADIUS_ADJUST\t\t{cfg["atom_radius_adjust"]}',
        f'RADIUS_LENTH\t\t\t{cfg["radius_length"]}',
        'BOUNDARY\t\t\t0',
        'SINGLE_LIGAND\t\t\t1',
        'MONOMER\t\t\t\t0',
        'HETMETAL\t\t\t1',
        'HETWATER\t\t\t0',
        'CHAIN_VALID\t\t\tABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890',
        'ROTATE_X\t\t\t0',
        'ROTATE_Y\t\t\t0',
        'ROTATE_Z\t\t\t0',
        'JUDGE\t\t\t\t2',
        'ERASER\t\t\t\t2',
        'RUN_INFO\t\t\t0',
        'VISUAL_OUTPUT\t\t\t1',
        'PDB_OUTPUT\t\t\t2',
        'LIGBUILDER\t\t\t1',
        'DISTANCE\t\t\t5',
        'V_POINT_NUM\t\t\t1',
        'V_OUTPUT_MOD\t\t\t0',
        'OUTPUT_RANK\t\t\t1.5',
        'RULER_1\t\t\t\t100',
        # Standard cavity parameters
        'SEPARATE_MIN_DEPTH\t\t8',
        'MAX_ABSTRACT_LIMIT_V\t\t1500',
        'SEPARATE_MAX_LIMIT_V\t\t6000',
        'MIN_ABSTRACT_DEPTH\t\t2',
        'SEPARATE_CHIP_LIMIT\t\t20',
        'SEPARATE_MAX_LIMIT\t\t2000',
        'MAX_ABSTRACT_LIMIT\t\t800',
        'SEPARATE_CHIP_LIMIT_V\t\t200',
        'SEPARATE_CHIP_DEPTH\t\t1',
        'MAX_ABSTRACT_DEPTH\t\t20',
        'MAX_DEPTH_VACANT\t\t100',
        'RIGID_LIMIT\t\t\t0',
        'MINIMAL_FEATURE_DISTANCE\t3.50',
        'MAXIMAL_FEATURE_NUMBER\t\t12',
        'RADIUS_MIS\t\t\t0.7',
        'EDGE_ADJUST\t\t\t0',
        'VACANT_ADJUST\t\t\t0',
    ])

    return '\n'.join(lines) + '\n'


def _generate_build_input(pdb_wsl_path: str, pocket_file: str, grid_file: str,
                          lb_root: str, work_dir: str,
                          task_name: str, params: dict) -> str:
    """Generate build.input content for LigBuilder de novo design."""
    mode_map = {'explore': 0, 'grow': 1, 'link': 2}
    design_mode = mode_map.get(params.get('mode', 'explore'), 0)
    result_dir = f'{work_dir}/result'

    lines = [
        f'INCLUDE\t\t\t\t{lb_root}/default/default.input',
        '',
        f'$TASK_NAME$\t\t\t{task_name}',
        f'$RESULT_DIR$\t\t\t{result_dir}',
        '',
        # Path configuration (inline from path.input)
        f'RUN_LIST\t\t\t{work_dir}/run_{task_name}.list',
        f'ACTIVE_LIST\t\t\t{work_dir}/active_{task_name}.list',
        f'LIGAND_COLLECTION_FILE\t\t{result_dir}/output_{task_name}/ligand.lig',
        f'LEAD_COLLECTION_FILE\t\t{result_dir}/output_{task_name}/lead.lig',
        f'SEED_RECORD_FILE\t\t{result_dir}/output_{task_name}/seed.lig',
        f'CONTROL_FILE\t\t\t{result_dir}/output_{task_name}/control.lig',
        f'STATUS_FILE\t\t\t{result_dir}/output_{task_name}/status.lig',
        f'EXTERNAL_INPUT_FILE\t\t{result_dir}/output_{task_name}/external.mol2',
        f'EXTERNAL_OUTPUT_FILE\t\t{result_dir}/output_{task_name}/external.dat',
        f'PROCESS_FILE\t\t\t{result_dir}/output_{task_name}/ligand.lig',
        f'PROCESS_OUTPUT_DIRECTORY\t\t{result_dir}/process_{task_name}/',
        f'FILTER_LIST\t\t\t{result_dir}/process_{task_name}/INDEX',
        f'FILTER_LOG\t\t\t{result_dir}/process_{task_name}/INDEX.filter',
        f'SCORE_LIST\t\t\t{result_dir}/process_{task_name}/INDEX.filter',
        f'SCORE_LOG\t\t\t{result_dir}/process_{task_name}/INDEX.score',
        f'RECOMMEND_LIST\t\t\t{result_dir}/process_{task_name}/INDEX.score',
        f'RECOMMEND_LOG\t\t\t{result_dir}/process_{task_name}/INDEX.recommend',
        f'CLUSTER_LIST\t\t\t{result_dir}/process_{task_name}/INDEX.recommend',
        f'CLUSTER_OUTPUT_DIRECTORY\t\t{result_dir}/cluster_{task_name}/',
        f'REPORT_LIST\t\t\t{result_dir}/cluster_{task_name}/INDEX',
        f'REPORT_HTML\t\t\t{result_dir}/report_{task_name}/report.html',
        f'REPORT_OUTPUT_MOL2\t\t{result_dir}/cluster_{task_name}/total.mol2',
        f'REPORT_SYNTHESIZE_LOG\t\t{result_dir}/synthesize_{task_name}/synthesize.log',
        f'ANALYSIS_INPUT_FILE\t\t{result_dir}/cluster_{task_name}/total.mol2',
        f'ANALYSIS_LOG_FILE\t\t{result_dir}/synthesize_{task_name}/synthesize.log',
        f'ANALYSIS_OUTPUT_DIRECTORY\t{result_dir}/synthesize_{task_name}',
        '',
        # Design parameters
        f'DESIGN_MODE\t\t\t{design_mode}',
        '',
        'PROTEIN_NUM\t\t\t1',
        f'POCKET_ATOM_FILE[1]\t\t{pocket_file}',
        f'POCKET_GRID_FILE[1]\t\t{grid_file}',
        f'PROTEIN_FILE[1]\t\t\t{pdb_wsl_path}',
        'MULTI_WEIGHT[1]\t\t\t1.0',
        '',
        'APPLY_KEY_INTERACTION\t\tNO',
        'APPLY_EXTERNAL_CALL\t\tNO',
        '',
        'CONTINUE_MODE\t\t\tNO',
        f'SESSION_NUMBER\t\t\t{params["sessions"]}',
        'MINIMAL_PROCESS_SESSION\t\t1',
        'START_STAGE\t\t\t1',
        'END_STAGE\t\t\t9',
        f'MOLECULE_NUMBER\t\t\t{params["molecules"]}',
        '',
        # Drug-likeness filters
        f'MAXIMAL_MOLECULAR_WEIGHT\t{params["max_mw"]}',
        f'MINIMAL_MOLECULAR_WEIGHT\t{params["min_mw"]}',
        f'MAXIMAL_LOGP\t\t\t{params["max_logp"]}',
        f'MINIMAL_LOGP\t\t\t{params["min_logp"]}',
        f'MAXIMAL_HB_DONOR_ATOM\t\t{params["max_hbd"]}',
        'MINIMAL_HB_DONOR_ATOM\t\t0',
        f'MAXIMAL_HB_ACCEPTOR_ATOM\t{params["max_hba"]}',
        'MINIMAL_HB_ACCEPTOR_ATOM\t0',
        'MAXIMAL_PKD\t\t\t16.00',
        f'MINIMAL_PKD\t\t\t{params["min_pkd"]}',
        'MINIMAL_AVER_PKD\t\t0.0',
        'MAXIMAL_RESULTS\t\t\t5000',
        'MAXIMAL_BUMP\t\t\t15',
        'MAXIMAL_NORING\t\t\t10',
        'MAXIMAL_BRANCHPERCENT\t\t90',
        'MAXIMAL_SINGLEBRANCH\t\t9',
        'MAXIMAL_HMVSHB\t\t\t5',
        'MINIMAL_STRUCTURE_SCORE\t\t-20',
        'MINIMAL_CHEMICAL_SCORE\t\t-200',
        '',
        # GA parameters
        'NUMBER_OF_GENERATION\t\t10',
        'NUMBER_OF_POPULATION\t\t300',
        'NUMBER_OF_PARENTS\t\t30',
        'ELITISM_RATIO\t\t\t0.10',
        'SIMILARITY_CUTOFF\t\t0.85',
        'CYCLE_TURN\t\t\t5',
        'MULTI_COUNT\t\t\t10',
        'MAXIMAL_COUNT\t\t\t50',
        'MULTI_SEED_DESIGN\t\t5',
        'SEED_GENERATOR\t\t\t1',
        'BUILDING_BLOCK_MODE\t\t100',
        '',
        # Recommendation
        'RECOMMEND_NUMBER\t\t20',
        'RECOMMEND_SCORE\t\t\t4.0',
        'RECOMMEND_STABLE\t\t0.0',
        'RECOMMEND_BUMP\t\t\t20',
        'RECOMMEND_RT\t\t\t20',
        'RECOMMEND_HMHB\t\t\t10',
        'RECOMMEND_DHM\t\t\t-5',
        'RECOMMEND_IVDW\t\t\t3',
        'RECOMMEND_MATCH\t\t\t-20.0',
        'RECOMMEND_CHEMICAL\t\t-200',
        'MINIMAL_OUTPUT_PKD\t\t2.0',
        '',
        # Diversity (moderate)
        # Override all database paths with absolute paths (critical for daemon mode)
        f'PARAMETER_DIRECTORY\t\t{lb_root}/parameter/',
        f'BUILDING_BLOCK_LIBRARY\t\t{lb_root}/fragment.mdb/',
        f'BUILDING_BLOCK_LIBRARY_EXTEND\t{lb_root}/fragment.mdb/extend/',
        f'BUILDING_BLOCK_LIBRARY_ROTATABLE\t{lb_root}/fragment.mdb/rotate_bio_200/',
        f'FORBIDDEN_STRUCTURE_LIBRARY\t{lb_root}/forbidden.mdb/',
        f'TOXIC_STRUCTURE_LIBRARY\t\t{lb_root}/toxicity.mdb/',
        f'FINALCHECK_STRUCTURE_LIBRARY\t{lb_root}/finalcheck.mdb/',
        f'SYNTHESIZE_LIBRARY\t\t{lb_root}/synthesize.mdb/',
        f'FILTER_LIBRARY\t\t\t{lb_root}/filter.mdb/',
        f'DERIVATION_LIBRARY\t\t{lb_root}/derivation.mdb/',
        '',
        f'INCLUDE\t\t\t\t{lb_root}/default/diversity_moderate.input',
        f'INCLUDE\t\t\t\t{lb_root}/default/synthesize_rapid.input',
        f'INCLUDE\t\t\t\t{lb_root}/default/optimize_soft.input',
        f'INCLUDE\t\t\t\t{lb_root}/default/recommend_moderate.input',
        f'INCLUDE\t\t\t\t{lb_root}/default/analysis_rapid.input',
        f'INCLUDE\t\t\t\t{lb_root}/default/shared_memory.input',
        f'INCLUDE\t\t\t\t{lb_root}/default/design-memory_1G.input',
        f'INCLUDE\t\t\t\t{lb_root}/default/synthesize-memory_1G.input',
        f'INCLUDE\t\t\t\t{lb_root}/default/analysis-memory_2G.input',
        f'INCLUDE\t\t\t\t{lb_root}/default/speedup.input',
    ]

    return '\n'.join(lines) + '\n'


# ============================================================================
# WSL Backend
# ============================================================================

def _win_to_wsl(win_path: str) -> str:
    """Convert Windows path to WSL mount path."""
    p = os.path.abspath(win_path).replace('\\', '/')
    if len(p) >= 2 and p[1] == ':':
        drive = p[0].lower()
        return f'/mnt/{drive}{p[2:]}'
    return p


def _run_wsl_cmd(cmd: str, timeout: int = WSL_TIMEOUT, cwd: str = None) -> subprocess.CompletedProcess:
    """Run a command in WSL."""
    full_cmd = ['wsl', '-e', 'bash', '-c', cmd]
    return subprocess.run(
        full_cmd, capture_output=True, text=True, timeout=timeout,
        cwd=cwd,
    )


def _run_cavity_wsl(pdb_path: str, output_dir: str,
                    pocket: dict = None, config: dict = None) -> dict:
    """Run cavity detection via WSL.

    Note: LigBuilder outputs cavity files next to the RECEPTOR_FILE,
    so we copy the PDB into output_dir and point RECEPTOR_FILE there.
    """
    lb_root = (config or {}).get('ligbuilder_wsl_root', LIGBUILDER_WSL_ROOT)
    stem = Path(pdb_path).stem

    # Copy PDB to output_dir so cavity files land there
    pdb_copy = os.path.join(output_dir, os.path.basename(pdb_path))
    if os.path.abspath(pdb_path) != os.path.abspath(pdb_copy):
        shutil.copy2(pdb_path, pdb_copy)
    pdb_wsl = _win_to_wsl(pdb_copy)
    out_wsl = _win_to_wsl(output_dir)

    # Generate cavity.input
    cavity_input = _generate_cavity_input(pdb_wsl, lb_root, pocket)
    cavity_input_path = os.path.join(output_dir, 'cavity.input')
    with open(cavity_input_path, 'w', encoding='utf-8') as f:
        f.write(cavity_input)

    cavity_input_wsl = _win_to_wsl(cavity_input_path)

    # Run cavity
    cmd = f'ulimit -s unlimited && cd "{out_wsl}" && "{lb_root}/bin/cavity" "{cavity_input_wsl}"'
    logger.info("Running cavity: %s", cmd)
    proc = _run_wsl_cmd(cmd, timeout=WSL_TIMEOUT)

    if proc.returncode != 0:
        return {
            'error': f'Cavity detection failed (exit {proc.returncode})',
            'stderr': proc.stderr[:2000] if proc.stderr else '',
            'stdout': proc.stdout[:2000] if proc.stdout else '',
        }

    # Find output files (LigBuilder writes next to RECEPTOR_FILE = output_dir)
    pocket_files = sorted(Path(output_dir).glob(f'{stem}_pocket_*.txt'))
    grid_files = sorted(Path(output_dir).glob(f'{stem}_grid_*.txt'))

    if not pocket_files or not grid_files:
        pocket_files = sorted(Path(output_dir).glob('*_pocket_*.txt'))
        grid_files = sorted(Path(output_dir).glob('*_grid_*.txt'))

    # Also check PDB source directory as fallback
    if not pocket_files or not grid_files:
        src_dir = os.path.dirname(os.path.abspath(pdb_path))
        pocket_files = sorted(Path(src_dir).glob(f'{stem}_pocket_*.txt'))
        grid_files = sorted(Path(src_dir).glob(f'{stem}_grid_*.txt'))
        # Move them to output_dir
        for f in pocket_files + grid_files:
            shutil.move(str(f), output_dir)
        pocket_files = sorted(Path(output_dir).glob(f'{stem}_pocket_*.txt'))
        grid_files = sorted(Path(output_dir).glob(f'{stem}_grid_*.txt'))

    if not pocket_files or not grid_files:
        return {
            'error': 'No pocket/grid files generated',
            'stdout': proc.stdout[:2000] if proc.stdout else '',
            'hint': 'Cavity detection ran but produced no output',
        }

    return {
        'success': True,
        'pocket_file': str(pocket_files[0]),
        'grid_file': str(grid_files[0]),
        'n_pockets': len(pocket_files),
        'all_pockets': [str(p) for p in pocket_files],
        'all_grids': [str(g) for g in grid_files],
        'output_dir': output_dir,
        'stdout': proc.stdout[:2000] if proc.stdout else '',
    }


def _run_full_wsl(pdb_path: str, output_dir: str, pocket: dict = None,
                  params: dict = None, config: dict = None) -> dict:
    """Run full LigBuilder pipeline (cavity + build) via WSL."""
    lb_root = (config or {}).get('ligbuilder_wsl_root', LIGBUILDER_WSL_ROOT)
    pdb_wsl = _win_to_wsl(pdb_path)
    out_wsl = _win_to_wsl(output_dir)
    stem = Path(pdb_path).stem
    task_name = stem.replace(' ', '_').replace('-', '_')
    params = params or DESIGN_DEFAULTS

    # Step 1: Cavity detection (copies PDB to output_dir internally)
    logger.info("Step 1/5: Cavity detection...")
    cavity_result = _run_cavity_wsl(pdb_path, output_dir, pocket, config)
    if 'error' in cavity_result:
        return cavity_result

    # Use the PDB copy in output_dir
    pdb_copy = os.path.join(output_dir, os.path.basename(pdb_path))
    pdb_copy_wsl = _win_to_wsl(pdb_copy)
    pocket_file_wsl = _win_to_wsl(cavity_result['pocket_file'])
    grid_file_wsl = _win_to_wsl(cavity_result['grid_file'])

    # Step 2: Generate build.input
    logger.info("Step 2/5: Generating build configuration...")
    build_input = _generate_build_input(
        pdb_copy_wsl, pocket_file_wsl, grid_file_wsl,
        lb_root, out_wsl, task_name, params
    )
    build_input_path = os.path.join(output_dir, 'build.input')
    with open(build_input_path, 'w', encoding='utf-8') as f:
        f.write(build_input)

    build_input_wsl = _win_to_wsl(build_input_path)

    # Step 3: Run build -Automatic (master) + Daemon sessions (workers)
    # Note: Do NOT pre-create result directories — LigBuilder creates them
    logger.info("Step 3/5: Running de novo design (sessions=%d, molecules=%d)...",
                params['sessions'], params['molecules'])

    n_sessions = params['sessions']

    # Launch Automatic monitor + all Daemon sessions in a single shell script
    # The Automatic process monitors; Daemons do the actual design work
    script = (
        f'ulimit -s unlimited\n'
        f'cd "{out_wsl}"\n'
        f'"{lb_root}/bin/build" -Automatic "{build_input_wsl}" &\n'
        f'AUTO_PID=$!\n'
        f'sleep 10\n'  # Wait for Automatic to generate run list
    )
    for i in range(1, n_sessions + 1):
        script += f'"{lb_root}/bin/build" -Daemon "{build_input_wsl}" {i} &\n'
    script += (
        f'wait $AUTO_PID\n'
        f'echo "BUILD_COMPLETE"\n'
    )

    # Write and execute the script
    script_path = os.path.join(output_dir, '_run_ligbuilder.sh')
    with open(script_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(script)

    script_wsl = _win_to_wsl(script_path)
    cmd = f'bash "{script_wsl}"'
    proc = _run_wsl_cmd(cmd, timeout=WSL_TIMEOUT)

    # Step 5: Collect results
    logger.info("Step 4/5: Collecting results...")
    return _collect_results(output_dir, task_name, cavity_result, proc, params)


def _collect_results(output_dir: str, task_name: str,
                     cavity_result: dict, proc: subprocess.CompletedProcess,
                     params: dict) -> dict:
    """Collect and summarize LigBuilder results."""
    result_dir = os.path.join(output_dir, 'result')
    output_task = os.path.join(result_dir, f'output_{task_name}')
    cluster_task = os.path.join(result_dir, f'cluster_{task_name}')
    report_task = os.path.join(result_dir, f'report_{task_name}')

    # Count generated molecules
    ligand_lig = os.path.join(output_task, 'ligand.lig')
    lead_lig = os.path.join(output_task, 'lead.lig')
    n_ligands = _count_lig_entries(ligand_lig)
    n_leads = _count_lig_entries(lead_lig)

    # Find MOL2 files
    mol2_files = sorted(Path(cluster_task).glob('*.mol2')) if os.path.isdir(cluster_task) else []
    total_mol2 = os.path.join(cluster_task, 'total.mol2')

    # Convert total.mol2 to SDF if available
    sdf_files = []
    if os.path.isfile(total_mol2):
        sdf_path = os.path.join(output_dir, f'{task_name}_ligands.sdf')
        try:
            _mol2_to_sdf_wsl(total_mol2, sdf_path)
            sdf_files.append(sdf_path)
        except Exception as e:
            logger.warning("MOL2→SDF conversion failed: %s", e)

    # Individual MOL2 → SDF
    process_dir = os.path.join(result_dir, f'process_{task_name}')
    index_file = os.path.join(process_dir, 'INDEX')
    individual_mol2 = []
    if os.path.isfile(index_file):
        with open(index_file, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                mol2_path = line.strip()
                if mol2_path and os.path.isfile(mol2_path):
                    individual_mol2.append(mol2_path)

    # Check report
    report_html = os.path.join(report_task, 'report.html')
    has_report = os.path.isfile(report_html)

    # Parse stdout for pKd predictions
    pkd_info = _parse_pkd_from_output(proc.stdout or '')

    result = {
        'success': True,
        'output_dir': output_dir,
        'n_ligands_generated': n_ligands,
        'n_leads': n_leads,
        'n_mol2_files': len(mol2_files),
        'sdf_files': [str(s) for s in sdf_files],
        'mol2_files': [str(m) for m in mol2_files[:20]],
        'report_html': report_html if has_report else None,
        'pocket_file': cavity_result.get('pocket_file'),
        'grid_file': cavity_result.get('grid_file'),
        'n_pockets': cavity_result.get('n_pockets', 0),
        'design_params': params,
        'pkd_info': pkd_info,
        'build_stdout': (proc.stdout or '')[-3000:],
        'build_stderr': (proc.stderr or '')[-1000:],
        'build_exitcode': proc.returncode,
    }

    logger.info("LigBuilder complete: %d ligands, %d leads, %d SDF files",
                n_ligands, n_leads, len(sdf_files))
    return result


def _count_lig_entries(lig_file: str) -> int:
    """Count molecules in a .lig file."""
    if not os.path.isfile(lig_file):
        return 0
    count = 0
    with open(lig_file, 'r', errors='replace') as f:
        for line in f:
            if line.startswith('REMARK') and 'Predicted pKd' in line:
                count += 1
    return count


def _parse_pkd_from_output(stdout: str) -> dict:
    """Parse pKd information from LigBuilder output."""
    info = {'max_pkd': None, 'molecules_designed': 0}
    for line in stdout.split('\n'):
        if 'Output Number' in line:
            try:
                info['molecules_designed'] = int(line.split(':')[-1].strip())
            except (ValueError, IndexError):
                pass
        if 'Maximal Binding Score' in line:
            try:
                score = float(line.split(':')[-1].strip().split()[0])
                if info['max_pkd'] is None or score > info['max_pkd']:
                    info['max_pkd'] = score
            except (ValueError, IndexError):
                pass
    return info


def _mol2_to_sdf_wsl(mol2_path: str, sdf_path: str):
    """Convert MOL2 to SDF using OpenBabel via WSL."""
    mol2_wsl = _win_to_wsl(mol2_path)
    sdf_wsl = _win_to_wsl(sdf_path)
    cmd = f'obabel "{mol2_wsl}" -O "{sdf_wsl}" -m 2>/dev/null || obabel "{mol2_wsl}" -O "{sdf_wsl}"'
    proc = _run_wsl_cmd(cmd, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f'obabel failed: {proc.stderr}')


# ============================================================================
# Docker Backend
# ============================================================================

def _docker_available() -> bool:
    """Check if Docker image is available."""
    try:
        proc = subprocess.run(
            ['docker', 'image', 'inspect', DOCKER_IMAGE],
            capture_output=True, timeout=10)
        return proc.returncode == 0
    except Exception:
        return False


def _build_docker() -> bool:
    """Build the Docker image."""
    try:
        proc = subprocess.run(
            ['docker', 'build', '-t', DOCKER_IMAGE, DOCKER_DIR],
            capture_output=True, timeout=DOCKER_BUILD_TIMEOUT)
        return proc.returncode == 0
    except Exception:
        return False


def _run_cavity_docker(pdb_path: str, output_dir: str,
                       pocket: dict = None, config: dict = None) -> dict:
    """Run cavity detection via Docker."""
    if not _docker_available():
        if not _build_docker():
            return {'error': 'Docker image not available and build failed'}

    lb_root_win = (config or {}).get('ligbuilder_root', LIGBUILDER_ROOT)
    pdb_dir = os.path.dirname(os.path.abspath(pdb_path))
    pdb_name = os.path.basename(pdb_path)

    cmd = [
        'docker', 'run', '--rm',
        '-v', f'{pdb_dir}:/input:ro',
        '-v', f'{output_dir}:/output',
        '-v', f'{lb_root_win}:/ligbuilder:ro',
    ]

    # Add database volumes
    for db_dir in DATABASE_DIRS:
        db_path = os.path.join(lb_root_win, db_dir)
        if os.path.isdir(db_path):
            cmd.extend(['-v', f'{db_path}:/ligbuilder/{db_dir}:ro'])

    cmd.extend([
        DOCKER_IMAGE,
        'cavity',
        f'/input/{pdb_name}',
        '/output',
    ])

    if pocket:
        cmd.extend([
            '--center', str(pocket['center_x']), str(pocket['center_y']), str(pocket['center_z']),
            '--size', str(pocket.get('size_x', 20)), str(pocket.get('size_y', 20)), str(pocket.get('size_z', 20)),
        ])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=DOCKER_TIMEOUT)
        if proc.returncode != 0:
            return {'error': f'Docker cavity failed (exit {proc.returncode})', 'stderr': proc.stderr[:2000]}

        # Parse JSON output from container
        try:
            result = json.loads(proc.stdout)
            return result
        except json.JSONDecodeError:
            # Look for files directly
            pocket_files = sorted(Path(output_dir).glob('*_pocket_*.txt'))
            grid_files = sorted(Path(output_dir).glob('*_grid_*.txt'))
            if pocket_files and grid_files:
                return {
                    'success': True,
                    'pocket_file': str(pocket_files[0]),
                    'grid_file': str(grid_files[0]),
                    'n_pockets': len(pocket_files),
                }
            return {'error': 'Docker ran but no output files found'}
    except subprocess.TimeoutExpired:
        return {'error': 'Docker cavity timed out'}


def _run_full_docker(pdb_path: str, output_dir: str, pocket: dict = None,
                     params: dict = None, config: dict = None) -> dict:
    """Run full pipeline via Docker."""
    if not _docker_available():
        if not _build_docker():
            return {'error': 'Docker image not available and build failed'}

    lb_root_win = (config or {}).get('ligbuilder_root', LIGBUILDER_ROOT)
    pdb_dir = os.path.dirname(os.path.abspath(pdb_path))
    pdb_name = os.path.basename(pdb_path)
    params = params or DESIGN_DEFAULTS

    cmd = [
        'docker', 'run', '--rm',
        '-v', f'{pdb_dir}:/input:ro',
        '-v', f'{output_dir}:/output',
        '-v', f'{lb_root_win}:/ligbuilder:ro',
    ]

    cmd.extend([
        DOCKER_IMAGE,
        'full',
        f'/input/{pdb_name}',
        '/output',
        '--sessions', str(params['sessions']),
        '--molecules', str(params['molecules']),
        '--mode', params.get('mode', 'explore'),
        '--min-pkd', str(params['min_pkd']),
    ])

    if pocket:
        cmd.extend([
            '--center', str(pocket['center_x']), str(pocket['center_y']), str(pocket['center_z']),
            '--size', str(pocket.get('size_x', 20)), str(pocket.get('size_y', 20)), str(pocket.get('size_z', 20)),
        ])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=DOCKER_TIMEOUT)
        try:
            result = json.loads(proc.stdout)
            return result
        except json.JSONDecodeError:
            if proc.returncode != 0:
                return {'error': f'Docker design failed (exit {proc.returncode})', 'stderr': proc.stderr[:2000]}
            return {'error': 'Docker ran but returned no JSON', 'stdout': proc.stdout[:2000]}
    except subprocess.TimeoutExpired:
        return {'error': 'Docker design timed out (increase DOCKER_TIMEOUT)'}

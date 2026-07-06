#!/usr/bin/env python3
"""
LigBuilder V3 Container Entrypoint
====================================
Usage:
  python run_ligbuilder.py cavity <pdb> <output_dir> [--center X Y Z] [--size X Y Z]
  python run_ligbuilder.py full <pdb> <output_dir> [options]

Options:
  --center X Y Z      Pocket center coordinates
  --size X Y Z        Pocket box size (default: 20 20 20)
  --mode {explore,grow,link}   Design mode (default: explore)
  --sessions N        Number of sessions (default: 5)
  --molecules N       Target molecule count (default: 500)
  --min-pkd F         Minimum pKd threshold (default: 3.0)
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


LB_ROOT = os.environ.get('LIGBUILDER_ROOT', '/ligbuilder')


def generate_cavity_input(pdb_path, output_dir, center=None, size=None):
    """Generate cavity.input file."""
    lines = [
        f'PARAMETER_DIRECTORY\t\t{LB_ROOT}/parameter/',
        f'RECEPTOR_FILE\t\t\t{pdb_path}',
    ]

    if center:
        pad = 5.0
        sx, sy, sz = size or [20, 20, 20]
        lines.extend([
            'DETECT_MODE\t\t\t2',
            'ONLY_GENERATE_GRID\t\t0',
            f'MIN_X\t\t\t\t{center[0] - sx/2 - pad:.2f}',
            f'MAX_X\t\t\t\t{center[0] + sx/2 + pad:.2f}',
            f'MIN_Y\t\t\t\t{center[1] - sy/2 - pad:.2f}',
            f'MAX_Y\t\t\t\t{center[1] + sy/2 + pad:.2f}',
            f'MIN_Z\t\t\t\t{center[2] - sz/2 - pad:.2f}',
            f'MAX_Z\t\t\t\t{center[2] + sz/2 + pad:.2f}',
        ])
    else:
        lines.extend([
            'DETECT_MODE\t\t\t0',
            'ONLY_GENERATE_GRID\t\t0',
        ])

    lines.extend([
        'ATOM_RADIUS_ADJUST\t\t1.4',
        'RADIUS_LENTH\t\t\t15.0',
        'BOUNDARY\t\t\t0', 'SINGLE_LIGAND\t\t\t1',
        'MONOMER\t\t\t\t0', 'HETMETAL\t\t\t1', 'HETWATER\t\t\t0',
        'CHAIN_VALID\t\t\tABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890',
        'ROTATE_X\t\t\t0', 'ROTATE_Y\t\t\t0', 'ROTATE_Z\t\t\t0',
        'JUDGE\t\t\t\t2', 'ERASER\t\t\t\t2',
        'RUN_INFO\t\t\t0', 'VISUAL_OUTPUT\t\t\t1',
        'PDB_OUTPUT\t\t\t2', 'LIGBUILDER\t\t\t1',
        'DISTANCE\t\t\t5', 'V_POINT_NUM\t\t\t1', 'V_OUTPUT_MOD\t\t\t0',
        'OUTPUT_RANK\t\t\t1.5', 'RULER_1\t\t\t\t100',
        'SEPARATE_MIN_DEPTH\t\t8', 'MAX_ABSTRACT_LIMIT_V\t\t1500',
        'SEPARATE_MAX_LIMIT_V\t\t6000', 'MIN_ABSTRACT_DEPTH\t\t2',
        'SEPARATE_CHIP_LIMIT\t\t20', 'SEPARATE_MAX_LIMIT\t\t2000',
        'MAX_ABSTRACT_LIMIT\t\t800', 'SEPARATE_CHIP_LIMIT_V\t\t200',
        'SEPARATE_CHIP_DEPTH\t\t1', 'MAX_ABSTRACT_DEPTH\t\t20',
        'MAX_DEPTH_VACANT\t\t100', 'RIGID_LIMIT\t\t\t0',
        'MINIMAL_FEATURE_DISTANCE\t3.50', 'MAXIMAL_FEATURE_NUMBER\t\t12',
        'RADIUS_MIS\t\t\t0.7', 'EDGE_ADJUST\t\t\t0', 'VACANT_ADJUST\t\t\t0',
    ])

    return '\n'.join(lines) + '\n'


def run_cavity(pdb_path, output_dir, center=None, size=None):
    """Run cavity detection."""
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(pdb_path).stem

    cavity_input = generate_cavity_input(pdb_path, output_dir, center, size)
    input_path = os.path.join(output_dir, 'cavity.input')
    with open(input_path, 'w', encoding='utf-8') as f:
        f.write(cavity_input)

    cmd = f'ulimit -s unlimited && cd "{output_dir}" && "{LB_ROOT}/bin/cavity" "{input_path}"'
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)

    pocket_files = sorted(Path(output_dir).glob('*_pocket_*.txt'))
    grid_files = sorted(Path(output_dir).glob('*_grid_*.txt'))

    return {
        'success': proc.returncode == 0 and len(pocket_files) > 0,
        'pocket_files': [str(p) for p in pocket_files],
        'grid_files': [str(g) for g in grid_files],
        'exit_code': proc.returncode,
        'stdout': proc.stdout[-2000:],
        'stderr': proc.stderr[-1000:],
    }


def run_full(pdb_path, output_dir, center=None, size=None,
             mode='explore', sessions=5, molecules=500, min_pkd=3.0):
    """Run full cavity + build pipeline."""
    # Step 1: Cavity
    cavity_result = run_cavity(pdb_path, output_dir, center, size)
    if not cavity_result['success']:
        return {'error': 'Cavity detection failed', **cavity_result}

    pocket_file = cavity_result['pocket_files'][0]
    grid_file = cavity_result['grid_files'][0]
    stem = Path(pdb_path).stem
    task_name = stem.replace(' ', '_').replace('-', '_')

    # Step 2: Generate build.input
    mode_map = {'explore': 0, 'grow': 1, 'link': 2}
    result_dir = os.path.join(output_dir, 'result')
    for sub in [f'output_{task_name}', f'process_{task_name}',
                f'cluster_{task_name}', f'report_{task_name}',
                f'synthesize_{task_name}']:
        os.makedirs(os.path.join(result_dir, sub), exist_ok=True)

    build_lines = [
        f'INCLUDE\t\t\t\t{LB_ROOT}/default/default.input',
        f'$TASK_NAME$\t\t\t{task_name}',
        f'$RESULT_DIR$\t\t\t{result_dir}',
        f'INCLUDE\t\t\t\t{LB_ROOT}/default/path.input',
        f'DESIGN_MODE\t\t\t{mode_map.get(mode, 0)}',
        f'PROTEIN_NUM\t\t\t1',
        f'POCKET_ATOM_FILE[1]\t\t{pocket_file}',
        f'POCKET_GRID_FILE[1]\t\t{grid_file}',
        f'PROTEIN_FILE[1]\t\t\t{pdb_path}',
        f'MULTI_WEIGHT[1]\t\t\t1.0',
        f'APPLY_KEY_INTERACTION\t\tNO',
        f'APPLY_EXTERNAL_CALL\t\tNO',
        f'CONTINUE_MODE\t\t\tNO',
        f'SESSION_NUMBER\t\t\t{sessions}',
        f'MINIMAL_PROCESS_SESSION\t\t1',
        f'START_STAGE\t\t\t1',
        f'END_STAGE\t\t\t9',
        f'MOLECULE_NUMBER\t\t\t{molecules}',
        f'MINIMAL_PKD\t\t\t{min_pkd}',
        f'MAXIMAL_MOLECULAR_WEIGHT\t500',
        f'MINIMAL_MOLECULAR_WEIGHT\t100',
        f'MAXIMAL_LOGP\t\t\t5.0',
        f'MINIMAL_LOGP\t\t\t-1.0',
        f'MAXIMAL_HB_DONOR_ATOM\t\t5',
        f'MAXIMAL_HB_ACCEPTOR_ATOM\t10',
        f'INCLUDE\t\t\t\t{LB_ROOT}/default/diversity_moderate.input',
        f'INCLUDE\t\t\t\t{LB_ROOT}/default/synthesize_rapid.input',
        f'INCLUDE\t\t\t\t{LB_ROOT}/default/optimize_soft.input',
        f'INCLUDE\t\t\t\t{LB_ROOT}/default/recommend_moderate.input',
        f'INCLUDE\t\t\t\t{LB_ROOT}/default/analysis_rapid.input',
        f'INCLUDE\t\t\t\t{LB_ROOT}/default/shared_memory.input',
        f'INCLUDE\t\t\t\t{LB_ROOT}/default/design-memory_1G.input',
        f'INCLUDE\t\t\t\t{LB_ROOT}/default/synthesize-memory_1G.input',
        f'INCLUDE\t\t\t\t{LB_ROOT}/default/analysis-memory_2G.input',
        f'INCLUDE\t\t\t\t{LB_ROOT}/default/speedup.input',
    ]

    build_input_path = os.path.join(output_dir, 'build.input')
    with open(build_input_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(build_lines) + '\n')

    # Step 3: Run build
    cmd = (f'ulimit -s unlimited && cd "{output_dir}" && '
           f'"{LB_ROOT}/bin/build" -Automatic "{build_input_path}"')
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600)

    # Step 4: Collect results
    ligand_lig = os.path.join(result_dir, f'output_{task_name}', 'ligand.lig')
    lead_lig = os.path.join(result_dir, f'output_{task_name}', 'lead.lig')
    total_mol2 = os.path.join(result_dir, f'cluster_{task_name}', 'total.mol2')

    n_ligands = 0
    if os.path.isfile(ligand_lig):
        with open(ligand_lig, 'r', errors='replace') as f:
            n_ligands = sum(1 for l in f if 'Predicted pKd' in l)

    # Convert to SDF
    sdf_path = os.path.join(output_dir, f'{task_name}_ligands.sdf')
    if os.path.isfile(total_mol2):
        os.system(f'obabel "{total_mol2}" -O "{sdf_path}" 2>/dev/null')

    result = {
        'success': proc.returncode == 0 or n_ligands > 0,
        'n_ligands': n_ligands,
        'output_dir': output_dir,
        'sdf_file': sdf_path if os.path.isfile(sdf_path) else None,
        'ligand_lig': ligand_lig if os.path.isfile(ligand_lig) else None,
        'total_mol2': total_mol2 if os.path.isfile(total_mol2) else None,
        'report_html': os.path.join(result_dir, f'report_{task_name}', 'report.html'),
        'cavity': cavity_result,
        'build_exit_code': proc.returncode,
        'build_stdout': proc.stdout[-3000:],
        'build_stderr': proc.stderr[-1000:],
    }

    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description='LigBuilder V3 Container')
    parser.add_argument('command', choices=['cavity', 'full'], help='Command')
    parser.add_argument('pdb', help='Protein PDB file')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--center', nargs=3, type=float, help='Pocket center X Y Z')
    parser.add_argument('--size', nargs=3, type=float, default=[20, 20, 20], help='Box size X Y Z')
    parser.add_argument('--mode', default='explore', choices=['explore', 'grow', 'link'])
    parser.add_argument('--sessions', type=int, default=5)
    parser.add_argument('--molecules', type=int, default=500)
    parser.add_argument('--min-pkd', type=float, default=3.0)

    args = parser.parse_args()

    if not os.path.exists(args.pdb):
        print(json.dumps({'error': f'PDB not found: {args.pdb}'}))
        sys.exit(1)

    if args.command == 'cavity':
        result = run_cavity(args.pdb, args.output_dir, args.center, args.size)
    else:
        result = run_full(args.pdb, args.output_dir, args.center, args.size,
                          args.mode, args.sessions, args.molecules, args.min_pkd)

    if not result.get('success'):
        print(json.dumps(result, indent=2), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

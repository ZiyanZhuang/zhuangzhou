"""
QM Engine Pipeline — Full enzyme TS characterization workflow.

Pipeline:
  PDB (MD snapshot) → PLIP active site → QM cluster → DFT opt → TS search → Freq → SP → Report
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional

from .cluster import extract_qm_cluster
from .bridge import (
    run_dft_optimization as _run_opt,
    run_ts_search as _run_ts,
    run_frequency as _run_freq,
    run_single_point as _run_sp,
)
from .analysis import generate_report as _generate_report, plot_energy_profile

logger = logging.getLogger(__name__)

# Re-export bridge functions with same names
run_dft_optimization = _run_opt
run_ts_search = _run_ts
run_frequency = _run_freq
run_single_point = _run_sp


def run_ts_pipeline(
    pdb_path: str,
    output_dir: str,
    *,
    center_residues: List[str] = None,
    plip_result: dict = None,
    cutoff: float = 5.0,
    charge: int = None,
    spin: int = None,
    chain: str = 'A',
    functional: str = 'b3lyp',
    basis_opt: str = 'def2-svp',
    basis_sp: str = 'def2-tzvp',
    dispersion: str = 'd3bj',
    steps: List[str] = None,
    max_memory: int = 6000,
    try_gpu: bool = True,
) -> dict:
    """
    Full TS characterization pipeline for enzyme catalysis.

    Args:
        pdb_path: PDB file (MD snapshot or crystal structure)
        output_dir: Output directory for all results
        center_residues: Residue names for QM center (e.g. ['HEM', 'CYS'])
        plip_result: PLIP analysis result (auto-extract interacting residues)
        cutoff: Radius (A) for QM cluster selection
        charge: System charge (auto if None)
        spin: 2S value (auto if None)
        chain: Chain ID
        functional: DFT functional
        basis_opt: Basis set for optimization/TS/freq
        basis_sp: Basis set for single-point energy
        dispersion: Dispersion correction
        steps: Pipeline steps to run (default: all)
                Options: 'extract', 'opt', 'ts', 'freq', 'sp'
        max_memory: Max memory in MB
        try_gpu: Try GPU acceleration

    Returns:
        dict with results from each step, report_path
    """
    if steps is None:
        steps = ['extract', 'opt', 'ts', 'freq', 'sp']

    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    report = {
        'input_pdb': pdb_path,
        'functional': functional,
        'basis_opt': basis_opt,
        'basis_sp': basis_sp,
        'dispersion': dispersion,
        'steps': {},
    }

    # Step 1: Extract QM cluster
    if 'extract' in steps:
        logger.info("=== Step 1: Extract QM cluster ===")
        cluster_dir = os.path.join(output_dir, '01_cluster')
        cluster = extract_qm_cluster(
            pdb_path, cluster_dir,
            center_residues=center_residues,
            plip_result=plip_result,
            cutoff=cutoff,
            charge=charge,
            spin=spin,
            chain=chain,
        )
        report['steps']['extract'] = cluster

        if 'error' in cluster:
            report['error'] = f"Cluster extraction failed: {cluster['error']}"
            return report

        xyz_path = cluster['xyz_path']
        constraints_path = cluster['constraints_path']
        charge = cluster['charge']
        spin = cluster['spin']
        report['charge'] = charge
        report['spin'] = spin
        report['n_atoms'] = cluster['n_atoms']
    else:
        # Expect pre-existing cluster
        cluster_dir = os.path.join(output_dir, '01_cluster')
        xyz_path = os.path.join(cluster_dir, 'qm_cluster.xyz')
        constraints_path = os.path.join(cluster_dir, 'constraints.txt')
        if charge is None:
            charge = 0
        if spin is None:
            spin = 0

    # Step 2: Geometry optimization
    if 'opt' in steps:
        logger.info("=== Step 2: DFT Geometry Optimization ===")
        opt_dir = os.path.join(output_dir, '02_opt')
        opt_result = _run_opt(
            xyz_path, opt_dir, charge=charge, spin=spin,
            functional=functional, basis=basis_opt,
            constraints_path=constraints_path,
            dispersion=dispersion, max_memory=max_memory,
            try_gpu=try_gpu,
        )
        report['steps']['opt'] = opt_result

        if opt_result.get('success'):
            xyz_path = opt_result.get('optimized_xyz', xyz_path)
        elif 'error' in opt_result:
            report['error'] = f"Optimization failed: {opt_result.get('error', '')}"
            # Continue with TS if opt failed (TS search may work from initial guess)

    # Step 3: Transition state search
    if 'ts' in steps:
        logger.info("=== Step 3: Transition State Search ===")
        ts_dir = os.path.join(output_dir, '03_ts')
        ts_result = _run_ts(
            xyz_path, ts_dir, charge=charge, spin=spin,
            functional=functional, basis=basis_opt,
            constraints_path=constraints_path,
            dispersion=dispersion, max_memory=max_memory,
            try_gpu=try_gpu,
        )
        report['steps']['ts'] = ts_result

        if ts_result.get('success'):
            xyz_path = ts_result.get('optimized_xyz', xyz_path)
        elif 'error' in ts_result:
            report['error'] = f"TS search failed: {ts_result.get('error', '')}"
            return report

    # Step 4: Frequency analysis
    if 'freq' in steps:
        logger.info("=== Step 4: Frequency Analysis ===")
        freq_dir = os.path.join(output_dir, '04_freq')
        freq_result = _run_freq(
            xyz_path, freq_dir, charge=charge, spin=spin,
            functional=functional, basis=basis_opt,
            dispersion=dispersion, max_memory=max_memory,
            try_gpu=try_gpu,
        )
        report['steps']['freq'] = freq_result

        if freq_result.get('success'):
            report['is_ts'] = freq_result.get('is_ts', False)
            report['imaginary_freqs'] = freq_result.get('imaginary_freqs', [])

    # Step 5: Single-point at higher basis
    if 'sp' in steps:
        logger.info("=== Step 5: Single-point (%s) ===", basis_sp)
        sp_dir = os.path.join(output_dir, '05_sp')
        sp_result = _run_sp(
            xyz_path, sp_dir, charge=charge, spin=spin,
            functional=functional, basis=basis_sp,
            dispersion=dispersion, max_memory=max_memory,
            try_gpu=try_gpu,
        )
        report['steps']['sp'] = sp_result

        if sp_result.get('success'):
            report['sp_energy'] = sp_result['energy']
            report['sp_energy_kcal'] = sp_result['energy_kcal']

    elapsed = time.time() - t0
    report['elapsed_s'] = round(elapsed, 1)
    report['success'] = not any(
        'error' in report['steps'].get(s, {})
        for s in steps if s in report['steps']
    )

    # Generate comprehensive report + energy profile plot
    try:
        analysis = _generate_report(
            report, output_dir,
            title='QM Engine Report',
            plot_profile=True,
        )
        report['report_path'] = analysis.get('report_path')
        report['plot_path'] = analysis.get('plot_path')
        report['barriers'] = analysis.get('barriers', {})
    except Exception as e:
        logger.warning("Report generation failed: %s", e)
        # Fallback: save raw JSON
        json_path = os.path.join(output_dir, 'qm_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        report['report_path'] = json_path

    return report

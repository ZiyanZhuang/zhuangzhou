"""
Test metalloprotein MD pipeline with CYP P450 (PDB: 2CPP).

PDB 2CPP: Cytochrome P450cam (CYP101A1) from Pseudomonas putida
  - Resolution: 1.63 A (high quality)
  - Heme B (HEM) with Fe coordinated to Cys357 (axial thiolate)
  - Well-characterized benchmark system for P450 simulations

Test workflow:
  1. Download 2CPP from RCSB
  2. PDB prep: rename HEM->HEME, detect Fe-Cys357 coordination
  3. pdb2gmx with CHARMM27 (native HEME support)
  4. Solvate + ions
  5. EM -> NVT -> NPT -> 10 ns production MD

Expected: ~40K atom system, ~343 ns/day on RTX 5070
"""

import logging
import os
import sys
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_metalloprotein.log', encoding='utf-8'),
    ]
)
logger = logging.getLogger('test_metalloprotein')

# Ensure utf-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def test_pdb_prep():
    """Test PDB preparation only (fast, no GROMACS needed)."""
    from zhuangzhou.nonstandard_res.pdb_prep import prepare_metalloprotein_pdb

    work_dir = os.path.join(os.path.dirname(__file__), 'test_output', 'pdb_prep')
    os.makedirs(work_dir, exist_ok=True)

    logger.info("=== Testing PDB preparation with 2CPP ===")
    result = prepare_metalloprotein_pdb('2CPP', work_dir, ff='charmm27')

    if result.get('error'):
        logger.error("PDB prep failed: %s", result['error'])
        return False

    logger.info("Clean PDB: %s", result['clean_pdb'])
    logger.info("HEME detected: %s", result['heme_detected'])
    logger.info("HEME residues: %d", len(result['heme_residues']))
    logger.info("Coordination bonds: %d", len(result['coordination']))
    logger.info("Renamed residues: %s", result['renamed_residues'])

    for c in result['coordination']:
        logger.info("  Coordination: %s %s%d -- FE (%.2f A) [%s]",
                     c['protein_res'], c['protein_atom'], c['protein_resid'],
                     c['distance'], c['bond_type'])

    # Verify expected features
    assert result['heme_detected'], "HEME should be detected in 2CPP"
    assert len(result['heme_residues']) >= 1, "At least 1 HEME expected"

    # Check that Fe-Cys coordination was found
    cys_coord = [c for c in result['coordination'] if 'Cys' in c['bond_type']]
    assert len(cys_coord) >= 1, f"Fe-Cys coordination expected, found: {result['coordination']}"
    logger.info("Fe-Cys357 coordination: %.2f A", cys_coord[0]['distance'])

    # Verify clean PDB exists and has content
    assert os.path.exists(result['clean_pdb'])
    with open(result['clean_pdb'], encoding='utf-8', errors='replace') as f:
        lines = [l for l in f if l.startswith(('ATOM', 'HETATM'))]
    logger.info("Clean PDB: %d atom lines", len(lines))

    # Check HEME was renamed
    heme_lines = [l for l in lines if 'HEME' in l[17:21]]
    logger.info("HEME lines in clean PDB: %d", len(heme_lines))

    logger.info("=== PDB prep test PASSED ===")
    return True


def test_full_md(sim_time_ns=10.0):
    """Test full metalloprotein MD pipeline (requires GROMACS)."""
    from zhuangzhou.gromacs_md.pipeline import run_metalloprotein_md
    from zhuangzhou.gromacs_md.config import check_dependencies

    # Check dependencies
    deps = check_dependencies()
    if not deps['gmx']['ok']:
        logger.error("GROMACS not found at %s", deps['gmx']['path'])
        return False

    work_dir = os.path.join(os.path.dirname(__file__), 'test_output', 'cyp_md')
    os.makedirs(work_dir, exist_ok=True)

    logger.info("=== Testing full metalloprotein MD with 2CPP (%.1f ns) ===", sim_time_ns)
    t0 = time.time()

    result = run_metalloprotein_md(
        pdb_path='2CPP',
        work_dir=work_dir,
        sim_time_ns=sim_time_ns,
        temp=300,
        gpu=True,
    )

    elapsed = time.time() - t0

    if result.get('error'):
        logger.error("MD pipeline failed: %s", result['error'])
        if result.get('hint'):
            logger.error("Hint: %s", result['hint'])
        # Print stage details
        for stage, info in result.get('stages', {}).items():
            if info.get('error'):
                logger.error("  %s: %s", stage, info.get('error', '')[:200])
                if info.get('detail'):
                    logger.error("  Detail: %s", info['detail'][:500])
        return False

    logger.info("=== MD pipeline completed in %.1f s ===", elapsed)

    # Print stage summary
    for stage, info in result.get('stages', {}).items():
        status = 'OK' if info.get('success') else 'FAILED'
        perf = info.get('ns_per_day', '')
        elapsed_s = info.get('elapsed_s', '')
        line = f"  {stage:12s}: {status}"
        if elapsed_s:
            line += f"  ({elapsed_s:.1f}s)"
        if perf:
            line += f"  [{perf:.1f} ns/day]"
        logger.info(line)

    # Verify outputs
    outputs = result.get('outputs', {})
    for name, path in outputs.items():
        exists = os.path.exists(path)
        logger.info("  Output %-12s: %s %s", name,
                     'EXISTS' if exists else 'MISSING', path)

    logger.info("=== Full MD test PASSED ===")
    return True


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Test metalloprotein MD pipeline')
    parser.add_argument('--prep-only', action='store_true',
                        help='Test PDB prep only (no GROMACS needed)')
    parser.add_argument('--time', type=float, default=10.0,
                        help='Simulation time in ns (default: 10.0)')
    args = parser.parse_args()

    if args.prep_only:
        ok = test_pdb_prep()
    else:
        ok = test_pdb_prep() and test_full_md(args.time)

    sys.exit(0 if ok else 1)

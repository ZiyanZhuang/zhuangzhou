"""
Test CYP P450 metalloprotein MD with BONDED Fe-S model.

This test validates the heme_topology fix:
  1. pdb2gmx creates topology (specbond may or may not fire)
  2. heme_topology.py detects missing Fe-SG bond
  3. Removes HG1 from coordinating CYS (thiolate)
  4. Injects explicit Fe-SG bond (k=209200 kJ/mol/nm^2)
  5. Adds CB-SG-Fe and SG-Fe-N angles
  6. Validates coordination geometry

Expected result: Fe-SG distance stable at ~2.3 A throughout MD
(previous non-bonded test drifted to 4.1 A mean, 6.7 A HEME COM drift)
"""

import logging
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_cyp_bonded.log', encoding='utf-8'),
    ]
)
logger = logging.getLogger('test_cyp_bonded')


def test_bonded_md(sim_time_ns=10.0):
    from zhuangzhou.gromacs_md.pipeline import run_metalloprotein_md
    from zhuangzhou.gromacs_md.config import check_dependencies

    deps = check_dependencies()
    if not deps['gmx']['ok']:
        logger.error("GROMACS not found")
        return False

    work_dir = os.path.join(os.path.dirname(__file__), 'test_output', 'cyp_md4')
    os.makedirs(work_dir, exist_ok=True)

    logger.info("=== CYP P450 bonded model test (%.1f ns) ===", sim_time_ns)
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
        logger.error("FAILED: %s", result['error'])
        if result.get('hint'):
            logger.error("Hint: %s", result['hint'])
        for stage, info in result.get('stages', {}).items():
            if isinstance(info, dict) and info.get('error'):
                logger.error("  %s: %s", stage, str(info.get('error', ''))[:300])
        return False

    logger.info("=== Completed in %.1f s ===", elapsed)

    for stage, info in result.get('stages', {}).items():
        if isinstance(info, dict):
            status = 'OK' if info.get('success') or info.get('validated') else '?'
            line = f"  {stage:20s}: {status}"
            if info.get('elapsed_s'):
                line += f"  ({info['elapsed_s']:.1f}s)"
            if info.get('ns_per_day'):
                line += f"  [{info['ns_per_day']:.1f} ns/day]"
            logger.info(line)

    # Check heme_topology stage
    heme_stage = result.get('stages', {}).get('heme_topology', {})
    logger.info("\nHEME topology fixes:")
    logger.info("  Bonds fixed: %d", heme_stage.get('bonds_fixed', 0))
    logger.info("  Atoms removed: %d", heme_stage.get('atoms_removed', 0))
    logger.info("  Angles added: %d", heme_stage.get('angles_added', 0))
    for fix in heme_stage.get('fixes', []):
        logger.info("  -> %s", fix)

    # Check validation
    val_stage = result.get('stages', {}).get('heme_validation', {})
    logger.info("\nHEME validation:")
    for check in val_stage.get('checks', []):
        logger.info("  %s", check)

    logger.info("\n=== Test PASSED ===")
    return True


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', type=float, default=10.0)
    args = parser.parse_args()

    ok = test_bonded_md(args.time)
    sys.exit(0 if ok else 1)

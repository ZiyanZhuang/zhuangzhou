"""
CYP P450cam (2CPP) Metalloprotein MD Test — Bonded Model Fix Validation

Tests the corrected pipeline:
  - Bonded model (explicit Fe-S bond, k=209200 kJ/mol/nm²)
  - NO distance restraints (removed band-aid)
  - NO POSRES_COORD in production MD
  - Post-equilibration Fe coordination check

Expected: Fe-SG distance stays < 0.30 nm throughout 10ns production
Previous (cyp_md3, distance-restraints-only): mean Fe-SG = 0.435 nm (FAILED)
"""
import sys
import os
import logging

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(os.path.dirname(__file__), 'test_output', 'cyp_md5', 'run.log'),
            encoding='utf-8',
        ) if os.path.exists(os.path.join(os.path.dirname(__file__), 'test_output', 'cyp_md5'))
        else logging.StreamHandler(sys.stdout),
    ]
)

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from zhuangzhou.gromacs_md.pipeline import run_metalloprotein_md

# Test parameters
WORK_DIR = os.path.join(os.path.dirname(__file__), 'test_output', 'cyp_md5')
os.makedirs(WORK_DIR, exist_ok=True)

# Re-setup file handler now that dir exists
file_handler = logging.FileHandler(
    os.path.join(WORK_DIR, 'run.log'), encoding='utf-8', mode='w')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s'))
logging.getLogger().addHandler(file_handler)

print("=" * 70)
print("CYP P450cam (2CPP) Metalloprotein MD Test — Bonded Model Fix")
print("=" * 70)
print(f"Work dir: {WORK_DIR}")
print(f"PDB: 2CPP (auto-download)")
print(f"Force field: CHARMM27 (native HEME support)")
print(f"Simulation: 10 ns, 300 K, stepdown equilibration")
print(f"Key fix: Explicit Fe-S bonded model (no distance restraints)")
print("=" * 70)

# Run the simulation
pdb_path = os.path.join(WORK_DIR, '2CPP.pdb')
result = run_metalloprotein_md(
    pdb_path=pdb_path,
    work_dir=WORK_DIR,
    sim_time_ns=10.0,
    water='tip3p',
    temp=300,
    gpu=True,
    stepdown=True,
)

# Report results
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

if result.get('success'):
    print("STATUS: SUCCESS")
else:
    print(f"STATUS: FAILED — {result.get('error', 'unknown')}")

# Print stage results
for stage_name, stage_result in result.get('stages', {}).items():
    if isinstance(stage_result, dict):
        if 'error' in stage_result:
            print(f"  {stage_name:20s}: FAILED — {stage_result['error'][:80]}")
        elif stage_result.get('success') or stage_result.get('validated'):
            elapsed = stage_result.get('elapsed_s', '')
            perf = stage_result.get('ns_per_day', '')
            extra = f" ({elapsed}s" if elapsed else ""
            if perf:
                extra += f", {perf} ns/day"
            if extra:
                extra += ")"
            print(f"  {stage_name:20s}: OK{extra}")
        elif 'distances' in stage_result:
            # Coordination check
            for d in stage_result['distances']:
                print(f"  {stage_name:20s}: {d['bond_type']} "
                      f"{d['distance_A']:.2f} A (crystal: {d['crystal_A']:.2f} A) "
                      f"[{d['status']}]")

# Print heme fix details
heme_fix = result.get('stages', {}).get('heme_topology', {})
if heme_fix:
    print(f"\nHeme Topology Fixes:")
    print(f"  Bonds fixed: {heme_fix.get('bonds_fixed', 0)}")
    print(f"  Angles added: {heme_fix.get('angles_added', 0)}")
    print(f"  Atoms neutralized: {heme_fix.get('atoms_neutralized', 0)}")
    for fix in heme_fix.get('fixes', []):
        print(f"  — {fix}")

print("\n" + "=" * 70)

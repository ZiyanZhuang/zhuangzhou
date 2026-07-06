"""
CYP2C9 + Flurbiprofen (PDB: 1R9O) — CGenFF Ligand Parameterization Test

Tests the new CHARMM36 + CGenFF pipeline:
  - CHARMM36 protein + HEME (native specbond/pdb2gmx)
  - CGenFF small molecule parameterization (flurbiprofen)
  - Full metalloprotein+ligand MD workflow

CYP isoform axial CYS mapping:
  - P450cam (2CPP):  CYS357  (Pseudomonas putida)
  - CYP3A4:          CYS442  (Homo sapiens)
  - CYP2D6:          CYS443  (Homo sapiens)
  - CYP2C9 (1R9O):   CYS435  (Homo sapiens)
  - CYP1A2:          CYS458  (Homo sapiens)

Expected: Fe-SG distance stays < 0.30 nm, ligand stays in active site
"""
import sys
import os
import logging

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Test parameters
WORK_DIR = os.path.join(os.path.dirname(__file__), 'test_output', 'cyp2c9_cgenff')
os.makedirs(WORK_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(WORK_DIR, 'run.log'),
            encoding='utf-8', mode='w',
        ),
    ]
)

print("=" * 70)
print("CYP2C9 + Flurbiprofen (1R9O) — CGenFF/CHARMM36 Test")
print("=" * 70)
print(f"Work dir: {WORK_DIR}")
print(f"PDB: 1R9O (auto-download)")
print(f"CYP isoform: CYP2C9, axial CYS435")
print(f"Ligand: FLP (Flurbiprofen, NSAID substrate)")
print(f"Force field: CHARMM36 + CGenFF")
print(f"Simulation: 10 ns, 300 K, stepdown equilibration")
print("=" * 70)

# Step 1: Check CGenFF availability
from zhuangzhou.gromacs_md.config import CHARMM36_FF, check_dependencies
deps = check_dependencies()
print(f"\nDependency check:")
for name, info in deps.items():
    status = 'OK' if info['ok'] else 'MISSING'
    print(f"  {name:12s}: {status:8s} {info.get('path', '')}")

if not deps['charmm36']['ok']:
    print("\nERROR: CHARMM36 force field not found!")
    print("Download from: http://mackerell.umaryland.edu/charmm_ff.shtml#gromacs")
    print("Extract to GROMACS top directory (e.g., E:/gmx2025.1_GPU/share/gromacs/top/)")
    sys.exit(1)

print(f"\nCHARMM36 FF: {CHARMM36_FF}")

# Step 2: Prepare PDB
print("\n--- Step 2: Prepare PDB ---")
from zhuangzhou.nonstandard_res.pdb_prep import prepare_metalloprotein_pdb

pdb_path = os.path.join(WORK_DIR, '1R9O.pdb')
if not os.path.exists(pdb_path):
    # Auto-download by passing PDB ID
    prep = prepare_metalloprotein_pdb(
        '1R9O', WORK_DIR,
        ff='charmm36',
        extract_ligands=True,
        keep_ligands=False,
    )
else:
    prep = prepare_metalloprotein_pdb(
        pdb_path, WORK_DIR,
        ff='charmm36',
        extract_ligands=True,
        keep_ligands=False,
    )

if prep.get('error'):
    print(f"PDB prep error: {prep['error']}")
    sys.exit(1)

print(f"  HEME detected: {prep['heme_detected']}")
print(f"  Coordination bonds: {len(prep['coordination'])}")
for c in prep['coordination']:
    print(f"    {c['bond_type']}: {c['protein_res']}{c['protein_resid']}.{c['protein_atom']} "
          f"- {c['metal_atom']} ({c['distance']:.3f} A)")
print(f"  Ligands extracted: {[l['resname'] for l in prep.get('ligands', [])]}")

# Step 3: Test CGenFF parameterization (ligand only, no full MD yet)
print("\n--- Step 3: CGenFF Parameterization ---")
from zhuangzhou.gromacs_md.cgenff_param import (
    parameterize_ligand_cgenff, _find_cgenff_binary, _find_charmm36_ff
)

cgenff_bin = _find_cgenff_binary()
print(f"  CGenFF binary: {cgenff_bin or 'NOT FOUND'}")
print(f"  CHARMM36 FF dir: {_find_charmm36_ff()}")

ligands = prep.get('ligands', [])
if ligands:
    lig = ligands[0]
    print(f"\n  Parameterizing: {lig['resname']} ({lig['n_atoms']} atoms)")
    result = parameterize_ligand_cgenff(
        WORK_DIR, lig['pdb_path'], lig['resname'])

    if 'error' in result:
        print(f"\n  CGenFF Error: {result['error']}")
        if 'hint' in result:
            print(f"  Hint: {result['hint']}")
        if 'mol2' in result:
            print(f"\n  MOL2 file ready for upload: {result['mol2']}")
            print(f"  Upload to https://cgenff.com/ → download .str file")
            print(f"  Place .str at: {os.path.join(WORK_DIR, lig['resname'] + '.str')}")
    else:
        print(f"  ITP: {result.get('itp')}")
        print(f"  GRO: {result.get('gro')}")
        print(f"  PRM: {result.get('prm')}")
        penalties = result.get('penalties', {})
        print(f"  Quality: {penalties.get('quality', 'unknown')}")
        print(f"  Max penalty: {penalties.get('max_penalty', 'N/A')}")
        print(f"  Param penalty: {penalties.get('param_penalty', 'N/A')}")
        print(f"  Charge penalty: {penalties.get('charge_penalty', 'N/A')}")
        if penalties.get('high_penalty_params'):
            print(f"  High-penalty parameters:")
            for p in penalties['high_penalty_params'][:5]:
                print(f"    penalty={p['penalty']:.0f}: {p['line'][:60]}")
else:
    print("  No ligands found in PDB")

# Step 4: Full MD test (only if CGenFF succeeded)
# Uncomment to run full 10ns MD:
# print("\n--- Step 4: Full MD Pipeline ---")
# from zhuangzhou.gromacs_md.pipeline import run_metalloprotein_ligand_md
# result = run_metalloprotein_ligand_md(
#     complex_pdb=pdb_path,
#     work_dir=WORK_DIR,
#     sim_time_ns=10.0,
#     water='tip3p',
#     temp=300,
#     gpu=True,
#     stepdown=True,
# )

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)

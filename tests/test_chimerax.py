"""Quick test for chimerax_viz module."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from zhuangzhou.chimerax_viz import (export_defattr, generate_comparison_cxc,
                                     generate_porcupine_cxc, generate_trajectory_cxc,
                                     generate_morph_cxc)
import os

work = 'tests/test_output/cyp_md3'

# 1. Comparison result
from zhuangzhou.gromacs_md.analysis import compare_structures
comp = compare_structures(work, interval_ns=2)
print(f"compare_structures: {comp.get('stats', {}).get('n_total', '?')} residues")

# 2. Defattr
r = export_defattr(comp['per_residue'], attr_name='displacement',
                   output_path=os.path.join(work, 'chimerax_viz', 'displacement.defattr'))
print(f"defattr: {r}")

# 3. Comparison .cxc
r2 = generate_comparison_cxc(work, comparison_result=comp)
print(f"comparison: cxc={r2.get('cxc_path')}")

# 4. Porcupine
r3 = generate_porcupine_cxc(work)
print(f"porcupine: bild={r3.get('bild_path')}, arrows={r3.get('n_arrows')}")

# 5. Trajectory
traj_pdb = os.path.join(work, 'trajectory_nowater.pdb')
if os.path.exists(traj_pdb):
    r4 = generate_trajectory_cxc(traj_pdb, output_dir=os.path.join(work, 'chimerax_viz'))
    print(f"trajectory: cxc={r4.get('cxc_path')}")

# 6. Morph
frames = comp.get('frame_pdbs', [])
if len(frames) >= 2:
    r5 = generate_morph_cxc(frames[0], frames[-1],
                            output_dir=os.path.join(work, 'chimerax_viz'))
    print(f"morph: cxc={r5.get('cxc_path')}")

# 7. Check all generated files
cx_dir = os.path.join(work, 'chimerax_viz')
if os.path.exists(cx_dir):
    files = os.listdir(cx_dir)
    print(f"\nGenerated files in chimerax_viz/:")
    for f in sorted(files):
        sz = os.path.getsize(os.path.join(cx_dir, f))
        print(f"  {f}: {sz:,} bytes")

import os
import subprocess
import json
import logging
from ..utils.system import win_to_wsl, run_wsl, write_file

logger = logging.getLogger(__name__)

def run_pyscf_resp_wsl(ligand_path_wsl, output_dir, charge=0, spin=1):
    """Run PySCF inside WSL to generate a .molden file for RESP fitting."""
    logger.info("Running PySCF (HF/6-31G*) in WSL for RESP...")
    output_dir_wsl = win_to_wsl(os.path.abspath(output_dir))
    
    # 策略：先通过 obabel 转换为 XYZ，保证元素识别鲁棒，再跑 PySCF
    xyz_cmd = f"obabel '{ligand_path_wsl}' -oxyz -O '{output_dir_wsl}/ligand_qm.xyz' 2>/dev/null"
    r_xyz = run_wsl(xyz_cmd, cwd=output_dir_wsl, timeout=60)
    
    pyscf_script = f"""
import os, sys
from pyscf import gto, scf
from pyscf.tools import molden as molden_tool

def parse_xyz(path):
    atoms = []
    with open(path) as f: lines = f.readlines()
    n_atoms = int(lines[0].strip())
    for i in range(2, 2 + n_atoms):
        p = lines[i].split()
        atoms.append((p[0], float(p[1]), float(p[2]), float(p[3])))
    return atoms

try:
    atoms = parse_xyz("{output_dir_wsl}/ligand_qm.xyz")
    atom_str = "; ".join(f"{{e}} {{x}} {{y}} {{z}}" for e, x, y, z in atoms)
    mol = gto.Mole()
    mol.atom, mol.basis, mol.charge, mol.spin = atom_str, '6-31g*', {charge}, {spin - 1}
    mol.build()
    mf = scf.UHF(mol) if {spin - 1} > 0 else scf.RHF(mol)
    mf.kernel()
    with open('ligand_hf.molden', 'w') as f:
        molden_tool.header(mol, f)
        molden_tool.orbital_coeff(mol, f, mf.mo_coeff, ene=mf.mo_energy, occ=mf.mo_occ)
    print("MOLDEN_OK")
except Exception as e:
    print(f"PYSCF_ERROR: {{e}}")
    sys.exit(1)
"""
    write_file(os.path.join(output_dir, "run_pyscf.py"), pyscf_script)
    r = run_wsl(f"python3 run_pyscf.py", cwd=output_dir_wsl)
    if "MOLDEN_OK" not in r.stdout:
        return {"error": f"PySCF failed: {r.stdout[-2000:]}"}
    return {"status": "success", "molden": "ligand_hf.molden"}

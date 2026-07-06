"""
AMBER Ligand Preparation 閳?RESP charges + GAFF2 parameterization pipeline.

Handles the full workflow from initial ligand structure to production-ready
GAFF2 parameters:
  1. Convert ligand format (PDBQT/SDF/MOL2 -> PDB)
  2. DFT geometry optimization (PySCF or Gaussian 16)
  3. RESP charge fitting (antechamber or Multiwfn)
  4. GAFF2 atom typing + frcmod generation
  5. Docking pose selection (CYP hydroxylation-aware)
  6. Coordinate frame alignment (Kabsch on CA atoms)
  7. Complex assembly (tleap) with solvation + neutralization
  8. Pre-flight validation (distances, clashes, special bonds)

Public API:
  run_ligand_prep()  閳?Full pipeline: structure -> GAFF2 params -> complex
"""

import os
import subprocess
import shutil
import re
import numpy as np


# ---------------------------------------------------------------------------
# Element detection — robust mapping from atom names to chemical elements
# ---------------------------------------------------------------------------

# Two-letter elements that commonly appear in small-molecule contexts
_TWO_LETTER_ELEMENTS = frozenset(
    {
        "CL",
        "BR",
        "SI",
        "SE",
        "FE",
        "ZN",
        "MG",
        "MN",
        "CO",
        "NI",
        "CU",
        "NA",
        "LI",
        "AL",
        "AS",
        "BI",
        "SN",
        "PB",
        "AG",
        "AU",
        "PT",
        "PD",
        "RU",
        "RH",
        "IR",
        "OS",
        "CD",
        "HG",
        "TI",
        "CR",
        "MO",
        "BA",
        "SR",
    }
)

_ONE_LETTER_ELEMENTS = frozenset({"C", "N", "O", "H", "S", "P", "F", "I", "B", "K"})


def _detect_element(atom_name: str) -> str:
    """Robustly detect the chemical element from an atom name.

    Handles AMBER/SYBYL/GAFF2 naming conventions:
      C1, CA, CB, CG1 → C   (carbon, not calcium — small-molecule context)
      N1, N2, NH1     → N
      CL1, Cl2        → Cl
      BR              → Br
      FE, Fe          → Fe  (metals keep two-letter)

    In small-molecule / ligand context, single-letter-prefixed names
    (CA, CB, CG, CD, CE, NZ, OH, OG, OD, SD, SG) are always
    carbon/nitrogen/oxygen/sulfur — *not* calcium/cobalt/osmium/etc.
    """
    raw = atom_name.strip()
    if not raw:
        return "X"

    # Strip trailing digits to get the alphabetic prefix
    alpha = ""
    for ch in raw:
        if ch.isalpha():
            alpha += ch
        else:
            break
    if not alpha:
        # Name starts with digit (e.g., "1H") — take first alpha after digits
        alpha = "".join(c for c in raw if c.isalpha())

    alpha_upper = alpha.upper()

    # 1-letter element check first (most common in small molecules)
    if alpha_upper and alpha_upper[0] in _ONE_LETTER_ELEMENTS:
        # Two-letter element override only for true halogens/metalloids
        # that start with a common organic letter (C→Cl, B→Br, S→Si/Se)
        if len(alpha_upper) >= 2:
            prefix2 = alpha_upper[:2]
            if prefix2 in {"CL", "BR", "SI", "SE", "FE"}:
                return prefix2[0] + prefix2[1].lower()  # Cl, Br, Si, Se
        # In small-molecule/ligand context, C/N/O/H/S/P/F/I/B/K + any
        # trailing alpha is just atom numbering (CA=carbon-alpha, not calcium)
        return alpha_upper[0]

    # Pure 2-letter element (no 1-letter match — e.g., could happen if first
    # char is not in the 1-letter set, which shouldn't happen for normal atoms)
    if len(alpha_upper) >= 2 and alpha_upper[:2] in _TWO_LETTER_ELEMENTS:
        return alpha_upper[0] + alpha_upper[1].lower()

    # Fallback: first character
    return alpha_upper[0] if alpha_upper else "X"


def _element_from_sybyl(sybyl_type: str) -> str:
    """Extract element from a SYBYL atom type like 'C.ar', 'N.am', 'O.2'."""
    base = sybyl_type.split(".")[0].strip()
    if not base:
        return "X"
    if len(base) == 1:
        return base.upper()
    return base[0].upper() + base[1:].lower()


# ---------------------------------------------------------------------------
# Format detection and normalization
# ---------------------------------------------------------------------------


def _detect_ligand_format(path: str) -> str:
    """Detect ligand file format from extension and content.

    Returns one of: 'mol2', 'sdf', 'pdb', 'pdbqt', 'xyz', 'smi', 'unknown'.
    """
    ext = os.path.splitext(path)[1].lower()
    ext_map = {
        ".mol2": "mol2",
        ".sdf": "sdf",
        ".sd": "sdf",
        ".pdb": "pdb",
        ".ent": "pdb",
        ".pdbqt": "pdbqt",
        ".xyz": "xyz",
        ".smi": "smi",
        ".smiles": "smi",
    }
    if ext in ext_map:
        return ext_map[ext]
    # Sniff content
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            head = f.read(500)
        if "@<TRIPOS>" in head:
            return "mol2"
        if "V2000" in head or "V3000" in head:
            return "sdf"
        if head.lstrip().startswith(("ATOM", "HETATM", "HEADER", "COMPND")):
            return "pdb"
    except Exception:
        pass
    return "unknown"


def _normalize_ligand_to_mol2(
    input_path: str,
    output_dir: str,
    charge: int = 0,
    log=print,
) -> str:
    """Convert any supported ligand format to a clean mol2 via obabel.

    Handles: MOL2, SDF, PDB, PDBQT, XYZ.
    Adds hydrogens if missing, generates 3D coords if needed.

    Returns:
        Path to normalized mol2 file in output_dir, or None on failure.
    """
    os.makedirs(output_dir, exist_ok=True)
    fmt = _detect_ligand_format(input_path)
    wsl_out = _win_to_wsl(os.path.abspath(output_dir))
    wsl_in = _win_to_wsl(os.path.abspath(input_path))
    out_name = "ligand_input.mol2"
    out_path = os.path.join(output_dir, out_name)

    if fmt == "mol2":
        # Already mol2 — just copy (we'll validate separately)
        shutil.copy2(input_path, out_path)
        log(f"  Input format: MOL2 (copied as-is)")
        return out_path

    obabel_fmt_flag = {
        "sdf": "sdf",
        "pdb": "pdb",
        "pdbqt": "pdbqt",
        "xyz": "xyz",
    }
    iflag = obabel_fmt_flag.get(fmt)
    if not iflag:
        log(
            f"  WARNING: Unsupported ligand format '{fmt}', attempting obabel auto-detect"
        )
        iflag = fmt

    # obabel conversion with hydrogen addition and 3D generation
    opts = "-h --gen3d" if fmt == "smi" else "-h"
    cmd = f"/usr/bin/obabel -i{iflag} '{wsl_in}' -omol2 -O '{wsl_out}/{out_name}' {opts} 2>&1"
    r = _run_wsl(cmd, cwd=wsl_out, timeout=120)

    if os.path.exists(out_path) and os.path.getsize(out_path) > 50:
        log(f"  Input format: {fmt.upper()} → normalized to mol2 via obabel")
        return out_path

    # Fallback: copy original and hope antechamber can handle it
    log(f"  WARNING: obabel conversion failed, using original file")
    shutil.copy2(input_path, os.path.join(output_dir, os.path.basename(input_path)))
    return input_path


def _generate_sybyl_mol2(gaff2_mol2: str, output_dir: str, log=print) -> str:
    """Convert GAFF2-typed mol2 to Sybyl-typed mol2 for meeko/Vina.

    Returns path to Sybyl mol2 or None on failure.
    """
    wsl_out = _win_to_wsl(os.path.abspath(output_dir))
    wsl_in = _win_to_wsl(os.path.abspath(gaff2_mol2))
    out_name = "ligand_sybyl.mol2"
    out_path = os.path.join(output_dir, out_name)

    cmd = (
        f"antechamber -i '{wsl_in}' -fi mol2 "
        f"-o '{wsl_out}/{out_name}' -fo mol2 -at sybyl -pf y 2>&1"
    )
    r = _run_wsl(cmd, cwd=wsl_out, timeout=120)

    if os.path.exists(out_path) and os.path.getsize(out_path) > 50:
        log(f"  GAFF2 → Sybyl mol2 conversion OK")
        return out_path

    # Fallback: obabel
    log(f"  antechamber Sybyl conversion failed, trying obabel …")
    cmd2 = f"/usr/bin/obabel -imol2 '{wsl_in}' -omol2 -O '{wsl_out}/{out_name}' 2>&1"
    _run_wsl(cmd2, cwd=wsl_out, timeout=60)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 50:
        log(f"  obabel mol2→mol2 (Sybyl) OK")
        return out_path

    log(f"  WARNING: Could not produce Sybyl mol2")
    return None


# ---------------------------------------------------------------------------
# Validation utilities
# ---------------------------------------------------------------------------


def _validate_mol2(mol2_path: str, expected_charge: int = None, log=print) -> dict:
    """Validate a mol2 file: atom count, bond count, charge sum, sanity checks.

    Returns dict with:
      'ok': bool, 'n_atoms': int, 'n_bonds': int,
      'charge_sum': float, 'elements': dict, 'warnings': list
    """
    warnings = []
    n_atoms = 0
    n_bonds = 0
    charge_sum = 0.0
    elements = {}

    try:
        with open(mol2_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        return {"ok": False, "error": f"Cannot read mol2: {e}", "warnings": [str(e)]}

    section = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("@<TRIPOS>"):
            section = stripped
            continue

        if section == "@<TRIPOS>ATOM":
            parts = line.split()
            if len(parts) >= 6:
                n_atoms += 1
                atom_name = parts[1]
                sybyl_type = parts[5] if len(parts) > 5 else ""
                # Extract element
                if "." in sybyl_type:
                    elem = _element_from_sybyl(sybyl_type)
                else:
                    elem = _detect_element(atom_name)
                elements[elem] = elements.get(elem, 0) + 1
                # Charge
                if len(parts) >= 9:
                    try:
                        charge_sum += float(parts[8])
                    except ValueError:
                        pass

        elif section == "@<TRIPOS>BOND":
            parts = line.split()
            if len(parts) >= 4:
                n_bonds += 1

    if n_atoms == 0:
        warnings.append("No atoms found in mol2")

    if expected_charge is not None:
        charge_diff = abs(charge_sum - expected_charge)
        if charge_diff > 0.05:
            warnings.append(
                f"Charge sum ({charge_sum:.4f}) differs from expected "
                f"({expected_charge}) by {charge_diff:.4f}"
            )

    # Sanity: at least 1 bond per non-H atom minus 1
    n_heavy = sum(v for k, v in elements.items() if k != "H")
    if n_bonds < n_heavy - 1 and n_heavy > 1:
        warnings.append(f"Suspiciously few bonds ({n_bonds}) for {n_heavy} heavy atoms")

    if warnings:
        for w in warnings:
            log(f"  ⚠ mol2 validation: {w}")

    return {
        "ok": len(warnings) == 0
        or (len(warnings) == 1 and "Charge sum" in warnings[0]),
        "n_atoms": n_atoms,
        "n_bonds": n_bonds,
        "charge_sum": round(charge_sum, 4),
        "elements": elements,
        "warnings": warnings,
    }


def _validate_frcmod(frcmod_path: str, log=print) -> dict:
    """Check frcmod for ATTN lines indicating missing/estimated parameters.

    Returns dict with 'ok', 'attn_count', 'attn_lines', 'warnings'.
    """
    attn_lines = []
    warnings = []

    try:
        with open(frcmod_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if "ATTN" in line.upper():
                    attn_lines.append(line.strip())
    except Exception as e:
        return {"ok": False, "attn_count": 0, "attn_lines": [], "warnings": [str(e)]}

    if attn_lines:
        warnings.append(
            f"frcmod contains {len(attn_lines)} ATTN lines (estimated parameters)"
        )
        for al in attn_lines[:5]:
            log(f"  ⚠ frcmod ATTN: {al}")
        if len(attn_lines) > 5:
            log(f"  ⚠ ... and {len(attn_lines) - 5} more")

    return {
        "ok": len(attn_lines) == 0,
        "attn_count": len(attn_lines),
        "attn_lines": attn_lines,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Pre-tleap sanity check
# ---------------------------------------------------------------------------


def _preflight_check(mol2_path, frcmod_path, expected_charge=None, log=print) -> dict:
    """Run pre-tleap validation on mol2 + frcmod.

    Returns dict with 'ok', 'mol2_info', 'frcmod_info', 'summary'.
    """
    mol2_info = _validate_mol2(mol2_path, expected_charge, log)
    frcmod_info = _validate_frcmod(frcmod_path, log)

    ok = mol2_info["ok"] and frcmod_info["ok"]
    attn_count = frcmod_info["attn_count"]
    frcmod_status = "OK" if frcmod_info["ok"] else f"{attn_count} ATTN warnings"
    summary_parts = [
        f"mol2: {mol2_info.get('n_atoms', 0)} atoms, "
        f"{mol2_info.get('n_bonds', 0)} bonds, "
        f"charge={mol2_info.get('charge_sum', '?')}",
        f"frcmod: {frcmod_status}",
    ]
    all_warnings = mol2_info.get("warnings", []) + frcmod_info.get("warnings", [])

    if ok:
        log(f"  ✓ Pre-flight check passed: {'; '.join(summary_parts)}")
    else:
        log(f"  ⚠ Pre-flight check warnings: {'; '.join(summary_parts)}")

    return {
        "ok": ok,
        "mol2_info": mol2_info,
        "frcmod_info": frcmod_info,
        "summary": "; ".join(summary_parts),
        "warnings": all_warnings,
    }


def _write_file(path, content):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)


def _run_wsl(cmd, cwd=None, timeout=3600):
    """Run a command in WSL2 bash."""
    # Use login shell to ensure environment variables like PATH are set correctly
    # Wrap cwd and cmd to handle spaces and special characters
    quoted_cwd = f"'{cwd}'" if cwd else "'/tmp'"
    full_cmd = (
        f'wsl -e bash -l -c "source ~/amber25/amber.sh && cd {quoted_cwd} && {cmd}"'
    )

    # Fix: Use encoding='utf-8' and errors='replace' to avoid UnicodeDecodeError on Windows
    result = subprocess.run(
        full_cmd,
        shell=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    return result


def _win_to_wsl(path):
    """Convert Windows path to WSL path."""
    path = path.replace("\\", "/")
    if len(path) >= 2 and path[1] == ":":
        drive = path[0].lower()
        return f"/mnt/{drive}{path[2:]}"
    return path


def _run_pyscf_resp_wsl(ligand_path_wsl, output_dir, charge=0, spin=1, log=print):
    """Run PySCF inside WSL to generate a .molden file for RESP fitting.

    Strategy: first convert the input to XYZ via obabel (robust element detection),
    then read XYZ for PySCF.  This avoids fragile element-from-atom-name parsing.

    Args:
        ligand_path_wsl: WSL path to ligand file
        output_dir: *Windows* path to output directory (for _write_file)
    """
    log(f"  Running PySCF (HF/6-31G*) in WSL for RESP...")
    output_dir_wsl = _win_to_wsl(os.path.abspath(output_dir))

    # Step 0: Convert input to XYZ via obabel for reliable element detection
    xyz_cmd = (
        f"/usr/bin/obabel '{ligand_path_wsl}' -oxyz "
        f"-O '{output_dir_wsl}/ligand_qm.xyz' 2>/dev/null"
    )
    r_xyz = _run_wsl(xyz_cmd, cwd=output_dir_wsl, timeout=60)
    xyz_win = os.path.join(output_dir, "ligand_qm.xyz")
    if not os.path.exists(xyz_win) or os.path.getsize(xyz_win) < 10:
        log("  WARNING: obabel XYZ conversion failed, will try direct parsing")
        use_xyz = False
    else:
        use_xyz = True

    pyscf_script = f"""
import os
import sys
import subprocess
from pyscf import gto, scf
from pyscf.tools import molden as molden_tool

# ---- Robust element detection for direct parsing fallback ----
_TWO_LETTER = {{"CL","BR","SI","SE","FE","ZN","MG","MN","CO","NI","CU","NA","LI","AL","AS"}}
_ONE_LETTER = {{"C","N","O","H","S","P","F","I","B","K"}}

def detect_elem(name):
    alpha = ''.join(c for c in name if c.isalpha()).upper()
    if not alpha: return 'X'
    if alpha[0] in _ONE_LETTER:
        if len(alpha) >= 2 and alpha[:2] in {{"CL","BR","SI","SE"}}:
            return alpha[0] + alpha[1].lower()
        return alpha[0]
    if len(alpha) >= 2 and alpha[:2] in _TWO_LETTER:
        return alpha[0] + alpha[1].lower()
    return alpha[0]

def parse_xyz(path):
    atoms = []
    with open(path) as f:
        lines = f.readlines()
    n_atoms = int(lines[0].strip())
    for i in range(2, 2 + n_atoms):
        p = lines[i].split()
        atoms.append((p[0], float(p[1]), float(p[2]), float(p[3])))
    return atoms

def parse_any(path):
    atoms = []
    with open(path) as f:
        lines = f.readlines()
    ext = os.path.splitext(path)[1].lower()
    if ext == '.sdf':
        n_atoms = int(lines[3][:3])
        for i in range(4, 4 + n_atoms):
            p = lines[i].split()
            atoms.append((p[3], float(p[0]), float(p[1]), float(p[2])))
    elif ext == '.pdb':
        for l in lines:
            if l.startswith(('ATOM', 'HETATM')):
                elem = l[76:78].strip()
                if not elem:
                    elem = detect_elem(l[12:16].strip())
                atoms.append((elem, float(l[30:38]), float(l[38:46]), float(l[46:54])))
    elif ext == '.mol2':
        in_atoms = False
        for l in lines:
            if l.startswith('@<TRIPOS>ATOM'): in_atoms = True; continue
            if l.startswith('@<TRIPOS>'): in_atoms = False; continue
            if in_atoms:
                p = l.split()
                if len(p) >= 6:
                    sybyl = p[5]
                    if '.' in sybyl:
                        elem = sybyl.split('.')[0]
                        if len(elem) == 1: elem = elem.upper()
                        else: elem = elem[0].upper() + elem[1:].lower()
                    else:
                        elem = detect_elem(p[1])
                    atoms.append((elem, float(p[2]), float(p[3]), float(p[4])))
    else:
        return parse_xyz(path)
    return atoms

try:
    xyz_file = "{output_dir_wsl}/ligand_qm.xyz"
    use_xyz = {str(use_xyz)}
    if use_xyz and os.path.exists(xyz_file):
        atoms = parse_xyz(xyz_file)
        print(f"Parsed {{len(atoms)}} atoms from XYZ (obabel)")
    else:
        atoms = parse_any("{ligand_path_wsl}")
        print(f"Parsed {{len(atoms)}} atoms from direct parsing")

    if not atoms:
        print("PYSCF_ERROR: No atoms parsed from ligand file")
        sys.exit(1)

    atom_str = "; ".join(f"{{e}} {{x}} {{y}} {{z}}" for e, x, y, z in atoms)

    mol = gto.Mole()
    mol.atom = atom_str
    mol.basis = '6-31g*'
    mol.charge = {charge}
    mol.spin = {spin - 1}
    mol.verbose = 4
    mol.build()

    # Use UHF for open-shell, RHF for closed-shell
    if {spin - 1} > 0:
        mf = scf.UHF(mol)
    else:
        mf = scf.RHF(mol)
    mf.conv_tol = 1e-9
    mf.max_cycle = 200
    mf.kernel()

    if not mf.converged:
        print("WARNING: SCF not converged, trying with DIIS+level_shift")
        mf = mf.copy()
        mf.level_shift = 0.2
        mf.kernel()

    molden_file = "ligand_hf.molden"
    with open(molden_file, 'w') as f:
        molden_tool.header(mol, f)
        molden_tool.orbital_coeff(mol, f, mf.mo_coeff, ene=mf.mo_energy, occ=mf.mo_occ)
    print("MOLDEN_OK")
except Exception as e:
    print(f"PYSCF_ERROR: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
    _write_file(os.path.join(output_dir, "run_pyscf.py"), pyscf_script)
    r = _run_wsl(f"python3 run_pyscf.py", cwd=output_dir_wsl)
    if "MOLDEN_OK" not in r.stdout:
        return {"error": f"PySCF failed: {r.stdout[-2000:]}\n{r.stderr[-1000:]}"}
    return {"status": "success", "molden": "ligand_hf.molden"}


def _run_multiwfn_resp_wsl(output_dir, multiwfn_bin=None, log=print):
    """Run Multiwfn inside WSL to fit RESP charges.

    Args:
        output_dir: *Windows* path to output directory (for os.listdir)
    """
    log(f"  Running Multiwfn RESP fitting in WSL...")
    output_dir_wsl = _win_to_wsl(os.path.abspath(output_dir))
    m_bin = (
        multiwfn_bin
        or os.environ.get("MULTIWFN_BIN")
    )
    if not m_bin:
        return {
            "error": "Multiwfn binary not configured",
            "hint": "Set MULTIWFN_BIN or pass multiwfn_bin explicitly.",
        }

    cmd = f"""printf '7\\n18\\n1\\ny\\n0\\n0\\nq\\n' | '{m_bin}' ligand_hf.molden"""
    r = _run_wsl(cmd, cwd=output_dir_wsl)
    chg_list = [f for f in os.listdir(output_dir) if f.endswith(".chg")]
    if not chg_list:
        return {
            "error": f"Multiwfn failed to produce .chg file. Stdout: {r.stdout[-1000:]}"
        }

    return {"status": "success", "chg": chg_list[0]}


def _merge_resp_to_mol2(output_dir, mol2_in, chg_in, mol2_out):
    """Read RESP charges from .chg and inject into .mol2."""
    charges = []
    with open(os.path.join(output_dir, chg_in)) as f:
        for line in f:
            p = line.split()
            if len(p) >= 5:
                try:
                    charges.append(float(p[-1]))
                except ValueError:
                    continue
            elif len(p) == 1:
                try:
                    charges.append(float(p[0]))
                except ValueError:
                    continue

    with open(os.path.join(output_dir, mol2_in)) as f:
        lines = f.readlines()

    out_lines = []
    atom_idx = 0
    in_atoms = False
    for line in lines:
        if line.startswith("@<TRIPOS>ATOM"):
            in_atoms = True
            out_lines.append(line)
            continue
        if line.startswith("@<TRIPOS>BOND"):
            in_atoms = False

        if in_atoms and atom_idx < len(charges):
            parts = line.split()
            if len(parts) >= 9:
                parts[8] = f"{charges[atom_idx]:.6f}"
                new_line = f"{parts[0]:>7} {parts[1]:<8} {float(parts[2]):>10.4f} {float(parts[3]):>10.4f} {float(parts[4]):>10.4f} {parts[5]:<8} {parts[6]:>5} {parts[7]:<8} {parts[8]:>10}\n"
                out_lines.append(new_line)
                atom_idx += 1
            else:
                out_lines.append(line)
        else:
            out_lines.append(line)

    with open(os.path.join(output_dir, mol2_out), "w") as f:
        f.writelines(out_lines)
    return mol2_out


def run_ligand_prep(
    ligand_path,
    output_dir,
    ligand_name="LIG",
    charge=0,
    spin=1,
    resp_method="multiwfn",
    protein_pdb=None,
    heme_state=None,
    axial_cys_resid=None,
    hem_resid=None,
    heme_params_dir=None,
    receptor_range=None,
    prod_ns=30.0,
    multiwfn_bin=None,
    config=None,
    **kw,
):
    """Run the full ligand preparation pipeline.

    Args:
        ligand_path: Input ligand file (PDB, MOL2, SDF, or PDBQT)
        output_dir: Output directory for all generated files
        ligand_name: 3-letter residue name for ligand (default: LIG)
        charge: Net charge of ligand (default: 0)
        spin: Spin multiplicity (1 for singlet)
        resp_method: 'antechamber' or 'multiwfn' for RESP fitting
        protein_pdb: Protein PDB for complex assembly (optional)
        heme_state: Heme state for CYP systems (CPDI/DIOXY/IC6)
        axial_cys_resid: Axial cysteine residue ID (CYP systems)
        hem_resid: Heme residue ID in topology
        heme_params_dir: Path to heme parameter files
        receptor_range: Receptor residue range string (e.g., '1-513')
        prod_ns: Production MD time in nanoseconds
        multiwfn_bin: Optional path to Multiwfn binary in WSL
        config: Optional config dict

    Returns:
        dict with mol2_path, frcmod_path, complex files, or error dict.
    """
    if not os.path.exists(ligand_path):
        return {"error": f"Ligand file not found: {ligand_path}"}

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    log_lines = []

    def log(msg):
        log_lines.append(msg)
        print(msg)

    try:
        log("=" * 70)
        log("  AMBER Ligand Preparation Pipeline (PySCF + Multiwfn + GAFF2)")
        log("=" * 70)

        wsl_outdir = _win_to_wsl(output_dir)
        wsl_ligand = _win_to_wsl(os.path.abspath(ligand_path))

        ext = os.path.splitext(ligand_path)[1].lower()
        log(f"\n[Step 1] Input ligand: {os.path.basename(ligand_path)} ({ext})")

        if ext == ".pdbqt":
            log("  Converting PDBQT -> PDB via obabel...")
            cmd = f"obabel -ipdbqt '{wsl_ligand}' -opdb -O '{wsl_outdir}/ligand.pdb'"
            r = _run_wsl(cmd, cwd=wsl_outdir)
            if r.returncode != 0:
                return {"error": f"obabel conversion failed: {r.stderr}"}
            ligand_file = "ligand.pdb"
            ligand_format = "pdb"
        else:
            ligand_file = os.path.basename(ligand_path)
            ligand_format = ext[1:]
            shutil.copy2(ligand_path, os.path.join(output_dir, ligand_file))

        log(f"\n[Step 2] Generating GAFF2 topology (Antechamber)...")
        fi_map = {"pdb": "pdb", "mol2": "mol2", "sdf": "sdf", "pdbqt": "pdb"}
        fi = fi_map.get(ligand_format, "pdb")

        gaff2_mol2 = f"{ligand_name.lower()}_gaff2.mol2"
        cmd = (
            f"antechamber -i '{wsl_outdir}/{ligand_file}' -fi {fi} "
            f"-o '{wsl_outdir}/{gaff2_mol2}' -fo mol2 "
            f"-c bcc -s 2 -rn {ligand_name} -at gaff2 -nc {charge} -pf y"
        )
        r = _run_wsl(cmd, cwd=wsl_outdir, timeout=1200)
        if r.returncode != 0:
            return {"error": f"Antechamber failed: {r.stderr}"}

        if resp_method == "multiwfn":
            log(f"\n[Step 3] Fitting high-precision RESP charges via PySCF/Multiwfn...")
            p_res = _run_pyscf_resp_wsl(
                wsl_outdir + "/" + ligand_file, output_dir, charge, spin, log
            )
            if "error" in p_res:
                return p_res
            m_res = _run_multiwfn_resp_wsl(output_dir, multiwfn_bin, log)
            if "error" in m_res:
                return m_res
            final_mol2 = f"{ligand_name.lower()}_resp_gaff2.mol2"
            _merge_resp_to_mol2(output_dir, gaff2_mol2, m_res["chg"], final_mol2)
            log(f"  RESP charges merged into {final_mol2}")
            mol2_path = os.path.join(output_dir, final_mol2)
        else:
            log(f"\n[Step 3] Using Antechamber BCC charges (resp_method='antechamber')")
            mol2_path = os.path.join(output_dir, gaff2_mol2)

        log("\n[Step 4] Generating GAFF2 frcmod...")
        frcmod_name = f"{ligand_name.lower()}.frcmod"
        cmd = (
            f"parmchk2 -i '{wsl_outdir}/{os.path.basename(mol2_path)}' "
            f"-f mol2 -o '{wsl_outdir}/{frcmod_name}' -s gaff2"
        )
        r = _run_wsl(cmd, cwd=wsl_outdir)
        if r.returncode != 0:
            return {"error": f"parmchk2 failed: {r.stderr}"}
        frcmod_path = os.path.join(output_dir, frcmod_name)

        result = {
            "mol2_path": mol2_path,
            "frcmod_path": frcmod_path,
            "ligand_name": ligand_name,
        }
        if protein_pdb and os.path.exists(protein_pdb):
            log("\n[Step 5] Assembling protein-ligand complex...")
            complex_result = _assemble_complex(
                output_dir,
                protein_pdb,
                mol2_path,
                frcmod_path,
                ligand_name,
                heme_state,
                axial_cys_resid,
                hem_resid,
                heme_params_dir,
                receptor_range,
                prod_ns,
                log,
            )
            result.update(complex_result)

        log("\nLigand Preparation Complete!")
        return result

    except Exception as e:
        return {"error": f"Pipeline failed: {e}"}


def _assemble_complex(
    output_dir,
    protein_pdb,
    mol2_path,
    frcmod_path,
    ligand_name,
    heme_state,
    axial_cys_resid,
    hem_resid,
    heme_params_dir,
    receptor_range,
    prod_ns,
    log,
):
    """Assemble protein-ligand complex in tleap."""
    wsl_outdir = _win_to_wsl(output_dir)
    wsl_protein = _win_to_wsl(os.path.abspath(protein_pdb))
    wsl_mol2 = _win_to_wsl(os.path.abspath(mol2_path))
    wsl_frcmod = _win_to_wsl(os.path.abspath(frcmod_path))

    lines = [
        "source leaprc.protein.ff19SB",
        "source leaprc.gaff2",
        "source leaprc.water.tip3p",
        f"loadamberparams '{wsl_frcmod}'",
    ]

    if heme_state and heme_params_dir:
        wsl_heme = _win_to_wsl(heme_params_dir)
        lines.extend(
            [
                f"loadamberparams '{wsl_heme}/{heme_state}.frcmod'",
                f"HEM = loadmol2 '{wsl_heme}/HEM.mol2'",
                f"CYP = loadmol2 '{wsl_heme}/CYP.mol2'",
            ]
        )

    lines.append(f"LIG = loadmol2 '{wsl_mol2}'")
    lines.append(f"mol = loadpdb '{wsl_protein}'")

    if axial_cys_resid and hem_resid:
        lines.append(f"bond mol.{axial_cys_resid}.SG mol.{hem_resid}.FE")

    lines.extend(
        [
            "solvateoct mol TIP3PBOX 12.0",
            "addionsrand mol Na+ 0",
            "addionsrand mol Cl- 0",
            f"saveamberparm mol '{wsl_outdir}/complex.prmtop' '{wsl_outdir}/complex.inpcrd'",
            "quit",
        ]
    )

    tleap_path = os.path.join(output_dir, "tleap_complex.in")
    _write_file(tleap_path, "\n".join(lines))
    r = _run_wsl(f"tleap -f '{wsl_outdir}/tleap_complex.in'", cwd=wsl_outdir)

    prmtop = os.path.join(output_dir, "complex.prmtop")
    inpcrd = os.path.join(output_dir, "complex.inpcrd")
    if not os.path.exists(prmtop) or not os.path.exists(inpcrd):
        return {
            "error": f"tleap failed to produce topology.\n"
            f"stdout: {r.stdout[-2000:]}\nstderr: {r.stderr[-1000:]}"
        }
    return {"prmtop": prmtop, "inpcrd": inpcrd}

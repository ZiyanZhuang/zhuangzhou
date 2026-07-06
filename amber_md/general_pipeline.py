"""
General AMBER Protein-Ligand MD Pipeline
==========================================
Automated pipeline for *ordinary* protein + small-molecule systems
(no heme / metalloprotein special handling).

Workflow:
  1. Protein preparation   — pdb4amber / reduce (add H, fix naming)
  2. Ligand parameterization — PySCF+Multiwfn RESP + GAFF2  *or*  AM1-BCC + GAFF2
  3. (Optional) Vina docking — determine binding pose
  4. tleap complex assembly — ff14SB/ff19SB + GAFF2 + TIP3P, solvate, neutralize
  5. MD input generation    — continuous step-down protocol (via protocol.py)
  6. Run / analysis scripts

Public API:
  run_protein_ligand_md()      — Full protein+ligand pipeline
  run_apo_protein_md()         — Apo protein (no ligand) pipeline
  generate_general_md_inputs() — MD input files only (reusable, delegates to protocol)
  generate_general_run_script() — pmemd.cuda bash script (delegates to protocol)
  generate_general_tleap()     — tleap.in only
"""

import os
import shutil
import subprocess
import numpy as np

from .ligand_prep import (
    _write_file,
    _run_wsl,
    _win_to_wsl,
    _normalize_ligand_to_mol2,
    _generate_sybyl_mol2,
    _validate_mol2,
    _validate_frcmod,
    _preflight_check,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_file(path, label="File"):
    """Return error dict if file doesn't exist, else None."""
    if not os.path.exists(path):
        return {"error": f"{label} not found: {path}"}
    return None


# ---------------------------------------------------------------------------
# Step 1 – Protein preparation
# ---------------------------------------------------------------------------


def _clean_pdb_for_amber(pdb_path, output_path, log=print):
    """Fix common PDB atom/residue naming issues for AMBER ff14SB.

    Fixes applied:
      - ILE: CD → CD1  (AMBER ff14SB convention)
      - C-terminal carboxylate: OC1 → O, OC2 → OXT
      - CIS/CSS → CYS with proper disulfide naming
      - Remove duplicate H if reduce already added them
    """
    log("  Cleaning PDB atom names for AMBER …")

    # First pass: find last residue number per chain (C-terminal detection)
    last_resid_per_chain = {}
    with open(pdb_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("ATOM"):
                chain = line[21]
                try:
                    resid = int(line[22:26])
                except ValueError:
                    continue
                if (
                    chain not in last_resid_per_chain
                    or resid > last_resid_per_chain[chain]
                ):
                    last_resid_per_chain[chain] = resid

    fixes = {"ILE_CD": 0, "CTERM_OC": 0}

    with (
        open(pdb_path, encoding="utf-8", errors="replace") as fin,
        open(output_path, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            if not line.startswith(("ATOM", "HETATM")):
                fout.write(line)
                continue

            resname = line[17:20].strip()
            atom_name = line[12:16]
            chain = line[21]
            try:
                resid = int(line[22:26])
            except ValueError:
                fout.write(line)
                continue

            # Fix 1: ILE CD → CD1
            if resname == "ILE" and atom_name.strip() == "CD":
                line = line[:12] + " CD1" + line[16:]
                fixes["ILE_CD"] += 1

            # Fix 2: C-terminal OC1 → O, OC2 → OXT
            is_cterm = resid == last_resid_per_chain.get(chain, -1)
            if is_cterm:
                aname = atom_name.strip()
                if aname == "OC1":
                    line = line[:12] + " O  " + line[16:]
                    fixes["CTERM_OC"] += 1
                elif aname == "OC2":
                    line = line[:12] + " OXT" + line[16:]
                    fixes["CTERM_OC"] += 1

            fout.write(line)

    for k, v in fixes.items():
        if v > 0:
            log(f"    Fixed {v} atoms: {k}")

    return output_path


def prepare_protein_wsl(pdb_path, output_dir, force_field="ff14SB", log=print):
    """Run pdb4amber + reduce in WSL to clean / protonate the protein.

    Returns dict with 'prepared_pdb' (Windows path) or 'error'.
    """
    os.makedirs(output_dir, exist_ok=True)
    wsl_pdb = _win_to_wsl(os.path.abspath(pdb_path))
    wsl_out = _win_to_wsl(os.path.abspath(output_dir))
    out_name = "protein_prep.pdb"

    log("  Running pdb4amber → reduce in WSL …")
    cmd = (
        f"pdb4amber -i '{wsl_pdb}' -o '{wsl_out}/{out_name}' "
        f"--dry --reduce --no-conect 2>&1"
    )
    r = _run_wsl(cmd, cwd=wsl_out, timeout=120)

    win_out = os.path.join(output_dir, out_name)
    if not os.path.exists(win_out):
        # Fallback: just use reduce alone
        log("  pdb4amber failed or missing, trying reduce directly …")
        cmd2 = f"reduce '{wsl_pdb}' > '{wsl_out}/{out_name}' 2>/dev/null"
        r2 = _run_wsl(cmd2, cwd=wsl_out, timeout=120)

    if not os.path.exists(win_out):
        # Last resort: copy original — tleap will add H
        log("  WARNING: reduce unavailable, using raw PDB (tleap will add H)")
        shutil.copy2(pdb_path, win_out)

    # Always clean atom names for AMBER compatibility
    cleaned = os.path.join(output_dir, "protein_cleaned.pdb")
    _clean_pdb_for_amber(win_out, cleaned, log)

    return {"prepared_pdb": cleaned, "pdb4amber_log": r.stdout[-2000:] if r else ""}


# ---------------------------------------------------------------------------
# Step 2 – Ligand RESP + GAFF2  (delegates to ligand_prep module)
# ---------------------------------------------------------------------------


def prepare_ligand_gaff2(
    ligand_path,
    output_dir,
    ligand_name="LIG",
    charge=0,
    spin=1,
    resp_method="multiwfn",
    multiwfn_bin=None,
    log=print,
):
    """Parameterize ligand: GAFF2 typing + RESP (or BCC) charges.

    Workflow:
      1. Normalize input format → mol2 (via obabel if needed)
      2. Antechamber: GAFF2 atom types + BCC charges
      3. Validate GAFF2 mol2 (atoms, bonds, charge sum)
      4. (Optional) PySCF+Multiwfn high-precision RESP → merge charges
      5. parmchk2: generate frcmod
      6. Validate frcmod (check ATTN warnings)
      7. Generate Sybyl mol2 (for docking compatibility)
      8. Pre-flight check (combined mol2+frcmod validation)

    Returns dict with 'mol2_path', 'frcmod_path', 'sybyl_mol2', 'preflight' or 'error'.
    """
    from .ligand_prep import (
        _run_pyscf_resp_wsl,
        _run_multiwfn_resp_wsl,
        _merge_resp_to_mol2,
    )

    os.makedirs(output_dir, exist_ok=True)
    wsl_outdir = _win_to_wsl(os.path.abspath(output_dir))

    # ── Step 1: Normalize input to mol2 ──
    log("  Normalizing ligand input format …")
    normalized = _normalize_ligand_to_mol2(ligand_path, output_dir, charge, log)
    if normalized is None:
        return {"error": f"Cannot normalize ligand input: {ligand_path}"}

    # Copy normalized file into output_dir if it isn't already there
    ligand_file = os.path.basename(normalized)
    dst = os.path.join(output_dir, ligand_file)
    if os.path.abspath(normalized) != os.path.abspath(dst):
        shutil.copy2(normalized, dst)
        ligand_file = os.path.basename(dst)

    ext = os.path.splitext(ligand_file)[1].lower()
    fi_map = {"pdb": "pdb", "mol2": "mol2", "sdf": "sdf", "pdbqt": "pdb"}
    fi = fi_map.get(ext.lstrip("."), "mol2")

    # ── Step 2: Antechamber — GAFF2 typing + BCC charges ──
    log("  Antechamber: GAFF2 atom typing + BCC charges …")
    gaff2_mol2 = f"{ligand_name.lower()}_gaff2.mol2"
    cmd = (
        f"antechamber -i '{wsl_outdir}/{ligand_file}' -fi {fi} "
        f"-o '{wsl_outdir}/{gaff2_mol2}' -fo mol2 "
        f"-c bcc -s 2 -rn {ligand_name} -at gaff2 -nc {charge} -pf y"
    )
    r = _run_wsl(cmd, cwd=wsl_outdir, timeout=1200)
    if r.returncode != 0:
        return {"error": f"Antechamber failed: {r.stderr[-2000:]}"}

    mol2_path = os.path.join(output_dir, gaff2_mol2)
    if not os.path.exists(mol2_path) or os.path.getsize(mol2_path) < 50:
        return {"error": f"Antechamber produced empty mol2: {gaff2_mol2}"}

    # ── Step 3: Validate GAFF2 mol2 ──
    log("  Validating GAFF2 mol2 …")
    mol2_info = _validate_mol2(mol2_path, expected_charge=charge, log=log)
    if not mol2_info.get("ok") and mol2_info.get("n_atoms", 0) == 0:
        return {
            "error": f"GAFF2 mol2 validation failed: {mol2_info.get('warnings', [])}"
        }

    # ── Step 4: Optional RESP via PySCF + Multiwfn ──
    if resp_method == "multiwfn":
        log("  PySCF HF/6-31G* → Multiwfn RESP …")
        p_res = _run_pyscf_resp_wsl(
            wsl_outdir + "/" + ligand_file,
            output_dir,
            charge,
            spin,
            log,
        )
        if "error" in p_res:
            log(f"  RESP failed, falling back to BCC: {p_res['error'][:500]}")
        else:
            m_res = _run_multiwfn_resp_wsl(output_dir, multiwfn_bin, log)
            if "error" in m_res:
                log(f"  Multiwfn failed, falling back to BCC: {m_res['error'][:500]}")
            else:
                final_mol2 = f"{ligand_name.lower()}_resp_gaff2.mol2"
                _merge_resp_to_mol2(output_dir, gaff2_mol2, m_res["chg"], final_mol2)
                mol2_path = os.path.join(output_dir, final_mol2)
                log(f"  RESP charges merged → {final_mol2}")

                # Re-validate after RESP merge
                resp_info = _validate_mol2(mol2_path, expected_charge=charge, log=log)
                if resp_info.get("warnings"):
                    log(f"  Post-RESP charge sum: {resp_info.get('charge_sum', '?')}")

    # ── Step 5: parmchk2 — generate frcmod ──
    log("  parmchk2: generating frcmod …")
    frcmod_name = f"{ligand_name.lower()}.frcmod"
    cmd = (
        f"parmchk2 -i '{wsl_outdir}/{os.path.basename(mol2_path)}' "
        f"-f mol2 -o '{wsl_outdir}/{frcmod_name}' -s gaff2"
    )
    r = _run_wsl(cmd, cwd=wsl_outdir, timeout=120)
    frcmod_path = os.path.join(output_dir, frcmod_name)
    if r.returncode != 0 or not os.path.exists(frcmod_path):
        return {"error": f"parmchk2 failed: {r.stderr[-1000:]}"}

    # ── Step 6: Validate frcmod ──
    frcmod_info = _validate_frcmod(frcmod_path, log)

    # ── Step 7: Generate Sybyl mol2 for docking ──
    log("  Generating Sybyl mol2 for docking compatibility …")
    sybyl_mol2 = _generate_sybyl_mol2(mol2_path, output_dir, log)

    # ── Step 8: Pre-flight check ──
    preflight = _preflight_check(
        mol2_path, frcmod_path, expected_charge=charge, log=log
    )

    return {
        "mol2_path": mol2_path,
        "frcmod_path": frcmod_path,
        "sybyl_mol2": sybyl_mol2,
        "ligand_name": ligand_name,
        "mol2_info": mol2_info,
        "frcmod_info": frcmod_info,
        "preflight": preflight,
    }


# ---------------------------------------------------------------------------
# Step 3 – Vina docking (optional, for pose placement)
# ---------------------------------------------------------------------------


def dock_ligand_vina(
    protein_pdb,
    ligand_mol2,
    output_dir,
    exhaustiveness=32,
    n_poses=5,
    sybyl_mol2=None,
    log=print,
):
    """Run AutoDock Vina (Python API) to find binding pose.

    Uses the `vina` Python package (1.2.x) + `meeko` for PDBQT preparation.
    Falls back to obabel for receptor if meeko's prepare_receptor is unavailable.

    Args:
        protein_pdb: Prepared protein PDB
        ligand_mol2: Ligand mol2 (GAFF2 or Sybyl typed)
        output_dir: Output directory
        exhaustiveness: Vina exhaustiveness
        n_poses: Number of poses to generate
        sybyl_mol2: Pre-prepared Sybyl mol2 (skips inline conversion if provided)
        log: Logging function

    Returns dict with 'docked_pdb', 'best_score', 'center', 'size' or 'error'.
    """
    os.makedirs(output_dir, exist_ok=True)
    wsl_out = _win_to_wsl(os.path.abspath(output_dir))
    wsl_prot = _win_to_wsl(os.path.abspath(protein_pdb))

    # Step A: Prepare receptor PDBQT (obabel, works reliably for proteins)
    log("  Preparing Vina receptor (obabel → PDBQT) …")
    cmd_rec = (
        f"/usr/bin/obabel -ipdb '{wsl_prot}' -opdbqt -O '{wsl_out}/receptor.pdbqt' "
        f"--partialcharge gasteiger -xr 2>/dev/null"
    )
    _run_wsl(cmd_rec, cwd=wsl_out, timeout=120)

    rec_pdbqt = os.path.join(output_dir, "receptor.pdbqt")
    if not os.path.exists(rec_pdbqt) or os.path.getsize(rec_pdbqt) < 100:
        return {"error": "Failed to prepare receptor PDBQT (obabel)"}

    # Step B: Determine the best mol2 for ligand PDBQT preparation
    # Priority: pre-prepared Sybyl mol2 > inline detection > original
    lig_for_pdbqt = None

    if sybyl_mol2 and os.path.exists(sybyl_mol2):
        lig_for_pdbqt = sybyl_mol2
        log("  Using pre-prepared Sybyl mol2 for ligand PDBQT")
    else:
        # Check if the provided mol2 has GAFF2 types and needs conversion
        lig_for_pdbqt = ligand_mol2
        if ligand_mol2.lower().endswith(".mol2"):
            has_gaff2 = False
            try:
                with open(ligand_mol2, encoding="utf-8", errors="replace") as f:
                    in_atom = False
                    for line in f:
                        if "@<TRIPOS>ATOM" in line:
                            in_atom = True
                            continue
                        if in_atom and line.startswith("@"):
                            break
                        if in_atom:
                            parts = line.split()
                            if len(parts) >= 6:
                                atype = parts[5]
                                if atype.islower() and "." not in atype:
                                    has_gaff2 = True
                                    break
            except Exception:
                pass
            if has_gaff2:
                log("  Detected GAFF2 atom types → converting to Sybyl via utility …")
                converted = _generate_sybyl_mol2(ligand_mol2, output_dir, log)
                if converted:
                    lig_for_pdbqt = converted

    wsl_lig = _win_to_wsl(os.path.abspath(lig_for_pdbqt))

    # Step C: Prepare ligand PDBQT via meeko → obabel fallback
    log("  Preparing Vina ligand (meeko → PDBQT) …")
    cmd_lig = f"mk_prepare_ligand.py -i '{wsl_lig}' -o '{wsl_out}/ligand.pdbqt' 2>&1"
    r_lig = _run_wsl(cmd_lig, cwd=wsl_out, timeout=120)

    lig_pdbqt = os.path.join(output_dir, "ligand.pdbqt")
    if not os.path.exists(lig_pdbqt) or os.path.getsize(lig_pdbqt) < 50:
        # Fallback: obabel
        ext = os.path.splitext(lig_for_pdbqt)[1].lower()
        fi = "mol2" if ext == ".mol2" else "pdb"
        log("  meeko failed, falling back to obabel for ligand …")
        cmd_lig2 = (
            f"/usr/bin/obabel -i{fi} '{wsl_lig}' -opdbqt "
            f"-O '{wsl_out}/ligand.pdbqt' --partialcharge gasteiger 2>/dev/null"
        )
        _run_wsl(cmd_lig2, cwd=wsl_out, timeout=120)
        if not os.path.exists(lig_pdbqt) or os.path.getsize(lig_pdbqt) < 50:
            return {"error": "Failed to prepare ligand PDBQT"}

    # Step C: Compute box center from protein COM
    log("  Computing docking box (whole-protein center, 30 Å box) …")
    cx, cy, cz, n = 0, 0, 0, 0
    with open(protein_pdb, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("ATOM"):
                try:
                    cx += float(line[30:38])
                    cy += float(line[38:46])
                    cz += float(line[46:54])
                    n += 1
                except ValueError:
                    continue
    if n == 0:
        return {"error": "No ATOM records in protein PDB"}
    cx, cy, cz = cx / n, cy / n, cz / n
    box_size = 30.0  # Å — captures most active sites

    # Step D: Run Vina via Python API (in WSL)
    log(
        f"  Running Vina Python API (center=[{cx:.1f},{cy:.1f},{cz:.1f}], "
        f"box={box_size}Å, exhaust={exhaustiveness}) …"
    )
    wsl_rec_pdbqt = f"{wsl_out}/receptor.pdbqt"
    wsl_lig_pdbqt = f"{wsl_out}/ligand.pdbqt"
    wsl_docked = f"{wsl_out}/docked.pdbqt"

    vina_script = f"""
import sys
from vina import Vina

v = Vina(sf_name='vina')
v.set_receptor('{wsl_rec_pdbqt}')
v.set_ligand_from_file('{wsl_lig_pdbqt}')
v.compute_vina_maps(
    center=[{cx:.3f}, {cy:.3f}, {cz:.3f}],
    box_size=[{box_size}, {box_size}, {box_size}]
)
v.dock(exhaustiveness={exhaustiveness}, n_poses={n_poses})

# Write docked poses
v.write_poses('{wsl_docked}', n_poses={n_poses}, overwrite=True)

# Print best score
energies = v.energies()
if energies is not None and len(energies) > 0:
    print(f'BEST_SCORE={{energies[0][0]:.3f}}')
else:
    print('BEST_SCORE=NA')
"""
    # Write the script to file and run in WSL
    vina_py_path = os.path.join(output_dir, "run_vina.py")
    _write_file(vina_py_path, vina_script)

    vina_cmd = f"python3 '{wsl_out}/run_vina.py'"
    r = _run_wsl(vina_cmd, cwd=wsl_out, timeout=3600)

    docked_pdbqt = os.path.join(output_dir, "docked.pdbqt")
    if not os.path.exists(docked_pdbqt):
        return {
            "error": f"Vina docking failed:\nstdout: {r.stdout[-1500:]}\nstderr: {r.stderr[-1000:]}"
        }

    # Parse best score from stdout
    best_score = None
    if r.stdout:
        for line in r.stdout.splitlines():
            if line.startswith("BEST_SCORE="):
                val = line.split("=", 1)[1].strip()
                if val != "NA":
                    try:
                        best_score = float(val)
                    except ValueError:
                        pass

    # Step E: Extract best pose → PDB
    log("  Extracting best Vina pose → PDB …")
    cmd_conv = (
        f"/usr/bin/obabel -ipdbqt '{wsl_out}/docked.pdbqt' -opdb "
        f"-O '{wsl_out}/docked_best.pdb' -f 1 -l 1 2>/dev/null"
    )
    _run_wsl(cmd_conv, cwd=wsl_out, timeout=60)

    return {
        "docked_pdbqt": docked_pdbqt,
        "docked_pdb": os.path.join(output_dir, "docked_best.pdb"),
        "best_score": best_score,
        "center": [cx, cy, cz],
        "box_size": box_size,
    }


# ---------------------------------------------------------------------------
# Step 4 – tleap complex assembly
# ---------------------------------------------------------------------------


def generate_general_tleap(
    output_dir,
    protein_pdb,
    mol2_path,
    frcmod_path,
    ligand_name="LIG",
    force_field="ff14SB",
    water_model="TIP3P",
    box_buffer=12.0,
    prmtop_name="complex.prmtop",
    inpcrd_name="complex.inpcrd",
    log=print,
):
    """Generate tleap.in for a general protein-ligand complex.

    Returns dict with 'tleap_path', 'prmtop', 'inpcrd' or 'error'.
    """
    wsl_out = _win_to_wsl(os.path.abspath(output_dir))
    wsl_prot = _win_to_wsl(os.path.abspath(protein_pdb))
    wsl_mol2 = _win_to_wsl(os.path.abspath(mol2_path))
    wsl_frcmod = _win_to_wsl(os.path.abspath(frcmod_path))

    # Map force field names
    ff_map = {
        "ff14SB": "leaprc.protein.ff14SB",
        "ff19SB": "leaprc.protein.ff19SB",
    }
    ff_source = ff_map.get(force_field, f"leaprc.protein.{force_field}")

    water_map = {
        "TIP3P": ("leaprc.water.tip3p", "TIP3PBOX"),
        "OPC": ("leaprc.water.opc", "OPCBOX"),
        "SPC/E": ("leaprc.water.spce", "SPCBOX"),
    }
    water_source, water_box = water_map.get(
        water_model, ("leaprc.water.tip3p", "TIP3PBOX")
    )

    # Detect whether protein PDB already contains the ligand residue
    has_ligand_in_pdb = False
    with open(protein_pdb, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                rn = line[17:20].strip()
                if rn == ligand_name:
                    has_ligand_in_pdb = True
                    break

    if has_ligand_in_pdb:
        load_section = f"""# Load protein-ligand complex PDB (ligand {ligand_name} inside)
mol = loadpdb {wsl_prot}"""
    else:
        load_section = f"""# Load protein and ligand separately, then combine
prot = loadpdb {wsl_prot}
mol = combine {{prot {ligand_name}}}"""

    content = f"""# tleap input for Protein-Ligand Complex
# Force field: {force_field} + GAFF2 + {water_model}
# Auto-generated by zhuangzhou.amber_md.general_pipeline

# Load force fields
source {ff_source}
source leaprc.gaff2
source {water_source}

# Load ligand parameters
loadamberparams {wsl_frcmod}
{ligand_name} = loadmol2 {wsl_mol2}

{load_section}

# Check topology
check mol

# Solvate in truncated octahedron (buffer = {box_buffer} A)
solvateoct mol {water_box} {box_buffer}

# Neutralize with counterions
addionsrand mol Na+ 0
addionsrand mol Cl- 0

# Final verification
check mol
charge mol

# Save topology and coordinates
saveamberparm mol {wsl_out}/{prmtop_name} {wsl_out}/{inpcrd_name}

quit
"""
    tleap_path = os.path.join(output_dir, "tleap.in")
    _write_file(tleap_path, content)

    # Run tleap
    log("  Running tleap in WSL …")
    r = _run_wsl(f"tleap -f '{wsl_out}/tleap.in'", cwd=wsl_out, timeout=300)

    # Verify outputs
    prmtop_win = os.path.join(output_dir, prmtop_name)
    inpcrd_win = os.path.join(output_dir, inpcrd_name)

    if not os.path.exists(prmtop_win) or not os.path.exists(inpcrd_win):
        return {
            "error": f"tleap failed to produce topology files.\n"
            f"stdout: {r.stdout[-2000:]}\nstderr: {r.stderr[-1000:]}",
            "tleap_path": tleap_path,
        }

    log(f"  tleap OK → {prmtop_name} + {inpcrd_name}")
    return {
        "tleap_path": tleap_path,
        "prmtop": prmtop_win,
        "inpcrd": inpcrd_win,
        "tleap_log": r.stdout[-3000:],
    }


# ---------------------------------------------------------------------------
# Step 5 – MD input files (general, no CYP-specific masks)
# ---------------------------------------------------------------------------


def generate_general_md_inputs(
    output_dir,
    ligand_name="LIG",
    prod_ns=50.0,
    log=print,
    system_type="protein_ligand",
    protocol_overrides=None,
):
    """Generate AMBER MD input files — Continuous Step-down protocol.

    Delegates to ``amber_md.protocol.generate_inputs()`` for a clean,
    protocol-driven generation.  All hyperparameters can be overridden via
    ``protocol_overrides`` dict (any ``AmberMDProtocol`` field).

    Returns path to md_input/ directory.
    """
    from .protocol import default_protocol, generate_inputs

    proto = default_protocol(
        system_type=system_type,
        ligand_name=ligand_name,
        prod_ns=prod_ns,
        **(protocol_overrides or {}),
    )
    return generate_inputs(proto, output_dir, log)


# ---------------------------------------------------------------------------
# Step 6 – Run script
# ---------------------------------------------------------------------------


def generate_general_run_script(
    output_dir,
    prmtop="complex.prmtop",
    inpcrd="complex.inpcrd",
    amber_source="source ~/amber25/amber.sh",
    engine="pmemd.cuda",
    log=print,
    ligand_name="LIG",
    prod_ns=50.0,
    system_type="protein_ligand",
    protocol_overrides=None,
):
    """Generate bash script for continuous step-down MD protocol.

    Delegates to ``amber_md.protocol.generate_script()``.
    Returns path to run_md.sh.
    """
    from .protocol import default_protocol, generate_script

    proto = default_protocol(
        system_type=system_type,
        ligand_name=ligand_name,
        prod_ns=prod_ns,
        engine=engine,
        amber_source=amber_source,
        **(protocol_overrides or {}),
    )
    return generate_script(proto, output_dir, prmtop, inpcrd, log)


# ---------------------------------------------------------------------------
# Master orchestrator
# ---------------------------------------------------------------------------


def run_protein_ligand_md(
    protein_pdb,
    ligand_path,
    output_dir=None,
    ligand_name="LIG",
    charge=0,
    spin=1,
    resp_method="multiwfn",
    force_field="ff14SB",
    water_model="TIP3P",
    box_buffer=12.0,
    prod_ns=50.0,
    do_docking=True,
    docking_exhaustiveness=32,
    multiwfn_bin=None,
    engine="pmemd.cuda",
    protocol_overrides=None,
    config=None,
    **kw,
):
    """Full pipeline: protein + ligand → ready-to-run AMBER MD.

    Args:
        protein_pdb: Input protein PDB (can be un-protonated)
        ligand_path: Input ligand file (mol2, sdf, pdb)
        output_dir: Output directory (auto if None)
        ligand_name: 3-letter residue name for ligand
        charge: Ligand net charge
        spin: Ligand spin multiplicity
        resp_method: 'multiwfn' or 'antechamber'
        force_field: 'ff14SB' or 'ff19SB'
        water_model: 'TIP3P', 'OPC', or 'SPC/E'
        box_buffer: Solvation box buffer in Å
        prod_ns: Production run time in ns
        do_docking: If True, use Vina to place ligand
        docking_exhaustiveness: Vina exhaustiveness
        multiwfn_bin: Optional custom Multiwfn path
        engine: 'pmemd.cuda' or 'pmemd' or 'sander'
        protocol_overrides: dict of AmberMDProtocol fields to override
        config: Optional config dict

    Returns:
        dict with all output paths and next_steps, or error dict.
    """
    # Validate inputs
    err = _check_file(protein_pdb, "Protein PDB")
    if err:
        return err
    err = _check_file(ligand_path, "Ligand file")
    if err:
        return err

    if output_dir is None:
        stem = os.path.splitext(os.path.basename(protein_pdb))[0]
        output_dir = os.path.join(os.getcwd(), f"{stem}_amber_md")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    log_lines = []

    def log(msg):
        log_lines.append(msg)
        print(msg)

    try:
        log("=" * 70)
        log("  General Protein-Ligand AMBER MD Pipeline")
        log(f"  Force field: {force_field} + GAFF2 + {water_model}")
        log("=" * 70)

        # ----- Step 1: Protein prep -----
        log("\n[Step 1] Preparing protein (pdb4amber / reduce) …")
        prot_result = prepare_protein_wsl(protein_pdb, output_dir, force_field, log)
        prepared_pdb = prot_result["prepared_pdb"]
        log(f"  → {os.path.basename(prepared_pdb)}")

        # ----- Step 2: Ligand parameterization -----
        log(f"\n[Step 2] Parameterizing ligand ({resp_method} + GAFF2) …")
        lig_dir = os.path.join(output_dir, "ligand_params")
        lig_result = prepare_ligand_gaff2(
            ligand_path,
            lig_dir,
            ligand_name,
            charge,
            spin,
            resp_method,
            multiwfn_bin,
            log,
        )
        if "error" in lig_result:
            return lig_result
        mol2_path = lig_result["mol2_path"]
        frcmod_path = lig_result["frcmod_path"]
        sybyl_mol2 = lig_result.get("sybyl_mol2")
        log(f"  → mol2: {os.path.basename(mol2_path)}")
        log(f"  → frcmod: {os.path.basename(frcmod_path)}")
        if sybyl_mol2:
            log(f"  → sybyl: {os.path.basename(sybyl_mol2)}")
        if lig_result.get("preflight", {}).get("warnings"):
            for w in lig_result["preflight"]["warnings"]:
                log(f"  ⚠ {w}")

        # ----- Step 3: Docking (optional) -----
        complex_pdb = None
        if do_docking:
            log(
                f"\n[Step 3] Docking ligand with Vina (exhaustiveness={docking_exhaustiveness}) …"
            )
            dock_dir = os.path.join(output_dir, "docking")
            # Use Sybyl mol2 for docking (meeko compatibility), fall back to GAFF2
            dock_lig = sybyl_mol2 if sybyl_mol2 else mol2_path
            dock_result = dock_ligand_vina(
                prepared_pdb,
                dock_lig,
                dock_dir,
                exhaustiveness=docking_exhaustiveness,
                sybyl_mol2=sybyl_mol2,
                log=log,
            )
            if "error" in dock_result:
                log(f"  WARNING: Docking failed: {dock_result['error']}")
                log("  Skipping docking — will use raw ligand coordinates in tleap")
            else:
                log(f"  Best Vina score: {dock_result['best_score']} kcal/mol")
                # Update mol2 coordinates from docked pose (preserves GAFF2 types + RESP charges)
                updated = _update_mol2_coords_from_docked(
                    mol2_path,
                    dock_result["docked_pdbqt"],
                    os.path.join(output_dir, "ligand_docked.mol2"),
                    log,
                )
                if updated:
                    mol2_path = updated
                    log(
                        f"  → Updated mol2 with docked coordinates: {os.path.basename(mol2_path)}"
                    )
        else:
            log("\n[Step 3] Skipping docking (do_docking=False)")

        # Always use tleap combine (protein PDB + ligand mol2 separately)
        complex_pdb = prepared_pdb

        # ----- Pre-tleap validation -----
        log("\n  Running pre-tleap validation …")
        preflight = _preflight_check(
            mol2_path, frcmod_path, expected_charge=charge, log=log
        )
        if not preflight["ok"]:
            log("  ⚠ Pre-flight warnings detected (non-fatal, continuing) …")

        # ----- Step 4: tleap assembly -----
        log(
            f"\n[Step 4] Assembling complex in tleap ({force_field}+GAFF2+{water_model}) …"
        )
        tleap_result = generate_general_tleap(
            output_dir,
            complex_pdb,
            mol2_path,
            frcmod_path,
            ligand_name,
            force_field,
            water_model,
            box_buffer,
            log=log,
        )
        if "error" in tleap_result:
            return tleap_result

        # ----- Step 5 + 6: MD inputs and run script via protocol engine -----
        log(
            f"\n[Step 5-6] Generating MD inputs + run script (production = {prod_ns} ns) …"
        )
        from .protocol import default_protocol, setup_md

        proto = default_protocol(
            system_type="protein_ligand",
            ligand_name=ligand_name,
            prod_ns=prod_ns,
            engine=engine,
            force_field=force_field,
            water_model=water_model,
            box_buffer=box_buffer,
            **(protocol_overrides or {}),
        )
        md_result = setup_md(proto, output_dir, log=log)
        md_dir = md_result["md_input_dir"]
        run_path = md_result["run_script"]

        # Save log
        log_path = os.path.join(output_dir, "pipeline.log")
        _write_file(log_path, "\n".join(log_lines))

        wsl_dir = _win_to_wsl(output_dir)

        log("\n" + "=" * 70)
        log("  Pipeline Complete!")
        log("=" * 70)
        log(f"\n  Output directory: {output_dir}")
        log(f"  Topology: complex.prmtop")
        log(f"  Coordinates: complex.inpcrd")
        log(f"  Run: cd {wsl_dir} && bash run_md.sh")

        return {
            "output_dir": output_dir,
            "prepared_pdb": prepared_pdb,
            "mol2_path": mol2_path,
            "frcmod_path": frcmod_path,
            "sybyl_mol2": sybyl_mol2,
            "complex_pdb": complex_pdb,
            "prmtop": tleap_result["prmtop"],
            "inpcrd": tleap_result["inpcrd"],
            "tleap_input": tleap_result["tleap_path"],
            "md_input_dir": md_dir,
            "run_script": run_path,
            "tleap_log": tleap_result.get("tleap_log", ""),
            "preflight": preflight,
            "next_steps": [
                f"(In WSL) cd {wsl_dir} && bash run_md.sh",
                "Monitor md_output/ for stage outputs",
                "After production: use amber_ligand_analysis for full analysis",
            ],
        }

    except Exception as e:
        import traceback

        return {"error": f"Pipeline failed: {e}", "traceback": traceback.format_exc()}


# ---------------------------------------------------------------------------
# Apo-protein pipeline (no ligand)
# ---------------------------------------------------------------------------


def _generate_apo_tleap(
    output_dir,
    protein_pdb,
    force_field="ff14SB",
    water_model="TIP3P",
    box_buffer=12.0,
    prmtop_name="protein.prmtop",
    inpcrd_name="protein.inpcrd",
    log=print,
):
    """Generate tleap.in for an apo-protein system (no ligand).

    Returns dict with 'tleap_path', 'prmtop', 'inpcrd' or 'error'.
    """
    wsl_out = _win_to_wsl(os.path.abspath(output_dir))
    wsl_prot = _win_to_wsl(os.path.abspath(protein_pdb))

    ff_map = {
        "ff14SB": "leaprc.protein.ff14SB",
        "ff19SB": "leaprc.protein.ff19SB",
    }
    ff_source = ff_map.get(force_field, f"leaprc.protein.{force_field}")

    water_map = {
        "TIP3P": ("leaprc.water.tip3p", "TIP3PBOX"),
        "OPC": ("leaprc.water.opc", "OPCBOX"),
        "SPC/E": ("leaprc.water.spce", "SPCBOX"),
    }
    water_source, water_box = water_map.get(
        water_model, ("leaprc.water.tip3p", "TIP3PBOX")
    )

    content = f"""# tleap input for Apo-Protein System
# Force field: {force_field} + {water_model}
# Auto-generated by zhuangzhou.amber_md.general_pipeline

source {ff_source}
source {water_source}

mol = loadpdb {wsl_prot}

check mol

solvateoct mol {water_box} {box_buffer}

addionsrand mol Na+ 0
addionsrand mol Cl- 0

check mol
charge mol

saveamberparm mol {wsl_out}/{prmtop_name} {wsl_out}/{inpcrd_name}

quit
"""
    tleap_path = os.path.join(output_dir, "tleap.in")
    _write_file(tleap_path, content)

    log("  Running tleap in WSL …")
    r = _run_wsl(f"tleap -f '{wsl_out}/tleap.in'", cwd=wsl_out, timeout=300)

    prmtop_win = os.path.join(output_dir, prmtop_name)
    inpcrd_win = os.path.join(output_dir, inpcrd_name)

    if not os.path.exists(prmtop_win) or not os.path.exists(inpcrd_win):
        return {
            "error": f"tleap failed to produce topology files.\n"
            f"stdout: {r.stdout[-2000:]}\nstderr: {r.stderr[-1000:]}",
            "tleap_path": tleap_path,
        }

    log(f"  tleap OK → {prmtop_name} + {inpcrd_name}")
    return {
        "tleap_path": tleap_path,
        "prmtop": prmtop_win,
        "inpcrd": inpcrd_win,
        "tleap_log": r.stdout[-3000:],
    }


def run_apo_protein_md(
    protein_pdb,
    output_dir=None,
    force_field="ff14SB",
    water_model="TIP3P",
    box_buffer=12.0,
    prod_ns=50.0,
    engine="pmemd.cuda",
    protocol_overrides=None,
    config=None,
    **kw,
):
    """Full pipeline: apo-protein → ready-to-run AMBER MD (no ligand).

    Same continuous step-down protocol as protein-ligand, but without
    ligand parameterization, docking, or ligand-specific restraints.

    Args:
        protein_pdb: Input protein PDB
        output_dir: Output directory (auto if None)
        force_field: 'ff14SB' or 'ff19SB'
        water_model: 'TIP3P', 'OPC', or 'SPC/E'
        box_buffer: Solvation box buffer in Å
        prod_ns: Production run time in ns
        engine: 'pmemd.cuda' or 'pmemd' or 'sander'
        protocol_overrides: dict of AmberMDProtocol fields to override

    Returns:
        dict with output paths and next_steps, or error dict.
    """
    err = _check_file(protein_pdb, "Protein PDB")
    if err:
        return err

    if output_dir is None:
        stem = os.path.splitext(os.path.basename(protein_pdb))[0]
        output_dir = os.path.join(os.getcwd(), f"{stem}_amber_md")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    log_lines = []

    def log(msg):
        log_lines.append(msg)
        print(msg)

    try:
        log("=" * 70)
        log("  Apo-Protein AMBER MD Pipeline")
        log(f"  Force field: {force_field} + {water_model}")
        log("=" * 70)

        # Step 1: Protein prep
        log("\n[Step 1] Preparing protein (pdb4amber / reduce) …")
        prot_result = prepare_protein_wsl(protein_pdb, output_dir, force_field, log)
        prepared_pdb = prot_result["prepared_pdb"]
        log(f"  → {os.path.basename(prepared_pdb)}")

        # Step 2: tleap assembly
        log(f"\n[Step 2] Assembling system in tleap ({force_field}+{water_model}) …")
        tleap_result = _generate_apo_tleap(
            output_dir,
            prepared_pdb,
            force_field,
            water_model,
            box_buffer,
            log=log,
        )
        if "error" in tleap_result:
            return tleap_result

        # Step 3: MD inputs + run script via protocol engine
        log(
            f"\n[Step 3] Generating MD inputs + run script (production = {prod_ns} ns) …"
        )
        from .protocol import default_protocol, setup_md

        proto = default_protocol(
            system_type="protein",
            prod_ns=prod_ns,
            engine=engine,
            force_field=force_field,
            water_model=water_model,
            box_buffer=box_buffer,
            **(protocol_overrides or {}),
        )
        md_result = setup_md(
            proto,
            output_dir,
            prmtop="protein.prmtop",
            inpcrd="protein.inpcrd",
            log=log,
        )

        # Save log
        log_path = os.path.join(output_dir, "pipeline.log")
        _write_file(log_path, "\n".join(log_lines))

        wsl_dir = _win_to_wsl(output_dir)
        log("\n" + "=" * 70)
        log("  Pipeline Complete!")
        log("=" * 70)

        return {
            "output_dir": output_dir,
            "prepared_pdb": prepared_pdb,
            "prmtop": tleap_result["prmtop"],
            "inpcrd": tleap_result["inpcrd"],
            "tleap_input": tleap_result["tleap_path"],
            "md_input_dir": md_result["md_input_dir"],
            "run_script": md_result["run_script"],
            "protocol_summary": md_result["protocol_summary"],
            "tleap_log": tleap_result.get("tleap_log", ""),
            "next_steps": [
                f"(In WSL) cd {wsl_dir} && bash run_md.sh",
                "Monitor md_output/ for stage outputs",
            ],
        }

    except Exception as e:
        import traceback

        return {"error": f"Pipeline failed: {e}", "traceback": traceback.format_exc()}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _merge_protein_ligand_pdb(
    protein_pdb, ligand_pdb, output_path, ligand_name="LIG", log=print
):
    """Merge protein ATOM records + ligand HETATM into one PDB.

    Returns output_path.
    """
    log(f"  Merging protein + ligand → {os.path.basename(output_path)}")
    serial = 1
    with open(output_path, "w", encoding="utf-8") as out:
        # Copy CRYST1 if present
        with open(protein_pdb, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("CRYST1"):
                    out.write(line)
                    break

        # Protein atoms
        with open(protein_pdb, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("ATOM"):
                    out.write(f"{line[:6]}{serial:5d}{line[11:]}")
                    serial += 1
        out.write("TER\n")

        # Figure out what chain/resid to put ligand on
        last_resid = 0
        with open(protein_pdb, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("ATOM"):
                    try:
                        last_resid = int(line[22:26])
                    except ValueError:
                        pass
        lig_resid = last_resid + 1

        # Ligand atoms as HETATM
        with open(ligand_pdb, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    name = line[12:16].strip()
                    # Pad atom name
                    if len(name) <= 3:
                        fmt_name = f" {name:<3s}"
                    else:
                        fmt_name = f"{name:<4s}"

                    x = line[30:38]
                    y = line[38:46]
                    z = line[46:54]

                    out.write(
                        f"HETATM{serial:5d} {fmt_name} {ligand_name:>3s} A{lig_resid:4d}    "
                        f"{x}{y}{z}  1.00  0.00\n"
                    )
                    serial += 1
        out.write("TER\nEND\n")

    log(f"  → {serial - 1} atoms total, ligand resid = {lig_resid}")
    return output_path


def _update_mol2_coords_from_docked(mol2_path, docked_pdbqt, output_path, log=print):
    """Update mol2 XYZ coordinates using coordinates from docked PDBQT.

    This preserves GAFF2 atom types and RESP charges from the original mol2
    while replacing coordinates with the best docked pose from Vina.

    When PDBQT has fewer atoms than mol2 (heavy-atom only from meeko),
    we compute the rigid-body transformation (Kabsch) from old→new heavy
    atom positions, then apply that rotation+translation to ALL atoms
    including hydrogens.

    Returns output_path on success, None on failure.
    """
    import numpy as np

    # Parse coordinates from PDBQT first MODEL, separate heavy / H
    pdbqt_heavy_coords = []
    pdbqt_all_coords = []
    with open(docked_pdbqt, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    aname = line[12:16].strip()
                    is_h = aname.startswith("H")
                    pdbqt_all_coords.append((x, y, z))
                    if not is_h:
                        pdbqt_heavy_coords.append((x, y, z))
                except (ValueError, IndexError):
                    continue
            if line.startswith("ENDMDL"):
                break

    if not pdbqt_all_coords:
        log("    WARNING: No coordinates found in docked PDBQT")
        return None
    log(f"    PDBQT: {len(pdbqt_all_coords)} total, {len(pdbqt_heavy_coords)} heavy")

    # Parse mol2 atom section
    with open(mol2_path, encoding="utf-8", errors="replace") as f:
        mol2_lines = f.readlines()

    atom_start = None
    atom_end = None
    for i, line in enumerate(mol2_lines):
        if "@<TRIPOS>ATOM" in line:
            atom_start = i + 1
        elif atom_start is not None and line.startswith("@"):
            atom_end = i
            break
    if atom_end is None:
        atom_end = len(mol2_lines)
    if atom_start is None:
        log("    WARNING: No @<TRIPOS>ATOM section in mol2")
        return None

    mol2_atom_count = atom_end - atom_start

    # Helper to format one mol2 ATOM line
    def _fmt(parts, x, y, z):
        parts[2] = f"{x:>10.4f}"
        parts[3] = f"{y:>10.4f}"
        parts[4] = f"{z:>10.4f}"
        return (
            f"{parts[0]:>7s} {parts[1]:<4s}"
            f"{parts[2]:>10s}{parts[3]:>10s}{parts[4]:>10s}"
            f" {parts[5]:<8s}{parts[6]:>4s} {parts[7]:<6s}"
            f"{parts[8]:>12s}\n"
        )

    if len(pdbqt_all_coords) == mol2_atom_count:
        # Direct 1:1 mapping — replace all coordinates
        for idx in range(mol2_atom_count):
            parts = mol2_lines[atom_start + idx].split()
            if len(parts) >= 6:
                x, y, z = pdbqt_all_coords[idx]
                mol2_lines[atom_start + idx] = _fmt(parts, x, y, z)
        log(f"    Mapped all {mol2_atom_count} atom coordinates 1:1")

    elif len(pdbqt_heavy_coords) <= mol2_atom_count:
        # PDBQT has only heavy atoms (or heavy + some polar H).
        # Strategy: compute Kabsch rotation+translation from the original
        # heavy-atom positions to the docked heavy-atom positions, then
        # apply the same transform to ALL atoms (including H).

        # Collect original heavy-atom coords from mol2
        old_heavy = []
        heavy_indices = []
        all_old = []
        for idx in range(mol2_atom_count):
            parts = mol2_lines[atom_start + idx].split()
            if len(parts) >= 6:
                cx, cy, cz = float(parts[2]), float(parts[3]), float(parts[4])
                all_old.append((cx, cy, cz))
                if not parts[1].startswith("H"):
                    old_heavy.append((cx, cy, cz))
                    heavy_indices.append(idx)
            else:
                all_old.append((0.0, 0.0, 0.0))

        n_heavy_mol2 = len(old_heavy)
        n_heavy_pdbqt = len(pdbqt_heavy_coords)

        if n_heavy_mol2 != n_heavy_pdbqt:
            log(
                f"    WARNING: Heavy atom count mismatch "
                f"(mol2={n_heavy_mol2}, pdbqt={n_heavy_pdbqt})"
            )
            return None

        # Kabsch: find rotation R and translation t such that
        #   new_coords ≈ R @ old_coords + t
        old_h = np.array(old_heavy)
        new_h = np.array(pdbqt_heavy_coords[:n_heavy_pdbqt])

        centroid_old = old_h.mean(axis=0)
        centroid_new = new_h.mean(axis=0)
        H = (old_h - centroid_old).T @ (new_h - centroid_new)
        U, _, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        S = np.diag([1.0, 1.0, 1.0 if d > 0 else -1.0])
        R = Vt.T @ S @ U.T
        t = centroid_new - R @ centroid_old

        # Apply R, t to ALL atoms
        all_old_arr = np.array(all_old)
        all_new_arr = (R @ all_old_arr.T).T + t

        for idx in range(mol2_atom_count):
            parts = mol2_lines[atom_start + idx].split()
            if len(parts) >= 6:
                x, y, z = all_new_arr[idx]
                mol2_lines[atom_start + idx] = _fmt(parts, x, y, z)

        # Report RMSD on heavy atoms
        mapped_new = all_new_arr[heavy_indices]
        rmsd = np.sqrt(np.mean(np.sum((mapped_new - new_h) ** 2, axis=1)))
        log(
            f"    Kabsch transform: {n_heavy_mol2} heavy atoms matched, "
            f"RMSD={rmsd:.4f} Å, all {mol2_atom_count} atoms transformed"
        )
    else:
        log(
            f"    WARNING: Atom count mismatch (mol2={mol2_atom_count}, "
            f"pdbqt_heavy={len(pdbqt_heavy_coords)}, pdbqt_all={len(pdbqt_all_coords)}), "
            f"skipping coordinate update"
        )
        return None

    _write_file(output_path, "".join(mol2_lines))
    return output_path

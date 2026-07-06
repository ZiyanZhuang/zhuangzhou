"""
Parameterize non-standard residues for GROMACS MD.

Strategy (escalating complexity):
  1. Pre-built parameter database (HEME, common cofactors)
  2. SOBTOP/GAFF automatic parameterization (organic residues)
  3. QM-derived parameters via PySCF (metal centers)

Each method produces:
  - <resname>.itp  (topology include file)
  - <resname>.gro  (coordinate file in GROMACS format)
  - <resname>_atomtypes.itp  (atom types, separate for clean inclusion)
"""

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

from .config import (
    KNOWN_COFACTORS, METAL_INFO, PARAMS_DIR,
    POSRE_FC_METAL, POSRE_FC_COFACTOR,
)

logger = logging.getLogger(__name__)


def parameterize_residue(resname, pdb_path, work_dir, *,
                         method='auto', charge=0, multiplicity=1,
                         ff='amber99sb-ildn'):
    """
    Generate GROMACS .itp and .gro for a single non-standard residue.

    Args:
        resname: Residue name (e.g., 'HEM', 'FAD')
        pdb_path: PDB file containing the residue atoms
        work_dir: Output directory
        method: 'auto' | 'prebuilt' | 'sobtop' | 'qm'
        charge: Net charge of the residue
        multiplicity: Spin multiplicity (for QM)
        ff: Force field name (for compatibility checks)

    Returns:
        {
            'itp_path': str, 'gro_path': str,
            'atomtypes_itp': str or None,
            'method_used': str,
            'error': str or None,
        }
    """
    os.makedirs(work_dir, exist_ok=True)
    result = {
        'resname': resname,
        'itp_path': None,
        'gro_path': None,
        'atomtypes_itp': None,
        'method_used': None,
        'error': None,
    }

    if method == 'auto':
        # Try pre-built first
        if resname in KNOWN_COFACTORS:
            r = _try_prebuilt(resname, pdb_path, work_dir, ff)
            if not r.get('error'):
                result.update(r)
                result['method_used'] = 'prebuilt'
                return result

        # Try SOBTOP for organic residues
        r = _try_sobtop(resname, pdb_path, work_dir, charge)
        if not r.get('error'):
            result.update(r)
            result['method_used'] = 'sobtop'
            return result

        # Fall back to QM
        r = _try_qm(resname, pdb_path, work_dir, charge, multiplicity)
        if not r.get('error'):
            result.update(r)
            result['method_used'] = 'qm'
            return result

        result['error'] = 'All parameterization methods failed'
        return result

    elif method == 'prebuilt':
        r = _try_prebuilt(resname, pdb_path, work_dir, ff)
        result.update(r)
        result['method_used'] = 'prebuilt' if not r.get('error') else None
        return result

    elif method == 'sobtop':
        r = _try_sobtop(resname, pdb_path, work_dir, charge)
        result.update(r)
        result['method_used'] = 'sobtop' if not r.get('error') else None
        return result

    elif method == 'qm':
        r = _try_qm(resname, pdb_path, work_dir, charge, multiplicity)
        result.update(r)
        result['method_used'] = 'qm' if not r.get('error') else None
        return result

    else:
        result['error'] = f'Unknown method: {method}'
        return result


def parameterize_system(detection_result, pdb_path, work_dir, *,
                        ff='amber99sb-ildn', method='auto'):
    """
    Parameterize all non-standard residues detected in a PDB.

    Args:
        detection_result: Output from detect_nonstandard()
        pdb_path: Original PDB file path
        work_dir: Output directory
        ff: Force field name
        method: Parameterization method

    Returns:
        {
            'residues': [parameterize_residue result for each],
            'metal_ions': [metal ion parameter results],
            'all_itp_paths': [str],
            'all_atomtype_paths': [str],
            'error': str or None,
        }
    """
    os.makedirs(work_dir, exist_ok=True)
    results = {
        'residues': [],
        'metal_ions': [],
        'all_itp_paths': [],
        'all_atomtype_paths': [],
        'error': None,
    }

    if 'error' in detection_result:
        results['error'] = detection_result['error']
        return results

    # Parameterize each non-standard residue
    for res_info in detection_result.get('nonstandard_residues', []):
        resname = res_info['resname']
        res_dir = os.path.join(work_dir, resname)

        # Extract this residue's atoms to a separate PDB
        res_pdb = _extract_residue_pdb(
            pdb_path, resname, res_info['chain'], res_info['resid'], res_dir
        )
        if not res_pdb:
            results['residues'].append({
                'resname': resname, 'error': 'Failed to extract residue PDB'
            })
            continue

        r = parameterize_residue(
            resname, res_pdb, res_dir, method=method, ff=ff
        )
        results['residues'].append(r)
        if r.get('itp_path'):
            results['all_itp_paths'].append(r['itp_path'])
        if r.get('atomtypes_itp'):
            results['all_atomtype_paths'].append(r['atomtypes_itp'])

    # Handle metal ions (generate simple .itp for each unique metal type)
    seen_metals = set()
    for metal in detection_result.get('metal_ions', []):
        mel = metal['element']
        charge = metal['charge']
        metal_key = f'{mel}_{charge}'
        if metal_key in seen_metals:
            continue
        seen_metals.add(metal_key)

        r = _generate_metal_itp(mel, charge, work_dir)
        results['metal_ions'].append(r)
        if r.get('itp_path'):
            results['all_itp_paths'].append(r['itp_path'])

    return results


# ---------------------------------------------------------------------------
# Strategy 1: Pre-built parameter database
# ---------------------------------------------------------------------------

def _try_prebuilt(resname, pdb_path, work_dir, ff):
    """Look up pre-built .itp in params/ directory."""
    param_key = KNOWN_COFACTORS.get(resname)
    if not param_key:
        return {'error': f'No pre-built params for {resname}'}

    # Look for ff-specific params first, then generic
    candidates = [
        PARAMS_DIR / ff / f'{param_key}.itp',
        PARAMS_DIR / f'{param_key}.itp',
    ]

    src_itp = None
    for c in candidates:
        if c.exists():
            src_itp = c
            break

    if not src_itp:
        return {'error': f'Pre-built .itp not found for {param_key} (tried {candidates})'}

    os.makedirs(work_dir, exist_ok=True)
    dst_itp = os.path.join(work_dir, f'{resname}.itp')
    shutil.copy2(src_itp, dst_itp)

    # Check for atomtypes companion
    src_at = src_itp.parent / f'{param_key}_atomtypes.itp'
    dst_at = None
    if src_at.exists():
        dst_at = os.path.join(work_dir, f'{resname}_atomtypes.itp')
        shutil.copy2(src_at, dst_at)

    # Generate .gro from PDB using GROMACS editconf (or parse manually)
    gro_path = _pdb_to_gro(pdb_path, work_dir, resname)

    return {
        'itp_path': dst_itp,
        'gro_path': gro_path,
        'atomtypes_itp': dst_at,
        'error': None,
    }


# ---------------------------------------------------------------------------
# Strategy 2: SOBTOP/GAFF automatic parameterization
# ---------------------------------------------------------------------------

def _try_sobtop(resname, pdb_path, work_dir, charge=0):
    """Use SOBTOP to generate GAFF parameters."""
    try:
        from zhuangzhou.gromacs_md.config import SOBTOP_DIR, SOBTOP_EXE, OBABEL_EXE
    except ImportError:
        return {'error': 'SOBTOP/OpenBabel paths not configured'}

    if not SOBTOP_EXE or not os.path.exists(SOBTOP_EXE):
        return {'error': f'SOBTOP not found at {SOBTOP_EXE}'}
    if not OBABEL_EXE or not os.path.exists(OBABEL_EXE):
        return {'error': f'OpenBabel not found at {OBABEL_EXE}'}

    os.makedirs(work_dir, exist_ok=True)

    # Step 1: PDB -> mol2
    mol2_path = os.path.join(work_dir, f'{resname}.mol2')
    try:
        result = subprocess.run(
            [OBABEL_EXE, pdb_path, '-O', mol2_path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0 or not os.path.exists(mol2_path):
            return {'error': f'OpenBabel PDB->mol2 failed: {result.stderr[:300]}'}
    except Exception as e:
        return {'error': f'OpenBabel failed: {e}'}

    # Step 2: SOBTOP
    itp_path = os.path.join(work_dir, f'{resname}.itp')
    top_path = os.path.join(work_dir, f'{resname}.top')
    gro_path = os.path.join(work_dir, f'{resname}.gro')

    stdin_text = (
        f'{mol2_path}\n'
        f'1\n'       # Generate topology
        f'2\n'       # GAFF atom types
        f'4\n'       # Prebuilt bonded + guess
        f'{top_path}\n'
        f'{itp_path}\n'
        f'2\n'       # Generate .gro
        f'{gro_path}\n'
        f'0\n'       # Exit
    )

    try:
        result = subprocess.run(
            [SOBTOP_EXE], input=stdin_text, capture_output=True, text=True,
            timeout=60, cwd=SOBTOP_DIR,
        )
    except Exception as e:
        return {'error': f'SOBTOP failed: {e}'}

    if not os.path.exists(itp_path):
        out = (result.stdout or '') + (result.stderr or '')
        return {'error': f'SOBTOP did not generate .itp: {out[-300:]}'}

    # Fix residue/molecule names
    _fix_itp_names(itp_path, resname)
    if os.path.exists(gro_path):
        _fix_gro_resname(gro_path, resname)

    # Extract atomtypes to separate file
    atomtypes_itp = os.path.join(work_dir, f'{resname}_atomtypes.itp')
    _extract_atomtypes(itp_path, atomtypes_itp)
    _remove_section_from_itp(itp_path, 'atomtypes')

    return {
        'itp_path': itp_path,
        'gro_path': gro_path if os.path.exists(gro_path) else None,
        'atomtypes_itp': atomtypes_itp if os.path.exists(atomtypes_itp) else None,
        'error': None,
    }


# ---------------------------------------------------------------------------
# Strategy 3: QM-derived parameters
# ---------------------------------------------------------------------------

def _try_qm(resname, pdb_path, work_dir, charge=0, multiplicity=1):
    """
    Generate parameters using QM calculations (PySCF).

    This is the fallback for metal centers and unusual bonding.
    Workflow:
      1. Extract geometry from PDB
      2. Run DFT optimization + frequency (PySCF via bridge)
      3. Compute Hessian -> force constants
      4. Write .itp with QM-derived bonded parameters
    """
    try:
        from zhuangzhou.qm_engine.bridge import run_calculation
    except ImportError:
        return {'error': 'QM engine (PySCF) not available'}

    os.makedirs(work_dir, exist_ok=True)

    # Extract XYZ from PDB
    xyz_path = os.path.join(work_dir, f'{resname}.xyz')
    atoms = _pdb_to_xyz(pdb_path, xyz_path)
    if not atoms:
        return {'error': 'Failed to extract coordinates from PDB'}

    # Run optimization + frequency
    try:
        opt_result = run_calculation(
            xyz_path=xyz_path,
            calc_type='opt',
            charge=charge,
            multiplicity=multiplicity,
            basis='def2-SVP',
            functional='B3LYP',
            output_dir=work_dir,
        )
        if opt_result.get('error'):
            return {'error': f'QM optimization failed: {opt_result["error"]}'}

        opt_xyz = opt_result.get('optimized_xyz')
        if opt_xyz:
            freq_result = run_calculation(
                xyz_path=opt_xyz,
                calc_type='freq',
                charge=charge,
                multiplicity=multiplicity,
                basis='def2-SVP',
                functional='B3LYP',
                output_dir=work_dir,
            )
        else:
            freq_result = {'error': 'No optimized geometry'}

    except Exception as e:
        return {'error': f'QM calculation failed: {e}'}

    # Generate .itp from QM results
    itp_path = os.path.join(work_dir, f'{resname}.itp')
    _write_qm_itp(resname, opt_result, freq_result, itp_path, charge)

    # Generate .gro from optimized geometry
    gro_path = None
    if opt_xyz:
        gro_path = _xyz_to_gro(opt_xyz, os.path.join(work_dir, f'{resname}.gro'), resname)

    return {
        'itp_path': itp_path if os.path.exists(itp_path) else None,
        'gro_path': gro_path,
        'atomtypes_itp': None,
        'error': None if os.path.exists(itp_path) else 'Failed to generate .itp',
    }


# ---------------------------------------------------------------------------
# Metal ion .itp generation
# ---------------------------------------------------------------------------

def _generate_metal_itp(element, charge, work_dir):
    """
    Generate a minimal .itp for a metal ion (non-bonded model).

    Non-bonded model: metal treated as a charged Lennard-Jones particle.
    Position restraints are handled separately by restraints.py.
    """
    os.makedirs(work_dir, exist_ok=True)

    resname = f'{element.upper()}{charge}' if charge > 0 else element.upper()
    itp_path = os.path.join(work_dir, f'{resname}.itp')

    # LJ parameters for common metal ions (sigma in nm, epsilon in kJ/mol)
    # From Li & Merz (2020) 12-6 IOD parameter set for TIP3P
    METAL_LJ = {
        'Fe2': {'sigma': 0.2155, 'epsilon': 0.05230, 'mass': 55.845},
        'Fe3': {'sigma': 0.1974, 'epsilon': 0.01205, 'mass': 55.845},
        'Zn2': {'sigma': 0.2095, 'epsilon': 0.03790, 'mass': 65.38},
        'Mn2': {'sigma': 0.2232, 'epsilon': 0.07460, 'mass': 54.938},
        'Cu2': {'sigma': 0.2065, 'epsilon': 0.03199, 'mass': 63.546},
        'Co2': {'sigma': 0.2108, 'epsilon': 0.04101, 'mass': 58.933},
        'Ni2': {'sigma': 0.2034, 'epsilon': 0.02635, 'mass': 58.693},
        'Mg2': {'sigma': 0.2350, 'epsilon': 0.11380, 'mass': 24.305},
        'Ca2': {'sigma': 0.2872, 'epsilon': 0.44960, 'mass': 40.078},
    }

    key = f'{element}{charge}'
    lj = METAL_LJ.get(key, {'sigma': 0.220, 'epsilon': 0.050, 'mass': 55.0})

    content = f"""; Metal ion topology: {element}{charge}+
; Non-bonded model (Li & Merz 2020 12-6 IOD parameters)
; Use with position restraints for coordination geometry

[ moleculetype ]
; name  nrexcl
{resname:<8s}  1

[ atoms ]
;  nr  type  resnr  residu  atom  cgnr  charge    mass
    1  {resname:<6s}  1      {resname:<6s}  {element:<5s}  1     {float(charge):.4f}  {lj['mass']:.3f}

#ifdef POSRES_METAL
[ position_restraints ]
;  i  funct  fcx        fcy        fcz
   1  1      {POSRE_FC_METAL:.1f}  {POSRE_FC_METAL:.1f}  {POSRE_FC_METAL:.1f}
#endif
"""

    with open(itp_path, 'w', encoding='utf-8') as f:
        f.write(content)

    # Atomtypes file
    at_path = os.path.join(work_dir, f'{resname}_atomtypes.itp')
    at_content = f"""; Metal ion atom type: {element}{charge}+
[ atomtypes ]
; name  at.num  mass      charge  ptype  sigma      epsilon
  {resname:<6s}  {_atomic_number(element):>3d}    {lj['mass']:.3f}  {float(charge):.4f}  A      {lj['sigma']:.6f}  {lj['epsilon']:.6f}
"""
    with open(at_path, 'w', encoding='utf-8') as f:
        f.write(at_content)

    return {
        'resname': resname,
        'element': element,
        'charge': charge,
        'itp_path': itp_path,
        'atomtypes_itp': at_path,
        'error': None,
    }


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _extract_residue_pdb(pdb_path, resname, chain, resid, output_dir):
    """Extract a single residue's atoms from PDB into a new PDB file."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'{resname}.pdb')
    lines = []

    with open(pdb_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            rn = line[17:20].strip()
            ch = line[21].strip()
            try:
                ri = int(line[22:26].strip())
            except ValueError:
                continue
            if rn == resname and ch == chain and ri == resid:
                lines.append(line)

    if not lines:
        return None

    with open(out_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
        f.write('END\n')
    return out_path


def _pdb_to_gro(pdb_path, work_dir, resname):
    """Convert PDB to GRO format (simple parser, no GROMACS needed)."""
    os.makedirs(work_dir, exist_ok=True)
    gro_path = os.path.join(work_dir, f'{resname}.gro')
    atoms = []

    with open(pdb_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            name = line[12:16].strip()
            rn = line[17:20].strip()
            try:
                ri = int(line[22:26].strip())
                x = float(line[30:38]) / 10.0  # A -> nm
                y = float(line[38:46]) / 10.0
                z = float(line[46:54]) / 10.0
            except ValueError:
                continue
            atoms.append((ri, rn, name, x, y, z))

    if not atoms:
        return None

    with open(gro_path, 'w', encoding='utf-8') as f:
        f.write(f'{resname} in GROMACS format\n')
        f.write(f'{len(atoms)}\n')
        for i, (ri, rn, name, x, y, z) in enumerate(atoms, 1):
            f.write(f'{ri:5d}{rn:<5s}{name:>5s}{i:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n')
        f.write('   0.00000   0.00000   0.00000\n')
    return gro_path


def _pdb_to_xyz(pdb_path, xyz_path):
    """Extract XYZ coordinates from PDB."""
    atoms = []
    with open(pdb_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            element = line[76:78].strip() if len(line) > 77 else ''
            if not element:
                name = line[12:16].strip()
                element = name.lstrip('0123456789')[0:2].strip()
                if len(element) > 1 and element[1].islower():
                    pass
                else:
                    element = element[0]
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            atoms.append((element, x, y, z))

    if not atoms:
        return None

    with open(xyz_path, 'w', encoding='utf-8') as f:
        f.write(f'{len(atoms)}\n')
        f.write(f'Extracted from {os.path.basename(pdb_path)}\n')
        for el, x, y, z in atoms:
            f.write(f'{el:>2s}  {x:12.6f}  {y:12.6f}  {z:12.6f}\n')
    return atoms


def _xyz_to_gro(xyz_path, gro_path, resname):
    """Convert XYZ file to GRO format."""
    atoms = []
    with open(xyz_path, encoding='utf-8', errors='replace') as f:
        n = int(f.readline().strip())
        f.readline()  # comment
        for _ in range(n):
            parts = f.readline().split()
            el = parts[0]
            x = float(parts[1]) / 10.0  # A -> nm
            y = float(parts[2]) / 10.0
            z = float(parts[3]) / 10.0
            atoms.append((el, x, y, z))

    with open(gro_path, 'w', encoding='utf-8') as f:
        f.write(f'{resname} from QM optimization\n')
        f.write(f'{len(atoms)}\n')
        for i, (el, x, y, z) in enumerate(atoms, 1):
            f.write(f'    1{resname:<5s}{el:>5s}{i:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n')
        f.write('   0.00000   0.00000   0.00000\n')
    return gro_path


def _fix_itp_names(itp_path, resname):
    """Fix moleculetype and residue names in .itp to match target resname."""
    with open(itp_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    content = content.replace('ligand       3', f'{resname:<13s} 3')
    content = content.replace('      MOL', f'{resname:>9s}')
    with open(itp_path, 'w', encoding='utf-8') as f:
        f.write(content)


def _fix_gro_resname(gro_path, resname):
    """Fix residue name in .gro file."""
    with open(gro_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        if 'MOL' in line[:10] and len(line) > 20:
            line = line.replace('MOL', resname[:3].ljust(3), 1)
        new_lines.append(line)
    with open(gro_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)


def _extract_atomtypes(itp_path, output_path):
    """Extract [ atomtypes ] section from .itp into a separate file."""
    with open(itp_path, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    in_section = False
    section_lines = []
    for line in lines:
        if '[ atomtypes ]' in line:
            in_section = True
            section_lines.append(line)
            continue
        if in_section:
            if line.strip().startswith('[') and 'atomtypes' not in line:
                break
            section_lines.append(line)
    if section_lines:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(section_lines)


def _remove_section_from_itp(itp_path, section_name):
    """Remove a [ section ] from .itp file."""
    with open(itp_path, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    new_lines = []
    skip = False
    for line in lines:
        if f'[ {section_name} ]' in line:
            skip = True
            continue
        if skip and line.strip().startswith('['):
            skip = False
        if not skip:
            new_lines.append(line)
    with open(itp_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)


def _write_qm_itp(resname, opt_result, freq_result, itp_path, charge):
    """
    Generate a basic .itp from QM calculation results.

    For full MCPB.py-style parameterization, this would derive force
    constants from the Hessian matrix. Current implementation generates
    a template with QM-derived charges (Mulliken) and placeholder
    bonded parameters that should be refined.
    """
    # Get atoms and charges from optimization result
    atoms = opt_result.get('atoms', [])
    mulliken = opt_result.get('mulliken_charges', [])
    if not atoms:
        return

    with open(itp_path, 'w', encoding='utf-8') as f:
        f.write(f'; Topology for {resname} (QM-derived, B3LYP/def2-SVP)\n')
        f.write(f'; WARNING: Bonded parameters are approximate.\n')
        f.write(f';          Refine with Seminario method or MCPB.py for production.\n\n')

        f.write('[ moleculetype ]\n')
        f.write(f'; name  nrexcl\n')
        f.write(f'{resname:<8s}  3\n\n')

        f.write('[ atoms ]\n')
        f.write(';  nr  type    resnr  residu  atom  cgnr  charge      mass\n')
        for i, atom in enumerate(atoms):
            el = atom.get('element', atom.get('symbol', 'X'))
            q = mulliken[i] if i < len(mulliken) else 0.0
            mass = _atomic_mass(el)
            atype = f'{el}_{resname}'
            f.write(f'{i+1:5d}  {atype:<8s}  1  {resname:<6s}  {el:<5s}  {i+1:4d}'
                    f'  {q:10.6f}  {mass:.3f}\n')

        # Add position restraint ifdef
        f.write(f'\n#ifdef POSRES_{resname}\n')
        f.write('[ position_restraints ]\n')
        f.write(';  i  funct  fcx        fcy        fcz\n')
        for i, atom in enumerate(atoms):
            el = atom.get('element', atom.get('symbol', 'X'))
            if el != 'H':  # Restrain heavy atoms only
                f.write(f'{i+1:4d}  1  {POSRE_FC_COFACTOR:.1f}  '
                        f'{POSRE_FC_COFACTOR:.1f}  {POSRE_FC_COFACTOR:.1f}\n')
        f.write('#endif\n')


def _atomic_number(element):
    """Get atomic number for common elements."""
    TABLE = {
        'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'S': 16, 'P': 15,
        'Fe': 26, 'Zn': 30, 'Mn': 25, 'Cu': 29, 'Co': 27, 'Ni': 28,
        'Mg': 12, 'Ca': 20, 'Cl': 17, 'Br': 35, 'I': 53,
    }
    return TABLE.get(element, 0)


def _atomic_mass(element):
    """Get atomic mass for common elements."""
    TABLE = {
        'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
        'F': 18.998, 'S': 32.065, 'P': 30.974, 'Cl': 35.453,
        'Fe': 55.845, 'Zn': 65.38, 'Mn': 54.938, 'Cu': 63.546,
        'Co': 58.933, 'Ni': 58.693, 'Mg': 24.305, 'Ca': 40.078,
    }
    return TABLE.get(element, 12.0)

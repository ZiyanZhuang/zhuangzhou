"""
System preparation for protein-ligand MD.

Workflow:
  1. Split complex PDB into protein + ligand
  2. Protein: gmx pdb2gmx (AMBER99SB-ILDN)
  3. Ligand: OpenBabel → mol2, SOBTOP → GAFF itp/gro
  4. Merge: combine coordinates, edit topology
  5. Box, solvate, add ions
  6. Create index groups
"""

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

from .config import (
    GMX_BIN, SOBTOP_DIR, SOBTOP_EXE, OBABEL_EXE,
    DEFAULT_FF, DEFAULT_WATER, DEFAULT_BOX_DIST, DEFAULT_BOX_TYPE,
    TIMEOUT_SETUP,
)

logger = logging.getLogger(__name__)


def run_gmx(cmd, work_dir, input_text=None, timeout=TIMEOUT_SETUP, sandbox=None):
    """Run a GROMACS command, supporting local or remote execution.
    
    Args:
        sandbox: Optional Sandbox instance (Native, WSL, or SSH).
    """
    if sandbox:
        # Remote or sandboxed execution — prefix with 'gmx' (GROMACS 5+ subcommand style)
        gmx_cmd = f'gmx {cmd}' if not cmd.strip().startswith('gmx') else cmd
        logger.info("[%s-GMX] %s", getattr(sandbox, 'host_name', 'Sandboxed'), cmd.split()[0])
        result = sandbox.run(gmx_cmd, work_dir=work_dir, timeout=timeout)
        if not result.success:
            return result.stdout + result.stderr, False
        return result.stdout, True

    # Local execution (Legacy/Default)
    full_cmd = f'"{GMX_BIN}" {cmd}'
    logger.info("[Local-GMX] %s", cmd.split()[0])
    try:
        result = subprocess.run(
            full_cmd, shell=True, capture_output=True, text=True,
            input=input_text, timeout=timeout, cwd=work_dir,
        )
        output = (result.stdout or '') + (result.stderr or '')
        if result.returncode != 0:
            logger.error("[Local-GMX] %s failed:\n%s", cmd.split()[0], output[-800:])
            return output, False
        return output, True
    except subprocess.TimeoutExpired:
        return f"Timeout after {timeout}s", False
    except Exception as e:
        return str(e), False


def split_complex(complex_pdb, work_dir):
    """Split complex PDB into protein.pdb and ligand.pdb.

    Returns: (protein_pdb, ligand_pdb, ligand_resname) or raises.
    """
    protein_lines = []
    ligand_lines = []
    ligand_resnames = set()

    _SKIP_RESNAMES = {
        'HOH', 'WAT', 'DOD', 'TIP', 'SPC', 'SOL',   # water
        'NA', 'CL', 'K', 'BR', 'IOD', 'CS',          # monatomic ions
        'NA+', 'CL-', 'K+', 'BR-',                    # charged ions
        'MG', 'ZN', 'CA', 'FE', 'MN', 'CU', 'CO', 'NI',  # metals
        'FE2', 'FE3', 'ZN2', 'MN2', 'MG2', 'CU2', 'CA2', # divalent
    }
    with open(complex_pdb, encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('ATOM'):
                protein_lines.append(line)
            elif line.startswith('HETATM'):
                # Try both 3-char (PDB standard) and 4-char (extended) resname
                rn3 = line[17:20].strip()
                rn4 = line[17:21].strip()
                resname = rn4 if len(rn4) > len(rn3) else rn3
                # Skip water/ions/metals
                if resname in _SKIP_RESNAMES or rn3 in _SKIP_RESNAMES:
                    continue
                ligand_lines.append(line)
                ligand_resnames.add(resname)
            elif line.startswith('CONECT'):
                ligand_lines.append(line)

    if not ligand_lines:
        return None, None, None

    protein_pdb = os.path.join(work_dir, 'protein.pdb')
    ligand_pdb = os.path.join(work_dir, 'ligand.pdb')

    with open(protein_pdb, 'w', encoding='utf-8') as f:
        f.writelines(protein_lines)
        f.write('END\n')

    with open(ligand_pdb, 'w', encoding='utf-8') as f:
        f.writelines(ligand_lines)
        f.write('END\n')

    resname = list(ligand_resnames)[0] if ligand_resnames else 'LIG'
    logger.info("Split: %d protein atoms, %d ligand atoms (resname=%s)",
                len(protein_lines), len([l for l in ligand_lines if l.startswith('HETATM')]),
                resname)
    return protein_pdb, ligand_pdb, resname


def parameterize_protein(work_dir, protein_pdb, ff=DEFAULT_FF, water=DEFAULT_WATER):
    """Run pdb2gmx on protein. Returns (gro_path, success)."""
    cmd = (f'pdb2gmx -f "{protein_pdb}" -o protein.gro '
           f'-water {water} -ff {ff} -ignh')
    output, ok = run_gmx(cmd, work_dir)
    if not ok:
        return None, output
    return os.path.join(work_dir, 'protein.gro'), None


def parameterize_ligand(work_dir, ligand_pdb, resname='MOL'):
    """Generate GAFF parameters for ligand using SOBTOP.

    Steps: PDB → mol2 (OpenBabel) → SOBTOP (GAFF + prebuilt bonded params)
    Returns: (itp_path, gro_path, error_msg)
    """
    # SOBTOP runs from SOBTOP_DIR — all paths must be absolute
    work_dir = os.path.abspath(work_dir)
    ligand_pdb = os.path.abspath(ligand_pdb)
    ligand_mol2 = os.path.join(work_dir, 'ligand.mol2')
    ligand_itp = os.path.join(work_dir, 'ligand.itp')
    ligand_gro = os.path.join(work_dir, 'ligand.gro')

    # Step 1: Convert PDB → mol2 via OpenBabel
    if not os.path.exists(OBABEL_EXE):
        return None, None, f'OpenBabel not found at {OBABEL_EXE}'

    result = subprocess.run(
        [OBABEL_EXE, ligand_pdb, '-O', ligand_mol2],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0 or not os.path.exists(ligand_mol2):
        return None, None, f'OpenBabel conversion failed: {result.stderr}'

    # Step 2: Run SOBTOP for GAFF topology + .gro
    if not os.path.exists(SOBTOP_EXE):
        return None, None, f'SOBTOP not found at {SOBTOP_EXE}'

    # SOBTOP stdin protocol:
    # <mol2_path>  → load molecule
    # 1            → Generate GROMACS topology file
    # 2            → Assign GAFF types (option 2 in atom type submenu)
    # 4            → Prebuilt bonded params + guess missing
    # <top_path>   → output .top file
    # <itp_path>   → output .itp file
    # 2            → Generate GROMACS .gro file (back in main menu)
    # <gro_path>   → output .gro file
    # 0            → Exit
    ligand_top = os.path.join(work_dir, 'ligand.top')
    # SOBTOP (Fortran) cannot parse Windows backslash paths — use forward slashes
    _mol2 = ligand_mol2.replace(os.sep, '/')
    _top = ligand_top.replace(os.sep, '/')
    _itp = ligand_itp.replace(os.sep, '/')
    _gro = ligand_gro.replace(os.sep, '/')
    stdin_text = (
        f'{_mol2}\n'
        f'1\n'       # Generate topology
        f'2\n'       # GAFF atom types
        f'4\n'       # Prebuilt bonded + guess
        f'{_top}\n'
        f'{_itp}\n'
        f'2\n'       # Generate .gro
        f'{_gro}\n'
        f'0\n'       # Exit
    )

    try:
        result = subprocess.run(
            [SOBTOP_EXE], input=stdin_text, capture_output=True, text=True,
            timeout=60, cwd=SOBTOP_DIR,
        )
    except Exception as e:
        return None, None, f'SOBTOP failed: {e}'

    sobtop_output = (result.stdout or '') + (result.stderr or '')

    if not os.path.exists(ligand_itp):
        return None, None, f'SOBTOP did not generate .itp:\n{sobtop_output[-500:]}'
    if not os.path.exists(ligand_gro):
        return None, None, f'SOBTOP did not generate .gro:\n{sobtop_output[-500:]}'

    # Fix residue name in .itp and .gro to match the original
    _fix_resname(ligand_itp, resname)
    _fix_resname_gro(ligand_gro, resname)

    logger.info("Ligand parameterized: %s, %s", ligand_itp, ligand_gro)
    return ligand_itp, ligand_gro, None


def parameterize_ligand_acpype(work_dir, ligand_pdb, resname='MOL',
                                charge_method='bcc', net_charge=0):
    """Generate GAFF2 parameters for ligand using ACPYPE (AnteChamber PYthon Parser).

    ACPYPE is the gold standard for AMBER/GAFF ligand parameterization.
    Uses AM1-BCC charges by default (recommended by literature).

    Args:
        work_dir: output directory
        ligand_pdb: ligand PDB or SDF file
        resname: 3-letter residue name
        charge_method: 'bcc' (AM1-BCC, recommended) or 'gas' (Gasteiger)
        net_charge: net charge of the molecule

    Returns: (itp_path, gro_path, error_msg)
    """
    try:
        import acpype
    except ImportError:
        return None, None, 'ACPYPE not installed (pip install acpype)'

    from acpype.topol import ACTopol
    import tempfile

    ligand_itp = os.path.join(work_dir, f'{resname}.itp')
    ligand_gro = os.path.join(work_dir, f'{resname}.gro')

    try:
        # ACPYPE works from a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy input file
            input_ext = os.path.splitext(ligand_pdb)[1].lower()
            input_copy = os.path.join(tmpdir, f'{resname}{input_ext}')
            shutil.copy(ligand_pdb, input_copy)

            # Run ACPYPE
            mol = ACTopol(
                input_copy,
                chargeType=charge_method,
                chargeVal=net_charge,
                basename=resname,
            )
            mol.createACTopol()
            mol.createMolTopol()

            # Find outputs in ACPYPE output directory
            acpype_dir = os.path.join(tmpdir, f'{resname}.acpype')
            if not os.path.isdir(acpype_dir):
                # Try alternative naming
                for d in os.listdir(tmpdir):
                    if d.endswith('.acpype'):
                        acpype_dir = os.path.join(tmpdir, d)
                        break

            # Copy GROMACS files
            gmx_itp = os.path.join(acpype_dir, f'{resname}_GMX.itp')
            gmx_gro = os.path.join(acpype_dir, f'{resname}_GMX.gro')
            gmx_top = os.path.join(acpype_dir, f'{resname}_GMX.top')

            # Fallback: look for any .itp/.gro
            if not os.path.exists(gmx_itp):
                for f in os.listdir(acpype_dir):
                    if f.endswith('.itp') and 'GMX' in f:
                        gmx_itp = os.path.join(acpype_dir, f)
                    elif f.endswith('.gro') and 'GMX' in f:
                        gmx_gro = os.path.join(acpype_dir, f)

            if not os.path.exists(gmx_itp):
                return None, None, f'ACPYPE did not produce .itp in {acpype_dir}'

            shutil.copy(gmx_itp, ligand_itp)
            if os.path.exists(gmx_gro):
                shutil.copy(gmx_gro, ligand_gro)

            # Also copy atomtypes if present (needed for topology)
            atomtypes_itp = os.path.join(acpype_dir, f'{resname}_GMX_atomtypes.itp')
            if os.path.exists(atomtypes_itp):
                shutil.copy(atomtypes_itp,
                            os.path.join(work_dir, f'{resname}_atomtypes.itp'))

    except Exception as e:
        return None, None, f'ACPYPE failed: {e}'

    # Validate charge sum
    charge_err = _validate_itp_charge(ligand_itp)
    if charge_err:
        logger.warning("ACPYPE charge validation: %s", charge_err)

    # Fix resname
    _fix_resname(ligand_itp, resname)
    if os.path.exists(ligand_gro):
        _fix_resname_gro(ligand_gro, resname)

    logger.info("Ligand parameterized (ACPYPE/GAFF2): %s", ligand_itp)
    return ligand_itp, ligand_gro, None


def _validate_itp_charge(itp_path):
    """Validate that total charge in .itp is close to an integer."""
    total_charge = 0.0
    in_atoms = False
    with open(itp_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            stripped = line.strip()
            if '[ atoms ]' in stripped:
                in_atoms = True
                continue
            if in_atoms and stripped.startswith('['):
                break
            if in_atoms and stripped and not stripped.startswith(';'):
                parts = stripped.split()
                if len(parts) >= 7:
                    try:
                        total_charge += float(parts[6])
                    except (ValueError, IndexError):
                        pass
    rounded = round(total_charge)
    if abs(total_charge - rounded) > 0.01:
        return (f'Total charge {total_charge:.4f} is not an integer '
                f'(nearest: {rounded}). Check parameterization.')
    return None


def _fix_resname(itp_path, target_resname):
    """Ensure residue name and moleculetype in .itp match the target resname."""
    import re
    with open(itp_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    # Fix moleculetype name: various default names → target_resname
    # SOBTOP outputs "ligand", ACPYPE outputs "MOL" or actual resname
    content = re.sub(
        r'^(\s*)(ligand|MOL|UNK|LIG)\s+(\d+)\s*$',
        lambda m: f'{m.group(1)}{target_resname:<13s} {m.group(3)}',
        content, flags=re.MULTILINE)
    # Fix residue name in [ atoms ] section (column-based, any of MOL/ligand/UNK)
    content = re.sub(
        r'(\s+\d+\s+\S+\s+\d+\s+)(MOL|ligand|UNK|LIG)(\s+)',
        lambda m: f'{m.group(1)}{target_resname:>3s}{m.group(3)}',
        content)
    with open(itp_path, 'w', encoding='utf-8') as f:
        f.write(content)


def _fix_resname_gro(gro_path, target_resname):
    """Fix residue name in .gro file."""
    with open(gro_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        if 'MOL' in line[:10] and len(line) > 20:
            line = line.replace('MOL', target_resname[:3].ljust(3), 1)
        new_lines.append(line)
    with open(gro_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)


def merge_system(work_dir, protein_gro, ligand_gro, ligand_itp, resname='MOL'):
    """Merge protein + ligand into a single system.

    1. Combine .gro coordinates
    2. Edit topol.top to include ligand.itp and add ligand molecule
    Returns: (complex_gro, success)
    """
    complex_gro = os.path.join(work_dir, 'complex.gro')

    # Read protein .gro
    with open(protein_gro, encoding='utf-8', errors='replace') as f:
        prot_lines = f.readlines()
    prot_title = prot_lines[0]
    prot_natoms = int(prot_lines[1].strip())
    prot_atoms = prot_lines[2:2 + prot_natoms]
    prot_box = prot_lines[2 + prot_natoms]

    # Read ligand .gro
    with open(ligand_gro, encoding='utf-8', errors='replace') as f:
        lig_lines = f.readlines()
    lig_natoms = int(lig_lines[1].strip())
    lig_atoms = lig_lines[2:2 + lig_natoms]

    # Renumber ligand atoms
    start_num = prot_natoms + 1
    renumbered_lig = []
    for i, line in enumerate(lig_atoms):
        # GRO format: residue number (5), residue name (5), atom name (5), atom number (5), ...
        new_num = start_num + i
        # Replace atom number (positions 15-20)
        new_line = line[:15] + f'{new_num:5d}' + line[20:]
        renumbered_lig.append(new_line)

    # Write combined .gro
    total_atoms = prot_natoms + lig_natoms
    with open(complex_gro, 'w', encoding='utf-8') as f:
        f.write(f'Protein-Ligand Complex\n')
        f.write(f'{total_atoms}\n')
        f.writelines(prot_atoms)
        f.writelines(renumbered_lig)
        f.write(prot_box)

    # Edit topol.top to include ligand
    topol_path = os.path.join(work_dir, 'topol.top')
    if os.path.exists(topol_path):
        with open(topol_path, encoding='utf-8', errors='replace') as f:
            content = f.read()

        # Insert #include "ligand.itp" before [ system ]
        # Also need to include atomtypes from ligand.itp
        # Extract atomtypes from ligand.itp and put in separate file
        _extract_atomtypes(ligand_itp, os.path.join(work_dir, 'ligand_atomtypes.itp'))
        _remove_atomtypes_from_itp(ligand_itp)

        # Insert includes
        insert_point = content.find('[ system ]')
        if insert_point == -1:
            insert_point = content.find('[system]')

        # Insert atomtypes after forcefield include, ligand.itp before [ system ]
        ff_include_end = content.find('.ff/forcefield.itp"')
        if ff_include_end != -1:
            ff_include_end = content.find('\n', ff_include_end) + 1
            content = (content[:ff_include_end] +
                      '\n; Ligand GAFF atom types\n#include "ligand_atomtypes.itp"\n' +
                      content[ff_include_end:])
            # Re-find [ system ] after insertion
            insert_point = content.find('[ system ]')

        if insert_point != -1:
            content = (content[:insert_point] +
                      '; Ligand topology\n#include "ligand.itp"\n\n' +
                      content[insert_point:])

        # Add ligand to [ molecules ]
        content = content.rstrip() + f'\n{resname:20s} 1\n'

        with open(topol_path, 'w', encoding='utf-8') as f:
            f.write(content)

    logger.info("Merged system: %d atoms (%d protein + %d ligand)",
                total_atoms, prot_natoms, lig_natoms)
    return complex_gro, True


def _extract_atomtypes(itp_path, output_path):
    """Extract [ atomtypes ] section from .itp into a separate file."""
    with open(itp_path, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    in_atomtypes = False
    atomtype_lines = []
    for line in lines:
        if '[ atomtypes ]' in line:
            in_atomtypes = True
            atomtype_lines.append(line)
            continue
        if in_atomtypes:
            if line.strip().startswith('[') and 'atomtypes' not in line:
                break
            atomtype_lines.append(line)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(atomtype_lines)


def _remove_atomtypes_from_itp(itp_path):
    """Remove [ atomtypes ] section from .itp (it goes in separate file)."""
    with open(itp_path, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    new_lines = []
    skip = False
    for line in lines:
        if '[ atomtypes ]' in line:
            skip = True
            continue
        if skip and line.strip().startswith('['):
            skip = False
        if not skip:
            new_lines.append(line)

    with open(itp_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)


def solvate_system(work_dir, input_gro, box_dist=DEFAULT_BOX_DIST,
                   box_type=DEFAULT_BOX_TYPE):
    """Box + solvate + add ions. Returns (system_gro, success, error)."""
    # editconf: create box
    cmd = (f'editconf -f "{input_gro}" -o box.gro -c '
           f'-d {box_dist} -bt {box_type}')
    output, ok = run_gmx(cmd, work_dir)
    if not ok:
        return None, False, f'editconf failed: {output[-300:]}'

    # solvate
    cmd = 'solvate -cp box.gro -cs spc216.gro -o solv.gro -p topol.top'
    output, ok = run_gmx(cmd, work_dir)
    if not ok:
        return None, False, f'solvate failed: {output[-300:]}'

    # grompp for genion
    ions_mdp = os.path.join(work_dir, 'ions.mdp')
    _write_template('ions.mdp', ions_mdp)
    cmd = 'grompp -f ions.mdp -c solv.gro -p topol.top -o ions.tpr -maxwarn 5'
    output, ok = run_gmx(cmd, work_dir)
    if not ok:
        return None, False, f'grompp (ions) failed: {output[-300:]}'

    # genion: neutralize system — detect water group name (SOL, Water, etc.)
    water_group = 'SOL'
    ndx_out, _ = run_gmx('make_ndx -f ions.tpr', work_dir, input_text='q\n')
    if ndx_out:
        for grp_name in ['SOL', 'Water', 'SOL_', 'W']:
            if f' {grp_name} ' in ndx_out or f' {grp_name}\n' in ndx_out:
                water_group = grp_name
                break
    cmd = 'genion -s ions.tpr -o system.gro -p topol.top -pname NA -nname CL -neutral'
    output, ok = run_gmx(cmd, work_dir, input_text=f'{water_group}\n')
    if not ok:
        return None, False, f'genion failed: {output[-300:]}'

    system_gro = os.path.join(work_dir, 'system.gro')
    return system_gro, True, None


def create_index(work_dir, system_gro, resname='MOL'):
    """Create index.ndx with proper groups for protein-ligand system.

    Creates groups:
      - Protein_LIG: protein + ligand (for thermostat coupling)
      - Water_and_ions: solvent + ions

    Returns: (ndx_path, tc_grps_string)
    """
    ndx_path = os.path.join(work_dir, 'index.ndx')

    # make_ndx: merge Protein with ligand, create Water_and_ions
    # Commands:
    #   "Protein" | r <resname>   → creates Protein_<resname> group
    #   "Water" | "Ion"           → creates Water_Ion group
    #   q                         → quit
    stdin = (
        f'"Protein" | r {resname}\n'    # → Protein_MOL (group ~19)
        f'"Water" | "Ion"\n'            # → Water_Ion (group ~20)
        f'q\n'
    )
    cmd = f'make_ndx -f "{system_gro}" -o index.ndx'
    output, ok = run_gmx(cmd, work_dir, input_text=stdin)

    if ok and os.path.exists(ndx_path):
        # Read group names to determine tc_grps
        with open(ndx_path, encoding='utf-8', errors='replace') as f:
            groups = [l.strip('[] \n') for l in f if l.startswith('[')]

        # Find the protein+ligand group and water+ions group
        prot_lig = None
        wat_ion = None
        for g in groups:
            if 'Protein' in g and resname in g:
                prot_lig = g
            elif 'Water' in g and 'Ion' in g:
                wat_ion = g
            elif g == 'Water_Ion':
                wat_ion = g

        if not prot_lig:
            prot_lig = f'Protein_{resname}'
        if not wat_ion:
            wat_ion = 'Water_Ion'

        return ndx_path, f'{prot_lig} {wat_ion}'

    # Fallback: generate manually with Python
    _generate_index_python(work_dir, system_gro, resname, ndx_path)
    return ndx_path, f'Protein_{resname} Water_and_ions'


def _generate_index_python(work_dir, gro_file, resname, ndx_path):
    """Generate index.ndx manually by parsing .gro file."""
    protein_atoms = []
    ligand_atoms = []
    solvent_atoms = []

    with open(gro_file, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    natoms = int(lines[1].strip())
    for i, line in enumerate(lines[2:2 + natoms], start=1):
        res = line[5:10].strip()
        if res == resname:
            ligand_atoms.append(i)
        elif res in ('SOL', 'NA', 'CL', 'Na+', 'Cl-'):
            solvent_atoms.append(i)
        else:
            protein_atoms.append(i)

    def write_group(f, name, atoms):
        f.write(f'[ {name} ]\n')
        for j in range(0, len(atoms), 15):
            chunk = atoms[j:j+15]
            f.write(' '.join(str(a) for a in chunk) + '\n')

    with open(ndx_path, 'w', encoding='utf-8') as f:
        write_group(f, f'Protein_{resname}', protein_atoms + ligand_atoms)
        write_group(f, 'Water_and_ions', solvent_atoms)

    logger.info("Generated index.ndx: Protein_LIG=%d, Water_and_ions=%d",
                len(protein_atoms) + len(ligand_atoms), len(solvent_atoms))


def merge_system_cgenff(work_dir, protein_gro, ligand_gro, ligand_itp,
                        ligand_prm=None, resname='LIG'):
    """Merge protein + CGenFF ligand into a single system.

    Unlike GAFF merge, CGenFF uses a .prm file instead of extracted atomtypes.
    The .prm contains CGenFF-specific bonded/nonbonded parameters.

    Returns: (complex_gro, success)
    """
    complex_gro = os.path.join(work_dir, 'complex.gro')

    # Read protein .gro
    with open(protein_gro, encoding='utf-8', errors='replace') as f:
        prot_lines = f.readlines()
    prot_natoms = int(prot_lines[1].strip())
    prot_atoms = prot_lines[2:2 + prot_natoms]
    prot_box = prot_lines[2 + prot_natoms]

    # Read ligand .gro
    with open(ligand_gro, encoding='utf-8', errors='replace') as f:
        lig_lines = f.readlines()
    lig_natoms = int(lig_lines[1].strip())
    lig_atoms = lig_lines[2:2 + lig_natoms]

    # Renumber ligand atoms
    start_num = prot_natoms + 1
    renumbered_lig = []
    for i, line in enumerate(lig_atoms):
        new_num = start_num + i
        new_line = line[:15] + f'{new_num:5d}' + line[20:]
        renumbered_lig.append(new_line)

    # Write combined .gro
    total_atoms = prot_natoms + lig_natoms
    with open(complex_gro, 'w', encoding='utf-8') as f:
        f.write('Protein-Ligand Complex\n')
        f.write(f'{total_atoms}\n')
        f.writelines(prot_atoms)
        f.writelines(renumbered_lig)
        f.write(prot_box)

    # Edit topol.top for CGenFF topology
    topol_path = os.path.join(work_dir, 'topol.top')
    if os.path.exists(topol_path):
        from .cgenff_param import integrate_cgenff_topology
        integrate_cgenff_topology(topol_path, ligand_itp, ligand_prm, resname)

    logger.info("Merged system (CGenFF): %d atoms (%d protein + %d ligand)",
                total_atoms, prot_natoms, lig_natoms)
    return complex_gro, True


def parameterize_ligand_cgenff(work_dir, ligand_pdb, resname='LIG',
                                str_file=None, ff_dir=None):
    """Parameterize ligand using CGenFF for CHARMM36 compatibility.

    Delegates to cgenff_param module. Returns (itp, gro, error).
    Falls back to SOBTOP/GAFF if CGenFF is unavailable.
    """
    from .cgenff_param import parameterize_ligand_cgenff as _cgenff_param

    result = _cgenff_param(work_dir, ligand_pdb, resname,
                           str_file=str_file, ff_dir=ff_dir)

    if 'error' in result:
        # Return error info in the same format as parameterize_ligand
        hint = result.get('hint', '')
        return None, None, f"CGenFF failed: {result['error']}\n{hint}"

    itp = result.get('itp')
    gro = result.get('gro')
    return itp, gro, None


def _write_template(template_name, output_path, **kwargs):
    """Copy MDP template to work dir, optionally substituting values."""
    from .config import TEMPLATE_DIR
    src = TEMPLATE_DIR / template_name
    if not src.exists():
        raise FileNotFoundError(f'Template not found: {src}')

    with open(src, encoding='utf-8', errors='replace') as f:
        content = f.read()

    for key, value in kwargs.items():
        content = content.replace(f'__{key}__', str(value))

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

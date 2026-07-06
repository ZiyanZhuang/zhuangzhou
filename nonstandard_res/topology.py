"""
GROMACS topology assembly for systems with non-standard residues.

Handles:
  - Merging non-standard residue .itp files into topol.top
  - Inserting atom types before force field includes
  - Adding molecules to [ molecules ] section
  - Resolving atom indices for position/distance restraints
  - Creating proper index groups for tc-grps
"""

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def build_topology(detection_result, param_results, topol_path, work_dir, *,
                   restraint_result=None):
    """
    Assemble a complete GROMACS topology with all non-standard residues.

    This modifies an existing topol.top (from pdb2gmx) to include
    non-standard residues, their parameters, and restraint files.

    Args:
        detection_result: Output from detect_nonstandard()
        param_results: Output from parameterize_system()
        topol_path: Path to existing topol.top from pdb2gmx
        work_dir: Working directory
        restraint_result: Output from generate_restraints() (optional)

    Returns:
        {
            'topol_path': str,
            'includes_added': [str],
            'molecules_added': [str],
            'error': str or None,
        }
    """
    result = {
        'topol_path': topol_path,
        'includes_added': [],
        'molecules_added': [],
        'error': None,
    }

    if not os.path.exists(topol_path):
        result['error'] = f'topol.top not found: {topol_path}'
        return result

    with open(topol_path, encoding='utf-8', errors='replace') as f:
        content = f.read()

    # 1. Insert atom type includes (after forcefield.itp include)
    atomtype_includes = []
    for r in param_results.get('residues', []):
        if r.get('atomtypes_itp'):
            rel = _relative_path(r['atomtypes_itp'], work_dir)
            atomtype_includes.append(f'#include "{rel}"')
    for m in param_results.get('metal_ions', []):
        if m.get('atomtypes_itp'):
            rel = _relative_path(m['atomtypes_itp'], work_dir)
            atomtype_includes.append(f'#include "{rel}"')

    if atomtype_includes:
        ff_end = _find_ff_include_end(content)
        if ff_end >= 0:
            insert = '\n; Non-standard residue atom types\n'
            insert += '\n'.join(atomtype_includes) + '\n'
            content = content[:ff_end] + insert + content[ff_end:]
            result['includes_added'].extend(atomtype_includes)

    # 2. Insert residue .itp includes (before [ system ])
    itp_includes = []
    for r in param_results.get('residues', []):
        if r.get('itp_path'):
            rel = _relative_path(r['itp_path'], work_dir)
            itp_includes.append(f'#include "{rel}"')
    for m in param_results.get('metal_ions', []):
        if m.get('itp_path'):
            rel = _relative_path(m['itp_path'], work_dir)
            itp_includes.append(f'#include "{rel}"')

    if itp_includes:
        sys_pos = content.find('[ system ]')
        if sys_pos >= 0:
            insert = '\n; Non-standard residue topologies\n'
            insert += '\n'.join(itp_includes) + '\n\n'
            content = content[:sys_pos] + insert + content[sys_pos:]
            result['includes_added'].extend(itp_includes)

    # 3. Insert restraint includes
    if restraint_result:
        inserts = restraint_result.get('topol_inserts', [])
        if inserts:
            sys_pos = content.find('[ system ]')
            if sys_pos >= 0:
                insert = '\n; Restraints for non-standard residues\n'
                insert += '\n'.join(inserts) + '\n\n'
                content = content[:sys_pos] + insert + content[sys_pos:]

    # 4. Add molecules to [ molecules ] section
    molecules_to_add = []
    for r in param_results.get('residues', []):
        rn = r.get('resname')
        if rn:
            molecules_to_add.append(rn)
    for m in param_results.get('metal_ions', []):
        rn = m.get('resname')
        if rn:
            molecules_to_add.append(rn)

    # Count occurrences from detection result
    nonstandard = detection_result.get('nonstandard_residues', [])
    metals = detection_result.get('metal_ions', [])

    for rn in molecules_to_add:
        # Count how many of this residue exist
        count = sum(1 for r in nonstandard if r['resname'] == rn)
        count += sum(1 for m in metals
                     if f'{m["element"].upper()}{m["charge"]}' == rn
                     or m['resname'] == rn)
        if count == 0:
            count = 1
        content = content.rstrip() + f'\n{rn:20s} {count}\n'
        result['molecules_added'].append(f'{rn} {count}')

    # Write modified topology
    with open(topol_path, 'w', encoding='utf-8') as f:
        f.write(content)

    logger.info("Topology updated: %d includes, %d molecules added",
                len(result['includes_added']), len(result['molecules_added']))
    return result


def merge_itp_into_topol(itp_path, topol_path, resname, work_dir, *,
                         atomtypes_itp=None, count=1):
    """
    Simple helper: merge a single .itp file into existing topol.top.

    This is the minimal version for adding one non-standard residue.
    For complex systems, use build_topology() instead.

    Args:
        itp_path: Path to the .itp file
        topol_path: Path to topol.top
        resname: Residue name for [ molecules ]
        work_dir: Working directory
        atomtypes_itp: Optional atom types .itp path
        count: Number of this molecule to add
    """
    if not os.path.exists(topol_path):
        return {'error': f'topol.top not found: {topol_path}'}

    with open(topol_path, encoding='utf-8', errors='replace') as f:
        content = f.read()

    # Insert atomtypes after FF include
    if atomtypes_itp and os.path.exists(atomtypes_itp):
        rel = _relative_path(atomtypes_itp, work_dir)
        ff_end = _find_ff_include_end(content)
        if ff_end >= 0:
            content = (content[:ff_end] +
                       f'\n; {resname} atom types\n#include "{rel}"\n' +
                       content[ff_end:])

    # Insert itp before [ system ]
    rel = _relative_path(itp_path, work_dir)
    sys_pos = content.find('[ system ]')
    if sys_pos >= 0:
        content = (content[:sys_pos] +
                   f'; {resname} topology\n#include "{rel}"\n\n' +
                   content[sys_pos:])

    # Add to [ molecules ]
    content = content.rstrip() + f'\n{resname:20s} {count}\n'

    with open(topol_path, 'w', encoding='utf-8') as f:
        f.write(content)

    return {'topol_path': topol_path, 'error': None}


def merge_gro_coordinates(system_gro, residue_gros, output_gro):
    """
    Append non-standard residue coordinates to system .gro file.

    Args:
        system_gro: Existing system .gro (protein after pdb2gmx)
        residue_gros: List of .gro files for non-standard residues
        output_gro: Output merged .gro path

    Returns: output_gro path or None on error
    """
    # Read system
    with open(system_gro, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    title = lines[0]
    natoms = int(lines[1].strip())
    atom_lines = lines[2:2 + natoms]
    box_line = lines[2 + natoms] if len(lines) > 2 + natoms else '   0.00000   0.00000   0.00000\n'

    # Append each residue
    for gro in residue_gros:
        if not os.path.exists(gro):
            continue
        with open(gro, encoding='utf-8', errors='replace') as f:
            g_lines = f.readlines()
        g_natoms = int(g_lines[1].strip())
        g_atoms = g_lines[2:2 + g_natoms]

        # Renumber atoms
        start = natoms + 1
        for i, line in enumerate(g_atoms):
            if len(line) >= 20:
                new_num = start + i
                line = line[:15] + f'{new_num:5d}' + line[20:]
            atom_lines.append(line)
        natoms += g_natoms

    # Write merged
    with open(output_gro, 'w', encoding='utf-8') as f:
        f.write(title)
        f.write(f'{natoms}\n')
        f.writelines(atom_lines)
        f.write(box_line)

    logger.info("Merged coordinates: %d total atoms -> %s", natoms, output_gro)
    return output_gro


def create_index_with_nonstandard(system_gro, detection_result, work_dir,
                                  gmx_bin=None):
    """
    Create index.ndx with proper groups for metalloprotein systems.

    Groups created:
      - Protein_Cofactor: protein + all non-standard residues
      - Metal: all metal ions
      - Water_Ion: solvent + counter ions

    Returns: (ndx_path, tc_grps_string)
    """
    ndx_path = os.path.join(work_dir, 'index.ndx')

    nonstandard = detection_result.get('nonstandard_residues', [])
    metals = detection_result.get('metal_ions', [])

    # Collect all non-standard residue names
    ns_resnames = set()
    for r in nonstandard:
        ns_resnames.add(r['resname'])
    for m in metals:
        ns_resnames.add(m['resname'])

    # Try using gmx make_ndx
    if gmx_bin and os.path.exists(gmx_bin):
        return _gmx_make_ndx(system_gro, ns_resnames, ndx_path, gmx_bin, work_dir)

    # Fallback: Python-based index generation
    return _python_make_ndx(system_gro, ns_resnames, ndx_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_ff_include_end(content):
    """Find the end of the forcefield.itp #include line."""
    match = re.search(r'#include\s+".*forcefield\.itp"\s*\n', content)
    if match:
        return match.end()
    return -1


def _relative_path(abs_path, base_dir):
    """Get relative path from base_dir to abs_path."""
    try:
        return os.path.relpath(abs_path, base_dir)
    except ValueError:
        # Different drives on Windows
        return abs_path


def _gmx_make_ndx(system_gro, ns_resnames, ndx_path, gmx_bin, work_dir):
    """Create index using gmx make_ndx."""
    import subprocess
    from .config import POSRE_FC_METAL

    # Build make_ndx commands
    # Merge protein with all non-standard residues
    ns_list = ' '.join(f'r {rn}' for rn in ns_resnames)
    stdin = f'"Protein" | {ns_list}\n' if ns_resnames else ''
    stdin += '"Water" | "Ion"\nq\n'

    cmd = f'"{gmx_bin}" make_ndx -f "{system_gro}" -o "{ndx_path}"'
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            input=stdin, timeout=30, cwd=work_dir,
        )
    except Exception as e:
        logger.warning("gmx make_ndx failed: %s, using Python fallback", e)
        return _python_make_ndx(system_gro, ns_resnames, ndx_path)

    if os.path.exists(ndx_path):
        # Determine group names
        with open(ndx_path, encoding='utf-8', errors='replace') as f:
            groups = [l.strip('[] \n') for l in f if l.startswith('[')]

        prot_ns = None
        wat_ion = None
        for g in groups:
            if 'Protein' in g and any(rn in g for rn in ns_resnames):
                prot_ns = g
            elif 'Water' in g and 'Ion' in g:
                wat_ion = g

        prot_ns = prot_ns or 'Protein'
        wat_ion = wat_ion or 'Water_Ion'
        return ndx_path, f'{prot_ns} {wat_ion}'

    return _python_make_ndx(system_gro, ns_resnames, ndx_path)


def _python_make_ndx(system_gro, ns_resnames, ndx_path):
    """Generate index.ndx by parsing .gro file."""
    protein_atoms = []
    ns_atoms = []
    solvent_atoms = []

    SOLVENT = {'SOL', 'HOH', 'WAT', 'NA', 'CL', 'Na+', 'Cl-', 'K', 'K+'}

    with open(system_gro, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    natoms = int(lines[1].strip())
    for i, line in enumerate(lines[2:2 + natoms], start=1):
        res = line[5:10].strip()
        if res in ns_resnames:
            ns_atoms.append(i)
        elif res in SOLVENT:
            solvent_atoms.append(i)
        else:
            protein_atoms.append(i)

    def write_group(f, name, atoms):
        f.write(f'[ {name} ]\n')
        for j in range(0, len(atoms), 15):
            chunk = atoms[j:j + 15]
            f.write(' '.join(str(a) for a in chunk) + '\n')

    with open(ndx_path, 'w', encoding='utf-8') as f:
        write_group(f, 'Protein', protein_atoms)
        write_group(f, 'Non-Standard', ns_atoms)
        write_group(f, 'Protein_NS', protein_atoms + ns_atoms)
        write_group(f, 'Water_Ion', solvent_atoms)

    tc_grps = 'Protein_NS Water_Ion'
    logger.info("Generated index: Protein=%d, Non-Standard=%d, Water_Ion=%d",
                len(protein_atoms), len(ns_atoms), len(solvent_atoms))
    return ndx_path, tc_grps

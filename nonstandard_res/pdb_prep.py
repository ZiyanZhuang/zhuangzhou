"""
Metalloprotein PDB preparation for GROMACS MD.

Handles the critical preprocessing needed before pdb2gmx:
  - Download PDB from RCSB if given a PDB ID
  - Remove alternate conformations (keep A)
  - Remove crystallographic waters (optional)
  - Detect HEME and metal coordination
  - Rename residues to match force field conventions (CHARMM27: HEM->HEME)
  - Ensure coordinated Cys/His naming for specbond.dat compatibility
  - Write clean PDB ready for pdb2gmx

Reference: GROMACS specbond.dat defines special bonds:
  CYS SG -- HEM FE  (0.25 nm) -> CYS2 + HEME
  HIS NE2 -- HEM FE (0.20 nm) -> HIS1 + HEME
  MET SD -- HEM FE  (0.24 nm) -> MET  + HEME
"""

import logging
import math
import os
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Heme residue names recognized in PDB files
# NOTE: Do NOT rename these before pdb2gmx — specbond.dat expects original
# PDB names (e.g., HEM) and does the renaming automatically (HEM -> HEME)
HEME_RESNAMES = {'HEM', 'HEME', 'HEC', 'HEA', 'HEO2'}

# Standard amino acid names that pdb2gmx handles
STANDARD_AA = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'HID', 'HIE', 'HIP', 'HSD', 'HSE', 'HSP', 'CYX', 'CYM', 'CYS2',
    'ASH', 'GLH', 'LYN', 'LSN', 'ASPP', 'GLUP',
}

# Waters and simple ions to remove
REMOVE_RESIDUES = {'HOH', 'WAT', 'DOD', 'TIP', 'SPC'}

# CHARMM27 HEME atom name mapping (PDB standard -> CHARMM27)
# Most names are the same, but some PDBs use different conventions
HEME_ATOM_RENAME = {
    # Common alternative names in PDB files
    'N A': 'NA', 'N B': 'NB', 'N C': 'NC', 'N D': 'ND',
    'FE2': 'FE',  # Some PDBs use FE2 for Fe2+
    'FE1': 'FE',
}


def prepare_metalloprotein_pdb(pdb_path, work_dir, *,
                                keep_waters=False,
                                keep_ions=True,
                                keep_ligands=False,
                                extract_ligands=False,
                                remove_altloc=True,
                                ff='charmm27'):
    """
    Clean and prepare a metalloprotein PDB for GROMACS simulation.

    Args:
        pdb_path: Input PDB file path (or 4-letter PDB ID to download)
        work_dir: Output directory
        keep_waters: Keep crystallographic waters
        keep_ions: Keep metal ions (NA, CL, etc.)
        keep_ligands: Keep non-heme HETATM ligands in the clean PDB
        extract_ligands: Extract non-heme ligands to separate PDB files
                         (for SOBTOP/GAFF parameterization). Implies keep_ligands=False.
        remove_altloc: Remove alternate conformations (keep A)
        ff: Target force field (affects residue renaming)

    Returns:
        {
            'clean_pdb': str,          # Path to cleaned PDB
            'original_pdb': str,       # Path to original/downloaded PDB
            'heme_detected': bool,
            'heme_residues': list,     # [{resname, chain, resid, fe_coord}]
            'coordination': list,      # [{metal, protein_res, distance, bond_type}]
            'renamed_residues': list,  # What was renamed
            'warnings': list,
            'ligands': list,           # [{resname, pdb_path, n_atoms}] if extract_ligands
            'error': str or None,
        }
    """
    work_dir = os.path.abspath(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    result = {
        'clean_pdb': None,
        'original_pdb': None,
        'heme_detected': False,
        'heme_residues': [],
        'coordination': [],
        'renamed_residues': [],
        'warnings': [],
        'ligands': [],
        'error': None,
    }

    # Handle PDB ID input
    if len(pdb_path) == 4 and pdb_path.isalnum() and not os.path.exists(pdb_path):
        pdb_path = _download_pdb(pdb_path, work_dir)
        if not pdb_path:
            result['error'] = f'Failed to download PDB'
            return result

    if not os.path.isfile(pdb_path):
        result['error'] = f'PDB file not found: {pdb_path}'
        return result

    result['original_pdb'] = pdb_path

    # Parse PDB
    lines = _parse_pdb_lines(pdb_path)
    if not lines:
        result['error'] = 'No ATOM/HETATM lines found in PDB'
        return result

    # Step 1: Remove alternate conformations
    if remove_altloc:
        lines, n_removed = _remove_altloc(lines)
        if n_removed > 0:
            result['warnings'].append(f'Removed {n_removed} alternate conformations')

    # Step 2: Remove waters (optionally)
    if not keep_waters:
        lines = [l for l in lines if _get_resname(l) not in REMOVE_RESIDUES]

    # Step 3: Handle non-heme ligands
    heme_names = HEME_RESNAMES
    ion_names = {'NA', 'CL', 'K', 'MG', 'CA', 'ZN', 'FE', 'MN', 'CU',
                 'NA+', 'CL-', 'K+', 'MG2', 'CA2', 'ZN2', 'FE2', 'FE3',
                 'CO', 'NI', 'IOD', 'BR'}

    if extract_ligands or not keep_ligands:
        extracted_ligands = {}  # resname -> [lines]
        removed_ligands = set()
        new_lines = []
        for l in lines:
            if l[:6].strip() == 'HETATM':
                rn = _get_resname(l)
                if rn in heme_names or (keep_ions and rn in ion_names):
                    new_lines.append(l)
                elif rn in STANDARD_AA:
                    new_lines.append(l)
                else:
                    if extract_ligands:
                        extracted_ligands.setdefault(rn, []).append(l)
                    removed_ligands.add(rn)
            else:
                new_lines.append(l)
        lines = new_lines

        # Write extracted ligand PDB files
        if extract_ligands and extracted_ligands:
            for lig_resname, lig_lines in extracted_ligands.items():
                lig_pdb = os.path.join(work_dir, f'ligand_{lig_resname}.pdb')
                with open(lig_pdb, 'w', encoding='utf-8') as f:
                    f.writelines(lig_lines)
                    f.write('END\n')
                n_atoms = len([l for l in lig_lines
                              if l[:6].strip() == 'HETATM'])
                result['ligands'].append({
                    'resname': lig_resname,
                    'pdb_path': lig_pdb,
                    'n_atoms': n_atoms,
                })
                logger.info("Extracted ligand %s: %d atoms -> %s",
                            lig_resname, n_atoms, lig_pdb)

        if removed_ligands:
            action = 'Extracted' if extract_ligands else 'Removed'
            result['warnings'].append(
                f'{action} non-heme ligands: {", ".join(sorted(removed_ligands))}')

    # Step 4: Detect HEME and coordination
    heme_info = _detect_heme(lines)
    result['heme_detected'] = len(heme_info['heme_residues']) > 0
    result['heme_residues'] = heme_info['heme_residues']
    result['coordination'] = heme_info['coordination']

    # Step 5: Rename residues for target FF
    if ff and ff.startswith(('charmm', 'CHARMM')):
        lines, renames = _rename_for_charmm27(lines, heme_info)
        result['renamed_residues'] = renames

    # Step 6: Fix HEME atom names
    lines = _fix_heme_atom_names(lines)

    # Step 7: Ensure HEME records are HETATM (required by pdb2gmx)
    lines = _ensure_hetatm(lines)

    # Step 8: Write clean PDB
    clean_pdb = os.path.join(work_dir, 'clean.pdb')
    _write_clean_pdb(lines, clean_pdb)
    result['clean_pdb'] = clean_pdb

    # Summary logging
    n_atoms = len([l for l in lines if l[:6].strip() in ('ATOM', 'HETATM')])
    n_heme = len(result['heme_residues'])
    n_coord = len(result['coordination'])
    logger.info("PDB prep: %d atoms, %d HEME, %d coordination bonds -> %s",
                n_atoms, n_heme, n_coord, clean_pdb)

    return result


def _download_pdb(pdb_id, work_dir):
    """Download PDB from RCSB."""
    import urllib.request
    url = f'https://files.rcsb.org/download/{pdb_id.upper()}.pdb'
    out_path = os.path.join(work_dir, f'{pdb_id.upper()}.pdb')
    try:
        urllib.request.urlretrieve(url, out_path)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
            logger.info("Downloaded PDB %s -> %s", pdb_id, out_path)
            return out_path
    except Exception as e:
        logger.error("Failed to download PDB %s: %s", pdb_id, e)
    return None


def _parse_pdb_lines(pdb_path):
    """Read PDB and return ATOM/HETATM/TER/END lines."""
    lines = []
    with open(pdb_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            record = line[:6].strip()
            if record in ('ATOM', 'HETATM', 'TER', 'END', 'CONECT'):
                lines.append(line)
    return lines


def _get_resname(line):
    """Extract residue name from PDB line."""
    if len(line) < 20:
        return ''
    return line[17:20].strip()


def _get_resid(line):
    """Extract residue ID from PDB line."""
    try:
        return int(line[22:26].strip())
    except (ValueError, IndexError):
        return -1


def _get_chain(line):
    """Extract chain from PDB line."""
    if len(line) < 22:
        return ''
    return line[21].strip()


def _get_atom_name(line):
    """Extract atom name from PDB line."""
    return line[12:16].strip()


def _get_coords(line):
    """Extract x,y,z coordinates from PDB line."""
    try:
        return (float(line[30:38]), float(line[38:46]), float(line[46:54]))
    except (ValueError, IndexError):
        return None


def _remove_altloc(lines):
    """Remove alternate conformations, keeping altloc A (or blank)."""
    new_lines = []
    n_removed = 0
    for line in lines:
        if line[:6].strip() not in ('ATOM', 'HETATM'):
            new_lines.append(line)
            continue
        altloc = line[16] if len(line) > 16 else ' '
        if altloc in (' ', 'A', '1'):
            # Clear altloc indicator
            if altloc != ' ':
                line = line[:16] + ' ' + line[17:]
            new_lines.append(line)
        else:
            n_removed += 1
    return new_lines, n_removed


def _detect_heme(lines):
    """Detect HEME residues and their coordination bonds."""
    result = {
        'heme_residues': [],
        'coordination': [],
        'fe_atoms': [],
    }

    # Collect all atoms by residue key
    atoms_by_res = {}
    all_atoms = []
    for line in lines:
        if line[:6].strip() not in ('ATOM', 'HETATM'):
            continue
        resname = _get_resname(line)
        chain = _get_chain(line)
        resid = _get_resid(line)
        atom_name = _get_atom_name(line)
        coords = _get_coords(line)
        if not coords:
            continue
        atom = {
            'name': atom_name, 'resname': resname, 'chain': chain,
            'resid': resid, 'coords': coords, 'line': line,
        }
        key = (resname, chain, resid)
        atoms_by_res.setdefault(key, []).append(atom)
        all_atoms.append(atom)

    # Find HEME residues and their Fe atoms
    fe_atoms = []
    for (resname, chain, resid), atoms in atoms_by_res.items():
        if resname not in HEME_RESNAMES:
            continue
        fe_coord = None
        for a in atoms:
            if a['name'] in ('FE', 'FE1', 'FE2'):
                fe_coord = a['coords']
                fe_atoms.append(a)
                break
        result['heme_residues'].append({
            'resname': resname, 'chain': chain, 'resid': resid,
            'n_atoms': len(atoms), 'fe_coord': fe_coord,
        })

    result['fe_atoms'] = fe_atoms

    # Detect coordination bonds (Fe -- protein residues)
    for fe in fe_atoms:
        fx, fy, fz = fe['coords']

        for (resname, chain, resid), atoms in atoms_by_res.items():
            if resname not in STANDARD_AA:
                continue

            for a in atoms:
                # CYS SG coordination (Fe-S, ~2.3 A)
                if resname in ('CYS', 'CYS2', 'CYM') and a['name'] == 'SG':
                    d = _dist(fx, fy, fz, *a['coords'])
                    if d <= 3.0:  # generous cutoff for detection
                        result['coordination'].append({
                            'metal_atom': fe['name'],
                            'metal_resid': fe['resid'],
                            'metal_chain': fe['chain'],
                            'protein_res': resname,
                            'protein_resid': resid,
                            'protein_chain': chain,
                            'protein_atom': 'SG',
                            'distance': round(d, 3),
                            'bond_type': 'Fe-S(Cys)',
                            'specbond_cutoff': 2.5,  # specbond.dat: 0.25 nm
                        })

                # HIS NE2/ND1 coordination (Fe-N, ~2.1 A)
                if resname in ('HIS', 'HSD', 'HSE', 'HSP', 'HID', 'HIE', 'HIP'):
                    if a['name'] in ('NE2', 'ND1'):
                        d = _dist(fx, fy, fz, *a['coords'])
                        if d <= 2.8:
                            result['coordination'].append({
                                'metal_atom': fe['name'],
                                'metal_resid': fe['resid'],
                                'metal_chain': fe['chain'],
                                'protein_res': resname,
                                'protein_resid': resid,
                                'protein_chain': chain,
                                'protein_atom': a['name'],
                                'distance': round(d, 3),
                                'bond_type': f'Fe-N({a["name"]})',
                                'specbond_cutoff': 2.0,
                            })

                # MET SD coordination (Fe-S, ~2.3 A)
                if resname == 'MET' and a['name'] == 'SD':
                    d = _dist(fx, fy, fz, *a['coords'])
                    if d <= 2.8:
                        result['coordination'].append({
                            'metal_atom': fe['name'],
                            'metal_resid': fe['resid'],
                            'metal_chain': fe['chain'],
                            'protein_res': resname,
                            'protein_resid': resid,
                            'protein_chain': chain,
                            'protein_atom': 'SD',
                            'distance': round(d, 3),
                            'bond_type': 'Fe-S(Met)',
                            'specbond_cutoff': 2.4,
                        })

                # TYR OH coordination (Fe-O, rare)
                if resname == 'TYR' and a['name'] == 'OH':
                    d = _dist(fx, fy, fz, *a['coords'])
                    if d <= 2.5:
                        result['coordination'].append({
                            'metal_atom': fe['name'],
                            'metal_resid': fe['resid'],
                            'metal_chain': fe['chain'],
                            'protein_res': resname,
                            'protein_resid': resid,
                            'protein_chain': chain,
                            'protein_atom': 'OH',
                            'distance': round(d, 3),
                            'bond_type': 'Fe-O(Tyr)',
                            'specbond_cutoff': 2.5,
                        })

    return result


def _rename_for_charmm27(lines, heme_info):
    """Rename residues to match CHARMM27 conventions."""
    renames = []

    # Build set of residues that coordinate Fe
    coord_residues = set()
    for c in heme_info['coordination']:
        key = (c['protein_res'], c['protein_chain'], c['protein_resid'])
        coord_residues.add((key, c['bond_type'], c['protein_atom']))

    new_lines = []
    for line in lines:
        if line[:6].strip() not in ('ATOM', 'HETATM'):
            new_lines.append(line)
            continue

        resname = _get_resname(line)
        chain = _get_chain(line)
        resid = _get_resid(line)
        new_resname = None

        # Rename HEM -> HEME: both CHARMM27 and CHARMM36 rtp use 'HEME'.
        # specbond.dat expects 'HEM' but GROMACS 2025 may check rtp BEFORE
        # specbond renaming, causing "Residue 'HEM' not found" errors.
        # Our heme_topology.py handles Fe-S bond injection regardless.
        if resname == 'HEM':
            new_resname = 'HEME'

        # Coordinated HIS -> HSD (NE2 coordinates Fe -> delta nitrogen is protonated)
        # In CHARMM27, HSD = His with HD1 on ND1 (delta protonated)
        # When NE2 coordinates Fe, ND1 carries the proton -> HSD
        # When ND1 coordinates Fe, NE2 carries the proton -> HSE
        for (key, bond_type, patom) in coord_residues:
            r, c, ri = key
            if resname in ('HIS', 'HSD', 'HSE', 'HSP', 'HID', 'HIE', 'HIP'):
                if chain == c and resid == ri:
                    if patom == 'NE2':
                        new_resname = 'HSD'  # ND1 protonated, NE2 to Fe
                    elif patom == 'ND1':
                        new_resname = 'HSE'  # NE2 protonated, ND1 to Fe

        if new_resname and new_resname != resname:
            # Pad/truncate to 3 or 4 chars matching PDB column width (17:20 = 3 chars, but 17:21 = 4)
            # PDB format: columns 18-20 (0-indexed 17-19) for residue name (right-justified in 17:20)
            padded = f'{new_resname:>4s}' if len(new_resname) > 3 else f'{new_resname:>3s}'
            if len(new_resname) <= 3:
                line = line[:17] + f'{new_resname:>3s}' + line[20:]
            else:
                # 4-char resname: use columns 17-20
                line = line[:17] + f'{new_resname:<4s}' + line[21:]

            rename_key = f'{resname}_{resid}_{chain}'
            rename_entry = f'{resname} -> {new_resname} (chain {chain}, resid {resid})'
            if rename_entry not in renames:
                renames.append(rename_entry)

        new_lines.append(line)

    return new_lines, renames


def _fix_heme_atom_names(lines):
    """Fix HEME atom names to match CHARMM27 conventions."""
    new_lines = []
    for line in lines:
        if line[:6].strip() not in ('ATOM', 'HETATM'):
            new_lines.append(line)
            continue

        resname = _get_resname(line)
        if resname not in ('HEME', 'HEM', 'HEC'):
            new_lines.append(line)
            continue

        atom_name = _get_atom_name(line)
        if atom_name in HEME_ATOM_RENAME:
            new_name = HEME_ATOM_RENAME[atom_name]
            # PDB atom name format: columns 13-16 (0-indexed 12-15)
            line = line[:12] + f'{new_name:>4s}' + line[16:]

        new_lines.append(line)
    return new_lines


def _ensure_hetatm(lines):
    """Ensure HEME and metal residues use HETATM records."""
    non_standard = {'HEME', 'HEM', 'HEC', 'HEA', 'HEO2'}
    new_lines = []
    for line in lines:
        if line[:4] == 'ATOM':
            resname = _get_resname(line)
            if resname in non_standard:
                line = 'HETATM' + line[6:]
        new_lines.append(line)
    return new_lines


def _write_clean_pdb(lines, output_path):
    """Write cleaned PDB with proper formatting.

    Inserts TER between protein and HEME/cofactor to satisfy GROMACS 2025's
    strict chain-type consistency checks (needed for CHARMM36 where HEME is
    in merged.rtp and classified as 'Other', not 'Protein').
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write header
        f.write('REMARK   Prepared by zhuangzhou metalloprotein PDB prep\n')
        f.write('REMARK   Force field: CHARMM (native HEME support)\n')

        prev_is_protein = False
        for line in lines:
            record = line[:6].strip()
            if record in ('ATOM', 'HETATM'):
                resname = line[17:20].strip()
                cur_is_protein = (record == 'ATOM' or
                                  resname in STANDARD_AA)
                cur_is_heme = resname in HEME_RESNAMES
                # Insert TER when transitioning from protein to HEME
                if prev_is_protein and cur_is_heme:
                    f.write('TER\n')
                prev_is_protein = cur_is_protein
            elif record == 'TER':
                prev_is_protein = False
            f.write(line if line.endswith('\n') else line + '\n')

        # Ensure END record
        if not any(l.startswith('END') for l in lines):
            f.write('END\n')


def _dist(x1, y1, z1, x2, y2, z2):
    """Euclidean distance."""
    return math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

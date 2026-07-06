"""
QM Cluster Extraction - Build QM model from enzyme active site.

Implements the quantum chemical cluster approach (Himo, Siegbahn):
  1. Read PDB (from MD snapshot or crystal structure)
  2. Identify reactive center (substrate/cofactor from PLIP or user-specified)
  3. Select residues within cutoff radius
  4. Truncate amino acids at CA-CB bond:
     - Keep sidechain (CB and beyond)
     - Remove backbone (N, CA, C, O, H, HA)
     - Cap dangling CB with H along CB->CA direction at 1.09 A
  5. Freeze capping H atoms (backbone anchor points)
  6. Generate XYZ + geomeTRIC constraints
  7. Assign charge and spin

References:
  Himo, F. JACS 2017, 139, 6780-6786 (cluster approach review)
  Siegbahn, P.E.M.; Himo, F. WCMS 2011, 1, 323-336
"""

import logging
import math
import os
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Standard amino acid net charges at pH 7 (for QM cluster)
RESIDUE_CHARGE = {
    'ARG': 1, 'LYS': 1,
    'ASP': -1, 'GLU': -1,
    'HIS': 0,  # neutral by default; HIP(+1) handled separately
}

# Common metal ion charges
METAL_CHARGE = {
    'FE': 2, 'FE2': 2, 'FE3': 3,
    'ZN': 2, 'MG': 2, 'MN': 2, 'CA': 2,
    'CU': 2, 'CO': 2, 'NI': 2,
    'NA': 1, 'K': 1, 'CL': -1,
}

# Backbone atom names to REMOVE during truncation
BACKBONE_ATOMS = {
    'N', 'CA', 'C', 'O', 'OXT',
    'H', 'H1', 'H2', 'H3',       # N-terminal / amide H
    'HA', 'HA2', 'HA3',           # CA hydrogen
    'HN', 'HT1', 'HT2', 'HT3',  # various naming conventions
}

# Standard amino acids
STD_AMINO_ACIDS = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
    'HIS', 'HID', 'HIE', 'HIP',
    'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO',
    'SER', 'THR', 'TRP', 'TYR', 'VAL',
}

# Ion / small molecule residue names
ION_RESIDUES = {'NA', 'CL', 'MG', 'ZN', 'CA', 'FE', 'MN', 'CU', 'CO', 'NI', 'K'}
WATER_RESIDUES = {'HOH', 'WAT', 'TIP', 'TIP3', 'SOL'}

# C-H capping bond length (Angstrom)
CAP_CH_LENGTH = 1.09


def extract_qm_cluster(
    pdb_path: str,
    output_dir: str,
    center_residues: List[str] = None,
    center_atoms: List[Tuple[str, int]] = None,
    plip_result: dict = None,
    cutoff: float = 5.0,
    include_waters: bool = True,
    max_waters: int = 5,
    charge: int = None,
    spin: int = None,
    chain: str = 'A',
    keep_backbone: List[Tuple[str, int]] = None,
    keep_residues: List[Tuple[str, int]] = None,
    keep_waters: List[int] = None,
) -> dict:
    """
    Extract QM cluster from PDB around active site.

    Implements standard cluster approach with CA-CB truncation:
    - Amino acids: keep sidechain (CB onward), cap CB with H
    - Substrate/cofactor (HETATM): keep intact
    - Waters: keep up to max_waters closest to center
    - Ions: keep as-is

    Args:
        pdb_path: Input PDB file
        output_dir: Output directory for XYZ, constraints
        center_residues: Residue names defining center (e.g. ['HEM', 'CYS'])
        center_atoms: Specific (chain, resnum) pairs for center
        plip_result: PLIP analysis result (auto-extract interacting residues)
        cutoff: Radius in Angstrom around center
        include_waters: Include crystallographic waters
        max_waters: Maximum waters to include
        charge: System charge (auto-computed if None)
        spin: 2S value (0=singlet, auto if None)
        chain: Chain ID
        keep_backbone: List of (chain, resnum) where backbone should be kept
                      (e.g. when backbone N-H or C=O participates in catalysis)

    Returns:
        dict with xyz_path, constraints_path, charge, spin, n_atoms, residues
    """
    if not os.path.exists(pdb_path):
        return {'error': f'PDB not found: {pdb_path}'}

    os.makedirs(output_dir, exist_ok=True)

    # Parse PDB
    atoms = _parse_pdb(pdb_path)
    if not atoms:
        return {'error': 'No atoms found in PDB'}

    # Determine center atoms
    center_coords = _find_center_atoms(atoms, center_residues, center_atoms,
                                        plip_result, chain)
    if not center_coords:
        return {'error': 'No center atoms found',
                'hint': 'Specify center_residues, center_atoms, or provide plip_result'}

    # Select residues within cutoff
    qm_residue_keys = _select_residues_in_cutoff(atoms, center_coords, cutoff,
                                                  include_waters)

    # Limit waters by distance to center
    if include_waters and max_waters > 0:
        qm_residue_keys = _limit_waters(atoms, center_coords, qm_residue_keys,
                                         max_waters, keep_waters)
    
    # Manually add explicitly kept residues and waters
    if keep_residues:
        kr_set = set(keep_residues)
        for atom in atoms:
            if (atom['chain'], atom['resnum']) in kr_set:
                qm_residue_keys.add((atom['chain'], atom['resnum'], atom['resname']))
    if keep_waters:
        kw_set = set(keep_waters)
        for atom in atoms:
            if atom['resname'] in WATER_RESIDUES and atom['resnum'] in kw_set:
                qm_residue_keys.add((atom['chain'], atom['resnum'], atom['resname']))

    # Group atoms by residue
    residue_atoms = {}
    for atom in atoms:
        rk = (atom['chain'], atom['resnum'], atom['resname'])
        if rk in qm_residue_keys:
            if rk not in residue_atoms:
                residue_atoms[rk] = []
            residue_atoms[rk].append(atom)

    # Build QM model: truncate amino acids, cap with H, keep others intact
    keep_bb = set(keep_backbone) if keep_backbone else set()
    qm_model_atoms = []  # list of (element, x, y, z, freeze, source_info)
    cap_h_indices = []    # indices of capping H atoms (to freeze)

    for rk in sorted(qm_residue_keys):
        res_chain, resnum, resname = rk
        res_atoms = residue_atoms.get(rk, [])

        if resname in STD_AMINO_ACIDS:
            if resname == 'GLY':
                # GLY has no sidechain - skip unless backbone is needed
                if (res_chain, resnum) in keep_bb:
                    for a in res_atoms:
                        elem = _get_element(a)
                        qm_model_atoms.append((elem, a['x'], a['y'], a['z'],
                                               False, rk))
                continue

            if (res_chain, resnum) in keep_bb:
                # Keep entire residue (backbone participates in catalysis) and cap peptide bonds
                _truncate_and_cap_backbone(atoms, res_atoms, resname, rk, qm_model_atoms, cap_h_indices)
            else:
                # Standard truncation: CA-CB cut
                _truncate_and_cap(res_atoms, resname, rk, qm_model_atoms, cap_h_indices)

        elif resname in WATER_RESIDUES:
            # Water: keep all atoms
            for a in res_atoms:
                elem = _get_element(a)
                qm_model_atoms.append((elem, a['x'], a['y'], a['z'],
                                       False, rk))

        elif resname in ION_RESIDUES:
            # Ion: single atom
            for a in res_atoms:
                elem = _get_element(a)
                qm_model_atoms.append((elem, a['x'], a['y'], a['z'],
                                       False, rk))

        else:
            # Non-standard residue (substrate, cofactor, HEM, etc.): keep intact
            for a in res_atoms:
                elem = _get_element(a)
                qm_model_atoms.append((elem, a['x'], a['y'], a['z'],
                                       False, rk))

    n_atoms = len(qm_model_atoms)
    if n_atoms == 0:
        return {'error': 'No atoms in QM model after truncation'}

    # Write XYZ
    xyz_path = os.path.join(output_dir, 'qm_cluster.xyz')
    with open(xyz_path, 'w', encoding='utf-8') as f:
        f.write(f'{n_atoms}\n')
        f.write(f'QM cluster from {os.path.basename(pdb_path)}, '
                f'cutoff={cutoff}A, {len(qm_residue_keys)} residues, '
                f'{len(cap_h_indices)} capping H\n')
        for elem, x, y, z, freeze, rk in qm_model_atoms:
            f.write(f'{elem:>2s}  {x:12.6f}  {y:12.6f}  {z:12.6f}\n')

    # Write constraints file for geomeTRIC (freeze capping H atoms)
    constraints_path = os.path.join(output_dir, 'constraints.txt')
    with open(constraints_path, 'w', encoding='utf-8') as f:
        if cap_h_indices:
            f.write('$freeze\n')
            for idx in cap_h_indices:
                f.write(f'xyz {idx}\n')
            f.write('$end\n')

    # Compute charge and spin
    if charge is None:
        charge = _compute_charge(qm_residue_keys, residue_atoms)
    if spin is None:
        spin = _guess_spin(residue_atoms)

    # Write PDB of QM region for visualization (pre-truncation)
    qm_pdb_path = os.path.join(output_dir, 'qm_cluster.pdb')
    _write_qm_pdb(qm_pdb_path, qm_model_atoms)

    # Residue summary
    res_summary = []
    for rk in sorted(qm_residue_keys):
        n = sum(1 for _, _, _, _, _, src in qm_model_atoms if src == rk)
        if n > 0:
            res_summary.append({
                'chain': rk[0], 'resnum': rk[1], 'resname': rk[2], 'atoms': n
            })

    return {
        'success': True,
        'xyz_path': xyz_path,
        'pdb_path': qm_pdb_path,
        'constraints_path': constraints_path,
        'charge': charge,
        'spin': spin,
        'n_atoms': n_atoms,
        'n_residues': len(qm_residue_keys),
        'n_capping_h': len(cap_h_indices),
        'n_frozen': len(cap_h_indices),
        'residues': res_summary,
    }


# =============================================================================
# Internal helpers
# =============================================================================

def _parse_pdb(pdb_path):
    """Parse PDB file, return list of atom dicts."""
    atoms = []
    with open(pdb_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            try:
                atom = {
                    'record': line[:6].strip(),
                    'serial': int(line[6:11].strip()),
                    'name': line[12:16].strip(),
                    'resname': line[17:20].strip(),
                    'chain': line[21].strip() or 'A',
                    'resnum': int(line[22:26].strip()),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                    'element': line[76:78].strip() if len(line) > 77 else '',
                    'line': line.rstrip(),
                }
                # Fallback element detection if column 77-78 empty
                if not atom['element']:
                    atom['element'] = _element_from_name(atom['name'], atom['resname'])
                atoms.append(atom)
            except (ValueError, IndexError):
                continue
    return atoms


def _find_center_atoms(atoms, center_residues, center_atoms, plip_result, chain):
    """Find atoms defining the QM cluster center."""
    center = []
    if plip_result and plip_result.get('interactions'):
        center_res_nums = _extract_plip_residues(plip_result)
        for atom in atoms:
            if atom['resnum'] in center_res_nums and atom['chain'] == chain:
                center.append(atom)
            if atom['record'] == 'HETATM' and atom['resname'] not in WATER_RESIDUES \
               and atom['resname'] not in ION_RESIDUES:
                center.append(atom)
    elif center_residues:
        for atom in atoms:
            if atom['resname'] in center_residues:
                center.append(atom)
    elif center_atoms:
        ca_set = set(center_atoms)
        for atom in atoms:
            if (atom['chain'], atom['resnum']) in ca_set:
                center.append(atom)
    else:
        for atom in atoms:
            if atom['record'] == 'HETATM' and atom['resname'] not in WATER_RESIDUES \
               and atom['resname'] not in ION_RESIDUES:
                center.append(atom)
    return center


def _select_residues_in_cutoff(atoms, center_coords, cutoff, include_waters):
    """Select residues with any atom within cutoff of center atoms."""
    cutoff_sq = cutoff * cutoff
    qm_residue_keys = set()

    for catom in center_coords:
        cx, cy, cz = catom['x'], catom['y'], catom['z']
        for atom in atoms:
            dx = atom['x'] - cx
            dy = atom['y'] - cy
            dz = atom['z'] - cz
            if dx*dx + dy*dy + dz*dz <= cutoff_sq:
                rname = atom['resname']
                if rname in WATER_RESIDUES and not include_waters:
                    continue
                rk = (atom['chain'], atom['resnum'], rname)
                qm_residue_keys.add(rk)

    return qm_residue_keys


def _limit_waters(atoms, center_coords, qm_residue_keys, max_waters, keep_waters=None):
    """Keep only the closest max_waters water molecules."""
    water_keys = [rk for rk in qm_residue_keys if rk[2] in WATER_RESIDUES]
    if len(water_keys) <= max_waters:
        return qm_residue_keys

    # Compute minimum distance from each water O to any center atom
    water_dists = []
    for wk in water_keys:
        # Find oxygen of this water
        min_dist = float('inf')
        for atom in atoms:
            if (atom['chain'], atom['resnum'], atom['resname']) == wk:
                if atom['name'].startswith('O'):
                    for ca in center_coords:
                        d = math.sqrt((atom['x']-ca['x'])**2 +
                                      (atom['y']-ca['y'])**2 +
                                      (atom['z']-ca['z'])**2)
                        min_dist = min(min_dist, d)
        water_dists.append((min_dist, wk))

    water_dists.sort()
    kw_set = set()
    if keep_waters:
        kw_set = {wk for _, wk in water_dists if wk[1] in keep_waters}
    keep_waters_selected = {wk for _, wk in water_dists[:max_waters]} | kw_set
    remove = {wk for wk in water_keys if wk not in keep_waters_selected}
    return qm_residue_keys - remove


def _truncate_and_cap(res_atoms, resname, rk, qm_model_atoms, cap_h_indices):
    """
    Truncate amino acid at CA-CB bond and cap with H.

    Standard cluster approach (Himo, Siegbahn):
    - Remove backbone atoms (N, CA, C, O, backbone H)
    - Keep sidechain atoms (CB and beyond)
    - Add capping H on CB in the CA direction at 1.09 A from CB
    - The capping H is frozen during optimization
    """
    # Find CA and CB coordinates
    ca_coord = None
    cb_coord = None
    sidechain_atoms = []

    for atom in res_atoms:
        name = atom['name']
        if name == 'CA':
            ca_coord = (atom['x'], atom['y'], atom['z'])
        elif name == 'CB':
            cb_coord = (atom['x'], atom['y'], atom['z'])

        # Collect sidechain atoms (everything NOT in backbone set)
        if name not in BACKBONE_ATOMS:
            sidechain_atoms.append(atom)

    if ca_coord is None or cb_coord is None:
        # Cannot truncate without CA/CB; keep sidechain as-is
        for a in sidechain_atoms:
            elem = _get_element(a)
            qm_model_atoms.append((elem, a['x'], a['y'], a['z'], False, rk))
        return

    if resname == 'PRO':
        # PRO: sidechain ring includes N (backbone), so keep N and its H too
        # Add back N and any H bonded to N
        for atom in res_atoms:
            if atom['name'] in ('N',):
                sidechain_atoms.append(atom)

    # Add sidechain atoms
    for a in sidechain_atoms:
        elem = _get_element(a)
        qm_model_atoms.append((elem, a['x'], a['y'], a['z'], False, rk))

    # Add capping H: placed at CB + 1.09 * (CA - CB) / |CA - CB|
    dx = ca_coord[0] - cb_coord[0]
    dy = ca_coord[1] - cb_coord[1]
    dz = ca_coord[2] - cb_coord[2]
    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
    if dist < 0.01:
        return  # degenerate geometry

    scale = CAP_CH_LENGTH / dist
    hx = cb_coord[0] + dx * scale
    hy = cb_coord[1] + dy * scale
    hz = cb_coord[2] + dz * scale

    cap_idx = len(qm_model_atoms) + 1  # 1-indexed for geomeTRIC
    qm_model_atoms.append(('H', hx, hy, hz, True, rk))
    cap_h_indices.append(cap_idx)


def _get_element(atom):
    """Get element symbol from atom dict, using PDB element column first."""
    elem = atom.get('element', '').strip()
    if elem and elem[0].isalpha():
        # Capitalize properly: 'FE' -> 'Fe', 'C' -> 'C'
        return elem.capitalize() if len(elem) == 2 else elem.upper()
    return _element_from_name(atom['name'], atom['resname'])


def _element_from_name(atom_name, resname):
    """Deduce element from PDB atom name."""
    name = atom_name.strip()

    # Two-letter elements (must check before single-letter)
    if resname in ION_RESIDUES:
        two = name[:2].upper()
        if two in ('FE', 'ZN', 'MG', 'MN', 'CU', 'CO', 'NI', 'CL', 'BR', 'NA'):
            return two.capitalize()

    # Standard: first alpha character is the element
    for c in name:
        if c.isalpha():
            return c.upper()
    return name[0].upper()


def _extract_plip_residues(plip_result):
    """Extract interacting residue numbers from PLIP result."""
    resnums = set()
    interactions = plip_result.get('interactions', {})
    for itype, ilist in interactions.items():
        if isinstance(ilist, list):
            for inter in ilist:
                if isinstance(inter, dict):
                    rn = inter.get('resnum') or inter.get('resnr')
                    if rn:
                        resnums.add(int(rn))
    return resnums


def _compute_charge(qm_residue_keys, residue_atoms):
    """Estimate total charge from residue types and metal ions."""
    charge = 0
    for rk in qm_residue_keys:
        resname = rk[2]

        # Amino acid charges
        if resname in RESIDUE_CHARGE:
            charge += RESIDUE_CHARGE[resname]

        # Metal ion charges
        if resname in METAL_CHARGE:
            charge += METAL_CHARGE[resname]
        elif resname in ION_RESIDUES:
            charge += METAL_CHARGE.get(resname, 0)

        # HIP (protonated histidine)
        if resname == 'HIP':
            charge += 1

    return charge


def _guess_spin(residue_atoms):
    """Guess spin from metal presence. Returns 2S (0=singlet)."""
    for rk, atoms in residue_atoms.items():
        for atom in atoms:
            name = atom['name'].strip().upper()
            resname = atom['resname'].strip().upper()
            # Check for transition metals
            if name in ('FE', 'MN', 'CO', 'NI', 'CU') or \
               resname in ('FE', 'FE2', 'FE3', 'MN', 'CO', 'NI', 'CU'):
                return 4  # High-spin quintet (2S=4) default for Fe
    return 0


def _write_qm_pdb(path, qm_model_atoms):
    """Write QM model as PDB for visualization."""
    with open(path, 'w', encoding='utf-8') as f:
        for i, (elem, x, y, z, freeze, rk) in enumerate(qm_model_atoms, 1):
            chain, resnum, resname = rk
            name = elem if len(elem) <= 2 else elem[:2]
            # Format atom name for PDB
            if len(name) < 4:
                name_fmt = f' {name:<3s}'
            else:
                name_fmt = f'{name:<4s}'
            record = 'HETATM' if resname not in STD_AMINO_ACIDS else 'ATOM  '
            marker = '*' if freeze else ' '
            f.write(f'{record}{i:>5d} {name_fmt} {resname:>3s} {chain:1s}'
                    f'{resnum:>4d}    {x:>8.3f}{y:>8.3f}{z:>8.3f}'
                    f'  1.00  0.00          {elem:>2s}\n')
        f.write('END\n')


def _truncate_and_cap_backbone(atoms, res_atoms, resname, rk, qm_model_atoms, cap_h_indices):
    '''
    Keep full amino acid backbone, but cap the N and C peptide bonds with H.
    '''
    res_chain, resnum, _ = rk
    
    # Add all current residue atoms
    for a in res_atoms:
        elem = _get_element(a)
        qm_model_atoms.append((elem, a['x'], a['y'], a['z'], False, rk))
        
    # Find N terminal cap direction (towards previous C)
    n_atom = next((a for a in res_atoms if a['name'] == 'N'), None)
    if n_atom:
        # try to find prev C
        prev_c = next((a for a in atoms if a['chain'] == res_chain and a['resnum'] == resnum - 1 and a['name'] == 'C'), None)
        if prev_c:
            dx = prev_c['x'] - n_atom['x']
            dy = prev_c['y'] - n_atom['y']
            dz = prev_c['z'] - n_atom['z']
        else:
            # fallback: use CA to N direction reversed... not perfect but ok
            ca_atom = next((a for a in res_atoms if a['name'] == 'CA'), None)
            if ca_atom:
                dx = n_atom['x'] - ca_atom['x']
                dy = n_atom['y'] - ca_atom['y']
                dz = n_atom['z'] - ca_atom['z']
            else:
                dx, dy, dz = 1.0, 0.0, 0.0
        
        import math
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dist > 0.01:
            scale = 1.01 / dist
            hx = n_atom['x'] + dx * scale
            hy = n_atom['y'] + dy * scale
            hz = n_atom['z'] + dz * scale
            cap_idx = len(qm_model_atoms) + 1
            qm_model_atoms.append(('H', hx, hy, hz, True, rk))
            cap_h_indices.append(cap_idx)
            
    # Find C terminal cap direction (towards next N)
    c_atom = next((a for a in res_atoms if a['name'] == 'C'), None)
    if c_atom:
        next_n = next((a for a in atoms if a['chain'] == res_chain and a['resnum'] == resnum + 1 and a['name'] == 'N'), None)
        if next_n:
            dx = next_n['x'] - c_atom['x']
            dy = next_n['y'] - c_atom['y']
            dz = next_n['z'] - c_atom['z']
        else:
            # fallback: sp2
            ca_atom = next((a for a in res_atoms if a['name'] == 'CA'), None)
            o_atom = next((a for a in res_atoms if a['name'] == 'O'), None)
            if ca_atom and o_atom:
                v1_x = ca_atom['x'] - c_atom['x']
                v1_y = ca_atom['y'] - c_atom['y']
                v1_z = ca_atom['z'] - c_atom['z']
                d1 = math.sqrt(v1_x**2 + v1_y**2 + v1_z**2)
                v1_x /= d1; v1_y /= d1; v1_z /= d1
                
                v2_x = o_atom['x'] - c_atom['x']
                v2_y = o_atom['y'] - c_atom['y']
                v2_z = o_atom['z'] - c_atom['z']
                d2 = math.sqrt(v2_x**2 + v2_y**2 + v2_z**2)
                v2_x /= d2; v2_y /= d2; v2_z /= d2
                
                dx = -(v1_x + v2_x)
                dy = -(v1_y + v2_y)
                dz = -(v1_z + v2_z)
            else:
                dx, dy, dz = 1.0, 0.0, 0.0
                
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dist > 0.01:
            scale = 1.09 / dist
            hx = c_atom['x'] + dx * scale
            hy = c_atom['y'] + dy * scale
            hz = c_atom['z'] + dz * scale
            cap_idx = len(qm_model_atoms) + 1
            qm_model_atoms.append(('H', hx, hy, hz, True, rk))
            cap_h_indices.append(cap_idx)

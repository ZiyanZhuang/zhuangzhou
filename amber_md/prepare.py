"""
AMBER MD Preparation — PDB parsing, heme detection, cysteine ID, Kabsch superposition.
Based on Shahrokh et al., J. Comp. Chem. (2011).
"""

import os
from itertools import permutations

import numpy as np

from .config import (
    AA3TO1, CYP_MOTIF, HEME_FE_NAMES, HEME_N_NAMES_MAP, HEME_RESNAMES,
)


def parse_pdb_atoms(pdb_path):
    """Parse PDB file, return list of atom dicts."""
    atoms = []
    with open(pdb_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                atom = {
                    'record': line[:6].strip(),
                    'serial': int(line[6:11]),
                    'name': line[12:16].strip(),
                    'altloc': line[16],
                    'resname': line[17:20].strip(),
                    'chain': line[21],
                    'resid': int(line[22:26]),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                    'raw': line,
                }
                atoms.append(atom)
            except (ValueError, IndexError):
                continue
    return atoms


def extract_sequence(atoms):
    """Extract protein sequence as dict {(chain, resid): (resname, 1-letter)}."""
    seq = {}
    for a in atoms:
        if a['record'] == 'ATOM' and a['name'] == 'CA':
            key = (a['chain'], a['resid'])
            if key not in seq:
                aa1 = AA3TO1.get(a['resname'], 'X')
                seq[key] = (a['resname'], aa1)
    return seq


def find_heme_residue(atoms):
    """Find the heme/porphyrin residue in PDB.

    Returns (resname, chain, resid, atoms_list) or (None, None, None, []).
    """
    fe_residues = set()
    for a in atoms:
        if a['name'] in HEME_FE_NAMES:
            fe_residues.add((a['resname'], a['chain'], a['resid']))

    for a in atoms:
        key = (a['resname'], a['chain'], a['resid'])
        if a['resname'] in HEME_RESNAMES or key in fe_residues:
            res_atoms = [x for x in atoms
                         if x['resname'] == key[0] and x['chain'] == key[1]
                         and x['resid'] == key[2]]
            return key[0], key[1], key[2], res_atoms

    return None, None, None, []


def find_heme_anchors(heme_atoms):
    """Extract Fe + 4 pyrrole N coordinates from heme residue atoms."""
    anchors = {}
    for a in heme_atoms:
        name = a['name']
        coord = np.array([a['x'], a['y'], a['z']])
        if name in HEME_FE_NAMES:
            anchors['FE'] = coord
        elif name in HEME_N_NAMES_MAP:
            canonical = HEME_N_NAMES_MAP[name]
            anchors[canonical] = coord
    return anchors


def identify_axial_cysteine(atoms, heme_anchors, seq):
    """Identify heme-binding cysteine via conserved motif + distance verification.

    Returns dict with chain, resid, motif, fe_sg_dist, method.
    Raises ValueError if no axial cysteine found.
    """
    fe_coord = heme_anchors.get('FE')
    if fe_coord is None:
        raise ValueError("No Fe atom found in heme residue")

    chains = {}
    for (ch, resid), (resname, aa1) in sorted(seq.items()):
        if ch not in chains:
            chains[ch] = []
        chains[ch].append((resid, aa1, resname))

    candidates = []
    for ch, residues in chains.items():
        residues.sort()
        seq_str = ''.join(r[1] for r in residues)
        resid_list = [r[0] for r in residues]

        for match in CYP_MOTIF.finditer(seq_str):
            cys_idx = match.start() + 7
            if cys_idx < len(resid_list):
                cys_resid = resid_list[cys_idx]
                motif_str = match.group()
                for a in atoms:
                    if a['chain'] == ch and a['resid'] == cys_resid and a['name'] == 'SG':
                        sg_coord = np.array([a['x'], a['y'], a['z']])
                        dist = np.linalg.norm(sg_coord - fe_coord)
                        candidates.append({
                            'chain': ch, 'resid': cys_resid,
                            'motif': motif_str, 'fe_sg_dist': dist,
                            'method': 'motif'
                        })

    # Fallback: closest CYS SG to Fe
    all_cys = []
    for a in atoms:
        if a['name'] == 'SG' and a['resname'] in ('CYS', 'CYX'):
            sg_coord = np.array([a['x'], a['y'], a['z']])
            dist = np.linalg.norm(sg_coord - fe_coord)
            if dist < 3.5:
                all_cys.append({
                    'chain': a['chain'], 'resid': a['resid'],
                    'motif': 'N/A', 'fe_sg_dist': dist,
                    'method': 'distance'
                })

    if candidates:
        return min(candidates, key=lambda x: x['fe_sg_dist'])
    elif all_cys:
        return min(all_cys, key=lambda x: x['fe_sg_dist'])
    else:
        raise ValueError("No axial cysteine found! Check PDB structure.")


def detect_heme_state(heme_atoms, fe_coord):
    """Determine heme oxidation/ligation state from geometry.

    Returns (state, o_atoms_list, reason_string).
    States: 'CPDI' (Compound I), 'DIOXY' (dioxygenated), 'IC6' (resting).
    """
    o_atoms = []
    for a in heme_atoms:
        if a['name'].startswith('O') and a['name'] not in (
                'O1A', 'O2A', 'O1D', 'O2D', 'O1', 'O2',
                'OA1', 'OA2', 'OD1', 'OD2'):
            coord = np.array([a['x'], a['y'], a['z']])
            dist = np.linalg.norm(coord - fe_coord)
            if dist < 2.5:
                o_atoms.append({'name': a['name'], 'dist': dist, 'coord': coord})

    for a in heme_atoms:
        if a['name'].startswith('O'):
            coord = np.array([a['x'], a['y'], a['z']])
            dist = np.linalg.norm(coord - fe_coord)
            if dist < 2.0 and not any(o['name'] == a['name'] for o in o_atoms):
                o_atoms.append({'name': a['name'], 'dist': dist, 'coord': coord})

    if not o_atoms:
        return 'IC6', o_atoms, "No proximal O found -> resting state"

    if len(o_atoms) >= 2:
        for i, o1 in enumerate(o_atoms):
            for o2 in o_atoms[i + 1:]:
                oo_dist = np.linalg.norm(o1['coord'] - o2['coord'])
                if oo_dist < 1.6:
                    return 'DIOXY', o_atoms, f"O-O bond {oo_dist:.2f} A -> dioxygenated"

    closest_o = min(o_atoms, key=lambda x: x['dist'])
    if closest_o['dist'] < 1.8:
        return 'CPDI', o_atoms, f"Fe-O = {closest_o['dist']:.2f} A -> Compound I"
    elif closest_o['dist'] < 2.2:
        return 'CPDI', o_atoms, f"Fe-O = {closest_o['dist']:.2f} A -> likely Compound I"
    else:
        return 'IC6', o_atoms, f"Fe-O = {closest_o['dist']:.2f} A (distant) -> resting state"


def kabsch_superpose(source_pts, target_pts):
    """Kabsch algorithm: find R, t such that target ~ R @ source + t.

    Returns (R, t, rmsd).
    """
    assert source_pts.shape == target_pts.shape
    src_center = source_pts.mean(axis=0)
    tgt_center = target_pts.mean(axis=0)

    src_c = source_pts - src_center
    tgt_c = target_pts - tgt_center

    H = src_c.T @ tgt_c
    U, S, Vt = np.linalg.svd(H)

    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])

    R = Vt.T @ sign_matrix @ U.T
    t = tgt_center - R @ src_center

    transformed = (R @ source_pts.T).T + t
    rmsd = np.sqrt(np.mean(np.sum((transformed - target_pts) ** 2, axis=1)))
    return R, t, rmsd


def superpose_heme(pdb_anchors, mol2_path):
    """Superpose HEM.mol2 atoms onto PDB heme position.

    Uses 5-point Kabsch (Fe + 4N) with all 24 N-permutations.
    Returns (transformed_coords_dict, best_rmsd, best_perm, mol2_atom_list).
    """
    mol2_atoms = {}
    mol2_all = []
    in_atoms = False
    with open(mol2_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if '@<TRIPOS>ATOM' in line:
                in_atoms = True
                continue
            if '@<TRIPOS>' in line and in_atoms:
                in_atoms = False
                continue
            if in_atoms and line.strip():
                parts = line.split()
                name = parts[1]
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                atype = parts[5]
                resid_info = parts[6] if len(parts) > 6 else ''
                resname_info = parts[7] if len(parts) > 7 else ''
                charge = parts[8] if len(parts) > 8 else ''
                mol2_atoms[name] = np.array([x, y, z])
                mol2_all.append({
                    'name': name, 'x': x, 'y': y, 'z': z,
                    'atype': atype, 'resid': resid_info,
                    'resname': resname_info, 'charge': charge
                })

    mol2_fe = mol2_atoms['FE']
    mol2_n_names = ['NA', 'NB', 'NC', 'ND']

    pdb_fe = pdb_anchors['FE']
    pdb_n_keys = [k for k in pdb_anchors if k != 'FE']

    if len(pdb_n_keys) != 4:
        raise ValueError(
            f"Expected 4 N atoms in PDB heme, found {len(pdb_n_keys)}: {pdb_n_keys}")

    best_rmsd = float('inf')
    best_R = best_t = best_perm = None

    for perm in permutations(mol2_n_names):
        target = np.array([pdb_fe] + [pdb_anchors[n] for n in pdb_n_keys])
        source = np.array([mol2_fe] + [mol2_atoms[n] for n in perm])
        R, t, rmsd = kabsch_superpose(source, target)
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_R, best_t = R, t
            best_perm = list(zip(pdb_n_keys, perm))

    transformed = {}
    for name, coord in mol2_atoms.items():
        transformed[name] = best_R @ coord + best_t

    return transformed, best_rmsd, best_perm, mol2_all


def write_prepared_pdb(output_path, atoms, heme_resname, heme_chain, heme_resid,
                       axial_cys_resid, axial_cys_chain, transformed_heme, mol2_all,
                       keep_chains=None):
    """Write prepared PDB: protein (H stripped, CYS->CYP) + superposed HEM.

    Returns (n_atoms, hem_resid_out).
    """
    serial = 1
    hem_resid_out = None

    with open(output_path, 'w', encoding='utf-8') as out:
        for a in atoms:
            if (a['resname'] == heme_resname and a['chain'] == heme_chain
                    and a['resid'] == heme_resid):
                continue
            if a['record'] == 'HETATM':
                continue
            if keep_chains and a['chain'] not in keep_chains:
                continue
            if (a['name'].startswith('H') or
                    (len(a['name']) > 1 and a['name'][0].isdigit()
                     and a['name'][1] == 'H')):
                continue

            line = a['raw']

            if (a['resname'] == 'CYS' and a['resid'] == axial_cys_resid
                    and a['chain'] == axial_cys_chain):
                if a['name'] == 'HG':
                    continue
                line = line[:17] + 'CYP' + line[20:]

            out.write(f"{line[:6]}{serial:5d}{line[11:]}")
            hem_resid_out = a['resid']
            serial += 1

        out.write("TER\n")
        hem_resid_out = (hem_resid_out or 0) + 1

        for ma in mol2_all:
            name = ma['name']
            if name.startswith('H'):
                continue

            coord = transformed_heme[name]
            if len(name) <= 3:
                fmt_name = f" {name:<3s}"
            else:
                fmt_name = f"{name:<4s}"

            if name == 'FE':
                element = "FE"
            elif name.startswith('O'):
                element = " O"
            elif name.startswith('N'):
                element = " N"
            elif name.startswith('S'):
                element = " S"
            else:
                element = " C"

            out.write(
                f"HETATM{serial:5d} {fmt_name} HEM A{hem_resid_out:4d}    "
                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                f"  1.00  0.00          {element}\n"
            )
            serial += 1

        out.write("TER\nEND\n")

    return serial - 1, hem_resid_out

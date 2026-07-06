import re
from pathlib import Path

target = Path(__file__).resolve().parent / "qm_engine" / "cluster.py"

with open(target, "r", encoding="utf-8") as f:
    content = f.read()

# Replace the block where keep_bb is handled
old_block = """            if (res_chain, resnum) in keep_bb:
                # Keep entire residue (backbone participates in catalysis)
                for a in res_atoms:
                    elem = _get_element(a)
                    qm_model_atoms.append((elem, a['x'], a['y'], a['z'],
                                           False, rk))
                continue

            # Standard truncation: CA-CB cut
            _truncate_and_cap(res_atoms, resname, rk, qm_model_atoms,
                              cap_h_indices)"""

new_block = """            if (res_chain, resnum) in keep_bb:
                # Keep entire residue (backbone participates in catalysis) and cap peptide bonds
                _truncate_and_cap_backbone(atoms, res_atoms, resname, rk, qm_model_atoms, cap_h_indices)
            else:
                # Standard truncation: CA-CB cut
                _truncate_and_cap(res_atoms, resname, rk, qm_model_atoms, cap_h_indices)"""

content = content.replace(old_block, new_block)

# Replace extract_qm_cluster signature to support keep_residues and keep_waters
old_sig = """def extract_qm_cluster(
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
) -> dict:"""

new_sig = """def extract_qm_cluster(
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
) -> dict:"""

content = content.replace(old_sig, new_sig)

# Add logic for keep_residues and keep_waters in _select_residues_in_cutoff step
old_select = """    if include_waters and max_waters > 0:
        qm_residue_keys = _limit_waters(atoms, center_coords, qm_residue_keys,
                                         max_waters)"""

new_select = """    if include_waters and max_waters > 0:
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
                qm_residue_keys.add((atom['chain'], atom['resnum'], atom['resname']))"""

content = content.replace(old_select, new_select)


# Modify _limit_waters to accept keep_waters
old_limit_waters_def = (
    """def _limit_waters(atoms, center_coords, qm_residue_keys, max_waters):"""
)
new_limit_waters_def = """def _limit_waters(atoms, center_coords, qm_residue_keys, max_waters, keep_waters=None):"""

old_limit_waters_body = """    water_dists.sort()
    keep_waters = {wk for _, wk in water_dists[:max_waters]}
    remove = {wk for wk in water_keys if wk not in keep_waters}
    return qm_residue_keys - remove"""

new_limit_waters_body = """    water_dists.sort()
    kw_set = set()
    if keep_waters:
        kw_set = {wk for _, wk in water_dists if wk[1] in keep_waters}
    keep_waters_selected = {wk for _, wk in water_dists[:max_waters]} | kw_set
    remove = {wk for wk in water_keys if wk not in keep_waters_selected}
    return qm_residue_keys - remove"""

content = content.replace(old_limit_waters_def, new_limit_waters_def)
content = content.replace(old_limit_waters_body, new_limit_waters_body)


# Append _truncate_and_cap_backbone
append_block = """

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
"""

content += append_block

with open(target, "w", encoding="utf-8") as f:
    f.write(content)

print("Patched cluster.py successfully")

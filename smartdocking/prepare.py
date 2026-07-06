"""
Protein & Ligand Preparation
=============================
- PDB preprocessing (chain selection, water removal, cofactor retention, ligand removal)
- Cofactor detection (HEME, FAD, NAD, etc.)
- Ligand detection (non-standard residues in ATOM/HETATM)
- PDB → PDBQT conversion (receptor)
- SDF → PDBQT conversion (ligand, via meeko)
- Extract ligand from PDB as SDF (for re-docking scenarios)

All residue sets and atom type mappings come from config.py.
"""

import logging
import os
import subprocess

import numpy as np

from .config import (
    STANDARD_RESIDUES, WATER_RESNAMES, COMMON_IONS,
    COFACTOR_RESNAMES, CRYSTALLOGRAPHIC_ADDITIVES, VINA_ATOM_TYPES,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Protein preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_protein(pdb_file: str, output_dir: str,
                       chain: str = None, remove_water: bool = True,
                       remove_ions: bool = True,
                       keep_cofactors: bool = True,
                       exclude_resnames: set = None) -> dict:
    """
    Preprocess protein PDB for docking: select chain, remove non-protein entities.

    Processing order (first match wins):
      1. Chain filter — reject atoms from non-target chains (cofactors on target chain kept)
      2. Crystallographic additives — always removed (GOL, EDO, PEG, etc.)
      3. Water — removed by default
      4. Ions — removed by default (SO4, PO4, NA, CL, etc.)
      5. Explicit exclusions — user-specified residue names
      6. Ligand auto-detection — anything not in STANDARD | WATER | IONS | COFACTOR
      7. Cofactors — kept on target chain only (HEM, FAD, etc.)

    Any residue NOT in STANDARD_RESIDUES | WATER | IONS | COFACTORS
    is treated as a ligand and removed from the receptor.
    """
    # Build the set of "recognized" residues (anything else = ligand)
    known_resnames = (STANDARD_RESIDUES | WATER_RESNAMES | COMMON_IONS
                      | COFACTOR_RESNAMES | CRYSTALLOGRAPHIC_ADDITIVES)

    stem = os.path.splitext(os.path.basename(pdb_file))[0]
    output_pdb = os.path.join(output_dir, f'{stem}_clean.pdb')

    kept_atoms = 0
    removed = {'water': 0, 'ions': 0, 'other_chains': 0, 'ligands': 0,
               'additives': 0}
    removed_ligand_names = set()
    removed_additive_names = set()

    with open(pdb_file, encoding='utf-8') as fin, \
         open(output_pdb, 'w', encoding='utf-8') as fout:
        for line in fin:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                if line.startswith('END'):
                    fout.write('END\n')
                elif line.startswith(('CRYST', 'REMARK')):
                    fout.write(line)
                continue

            resname = line[17:20].strip()
            line_chain = line[21].strip()

            # ── 1. Chain filter (first!) ──
            # Reject everything from non-target chains, including their
            # cofactors, ligands, waters, ions — all of it.
            if chain and line_chain and line_chain != chain:
                removed['other_chains'] += 1
                continue

            # ── 2. Crystallographic additives — always remove ──
            if resname in CRYSTALLOGRAPHIC_ADDITIVES:
                removed['additives'] += 1
                removed_additive_names.add(resname)
                continue

            # ── 3. Water ──
            if resname in WATER_RESNAMES:
                if remove_water:
                    removed['water'] += 1
                    continue
                # If keeping water, fall through to write

            # ── 4. Ions ──
            elif resname in COMMON_IONS:
                if remove_ions:
                    removed['ions'] += 1
                    continue

            # ── 5. Explicit exclusion ──
            elif exclude_resnames and resname in exclude_resnames:
                removed['ligands'] += 1
                removed_ligand_names.add(resname)
                continue

            # ── 6. Cofactors (on target chain) ──
            elif resname in COFACTOR_RESNAMES:
                if keep_cofactors:
                    pass  # keep — fall through to write
                else:
                    removed['ligands'] += 1
                    removed_ligand_names.add(resname)
                    continue

            # ── 7. Standard residues — keep ──
            elif resname in STANDARD_RESIDUES:
                pass  # keep

            # ── 8. Unknown residue = ligand → remove ──
            else:
                removed['ligands'] += 1
                removed_ligand_names.add(resname)
                continue

            fout.write(line)
            kept_atoms += 1

    all_removed = sorted(removed_ligand_names | removed_additive_names)
    logger.info("Preprocess: kept %d atoms, removed water=%d ions=%d "
                "other_chains=%d ligands=%d additives=%d %s",
                kept_atoms, removed['water'], removed['ions'],
                removed['other_chains'], removed['ligands'],
                removed['additives'], all_removed or '')
    return {'pdb_path': output_pdb, 'kept_atoms': kept_atoms, 'removed': removed,
            'removed_ligands': sorted(removed_ligand_names),
            'removed_additives': sorted(removed_additive_names)}


# ═══════════════════════════════════════════════════════════════════════════════
# Cofactor & Ligand detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_cofactors(pdb_file: str) -> list:
    """Detect cofactors (HEME, FAD, NAD, etc.) in PDB."""
    het_atoms = {}
    with open(pdb_file, encoding='utf-8') as f:
        for line in f:
            if not line.startswith('HETATM'):
                continue
            resname = line[17:20].strip()
            chain_id = line[21].strip()
            atom_name = line[12:16].strip()
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            het_atoms.setdefault((resname, chain_id), []).append((atom_name, x, y, z))

    cofactors = []
    for (resname, chain_id), atoms in het_atoms.items():
        if resname not in ('HEM', 'HEC', 'HEA', 'HEB'):
            continue
        info = {'resname': resname, 'chain': chain_id, 'type': 'HEME'}
        for name, x, y, z in atoms:
            if name == 'FE' or name.startswith('FE'):
                info['fe_coord'] = [x, y, z]
                break
        nitrogens = [[x, y, z] for name, x, y, z in atoms
                     if name in ('NA', 'NB', 'NC', 'ND')]
        if len(nitrogens) >= 3:
            info['heme_normal'] = _compute_plane_normal(nitrogens)
        cofactors.append(info)
        logger.info("Cofactor: %s chain=%s FE=%s", resname, chain_id, info.get('fe_coord'))
    return cofactors


def detect_ligands(pdb_file: str) -> list:
    """
    Detect non-standard residues (ligands) in PDB.
    Scans both ATOM and HETATM records.

    Excludes crystallographic additives (GOL, PEG, EDO, etc.) so they are
    never used as the active-site anchor for docking box placement.
    If only additives are present (no real ligand), define_active_site()
    will correctly fall through to HEME-based or protein-center detection.
    """
    all_non_ligand = (STANDARD_RESIDUES | WATER_RESNAMES | COMMON_IONS
                      | COFACTOR_RESNAMES | CRYSTALLOGRAPHIC_ADDITIVES)
    ligands = {}

    with open(pdb_file, encoding='utf-8') as f:
        for line in f:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            resname = line[17:20].strip()
            if not resname or resname in all_non_ligand:
                continue
            chain_id = line[21].strip()
            try:
                resid = int(line[22:26].strip())
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            atom_name = line[12:16].strip()
            ligands.setdefault((resname, chain_id, resid), []).append((atom_name, x, y, z))

    result = []
    for (resname, chain_id, resid), atoms in ligands.items():
        coords = np.array([(a[1], a[2], a[3]) for a in atoms])
        result.append({
            'resname': resname, 'chain': chain_id, 'resid': resid,
            'n_atoms': len(atoms), 'atoms': atoms,
            'center': coords.mean(axis=0).tolist(),
        })
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Ligand extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_ligand_sdf(pdb_file: str, resname: str, output_dir: str) -> str:
    """Extract a ligand from PDB and write as SDF using RDKit."""
    from rdkit import Chem

    lig_lines = []
    with open(pdb_file, encoding='utf-8') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                if line[17:20].strip() == resname:
                    lig_lines.append('HETATM' + line[6:])
    lig_lines.append('END\n')

    lig_pdb_str = ''.join(lig_lines)
    mol = Chem.MolFromPDBBlock(lig_pdb_str, sanitize=False, removeHs=False)

    sdf_path = os.path.join(output_dir, f'{resname}.sdf')

    if mol is not None:
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass
        w = Chem.SDWriter(sdf_path)
        w.write(mol)
        w.close()
        logger.info("Extracted ligand %s -> %s (%d atoms)", resname, sdf_path, mol.GetNumAtoms())
        return sdf_path

    # Fallback: raw PDB block
    logger.warning("RDKit failed to parse ligand %s, writing raw PDB", resname)
    raw_path = os.path.join(output_dir, f'{resname}_raw.pdb')
    with open(raw_path, 'w', encoding='utf-8') as f:
        f.write(lig_pdb_str)
    return raw_path


# ═══════════════════════════════════════════════════════════════════════════════
# Format conversion (PDB/SDF → PDBQT)
# ═══════════════════════════════════════════════════════════════════════════════

def pdb_to_pdbqt(pdb_path: str, output_pdbqt: str = None) -> str:
    """Protein PDB → PDBQT. Tries ADFR prepare_receptor, falls back to simple."""
    if output_pdbqt is None:
        output_pdbqt = pdb_path.replace('.pdb', '_receptor.pdbqt')

    # Try ADFR Suite
    try:
        result = subprocess.run(
            ['prepare_receptor', '-r', pdb_path, '-o', output_pdbqt],
            capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and os.path.exists(output_pdbqt):
            logger.info("PDBQT (ADFR): %s", output_pdbqt)
            return output_pdbqt
    except FileNotFoundError:
        logger.debug("Tool '%s' not found, using fallback", 'prepare_receptor')
    except subprocess.TimeoutExpired:
        logger.warning("Tool timed out after %ss, using fallback", 60)

    # Simple conversion
    lines = []
    with open(pdb_path, encoding='utf-8') as f:
        for line in f:
            if line.startswith('ATOM') or (
                    line.startswith('HETATM') and
                    line[17:20].strip() not in WATER_RESNAMES):
                elem = line[76:78].strip() if len(line) >= 78 else ''
                if not elem:
                    atom_name = line[12:16].strip()
                    elem = atom_name[0] if atom_name else 'C'
                ad_type = VINA_ATOM_TYPES.get(elem.upper(), elem)
                padded = line[:54].ljust(54)
                padded += f'  1.00  0.00    +0.000 {ad_type:>2s}'
                lines.append(padded.rstrip() + '\n')

    with open(output_pdbqt, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    logger.info("PDBQT (simple): %s", output_pdbqt)
    return output_pdbqt


def sdf_to_pdbqt(sdf_path: str, output_pdbqt: str = None):
    """Ligand SDF → PDBQT via meeko.

    Returns:
        str: path to output PDBQT on success
        dict: {'error': ..., 'hint': ...} on failure (project convention)
    """
    from rdkit import Chem
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    if output_pdbqt is None:
        output_pdbqt = sdf_path.replace('.sdf', '.pdbqt')

    # Validate input file
    if not os.path.exists(sdf_path):
        return {'error': f'SDF file not found: {sdf_path}',
                'hint': 'Check ligand file path or generate conformer first'}

    try:
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
        mol = next((m for m in suppl if m is not None), None)
    except Exception as e:
        return {'error': f'Cannot parse SDF: {sdf_path} — {e}',
                'hint': 'Ensure valid SDF format with 3D coordinates'}

    if mol is None:
        return {'error': f'No valid molecule in SDF: {sdf_path}',
                'hint': 'File may be empty or contain invalid chemistry'}

    try:
        mol = Chem.AddHs(mol, addCoords=True)
        preparator = MoleculePreparation()
        setup = list(preparator.prepare(mol))[0]
        pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setup)
        if not is_ok:
            return {'error': f'PDBQT conversion failed: {error_msg}',
                    'hint': 'Ligand may have unusual chemistry (charges, metals, radicals)'}
    except Exception as e:
        return {'error': f'Meeko preparation failed: {e}',
                'hint': 'Check ligand structure — may need manual curation'}

    with open(output_pdbqt, 'w', encoding='utf-8') as f:
        f.write(pdbqt_string)

    logger.info("PDBQT (ligand): %s", output_pdbqt)
    return output_pdbqt


# ═══════════════════════════════════════════════════════════════════════════════
# Complex generation (protein + docked ligand → single PDB)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_complexes(receptor_pdb: str, docked_pdbqt: str, output_dir: str,
                       stem: str, ligand_resname: str = 'LIG',
                       affinities: list = None) -> list:
    """
    Generate one complex PDB per Vina pose, each with full CONECT records.

    Uses meeko to restore bond topology from PDBQT. Each file gets a REMARK
    with its rank and affinity so it's self-documenting.

    Args:
        receptor_pdb: cleaned protein PDB (no ligand)
        docked_pdbqt: Vina output PDBQT (multi-model, up to 9 poses)
        output_dir: directory for output files
        stem: filename stem (e.g. 'M6_Best')
        ligand_resname: residue name for ligand in output
        affinities: list of (rank, affinity, rmsd_lb, rmsd_ub) from Vina

    Returns:
        list of dicts: [{'path': ..., 'rank': 1, 'affinity': -5.1}, ...]
    """
    from rdkit import Chem
    from meeko import PDBQTMolecule, RDKitMolCreate

    # 1. Parse all poses from PDBQT
    try:
        pdbqt_mol = PDBQTMolecule.from_file(docked_pdbqt)
        poses = list(pdbqt_mol)
    except Exception as e:
        logger.error("Failed to parse PDBQT: %s", e)
        return []

    # Build affinity lookup: rank → (affinity, rmsd_lb, rmsd_ub)
    aff_map = {}
    for entry in (affinities or []):
        aff_map[entry[0]] = entry[1:]  # rank → (aff, rmsd_lb, rmsd_ub)

    # 2. Read receptor once
    receptor_lines, max_serial, max_chain = _read_receptor(receptor_pdb)
    lig_chain = chr(min(ord(max_chain) + 1, ord('Z')))

    # 3. Generate one complex per pose
    results = []
    for i, pose_data in enumerate(poses):
        rank = i + 1
        try:
            mol = RDKitMolCreate.from_pdbqt_mol(pose_data)[0]
        except Exception:
            mol = None
        if mol is None:
            continue

        # RDKit mol may have multiple conformers (all 9); pick confId=i
        conf_id = i if i < mol.GetNumConformers() else 0
        lig_pdb_block = Chem.MolToPDBBlock(mol, confId=conf_id)
        if not lig_pdb_block:
            continue

        # Affinity info
        aff_info = aff_map.get(rank, (None, None, None))
        affinity = aff_info[0]
        aff_str = f'{affinity:.2f} kcal/mol' if affinity is not None else 'N/A'

        # Output path
        out_pdb = os.path.join(output_dir, f'{stem}_complex_rank{rank}.pdb')

        # Build REMARK header
        remarks = [
            f'REMARK  SmartDocking complex - rank {rank} of {len(poses)}',
            f'REMARK  Affinity: {aff_str}',
        ]
        if aff_info[1] is not None:
            remarks.append(f'REMARK  RMSD l.b.: {aff_info[1]:.3f}  u.b.: {aff_info[2]:.3f}')
        remarks.append(f'REMARK  Receptor: {os.path.basename(receptor_pdb)}')
        remarks.append(f'REMARK  Ligand resname: {ligand_resname}')

        # Write complex
        _write_complex(out_pdb, receptor_lines, max_serial, lig_chain,
                       lig_pdb_block, ligand_resname, remarks)

        results.append({'path': out_pdb, 'rank': rank, 'affinity': affinity})
        logger.info("  rank %d: %s  %s", rank, aff_str, os.path.basename(out_pdb))

    logger.info("Generated %d complex files in %s", len(results), output_dir)
    return results


def _read_receptor(receptor_pdb: str) -> tuple:
    """Read receptor PDB, return (lines, max_serial, max_chain)."""
    lines = []
    max_serial = 0
    max_chain = 'A'
    with open(receptor_pdb, encoding='utf-8') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                lines.append(line)
                try:
                    s = int(line[6:11].strip())
                    if s > max_serial:
                        max_serial = s
                except (ValueError, IndexError):
                    pass
                ch = line[21].strip()
                if ch and ch > max_chain:
                    max_chain = ch
            elif line.startswith(('CRYST', 'REMARK')):
                lines.append(line)
    return lines, max_serial, max_chain


def _write_complex(output_pdb: str, receptor_lines: list, serial_offset: int,
                   lig_chain: str, lig_pdb_block: str, ligand_resname: str,
                   remarks: list):
    """Write a single complex PDB from receptor lines + ligand PDB block."""
    # Parse ligand block, renumber, restamp resname/chain
    lig_hetatm = []
    lig_conect = []
    old_to_new = {}

    for line in lig_pdb_block.splitlines():
        if line.startswith(('HETATM', 'ATOM')):
            try:
                old_serial = int(line[6:11].strip())
            except ValueError:
                continue
            new_serial = serial_offset + old_serial
            old_to_new[old_serial] = new_serial
            lig_hetatm.append(
                f'HETATM{new_serial:5d}'
                f'{line[11:17]}'
                f'{ligand_resname:>3s} '
                f'{lig_chain:1s}'
                f'{1:4d}'
                f'{line[26:]}'
            )
        elif line.startswith('CONECT'):
            lig_conect.append(line)

    # Renumber CONECT
    renumbered = []
    for line in lig_conect:
        parts = line.split()
        fmt = 'CONECT'
        for p in parts[1:]:
            try:
                fmt += f'{old_to_new.get(int(p), int(p)):5d}'
            except ValueError:
                pass
        renumbered.append(fmt)

    with open(output_pdb, 'w', encoding='utf-8') as f:
        for r in remarks:
            f.write(r + '\n')
        for line in receptor_lines:
            f.write(line)
        f.write('TER\n')
        for line in lig_hetatm:
            f.write(line + '\n')
        for line in renumbered:
            f.write(line + '\n')
        f.write('END\n')


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_plane_normal(points):
    """Fit plane to points, return normal vector."""
    pts = np.array(points)
    centered = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centered)
    normal = vt[-1]
    if normal[2] < 0:
        normal = -normal
    return normal.tolist()

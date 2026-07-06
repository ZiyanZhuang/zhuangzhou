"""
Generate position restraints for non-standard residues and metal ions.

Position restraints are critical for:
  - Metal ions: Very strong restraints to maintain coordination geometry
  - Cofactor heavy atoms: Standard restraints during equilibration
  - Coordination bonds: Harmonic distance restraints between metal and ligands

Output files:
  - posre_metal.itp:    Position restraints for metal ions
  - posre_cofactor.itp: Position restraints for cofactor heavy atoms
  - disre_coord.itp:    Distance restraints for coordination bonds
"""

import logging
import os

from .config import POSRE_FC_METAL, POSRE_FC_COFACTOR, POSRE_FC_BACKBONE

logger = logging.getLogger(__name__)


def generate_restraints(detection_result, work_dir, *,
                        metal_fc=POSRE_FC_METAL,
                        cofactor_fc=POSRE_FC_COFACTOR):
    """
    Generate all restraint files for a system with non-standard residues.

    Args:
        detection_result: Output from detect_nonstandard()
        work_dir: Output directory
        metal_fc: Force constant for metal position restraints (kJ/mol/nm^2)
        cofactor_fc: Force constant for cofactor position restraints

    Returns:
        {
            'posre_metal': str or None,
            'posre_cofactor': str or None,
            'disre_coord': str or None,
            'topol_inserts': [str],  # Lines to add to topol.top
            'error': str or None,
        }
    """
    os.makedirs(work_dir, exist_ok=True)
    result = {
        'posre_metal': None,
        'posre_cofactor': None,
        'disre_coord': None,
        'topol_inserts': [],
        'error': None,
    }

    if 'error' in detection_result:
        result['error'] = detection_result['error']
        return result

    metals = detection_result.get('metal_ions', [])
    nonstandard = detection_result.get('nonstandard_residues', [])
    coord_bonds = detection_result.get('coordination_bonds', [])
    cofactor_bonds = detection_result.get('cofactor_metal_bonds', [])

    # 1. Metal position restraints
    if metals:
        path = generate_metal_restraints(metals, work_dir, fc=metal_fc)
        result['posre_metal'] = path
        result['topol_inserts'].append(
            '#ifdef POSRES_METAL\n#include "posre_metal.itp"\n#endif'
        )

    # 2. Cofactor position restraints (heavy atoms only)
    if nonstandard:
        path = _generate_cofactor_restraints(nonstandard, work_dir, fc=cofactor_fc)
        if path:
            result['posre_cofactor'] = path
            result['topol_inserts'].append(
                '#ifdef POSRES_COFACTOR\n#include "posre_cofactor.itp"\n#endif'
            )

    # 3. Distance restraints for coordination bonds
    all_coord = coord_bonds + cofactor_bonds
    if all_coord:
        path = _generate_distance_restraints(all_coord, work_dir)
        if path:
            result['disre_coord'] = path
            result['topol_inserts'].append(
                '#ifdef DISRE_COORD\n#include "disre_coord.itp"\n#endif'
            )

    return result


def generate_metal_restraints(metals, work_dir, *, fc=POSRE_FC_METAL):
    """
    Generate position restraint file for metal ions.

    Metal ions need very strong position restraints (10000 kJ/mol/nm^2)
    to maintain proper coordination geometry during equilibration.
    These can be relaxed or removed during production MD.

    Args:
        metals: List of metal dicts from detect_nonstandard()
        work_dir: Output directory
        fc: Force constant (kJ/mol/nm^2)

    Returns: Path to posre_metal.itp
    """
    os.makedirs(work_dir, exist_ok=True)
    path = os.path.join(work_dir, 'posre_metal.itp')

    with open(path, 'w', encoding='utf-8') as f:
        f.write('; Position restraints for metal ions\n')
        f.write(f'; Force constant: {fc:.1f} kJ/mol/nm^2\n')
        f.write('; Apply with: define = -DPOSRES_METAL in .mdp\n\n')
        f.write('[ position_restraints ]\n')
        f.write(';  i  funct  fcx        fcy        fcz\n')

        for i, metal in enumerate(metals, 1):
            f.write(f'; {metal.get("element", "M")} ({metal.get("key", "")})\n')
            # In practice, atom index must be determined from the merged topology
            # Here we write placeholders; topology.py resolves actual indices
            f.write(f'  {i:4d}  1  {fc:.1f}  {fc:.1f}  {fc:.1f}\n')

    logger.info("Generated metal restraints: %s (%d metals)", path, len(metals))
    return path


def _generate_cofactor_restraints(nonstandard, work_dir, fc=POSRE_FC_COFACTOR):
    """
    Generate position restraints for cofactor heavy atoms.

    During equilibration, cofactors (HEME, FAD, etc.) should be restrained
    to prevent large displacements before the system is properly equilibrated.
    Only heavy atoms (non-H) are restrained.
    """
    if not nonstandard:
        return None

    path = os.path.join(work_dir, 'posre_cofactor.itp')

    with open(path, 'w', encoding='utf-8') as f:
        f.write('; Position restraints for cofactor heavy atoms\n')
        f.write(f'; Force constant: {fc:.1f} kJ/mol/nm^2\n')
        f.write('; Apply with: define = -DPOSRES_COFACTOR in .mdp\n\n')

        for res in nonstandard:
            resname = res['resname']
            n_atoms = res.get('n_atoms', 0)
            f.write(f'; {resname} (chain {res["chain"]}, resid {res["resid"]})\n')
            f.write(f'; NOTE: Atom indices must match the merged topology.\n')
            f.write(f';       Use topology.py resolve_restraint_indices() to fix.\n')
            f.write('[ position_restraints ]\n')
            f.write(';  i  funct  fcx        fcy        fcz\n')
            # Write placeholder for all atoms (topology.py will filter H)
            for i in range(1, n_atoms + 1):
                f.write(f'  {i:4d}  1  {fc:.1f}  {fc:.1f}  {fc:.1f}\n')
            f.write('\n')

    return path


def _generate_distance_restraints(coord_bonds, work_dir):
    """
    Generate distance restraints for metal-ligand coordination bonds.

    Distance restraints maintain coordination geometry when using
    the non-bonded metal model. Parameters:
      - r0: equilibrium distance (from crystal structure)
      - r1: lower bound (r0 - 0.05 nm)
      - r2: upper bound (r0 + 0.05 nm)
      - fc: force constant (default 1000 kJ/mol/nm^2)

    For the bonded model, these are replaced by proper bond terms in .itp.
    """
    if not coord_bonds:
        return None

    path = os.path.join(work_dir, 'disre_coord.itp')
    fc = 1000.0  # kJ/mol/nm^2

    with open(path, 'w', encoding='utf-8') as f:
        f.write('; Distance restraints for metal-ligand coordination bonds\n')
        f.write('; Non-bonded metal model: use harmonic distance restraints\n')
        f.write('; Apply with: define = -DDISRE_COORD in .mdp\n')
        f.write('; Also set: disre = simple, disre-fc = 1000 in .mdp\n\n')

        f.write('[ distance_restraints ]\n')
        f.write('; i  j  type  index  type\'  low    up1    up2    fac\n')

        for idx, bond in enumerate(coord_bonds):
            d_nm = bond['distance'] / 10.0  # Angstrom -> nm
            low = max(0.0, d_nm - 0.05)
            up1 = d_nm + 0.05
            up2 = d_nm + 0.10

            f.write(f'; {bond.get("metal", "?")} -- {bond.get("protein_atom", bond.get("cofactor_atom", "?"))}'
                    f'  d={bond["distance"]:.3f} A\n')
            # Placeholder atom indices (i, j) -- resolved by topology.py
            f.write(f'  {idx*2+1:4d}  {idx*2+2:4d}  1  {idx:4d}  1'
                    f'  {low:.4f}  {up1:.4f}  {up2:.4f}  1.0\n')

    logger.info("Generated distance restraints: %s (%d bonds)", path, len(coord_bonds))
    return path


def generate_mdp_defines(detection_result, phase='equil'):
    """
    Return the -D defines needed for MDP files based on detected features.

    Args:
        detection_result: Output from detect_nonstandard()
        phase: 'equil' (all restraints) or 'prod' (metal only) or 'none'

    Returns: String like "-DPOSRES -DPOSRES_METAL -DPOSRES_COFACTOR -DDISRE_COORD"
    """
    defines = []

    if phase == 'none':
        return ''

    metals = detection_result.get('metal_ions', [])
    nonstandard = detection_result.get('nonstandard_residues', [])
    coord_bonds = detection_result.get('coordination_bonds', [])

    if phase in ('equil', 'prod'):
        defines.append('-DPOSRES')  # Standard protein backbone restraints
        if metals:
            defines.append('-DPOSRES_METAL')

    if phase == 'equil':
        if nonstandard:
            defines.append('-DPOSRES_COFACTOR')
        if coord_bonds or detection_result.get('cofactor_metal_bonds'):
            defines.append('-DDISRE_COORD')

    return ' '.join(defines)

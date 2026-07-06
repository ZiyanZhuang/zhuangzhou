"""
Professional position restraint management for GROMACS MD.

Implements multi-phase stepdown equilibration protocols used in
academic metalloprotein MD studies (CHARMM-GUI style).

Restraint hierarchy:
  - POSRES:          Protein backbone heavy atoms (fc=1000)
  - POSRES_SOFT:     Protein backbone heavy atoms (fc=100)
  - POSRES_METAL:    Metal ion positions (fc=10000, permanent or equil-only)
  - POSRES_COFACTOR: Cofactor heavy atoms (fc=1000, equil-only)
  - POSRES_COORD:    Metal-ligand distance restraints (coordination bonds)

Stepdown protocol (6 phases):
  Phase 0: EM          — all restraints active
  Phase 1: NVT (100ps) — all restraints active, velocity generation
  Phase 2: NPT1 (100ps) — all restraints, C-rescale barostat
  Phase 3: NPT2 (100ps) — soft backbone + metal + coord
  Phase 4: NPT3 (100ps) — metal + coord only
  Phase 5: Production   — coord only (or none)

References:
  - CHARMM-GUI equilibration protocol (Jo et al., 2008)
  - Best practices for metalloprotein MD (Shahrokh et al., 2012, JCTC)
  - Lu Tian's rules: C-rescale/Berendsen for NPT equilibration, not P-R
"""

import logging
import os
import re

logger = logging.getLogger(__name__)

# Force constants (kJ/mol/nm^2)
FC_BACKBONE = 1000.0    # Standard protein backbone
FC_SOFT = 100.0          # Soft backbone restraints (stepdown)
FC_METAL = 10000.0       # Metal ions (very strong)
FC_COFACTOR = 1000.0     # Cofactor heavy atoms
FC_SIDECHAIN = 500.0     # Protein sidechains (optional)


def get_equilibration_schedule(system_type='metalloprotein', sim_time_ns=10.0,
                                stepdown=True):
    """
    Return a multi-phase equilibration schedule.

    Each phase is a dict with:
        name, nsteps, dt, ensemble, barostat, defines, description

    Args:
        system_type: 'protein', 'protein_ligand', 'metalloprotein',
                     'metalloprotein_ligand'
        sim_time_ns: Production MD time
        stepdown: Use gradual restraint release (True) or single-phase (False)

    Returns:
        list of phase dicts
    """
    nsteps_md = int(sim_time_ns * 1e6 / 2)  # dt=0.002 ps

    has_metal = system_type in ('metalloprotein', 'metalloprotein_ligand')
    has_ligand = system_type in ('protein_ligand', 'metalloprotein_ligand')

    # Base defines for each component
    d_posres = '-DPOSRES'
    d_soft = '-DPOSRES_SOFT'
    d_metal = '-DPOSRES_METAL'
    d_cofac = '-DPOSRES_COFACTOR'
    # Note: POSRES_COORD (distance restraints) removed — bonded model
    # with explicit Fe-S bonds (k=209200 kJ/mol/nm²) is the correct
    # approach per industry standard (Shahrokh 2012, JCTC).

    if stepdown and has_metal:
        # Full stepdown protocol for metalloproteins
        phases = [
            {
                'name': 'em',
                'nsteps': 50000,
                'ensemble': 'minimize',
                'defines': _join_defines(d_posres, d_metal, d_cofac,
                                         active=has_metal),
                'description': 'Energy minimization (all restraints)',
            },
            {
                'name': 'nvt',
                'nsteps': 50000,  # 100 ps
                'ensemble': 'NVT',
                'defines': _join_defines(d_posres, d_metal, d_cofac,
                                         active=has_metal),
                'description': 'NVT equilibration (all restraints, gen_vel)',
            },
            {
                'name': 'npt1',
                'nsteps': 50000,  # 100 ps
                'ensemble': 'NPT',
                'barostat': 'C-rescale',
                'defines': _join_defines(d_posres, d_metal, d_cofac,
                                         active=has_metal),
                'description': 'NPT-1: full restraints (C-rescale)',
            },
            {
                'name': 'npt2',
                'nsteps': 50000,  # 100 ps
                'ensemble': 'NPT',
                'barostat': 'C-rescale',
                'defines': _join_defines(d_soft, d_metal,
                                         active=has_metal),
                'description': 'NPT-2: soft backbone + metal (cofactor free)',
            },
            {
                'name': 'npt3',
                'nsteps': 50000,  # 100 ps
                'ensemble': 'NPT',
                'barostat': 'C-rescale',
                'defines': _join_defines(d_metal,
                                         active=has_metal),
                'description': 'NPT-3: metal only (backbone+cofactor free)',
            },
            {
                'name': 'md',
                'nsteps': nsteps_md,
                'ensemble': 'NPT',
                'barostat': 'Parrinello-Rahman',
                'defines': '',  # No restraints — bonded model maintains coordination
                'description': f'Production MD ({sim_time_ns} ns, P-R, no restraints)',
            },
        ]
    elif stepdown and not has_metal:
        # Simpler stepdown for protein/protein-ligand (no metal)
        phases = [
            {
                'name': 'em',
                'nsteps': 50000,
                'ensemble': 'minimize',
                'defines': d_posres,
                'description': 'Energy minimization',
            },
            {
                'name': 'nvt',
                'nsteps': 50000,
                'ensemble': 'NVT',
                'defines': d_posres,
                'description': 'NVT equilibration (backbone restraints)',
            },
            {
                'name': 'npt',
                'nsteps': 50000,
                'ensemble': 'NPT',
                'barostat': 'C-rescale',
                'defines': d_posres,
                'description': 'NPT equilibration (C-rescale)',
            },
            {
                'name': 'md',
                'nsteps': nsteps_md,
                'ensemble': 'NPT',
                'barostat': 'Parrinello-Rahman',
                'defines': '',
                'description': f'Production MD ({sim_time_ns} ns)',
            },
        ]
    else:
        # Single-phase (legacy compatibility)
        phases = [
            {'name': 'em', 'nsteps': 50000, 'ensemble': 'minimize',
             'defines': d_posres if has_metal else d_posres,
             'description': 'Energy minimization'},
            {'name': 'nvt', 'nsteps': 50000, 'ensemble': 'NVT',
             'defines': d_posres, 'description': 'NVT equilibration'},
            {'name': 'npt', 'nsteps': 50000, 'ensemble': 'NPT',
             'barostat': 'C-rescale', 'defines': d_posres,
             'description': 'NPT equilibration'},
            {'name': 'md', 'nsteps': nsteps_md, 'ensemble': 'NPT',
             'barostat': 'Parrinello-Rahman', 'defines': '',
             'description': f'Production MD ({sim_time_ns} ns)'},
        ]

    return phases


def _join_defines(*defines, active=True):
    """Join define strings, filtering empty ones."""
    if not active:
        return defines[0] if defines else ''
    return ' '.join(d for d in defines if d)


def generate_soft_posre(topol_path, work_dir, fc=FC_SOFT):
    """
    Generate a 'soft' position restraint file from the standard posre.itp.

    Reads posre.itp (from pdb2gmx) and creates posre_soft.itp with lower fc.
    """
    posre_path = os.path.join(work_dir, 'posre.itp')
    soft_path = os.path.join(work_dir, 'posre_soft.itp')

    if not os.path.exists(posre_path):
        # Try to find any posre_*.itp file (handles chain_A, chain_B, etc.)
        import glob
        candidates = glob.glob(os.path.join(work_dir, 'posre_*.itp'))
        # Prefer protein chain files, exclude soft/cofactor/metal variants
        candidates = [c for c in candidates
                      if 'soft' not in os.path.basename(c)
                      and 'cofactor' not in os.path.basename(c)
                      and 'metal' not in os.path.basename(c)]
        if candidates:
            posre_path = candidates[0]
            soft_path = os.path.join(work_dir,
                                     os.path.basename(posre_path).replace('posre_', 'posre_soft_'))
        else:
            # Legacy fallback names
            for name in ('posre_Protein_chain_A.itp', 'posre_Protein.itp'):
                alt = os.path.join(work_dir, name)
                if os.path.exists(alt):
                    posre_path = alt
                    soft_path = os.path.join(work_dir,
                                             name.replace('posre_', 'posre_soft_'))
                    break

    if not os.path.exists(posre_path):
        logger.warning("No posre.itp found in %s, skipping soft restraints", work_dir)
        return None

    with open(posre_path, encoding='utf-8', errors='replace') as f:
        content = f.read()

    # Replace force constants: match lines like "   1  1  1000  1000  1000"
    new_content = re.sub(
        r'^(\s*\d+\s+1\s+)\d+\.?\d*\s+\d+\.?\d*\s+\d+\.?\d*',
        lambda m: f'{m.group(1)}{fc:.1f}  {fc:.1f}  {fc:.1f}',
        content, flags=re.MULTILINE,
    )

    # Update header comment
    new_content = new_content.replace(
        'Position restraint',
        f'Soft position restraint (fc={fc:.0f} kJ/mol/nm^2)',
        1
    )

    with open(soft_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    logger.info("Generated soft posre: %s (fc=%.0f)", soft_path, fc)
    return soft_path


def generate_metal_posre(metals_info, gro_path, work_dir, fc=FC_METAL):
    """
    Generate position restraints for metal ions from a .gro file.

    Args:
        metals_info: List of dicts with metal info (from pdb_prep or detection)
                     Each has 'resname', 'atom_name' or similar identifiers
        gro_path: System .gro file path
        work_dir: Output directory
        fc: Force constant (kJ/mol/nm^2)

    Returns: path to posre_metal.itp or None
    """
    if not metals_info or not os.path.exists(gro_path):
        return None

    # Parse GRO to find metal atom indices
    metal_atoms = _find_metal_atoms_in_gro(gro_path)
    if not metal_atoms:
        logger.warning("No metal atoms found in %s", gro_path)
        return None

    path = os.path.join(work_dir, 'posre_metal.itp')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'; Position restraints for metal ions (fc={fc:.0f})\n')
        f.write('; Apply with: define = -DPOSRES_METAL\n\n')
        f.write('[ position_restraints ]\n')
        f.write(';  i  funct  fcx        fcy        fcz\n')
        for idx, resname, atom_name in metal_atoms:
            f.write(f'; {resname} {atom_name}\n')
            f.write(f'  {idx:6d}  1  {fc:.1f}  {fc:.1f}  {fc:.1f}\n')

    logger.info("Generated metal posre: %s (%d atoms, fc=%.0f)",
                path, len(metal_atoms), fc)
    return path


def generate_cofactor_posre(gro_path, work_dir, cofactor_resnames=None,
                            fc=FC_COFACTOR):
    """
    Generate position restraints for cofactor heavy atoms.

    Restrains all non-hydrogen atoms of the specified cofactor residues.
    """
    if cofactor_resnames is None:
        cofactor_resnames = {
            'HEME', 'HEM', 'HEC', 'HEA', 'HEO2',   # heme variants
            'FAD', 'FMN', 'FE2', 'FES', 'F3S', 'SF4',  # iron-sulfur / flavin
            'NAD', 'NAP', 'NDP',                      # NAD+/NADP+
            'PLP', 'TPP', 'SAM', 'COA',               # other cofactors
            'BCL', 'BCB', 'BPH', 'CLA',               # chlorophylls
            'MTE', 'MGD',                              # molybdopterin
        }

    if not os.path.exists(gro_path):
        return None

    # Parse GRO for cofactor heavy atoms
    cofactor_atoms = []
    with open(gro_path, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    natoms = int(lines[1].strip())
    for i, line in enumerate(lines[2:2 + natoms], start=1):
        if len(line) < 20:
            continue
        resname = line[5:10].strip()
        atom_name = line[10:15].strip()
        if resname in cofactor_resnames:
            # Skip hydrogens
            if atom_name.startswith('H') or (len(atom_name) > 1 and
                                              atom_name[0].isdigit() and
                                              atom_name[1] == 'H'):
                continue
            cofactor_atoms.append((i, resname, atom_name))

    if not cofactor_atoms:
        return None

    path = os.path.join(work_dir, 'posre_cofactor.itp')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'; Position restraints for cofactor heavy atoms (fc={fc:.0f})\n')
        f.write('; Apply with: define = -DPOSRES_COFACTOR\n\n')
        f.write('[ position_restraints ]\n')
        f.write(';  i  funct  fcx        fcy        fcz\n')
        for idx, resname, atom_name in cofactor_atoms:
            f.write(f'  {idx:6d}  1  {fc:.1f}  {fc:.1f}  {fc:.1f}\n')

    logger.info("Generated cofactor posre: %s (%d atoms, fc=%.0f)",
                path, len(cofactor_atoms), fc)
    return path


def inject_restraints_into_topology(topol_path, work_dir, *,
                                     has_soft=False,
                                     has_metal=False,
                                     has_cofactor=False,
                                     has_coord=False):
    """
    Add all restraint #ifdef blocks to topol.top.

    Inserts after existing POSRES block (from pdb2gmx) or before [ system ].
    Idempotent: won't add duplicates.
    """
    if not os.path.exists(topol_path):
        return

    with open(topol_path, encoding='utf-8', errors='replace') as f:
        content = f.read()

    inserts = []

    if has_soft and 'POSRES_SOFT' not in content:
        # Find the matching soft posre file
        soft_file = _find_posre_file(work_dir, 'posre_soft')
        if soft_file:
            inserts.append(
                f'\n#ifdef POSRES_SOFT\n'
                f'#include "{os.path.basename(soft_file)}"\n'
                f'#endif\n'
            )

    if has_metal and 'POSRES_METAL' not in content:
        inserts.append(
            '\n#ifdef POSRES_METAL\n'
            '#include "posre_metal.itp"\n'
            '#endif\n'
        )

    if has_cofactor and 'POSRES_COFACTOR' not in content:
        inserts.append(
            '\n#ifdef POSRES_COFACTOR\n'
            '#include "posre_cofactor.itp"\n'
            '#endif\n'
        )

    if has_coord and 'POSRES_COORD' not in content:
        inserts.append(
            '\n#ifdef POSRES_COORD\n'
            '#include "posre_coord.itp"\n'
            '#endif\n'
        )

    if not inserts:
        return

    insert_text = ''.join(inserts)

    # Find insertion point: after existing POSRES block
    posres_match = re.search(r'#ifdef\s+POSRES\b.*?#endif', content, re.DOTALL)
    if posres_match:
        pos = posres_match.end()
        content = content[:pos] + insert_text + content[pos:]
    else:
        # Fallback: before [ system ]
        sys_pos = content.find('[ system ]')
        if sys_pos >= 0:
            content = content[:sys_pos] + insert_text + '\n' + content[sys_pos:]
        else:
            content += insert_text

    with open(topol_path, 'w', encoding='utf-8') as f:
        f.write(content)

    logger.info("Injected restraint blocks into topology: soft=%s metal=%s cofactor=%s coord=%s",
                has_soft, has_metal, has_cofactor, has_coord)


def _find_metal_atoms_in_gro(gro_path):
    """Find metal atom indices in a .gro file."""
    METAL_ELEMENTS = {'FE', 'ZN', 'MN', 'MG', 'CU', 'CO', 'NI', 'CA',
                      'FE2', 'FE3', 'ZN2', 'MN2', 'MG2', 'CU2', 'CA2'}
    metal_atoms = []
    with open(gro_path, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    natoms = int(lines[1].strip())
    for i, line in enumerate(lines[2:2 + natoms], start=1):
        if len(line) < 20:
            continue
        resname = line[5:10].strip()
        atom_name = line[10:15].strip()
        # Check if this is a metal: resname is metal, or atom is FE/ZN/etc in cofactor
        if resname in METAL_ELEMENTS:
            metal_atoms.append((i, resname, atom_name))
        elif atom_name.upper() in ('FE', 'FE1', 'FE2', 'ZN', 'CU', 'MN', 'MO', 'NI', 'CO'):
            metal_atoms.append((i, resname, atom_name))
    return metal_atoms


def _find_posre_file(work_dir, prefix):
    """Find a position restraint file matching a prefix."""
    for f in os.listdir(work_dir):
        if f.startswith(prefix) and f.endswith('.itp'):
            return os.path.join(work_dir, f)
    return None

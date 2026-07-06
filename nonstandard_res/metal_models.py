"""
Metal ion model selection and parameterization for GROMACS MD.

Decision framework: Given a metal ion type and its coordination environment,
select the optimal model and provide parameters.

Three models for metal ions in MD:
  1. Non-bonded (ion-in-solution): Simple Lennard-Jones + charge
     - Best for: Structural metals with weak/dynamic coordination (Ca2+, Mg2+, Na+)
     - Source: Li & Merz 2020 12-6 IOD parameters (TIP3P)
     - Pro: Simple, no topology changes. Con: Coordination geometry not enforced.

  2. Bonded (covalent): Explicit bonds, angles between metal and ligands
     - Best for: Catalytic metals with fixed coordination (Fe in HEME, Zn tetrahedral)
     - Source: CHARMM27 native params (HEME), ZAFF (Zn), MCPB.py (custom),
               Shahrokh et al. 2012 (Amber14ff CYP450)
     - Pro: Correct geometry. Con: Requires known coordination, no ligand exchange.

  3. Non-bonded + distance restraints (hybrid): LJ/charge + harmonic restraints
     - Best for: Metals with known coordination but dynamic binding (Fe-Cys in P450)
     - Source: Li & Merz params + crystal structure distances
     - Pro: Maintains coordination, allows flexibility. Con: Restraint tuning needed.

Decision table:
  Metal  | Coordination  | Recommended Model
  -------|---------------|------------------
  Fe(HEME) | Fixed 6-coord | Bonded (CHARMM27 native) + distance restraints
  Zn (4-coord) | His/Cys/Asp/Glu | Bonded (ZAFF) or non-bonded + restraints
  Zn (catalytic) | Dynamic | Non-bonded (12-6 IOD)
  Mg (enzyme) | 6-coord O-donors | Non-bonded (12-6 IOD) + restraints
  Ca (structural) | 6-8 coord | Non-bonded (12-6 IOD)
  Cu (blue copper) | His/Cys/Met | Bonded (custom) or non-bonded + restraints
  Mn (SOD) | His/Asp/water | Non-bonded + restraints
  Fe-S cluster | 4Fe-4S etc. | QM-derived bonded (requires MCPB.py or literature)

References:
  - Li & Merz, JCTC 2020, 16, 264-273 (12-6 IOD parameters)
  - Peters et al., JCTC 2010, 6, 2935-2947 (ZAFF)
  - Li & Merz, JCIM 2016, 56, 599-604 (MCPB.py)
  - Shahrokh et al., JCTC 2012, 8, 5167-5179 (P450 Fe distance restraints)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MetalModel:
    """Recommended model for a specific metal ion."""
    element: str
    charge: int
    model_type: str         # 'nonbonded', 'bonded', 'hybrid'
    description: str
    # Non-bonded parameters (Li & Merz 2020, 12-6 IOD, TIP3P)
    sigma_nm: Optional[float] = None    # LJ sigma (nm)
    epsilon_kj: Optional[float] = None  # LJ epsilon (kJ/mol)
    # Position restraint force constant
    posre_fc: float = 10000.0
    # Distance restraint recommended
    use_distance_restraints: bool = False
    # Bonded model source
    bonded_source: Optional[str] = None
    # Notes
    notes: str = ''


# Li & Merz 2020 12-6 IOD parameters for TIP3P water
# σ in nm, ε in kJ/mol
# Reference: JCTC 2020, 16, 264-273
LI_MERZ_PARAMS = {
    ('Fe', 2): MetalModel('Fe', 2, 'hybrid',
        'Fe2+ — use with distance restraints for coordination',
        sigma_nm=0.2268, epsilon_kj=0.1004,
        use_distance_restraints=True,
        notes='Fe2+ in proteins almost always needs explicit coordination handling'),
    ('Fe', 3): MetalModel('Fe', 3, 'hybrid',
        'Fe3+ — strong coordination, needs restraints or bonded model',
        sigma_nm=0.2200, epsilon_kj=0.0837,
        use_distance_restraints=True,
        notes='Fe3+ binds more tightly; consider bonded model for fixed sites'),
    ('Zn', 2): MetalModel('Zn', 2, 'hybrid',
        'Zn2+ — non-bonded for catalytic Zn, bonded (ZAFF) for structural',
        sigma_nm=0.2268, epsilon_kj=0.0460,
        use_distance_restraints=True,
        bonded_source='ZAFF (Peters 2010) or MCPB.py',
        notes='4-coord Zn: ZAFF strongly recommended. 6-coord: non-bonded OK.'),
    ('Mn', 2): MetalModel('Mn', 2, 'hybrid',
        'Mn2+ — non-bonded + restraints for most enzymes',
        sigma_nm=0.2414, epsilon_kj=0.0314,
        use_distance_restraints=True,
        notes='SOD, phosphatases. Restraints maintain coordination.'),
    ('Mg', 2): MetalModel('Mg', 2, 'nonbonded',
        'Mg2+ — non-bonded model usually sufficient',
        sigma_nm=0.1864, epsilon_kj=0.2930,
        notes='Strong O-donor preference. Non-bonded works for most enzymes.'),
    ('Ca', 2): MetalModel('Ca', 2, 'nonbonded',
        'Ca2+ — non-bonded, structural role, flexible coordination',
        sigma_nm=0.2696, epsilon_kj=0.0728,
        notes='EF-hand, calmodulin. Non-bonded with 12-6 IOD reproduces hydration.'),
    ('Cu', 1): MetalModel('Cu', 1, 'bonded',
        'Cu+ — requires bonded model (plastocyanin, blue copper)',
        sigma_nm=0.2040, epsilon_kj=0.0209,
        bonded_source='MCPB.py or custom QM-derived',
        notes='Cu+ coordination is sensitive; bonded model strongly preferred.'),
    ('Cu', 2): MetalModel('Cu', 2, 'hybrid',
        'Cu2+ — non-bonded + restraints or bonded (MCPB.py)',
        sigma_nm=0.2174, epsilon_kj=0.0209,
        use_distance_restraints=True,
        bonded_source='MCPB.py',
        notes='Jahn-Teller distortion makes Cu2+ challenging.'),
    ('Co', 2): MetalModel('Co', 2, 'hybrid',
        'Co2+ — non-bonded + restraints',
        sigma_nm=0.2268, epsilon_kj=0.0460,
        use_distance_restraints=True,
        notes='B12 enzymes may need bonded model for corrin ring.'),
    ('Ni', 2): MetalModel('Ni', 2, 'hybrid',
        'Ni2+ — non-bonded + restraints for most enzymes',
        sigma_nm=0.2112, epsilon_kj=0.0460,
        use_distance_restraints=True,
        notes='Urease, Ni-SOD. Square planar Ni needs bonded model.'),
    ('Na', 1): MetalModel('Na', 1, 'nonbonded',
        'Na+ — standard non-bonded (counter ion)',
        sigma_nm=0.2439, epsilon_kj=0.0874,
        posre_fc=0.0,
        notes='Counter ion only. No restraints needed.'),
    ('K', 1): MetalModel('K', 1, 'nonbonded',
        'K+ — standard non-bonded',
        sigma_nm=0.3038, epsilon_kj=0.0874,
        posre_fc=0.0,
        notes='Ion channels: non-bonded is standard.'),
}

# Special cases that bypass the general framework
SPECIAL_CASES = {
    'HEME': 'bonded',      # CHARMM27 native, always bonded
    'HEM': 'bonded',
    'HEC': 'bonded',       # Heme C
    'SF4': 'bonded_qm',    # 4Fe-4S cluster, requires QM params
    'FES': 'bonded_qm',    # 2Fe-2S cluster
    'F3S': 'bonded_qm',    # 3Fe-4S cluster
}


def recommend_metal_model(element, charge=None, coordination_number=None,
                           coordinating_residues=None, cofactor_resname=None):
    """
    Recommend the optimal metal ion model for MD simulation.

    Args:
        element: Metal element symbol (Fe, Zn, Mg, etc.)
        charge: Oxidation state (2, 3, etc.)
        coordination_number: Number of coordinating atoms
        coordinating_residues: List of coordinating residue names
        cofactor_resname: If metal is part of a cofactor (HEME, SF4, etc.)

    Returns:
        MetalModel with recommendation, or dict with error
    """
    # Special case: metal in cofactor
    if cofactor_resname and cofactor_resname in SPECIAL_CASES:
        model_type = SPECIAL_CASES[cofactor_resname]
        if model_type == 'bonded':
            return MetalModel(
                element, charge or 2, 'bonded',
                f'{element} in {cofactor_resname} — use native force field params',
                bonded_source=f'CHARMM27 native for {cofactor_resname}',
                use_distance_restraints=True,
                notes=f'CHARMM27 has built-in params for {cofactor_resname}. '
                      f'Use pdb2gmx with -merge all. Add distance restraints for safety.',
            )
        elif model_type == 'bonded_qm':
            return MetalModel(
                element, charge or 2, 'bonded',
                f'{element} in {cofactor_resname} — requires QM-derived parameters',
                bonded_source='MCPB.py or literature QM parameters',
                notes=f'Fe-S clusters need QM-derived bonded parameters. '
                      f'Use MCPB.py from AmberTools or find published params.',
            )

    # Default charge
    if charge is None:
        charge = 2

    # Look up in Li & Merz table
    key = (element, charge)
    if key in LI_MERZ_PARAMS:
        model = LI_MERZ_PARAMS[key]

        # Refine recommendation based on coordination
        if coordination_number and coordination_number <= 4 and element == 'Zn':
            return MetalModel(
                'Zn', 2, 'bonded',
                'Zn2+ tetrahedral coordination — ZAFF bonded model recommended',
                sigma_nm=model.sigma_nm, epsilon_kj=model.epsilon_kj,
                bonded_source='ZAFF (Peters 2010) or MCPB.py',
                use_distance_restraints=True,
                notes='Tetrahedral Zn (His/Cys/Asp/Glu): ZAFF preserves geometry. '
                      'Alternative: non-bonded + distance restraints.',
            )

        return model

    # Unknown metal
    logger.warning("No parameters for %s%d+, using generic non-bonded", element, charge)
    return MetalModel(
        element, charge, 'nonbonded',
        f'{element}{charge}+ — no pre-built parameters, use generic or QM',
        posre_fc=10000.0,
        use_distance_restraints=True,
        notes=f'No Li & Merz parameters for {element}{charge}+. '
              f'Consider MCPB.py for bonded model or use generic LJ with restraints.',
    )


def get_itp_for_nonbonded_metal(element, charge, ff='charmm27'):
    """
    Generate a minimal .itp file for a non-bonded metal ion.

    Uses Li & Merz 2020 12-6 IOD parameters.
    Returns itp content as string, or None if no params available.
    """
    key = (element, charge)
    if key not in LI_MERZ_PARAMS:
        return None

    model = LI_MERZ_PARAMS[key]
    if model.sigma_nm is None:
        return None

    resname = f'{element.upper()}{charge}' if charge > 1 else element.upper()
    atom_type = f'{element.upper()}{charge}+'

    itp = f"""; Non-bonded metal ion: {element}{charge}+
; Li & Merz 2020, JCTC 16, 264-273 (12-6 IOD, TIP3P)

[ atomtypes ]
; name  at_num  mass     charge  ptype  sigma(nm)    epsilon(kJ/mol)
  {atom_type:<6s}  {_atomic_number(element):3d}  {_atomic_mass(element):.4f}  {float(charge):.4f}  A  {model.sigma_nm:.6f}  {model.epsilon_kj:.6f}

[ moleculetype ]
; name  nrexcl
{resname:<6s}  1

[ atoms ]
; nr  type    resnr  residu  atom  cgnr  charge  mass
   1  {atom_type:<6s}  1      {resname:<6s}  {element.upper():<4s}  1     {float(charge):.4f}  {_atomic_mass(element):.4f}
"""
    return itp


def _atomic_number(element):
    """Get atomic number for common metals."""
    z = {'Fe': 26, 'Zn': 30, 'Mn': 25, 'Mg': 12, 'Ca': 20,
         'Cu': 29, 'Co': 27, 'Ni': 28, 'Na': 11, 'K': 19,
         'Mo': 42, 'W': 74, 'V': 23, 'Cr': 24}
    return z.get(element, 0)


def _atomic_mass(element):
    """Get atomic mass for common metals."""
    m = {'Fe': 55.845, 'Zn': 65.38, 'Mn': 54.938, 'Mg': 24.305,
         'Ca': 40.078, 'Cu': 63.546, 'Co': 58.933, 'Ni': 58.693,
         'Na': 22.990, 'K': 39.098, 'Mo': 95.95, 'W': 183.84,
         'V': 50.942, 'Cr': 51.996}
    return m.get(element, 0.0)

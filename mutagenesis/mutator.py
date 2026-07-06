"""
Protein Point Mutation Engine
=============================
Self-contained BioPython-based mutagenesis. No external script dependencies.

Supports:
  - Single-point mutation: mutate("protein.pdb", "V9L")
  - Multi-point mutation:  mutate("protein.pdb", ["V9L", "A5G"])
  - Chain selection, custom output path, batch mode

Strategy:
  1. Parse mutation string (e.g. "V9L" -> wt=VAL, pos=9, mut=LEU)
  2. Validate: check residue exists and matches wild-type
  3. Mutate: keep backbone (N, CA, C, O) + CB (if target != GLY),
     strip old side chain, rename residue
  4. Optionally rebuild side chains via PyMOL (if available)

Output naming:
  - Single:  <stem>_V9L.pdb
  - Multi:   <stem>_V9L_A5G_K3R.pdb
"""

import logging
import os
import re
import subprocess
import shutil
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# === Amino Acid Maps ===

AA_1TO3 = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
}

AA_3TO1 = {v: k for k, v in AA_1TO3.items()}

# Atoms to keep per residue type (backbone + side chain)
# Reference: standard PDB atom names
RESIDUE_ATOMS = {
    'GLY': {'N', 'CA', 'C', 'O', 'H', 'HA2', 'HA3', 'H1', 'H2', 'H3'},
    'ALA': {'N', 'CA', 'C', 'O', 'CB', 'H', 'HA', 'HB1', 'HB2', 'HB3', 'H1', 'H2', 'H3'},
    'VAL': {'N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'H', 'HA', 'HB',
            'HG11', 'HG12', 'HG13', 'HG21', 'HG22', 'HG23', 'H1', 'H2', 'H3'},
    'LEU': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'H', 'HA', 'HB2', 'HB3',
            'HG', 'HD11', 'HD12', 'HD13', 'HD21', 'HD22', 'HD23', 'H1', 'H2', 'H3'},
    'ILE': {'N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', 'H', 'HA', 'HB',
            'HG12', 'HG13', 'HG21', 'HG22', 'HG23', 'HD11', 'HD12', 'HD13', 'H1', 'H2', 'H3'},
    'PRO': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'HA', 'HB2', 'HB3',
            'HG2', 'HG3', 'HD2', 'HD3'},
    'PHE': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ',
            'H', 'HA', 'HB2', 'HB3', 'HD1', 'HD2', 'HE1', 'HE2', 'HZ', 'H1', 'H2', 'H3'},
    'TRP': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3',
            'CZ2', 'CZ3', 'CH2', 'H', 'HA', 'HB2', 'HB3', 'HD1', 'HE1', 'HE3',
            'HZ2', 'HZ3', 'HH2', 'H1', 'H2', 'H3'},
    'MET': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE', 'H', 'HA', 'HB2', 'HB3',
            'HG2', 'HG3', 'HE1', 'HE2', 'HE3', 'H1', 'H2', 'H3'},
    'SER': {'N', 'CA', 'C', 'O', 'CB', 'OG', 'H', 'HA', 'HB2', 'HB3', 'HG', 'H1', 'H2', 'H3'},
    'THR': {'N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', 'H', 'HA', 'HB', 'HG1',
            'HG21', 'HG22', 'HG23', 'H1', 'H2', 'H3'},
    'CYS': {'N', 'CA', 'C', 'O', 'CB', 'SG', 'H', 'HA', 'HB2', 'HB3', 'HG', 'H1', 'H2', 'H3'},
    'TYR': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH',
            'H', 'HA', 'HB2', 'HB3', 'HD1', 'HD2', 'HE1', 'HE2', 'HH', 'H1', 'H2', 'H3'},
    'ASN': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2', 'H', 'HA', 'HB2', 'HB3',
            'HD21', 'HD22', 'H1', 'H2', 'H3'},
    'GLN': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2', 'H', 'HA', 'HB2', 'HB3',
            'HG2', 'HG3', 'HE21', 'HE22', 'H1', 'H2', 'H3'},
    'ASP': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2', 'H', 'HA', 'HB2', 'HB3', 'H1', 'H2', 'H3'},
    'GLU': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2', 'H', 'HA', 'HB2', 'HB3',
            'HG2', 'HG3', 'H1', 'H2', 'H3'},
    'LYS': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', 'H', 'HA', 'HB2', 'HB3',
            'HG2', 'HG3', 'HD2', 'HD3', 'HE2', 'HE3', 'HZ1', 'HZ2', 'HZ3', 'H1', 'H2', 'H3'},
    'ARG': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2',
            'H', 'HA', 'HB2', 'HB3', 'HG2', 'HG3', 'HD2', 'HD3', 'HE',
            'HH11', 'HH12', 'HH21', 'HH22', 'H1', 'H2', 'H3'},
    'HIS': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2',
            'H', 'HA', 'HB2', 'HB3', 'HD1', 'HD2', 'HE1', 'HE2', 'H1', 'H2', 'H3'},
}

BACKBONE_ATOMS = {'N', 'CA', 'C', 'O'}


def parse_mutation(mut_str: str) -> Tuple[str, int, str]:
    """
    Parse mutation string like 'V9L' or 'VAL9LEU'.

    Returns:
        (wt_1letter, position, mut_1letter)

    Raises:
        ValueError if format is invalid.
    """
    mut_str = mut_str.strip().upper()

    # Format: single-letter like V9L, A5G
    m = re.match(r'^([A-Z])(\d+)([A-Z])$', mut_str)
    if m:
        return m.group(1), int(m.group(2)), m.group(3)

    # Format: three-letter like VAL9LEU
    m = re.match(r'^([A-Z]{3})(\d+)([A-Z]{3})$', mut_str)
    if m:
        wt3, pos, mut3 = m.group(1), int(m.group(2)), m.group(3)
        wt1 = AA_3TO1.get(wt3)
        mut1 = AA_3TO1.get(mut3)
        if wt1 and mut1:
            return wt1, pos, mut1

    raise ValueError(
        f"Invalid mutation format: '{mut_str}'. "
        f"Use single-letter (V9L) or three-letter (VAL9LEU)."
    )


def _validate_mutation(structure, chain_id: str, wt: str, pos: int, mut: str) -> str:
    """Validate that the target residue exists and matches wt. Returns error or ''."""
    model = structure[0]
    if chain_id not in [c.id for c in model.get_chains()]:
        return f"Chain '{chain_id}' not found. Available: {[c.id for c in model.get_chains()]}"

    chain = model[chain_id]
    res_id = (' ', pos, ' ')
    if res_id not in chain:
        return f"Residue {pos} not found in chain {chain_id}"

    residue = chain[res_id]
    actual_name = AA_3TO1.get(residue.resname.strip())
    if actual_name is None:
        return f"Residue {pos} is {residue.resname} (non-standard), cannot mutate"
    if actual_name != wt:
        return f"Position {pos}: expected {wt} ({AA_1TO3[wt]}) but found {actual_name} ({residue.resname})"
    if wt == mut:
        return f"Position {pos}: wild-type and mutant are the same ({wt})"

    return ""


def _apply_mutation_backbone(structure, chain_id: str, pos: int,
                             new_resname_3: str) -> List[str]:
    """
    Apply mutation by keeping backbone + CB, stripping incompatible side chain atoms.

    Returns list of removed atom names.
    """
    chain = structure[0][chain_id]
    residue = chain[(' ', pos, ' ')]

    # Determine which atoms to keep
    if new_resname_3 == 'GLY':
        keep = BACKBONE_ATOMS | {'H', 'HA2', 'HA3', 'H1', 'H2', 'H3'}
    else:
        # Keep backbone + CB (shared by all non-GLY residues)
        keep = BACKBONE_ATOMS | {'CB', 'H', 'HA', 'H1', 'H2', 'H3'}
        # If mutating to ALA, also keep CB hydrogens
        if new_resname_3 == 'ALA':
            keep |= {'HB1', 'HB2', 'HB3'}

    # Remove atoms not in keep set
    removed = []
    atoms_to_remove = []
    for atom in residue.get_atoms():
        if atom.name.strip() not in keep:
            atoms_to_remove.append(atom.id)
            removed.append(atom.name.strip())

    for atom_id in atoms_to_remove:
        residue.detach_child(atom_id)

    # Rename residue
    residue.resname = new_resname_3

    return removed


def _generate_output_path(pdb_path: str, mutations: List[str],
                          output: Optional[str], output_dir: Optional[str]) -> str:
    """Generate output file path from mutations."""
    if output:
        return output

    stem = os.path.splitext(os.path.basename(pdb_path))[0]
    mut_tag = '_'.join(mutations)
    filename = f"{stem}_{mut_tag}.pdb"

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)

    return os.path.join(os.path.dirname(pdb_path), filename)


def _try_pymol_rebuild(pdb_path: str, mutations: List[Tuple[str, int, str]],
                       output_path: str,
                       pymol_path: str = "D:/Soft/Pymol3.03/python.exe") -> Optional[str]:
    """
    Try to rebuild side chains using PyMOL mutagenesis wizard.
    Returns output_path on success, None on failure.
    """
    if not os.path.exists(pymol_path):
        return None

    # Build PyMOL script
    pdb_path_posix = pdb_path.replace('\\', '/')
    out_posix = output_path.replace('\\', '/')

    lines = [
        "import pymol",
        "from pymol import cmd",
        f'cmd.load("{pdb_path_posix}", "protein")',
        'cmd.wizard("mutagenesis")',
    ]

    for wt, pos, mut in mutations:
        new_3 = AA_1TO3[mut]
        lines.extend([
            f'cmd.get_wizard().do_select("resi {pos} and chain A")',
            f'cmd.get_wizard().set_mode("{new_3}")',
            'cmd.get_wizard().apply()',
        ])

    lines.extend([
        'cmd.set_wizard()',
        f'cmd.save("{out_posix}", "protein")',
        'cmd.quit()',
    ])

    script_path = output_path.replace('.pdb', '_mutate.py')
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    try:
        result = subprocess.run(
            [pymol_path, script_path],
            capture_output=True, text=True, timeout=120,
            cwd=os.path.dirname(pymol_path),
        )
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            os.remove(script_path)
            return output_path
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # Cleanup on failure
    if os.path.exists(script_path):
        os.remove(script_path)
    return None


def mutate_single(pdb_path: str, mutation: str, *,
                  chain: str = "A",
                  output: str = None,
                  output_dir: str = None,
                  rebuild: bool = True) -> dict:
    """
    Apply a single point mutation to a PDB file.

    Args:
        pdb_path: Input PDB file path
        mutation: Mutation string, e.g. "V9L" or "VAL9LEU"
        chain: Chain ID (default "A")
        output: Explicit output path (auto-generated if None)
        output_dir: Directory for output (auto if None)
        rebuild: If True, try PyMOL to rebuild side chains

    Returns:
        dict with keys: success, output_pdb, mutation, wt_residue, new_residue,
                       position, method, removed_atoms
    """
    return mutate(pdb_path, mutation, chain=chain, output=output,
                  output_dir=output_dir, rebuild=rebuild)


def mutate_multi(pdb_path: str, mutations: List[str], *,
                 chain: str = "A",
                 output: str = None,
                 output_dir: str = None,
                 rebuild: bool = True) -> dict:
    """
    Apply multiple point mutations to a PDB file simultaneously.

    Args:
        pdb_path: Input PDB file path
        mutations: List of mutation strings, e.g. ["V9L", "A5G"]
        chain: Chain ID (default "A")
        output: Explicit output path (auto-generated if None)
        output_dir: Directory for output (auto if None)
        rebuild: If True, try PyMOL to rebuild side chains

    Returns:
        dict with keys: success, output_pdb, mutations, details, method
    """
    return mutate(pdb_path, mutations, chain=chain, output=output,
                  output_dir=output_dir, rebuild=rebuild)


def mutate(pdb_path: str,
           mutations: Union[str, List[str]], *,
           chain: str = "A",
           output: str = None,
           output_dir: str = None,
           rebuild: bool = True) -> dict:
    """
    Apply single or multiple point mutations to a PDB structure.

    This is the main entry point. Automatically detects single vs multi mutation.

    Args:
        pdb_path: Input PDB file path
        mutations: Single mutation string or list of mutations
                   Format: "V9L" (single-letter) or "VAL9LEU" (three-letter)
        chain: Chain ID (default "A")
        output: Explicit output file path (auto-generated if None)
        output_dir: Output directory (uses input PDB directory if None)
        rebuild: If True, attempt PyMOL side-chain rebuild; else backbone-only

    Returns:
        dict:
          success: bool
          output_pdb: str — path to mutated PDB
          mutations: list of applied mutation strings
          details: list of per-mutation info dicts
          method: 'pymol' or 'backbone_only'
          error/hint: present on failure

    Examples:
        >>> from zhuangzhou.mutagenesis import mutate
        >>> mutate("protein.pdb", "V9L")
        >>> mutate("protein.pdb", ["V9L", "A5G", "K3R"])
    """
    if not os.path.exists(pdb_path):
        return {'error': f'PDB file not found: {pdb_path}', 'hint': 'Check file path'}

    # Normalize mutations to list
    if isinstance(mutations, str):
        mutations = [mutations]

    if not mutations:
        return {'error': 'No mutations provided', 'hint': 'Pass e.g. "V9L" or ["V9L", "A5G"]'}

    # Parse all mutations first
    parsed = []
    for mut_str in mutations:
        try:
            wt, pos, new = parse_mutation(mut_str)
            parsed.append((wt, pos, new, mut_str))
        except ValueError as e:
            return {'error': str(e), 'hint': 'Format: V9L (single-letter) or VAL9LEU (three-letter)'}

    # Load structure
    try:
        from Bio.PDB import PDBParser, PDBIO
    except ImportError:
        return {'error': 'BioPython not installed', 'hint': 'pip install biopython'}

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)

    # Validate all mutations
    details = []
    for wt, pos, new, mut_str in parsed:
        err = _validate_mutation(structure, chain, wt, pos, new)
        if err:
            return {'error': f'Mutation {mut_str}: {err}',
                    'hint': 'Check residue position and wild-type identity'}

    # Generate output path
    mut_labels = [m[3] for m in parsed]
    output_path = _generate_output_path(pdb_path, mut_labels, output, output_dir)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Try PyMOL rebuild first (better quality — proper rotamers)
    method = 'backbone_only'
    if rebuild:
        pymol_result = _try_pymol_rebuild(
            pdb_path, [(wt, pos, new) for wt, pos, new, _ in parsed], output_path)
        if pymol_result:
            method = 'pymol'
            details = [{
                'mutation': mut_str,
                'position': pos,
                'wt': f'{wt} ({AA_1TO3[wt]})',
                'new': f'{new} ({AA_1TO3[new]})',
                'removed_atoms': [],
            } for wt, pos, new, mut_str in parsed]

            return {
                'success': True,
                'output_pdb': output_path,
                'mutations': mut_labels,
                'details': details,
                'method': method,
            }

    # Fallback: backbone-preserving mutation
    for wt, pos, new, mut_str in parsed:
        new_3 = AA_1TO3[new]
        removed = _apply_mutation_backbone(structure, chain, pos, new_3)
        details.append({
            'mutation': mut_str,
            'position': pos,
            'wt': f'{wt} ({AA_1TO3[wt]})',
            'new': f'{new} ({new_3})',
            'removed_atoms': removed,
        })

    # Save
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_path)

    return {
        'success': True,
        'output_pdb': output_path,
        'mutations': mut_labels,
        'details': details,
        'method': method,
    }

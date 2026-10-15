"""
Zhuangzhou Tool Bridge
======================
Bridges Uni-Mol-main computational tools into the Zhuangzhou framework.

Each tool wrapper:
  1. Validates inputs
  2. Calls the underlying Uni-Mol-main script
  3. Returns structured results

Tools bridged:
  - ESM-2 mutation scanning (esm_mutation_scan.py)
  - Protein mutagenesis (biopython_mutate.py)
  - Smart docking (smart_docking.py)
  - PLIP interaction analysis (plip_analysis.py)
  - Vina docking (vina_docking.py)
  - GROMACS MD (gromacs_md.py)
  - Cofactor detection (cofactor_detector.py)
  - Mutation classification (mutation_classifier.py)
  - Batch mutant docking (batch_mutant_docking.py)
"""

import json
import logging
import math
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Default paths — overridden by config
_DEFAULT_UNIMOL_ROOT = os.path.normpath(
    os.environ.get('UNIMOL_ROOT', 'E:/Code/Uni-Mol-main')
)


def _scripts_dir(config: dict = None) -> str:
    root = (config or {}).get('unimol_root', _DEFAULT_UNIMOL_ROOT)
    return os.path.join(root, 'scripts')


def _add_scripts_path(config: dict = None):
    sd = _scripts_dir(config)
    if sd not in sys.path:
        sys.path.insert(0, sd)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ESM-2 Mutation Scan
# ═══════════════════════════════════════════════════════════════════════════════

def load_esm2_results(result_dirs: List[str]) -> List[dict]:
    """Load and merge ESM-2 multi-model results from JSON files."""
    all_mutations = {}
    for rdir in result_dirs:
        json_path = os.path.join(rdir, 'esm2_beneficial_mutations.json')
        if not os.path.exists(json_path):
            raise FileNotFoundError(f'ESM-2 results not found: {json_path}')
        with open(json_path, encoding='utf-8-sig') as f:
            data = json.load(f)
        if not isinstance(data, dict) or not isinstance(data.get('mutations'), list):
            raise ValueError(f'ESM-2 results require a mutations list: {json_path}')
        model_name = data.get('model', os.path.basename(rdir))
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(f'ESM-2 model identity must be a non-empty string: {json_path}')
        for index, m in enumerate(data['mutations']):
            if not isinstance(m, dict) or not {'position', 'wt', 'mut', 'mutation', 'delta_ll'} <= m.keys():
                raise ValueError(f'Invalid mutation record {index} in {json_path}')
            if type(m['position']) is not int or m['position'] <= 0:
                raise ValueError(f'Mutation position must be a positive integer: record {index}')
            if not all(isinstance(m[k], str) and len(m[k]) == 1 for k in ('wt', 'mut')):
                raise ValueError(f'Mutation residue codes must be single characters: record {index}')
            if m['mutation'] != f"{m['wt']}{m['position']}{m['mut']}":
                raise ValueError(f'Mutation identifier disagrees with position/residues: record {index}')
            if type(m['delta_ll']) not in (int, float) or not math.isfinite(m['delta_ll']):
                raise ValueError(f'Mutation delta_ll must be a finite number: record {index}')
            key = (m['position'], m['mut'])
            if key not in all_mutations:
                all_mutations[key] = {**m, 'scores': {}}
            if all_mutations[key]['wt'] != m['wt']:
                raise ValueError(f'Conflicting reference residue at position {m["position"]}')
            previous = all_mutations[key]['scores'].get(model_name)
            if previous is not None and previous != m['delta_ll']:
                raise ValueError(f'Conflicting scores for model {model_name}, mutation {m["mutation"]}')
            all_mutations[key]['scores'][model_name] = m['delta_ll']

    merged = []
    for key, info in all_mutations.items():
        scores = info['scores']
        n_models = len(scores)
        avg = sum(scores.values()) / n_models
        if not math.isfinite(avg) or not math.isfinite(avg * (1 + 0.3 * (n_models - 1))):
            raise ValueError('ESM-2 score aggregation overflowed; check saved score scales')
        merged.append({
            'position': info['position'],
            'wt': info['wt'],
            'mut': info['mut'],
            'mutation': info['mutation'],
            'delta_ll': max(scores.values()),
            'avg_delta_ll': avg,
            'consensus_score': avg * (1 + 0.3 * (n_models - 1)),
            'n_models': n_models,
            'model_scores': scores,
        })
    merged.sort(key=lambda x: -x['consensus_score'])
    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Protein Mutagenesis
# ═══════════════════════════════════════════════════════════════════════════════

def mutate_protein(pdb_path: str, mutations: List[str],
                   output_path: str, config: dict = None) -> dict:
    """
    Apply mutations to a PDB file using biopython_mutate.

    Args:
        pdb_path: Input PDB file
        mutations: List of mutation strings (e.g., ['V21L', 'N90G'])
        output_path: Output PDB file path
        config: Optional config dict with 'unimol_root'

    Returns:
        dict with 'success', 'output_pdb', 'mutations_applied'
    """
    _add_scripts_path(config)
    try:
        from biopython_mutate import parse_mutation, mutate_residue
        from Bio.PDB import PDBParser, PDBIO

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_path)

        AA_MAP = {
            'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
            'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
            'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
            'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
        }

        applied = []
        for mut_str in mutations:
            wt, pos, mut = parse_mutation(mut_str)
            new_resname = AA_MAP.get(mut)
            if new_resname:
                mutate_residue(structure, 'A', pos, new_resname)
                applied.append(mut_str)

        io = PDBIO()
        io.set_structure(structure)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        io.save(output_path)

        return {
            'success': True,
            'output_pdb': output_path,
            'mutations_applied': applied,
        }
    except Exception as e:
        logger.error("Mutagenesis failed: %s", e)
        return {'success': False, 'error': str(e), 'mutations_applied': []}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PLIP Interaction Analysis (delegated to zhuangzhou.plip_analysis module)
# ═══════════════════════════════════════════════════════════════════════════════

from .plip_analysis import run_plip_analysis


# ═══════════════════════════════════════════════════════════════════════════════
# 3b. PLIP Visualization (publication-quality PyMOL figures)
# ═══════════════════════════════════════════════════════════════════════════════

def run_plip_visualize(complex_pdb: str, plip_result: dict = None,
                       output_dir: str = None, ligand_resname: str = None,
                       affinity: float = None, rank: int = None,
                       width: int = 3600, height: int = 2700,
                       dpi: int = 600, config: dict = None) -> dict:
    """
    Generate publication-quality interaction figures from PLIP results.

    Uses fast direct-API renderer (Nature Chemical Biology / JACS style).
    Produces 600 DPI PNG with black H-bond dashes, 3-letter residue labels,
    cyan ligand, wheat receptor sticks, orthoscopic projection.

    Args:
        complex_pdb: Complex PDB (protein + ligand)
        plip_result: dict from run_plip_analysis() — if None, runs PLIP first
        output_dir: where to save images and scripts
        ligand_resname: ligand 3-letter code (auto-detected if None)
        affinity: docking affinity kcal/mol (optional annotation)
        rank: docking rank number (optional annotation)
        width/height/dpi: image dimensions and resolution

    Returns:
        dict with png_path, script_path, interactions, render_time
    """
    try:
        # Auto-run PLIP if no result provided
        if plip_result is None:
            plip_result = run_plip_analysis(complex_pdb, config=config)
            if 'error' in plip_result and not plip_result.get('interactions'):
                return {'error': f'PLIP failed: {plip_result["error"]}',
                        'hint': plip_result.get('hint', '')}

        from .pymol_viz import fast_visualize
        return fast_visualize(
            complex_pdb=complex_pdb,
            plip_result=plip_result,
            output_dir=output_dir,
            ligand_resname=ligand_resname,
            affinity=affinity,
            rank=rank,
            width=width,
            height=height,
            dpi=dpi,
        )
    except Exception as e:
        return {'error': str(e), 'hint': 'Check PyMOL at D:/Soft/Pymol3.03'}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Vina Docking
# ═══════════════════════════════════════════════════════════════════════════════

def run_vina_docking(protein_pdb: str, ligand_sdf: str,
                     center: tuple, box_size: tuple = (20, 20, 20),
                     exhaustiveness: int = 32, config: dict = None) -> dict:
    """
    Run Vina docking.

    Args:
        protein_pdb: Protein PDB file
        ligand_sdf: Ligand SDF/MOL2 file
        center: (x, y, z) grid center
        box_size: (sx, sy, sz) grid dimensions
        exhaustiveness: Vina exhaustiveness
        config: with 'vina_path' key

    Returns:
        dict with affinity, output poses, etc.
    """
    _add_scripts_path(config)
    try:
        from vina_docking import VinaDocking
        vina_path = (config or {}).get(
            'vina_path', 'E:/computing bio/MDS-gmx/Docking/vina/vina.exe')
        vd = VinaDocking(vina_path=vina_path)
        result = vd.dock(
            protein_pdb, ligand_sdf,
            center_x=center[0], center_y=center[1], center_z=center[2],
            size_x=box_size[0], size_y=box_size[1], size_z=box_size[2],
            exhaustiveness=exhaustiveness,
        )
        return result
    except ImportError:
        return {'error': 'vina_docking not available'}
    except Exception as e:
        logger.error("Vina docking failed: %s", e)
        return {'error': str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Cofactor Detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_cofactors(pdb_path: str, output_dir: str = None,
                     config: dict = None) -> dict:
    """
    Detect cofactors and metal ions in a PDB file.

    Returns:
        dict with 'cofactors', 'metals', 'md_requirements'
    """
    _add_scripts_path(config)
    try:
        from cofactor_detector import detect_from_pdb as _detect
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        result = _detect(pdb_path)
        return result
    except ImportError:
        return {'error': 'cofactor_detector not available'}
    except Exception as e:
        logger.error("Cofactor detection failed: %s", e)
        return {'error': str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Mutation Classification
# ═══════════════════════════════════════════════════════════════════════════════

def classify_mutations(pdb_path: str, mutations: List[str],
                       binding_site: Set[int] = None,
                       config: dict = None) -> List[dict] | dict:
    """
    Classify mutations by structural role (active_site, buried_core, surface, etc.)

    Returns:
        List of dicts with 'mutation', 'role', 'distance_to_site', etc.
    """
    import re as _re
    _MUT_RE = _re.compile(r'^([A-Za-z*])(\d+)([A-Za-z*-])$')

    def _norm(m):
        """Normalize string 'E45V' → dict required by mutation_classifier.py."""
        if isinstance(m, dict):
            return m
        s = str(m).strip()
        hit = _MUT_RE.match(s)
        if hit:
            return {'mutation': s,
                    'position': int(hit.group(2)),
                    'wt': hit.group(1).upper(),
                    'mut': hit.group(3).upper(),
                    'delta_ll': 0.0}
        return {'mutation': s, 'position': 0, 'wt': '?', 'mut': '?', 'delta_ll': 0.0}

    mutations_dicts = [_norm(m) for m in mutations]

    _add_scripts_path(config)
    try:
        from mutation_classifier import classify_mutations as _classify
        results = _classify(pdb_path, mutations_dicts, binding_site)
        return results
    except ImportError:
        return {'error': 'mutation_classifier backend is unavailable',
                'hint': 'Configure unimol_root with the required scripts; no structural classification was performed'}
    except Exception as e:
        logger.error("Classification failed: %s", e)
        return {'error': f'Mutation classification failed: {e}'}


# ═══════════════════════════════════════════════════════════════════════════════
# 7. GROMACS MD
# ═══════════════════════════════════════════════════════════════════════════════

def run_gromacs_md(pdb_path: str, output_dir: str,
                   simulation_time_ns: float = 10.0,
                   config: dict = None) -> dict:
    """
    Run GROMACS MD simulation pipeline (EM -> NVT -> NPT -> Production).
    Supports protein-ligand complexes via SOBTOP/GAFF parameterization.

    Returns:
        dict with stage results, trajectory paths, performance metrics
    """
    try:
        from zhuangzhou.gromacs_md import run_protein_ligand_md
        return run_protein_ligand_md(
            pdb_path, output_dir, sim_time_ns=simulation_time_ns,
            ff=(config or {}).get('ff', 'amber99sb-ildn'),
        )
    except Exception as e:
        logger.error("GROMACS MD failed: %s", e)
        return {'error': str(e), 'hint': 'Check GROMACS at E:/gmx2025.1_GPU/bin/gmx.exe'}


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Smart Docking (Full Pipeline)
# ═══════════════════════════════════════════════════════════════════════════════

def run_smart_docking(pdb_file: str, ligand_sdf: str = None,
                      ligand_resname: str = None, output_dir: str = None,
                      chain: str = None, exhaustiveness: int = 32,
                      num_modes: int = 9, padding: float = 10.0,
                      config: dict = None, *, center: list = None, box_size: list = None) -> dict:
    """
    Run the full SmartDocking pipeline (self-contained, no external imports).

    Args:
        pdb_file: protein PDB file (may contain ligand)
        ligand_sdf: external ligand SDF (if None, extracts from PDB)
        ligand_resname: which ligand to use from PDB (auto-detected if None)
        output_dir: output directory (default: <pdb_dir>/<stem>_docking/)
        chain: protein chain to keep (None = all)
        exhaustiveness: Vina search exhaustiveness
        num_modes: max docking poses
        padding: docking box padding around ligand (Angstrom)

    Returns:
        dict with docking results, report_path, etc.
    """
    try:
        from zhuangzhou.smartdocking import SmartDocking
        sd = SmartDocking(pdb_file, output_dir=output_dir)
        return sd.run(
            ligand_sdf=ligand_sdf, ligand_resname=ligand_resname,
            chain=chain, exhaustiveness=exhaustiveness,
            num_modes=num_modes, padding=padding, center=center, box_size=box_size)
    except Exception as e:
        return {'error': str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Batch Mutant Docking
# ═══════════════════════════════════════════════════════════════════════════════

def run_batch_mutant_docking(
    protein_pdb: str, ligand_sdf: str,
    mutations: List[str], output_dir: str,
    binding_site_center: tuple = None,
    config: dict = None,
) -> dict:
    """
    Dock a ligand against multiple mutant structures.

    Returns:
        dict mapping mutation → docking result
    """
    results = {}
    for mut in mutations:
        mut_pdb = os.path.join(output_dir, f'mut_{mut}.pdb')
        mut_result = mutate_protein(protein_pdb, [mut], mut_pdb, config)
        if not mut_result['success']:
            results[mut] = {'error': mut_result.get('error', 'mutation failed')}
            continue

        if binding_site_center:
            dock_result = run_vina_docking(
                mut_pdb, ligand_sdf, binding_site_center, config=config)
            results[mut] = dock_result
        else:
            results[mut] = {'mutant_pdb': mut_pdb, 'docking': 'skipped (no center)'}

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Mutate & Minimize — Structure optimization after mutations
# ═══════════════════════════════════════════════════════════════════════════════

def run_mutate_minimize(pdb_path: str, mutations: List[str],
                        output: str = None, mode: str = 'solvated',
                        chain: str = 'A', config: dict = None) -> dict:
    """
    Mutate protein + GROMACS energy minimization → optimized PDB.

    Args:
        pdb_path: Input PDB file
        mutations: List of mutation strings, e.g. ['V9L'] or ['V9L', 'A5G']
        output: Output PDB path (auto-generated if None)
        mode: 'vacuum' (fast, ~10s) or 'solvated' (physical, ~30s)
        chain: Chain ID (default 'A')

    Returns:
        dict with output_pdb, em_converged, potential_energy, max_force, etc.
    """
    try:
        from zhuangzhou.mutate_minimize import mutate_and_minimize
        return mutate_and_minimize(
            pdb_path, mutations, output=output, chain=chain, mode=mode)
    except Exception as e:
        logger.error("Mutate+minimize failed: %s", e)
        return {'error': str(e), 'hint': 'Check GROMACS at E:/gmx2025.1_GPU/bin/gmx.exe'}


def run_minimize_structure(pdb_path: str, output: str = None,
                           mode: str = 'solvated',
                           config: dict = None) -> dict:
    """
    Energy-minimize any protein PDB using GROMACS (no mutation step).

    Args:
        pdb_path: Input PDB file
        output: Output PDB path (auto-generated if None)
        mode: 'vacuum' or 'solvated'

    Returns:
        dict with output_pdb, em_converged, potential_energy, etc.
    """
    try:
        from zhuangzhou.mutate_minimize import minimize_structure
        return minimize_structure(pdb_path, output=output, mode=mode)
    except Exception as e:
        logger.error("Minimize failed: %s", e)
        return {'error': str(e), 'hint': 'Check GROMACS at E:/gmx2025.1_GPU/bin/gmx.exe'}


# ═══════════════════════════════════════════════════════════════════════════════
# 11. LigBuilder V3 — De Novo Ligand Design
# ═══════════════════════════════════════════════════════════════════════════════

def run_ligbuilder(protein_pdb: str, output_dir: str = None,
                   pocket: dict = None, design_params: dict = None,
                   config: dict = None) -> dict:
    """
    Run LigBuilder V3 de novo ligand design.

    Uses SmartDocking pocket definition → LigBuilder cavity + build pipeline.
    Backend: WSL (primary) → Docker (fallback).

    Args:
        protein_pdb: Path to protein PDB
        output_dir: Output directory
        pocket: Pocket definition {'center_x/y/z', 'size_x/y/z'}
        design_params: Override design parameters
        config: Override config

    Returns:
        dict with generated ligands, report, or error
    """
    try:
        from zhuangzhou.ligbuilder_gen import run_ligbuilder as _run
        return _run(protein_pdb, output_dir, pocket, design_params, config)
    except Exception as e:
        logger.error("LigBuilder failed: %s", e)
        return {'error': str(e), 'hint': 'Check WSL + LigBuilder installation'}


# ═══════════════════════════════════════════════════════════════════════════════
# 12. QM Engine (PySCF + geomeTRIC via WSL)
# ═══════════════════════════════════════════════════════════════════════════════

def run_qm_cluster(pdb_path: str, output_dir: str, center_residues: list = None,
                   cutoff: float = 5.0, config: dict = None, **kw) -> dict:
    """Extract QM cluster from PDB active site."""
    try:
        from zhuangzhou.qm_engine import extract_qm_cluster
        return extract_qm_cluster(pdb_path, output_dir,
                                  center_residues=center_residues,
                                  plip_result=kw.get('plip_result'),
                                  cutoff=cutoff, charge=kw.get('charge'),
                                  spin=kw.get('spin'))
    except Exception as e:
        return {'error': str(e), 'hint': 'Check PDB file and BioPython installation'}


def run_dft_optimization(xyz_path: str, output_dir: str, charge: int = 0,
                         spin: int = 0, functional: str = 'b3lyp',
                         basis: str = 'def2-svp', config: dict = None, **kw) -> dict:
    """Run DFT geometry optimization via PySCF + geomeTRIC in WSL."""
    try:
        from zhuangzhou.qm_engine import run_dft_optimization as _run
        return _run(xyz_path, output_dir, charge=charge, spin=spin,
                    functional=functional, basis=basis,
                    constraints_path=kw.get('constraints_path'))
    except Exception as e:
        return {'error': str(e), 'hint': 'Check WSL + PySCF installation'}


def run_dft_single_point(xyz_path: str, output_dir: str, charge: int = 0,
                         spin: int = 0, functional: str = 'b3lyp',
                         basis: str = 'def2-tzvp', config: dict = None, **kw) -> dict:
    """Run high-basis DFT single-point energy via PySCF in WSL."""
    try:
        from zhuangzhou.qm_engine import run_single_point as _run
        return _run(xyz_path, output_dir, charge=charge, spin=spin,
                    functional=functional, basis=basis)
    except Exception as e:
        return {'error': str(e), 'hint': 'Check WSL + PySCF installation'}


def run_ts_search(xyz_path: str, output_dir: str, charge: int = 0,
                  spin: int = 0, functional: str = 'b3lyp',
                  basis: str = 'def2-svp', config: dict = None, **kw) -> dict:
    """Run transition state search via PySCF + geomeTRIC in WSL."""
    try:
        from zhuangzhou.qm_engine import run_ts_search as _run
        return _run(xyz_path, output_dir, charge=charge, spin=spin,
                    functional=functional, basis=basis,
                    constraints_path=kw.get('constraints_path'))
    except Exception as e:
        return {'error': str(e), 'hint': 'Check WSL + PySCF installation'}


def run_frequency_analysis(xyz_path: str, output_dir: str, charge: int = 0,
                           spin: int = 0, functional: str = 'b3lyp',
                           basis: str = 'def2-svp', config: dict = None, **kw) -> dict:
    """Run vibrational frequency analysis via PySCF in WSL."""
    try:
        from zhuangzhou.qm_engine import run_frequency as _run
        return _run(xyz_path, output_dir, charge=charge, spin=spin,
                    functional=functional, basis=basis)
    except Exception as e:
        return {'error': str(e), 'hint': 'Check WSL + PySCF installation'}


def run_ts_pipeline(pdb_path: str, output_dir: str, center_residues: list = None,
                    cutoff: float = 5.0, functional: str = 'b3lyp',
                    config: dict = None, **kw) -> dict:
    """Full TS pipeline: PDB -> QM cluster -> DFT opt -> TS -> Freq -> SP."""
    try:
        from zhuangzhou.qm_engine import run_ts_pipeline as _run
        return _run(pdb_path, output_dir, center_residues=center_residues,
                    plip_result=kw.get('plip_result'), cutoff=cutoff,
                    charge=kw.get('charge'), spin=kw.get('spin'),
                    functional=functional)
    except Exception as e:
        return {'error': str(e), 'hint': 'Check WSL + PySCF + geomeTRIC installation'}


def run_qm_energy_profile(energies: dict, output_path: str,
                           title: str = 'Reaction Energy Profile',
                           level: str = '', config: dict = None, **kw) -> dict:
    """Plot reaction energy barrier diagram from stationary point energies."""
    try:
        from zhuangzhou.qm_engine import plot_energy_profile
        return plot_energy_profile(
            energies, output_path, title=title,
            level=level or kw.get('level', 'DFT'),
            dpi=kw.get('dpi', 300))
    except Exception as e:
        return {'error': str(e), 'hint': 'Check matplotlib installation'}


def run_qm_reaction_analysis(stationary_points: list, output_dir: str,
                              title: str = 'Reaction Energy Profile',
                              level: str = '', config: dict = None, **kw) -> dict:
    """Analyze reaction energies and generate comprehensive report + energy profile."""
    try:
        from zhuangzhou.qm_engine import analyze_reaction
        return analyze_reaction(
            stationary_points, output_dir, title=title, level=level)
    except Exception as e:
        return {'error': str(e), 'hint': 'Check matplotlib installation'}


# ═══════════════════════════════════════════════════════════════════════════════
# 14. Non-Standard Residue Parameterization
# ═══════════════════════════════════════════════════════════════════════════════

def run_detect_nonstandard(pdb_path: str, chain: str = None,
                           config: dict = None, **kw) -> dict:
    """Detect non-standard residues, metals, and coordination bonds in PDB."""
    try:
        from zhuangzhou.nonstandard_res import detect_nonstandard
        return detect_nonstandard(pdb_path, chain=chain)
    except Exception as e:
        return {'error': str(e)}


def run_parameterize_nonstandard(pdb_path: str, work_dir: str,
                                  method: str = 'auto', ff: str = 'amber99sb-ildn',
                                  config: dict = None, **kw) -> dict:
    """Detect and parameterize all non-standard residues in a PDB file."""
    try:
        from zhuangzhou.nonstandard_res import detect_nonstandard, parameterize_system
        detection = detect_nonstandard(pdb_path)
        if detection.get('error'):
            return detection
        return parameterize_system(detection, pdb_path, work_dir, ff=ff, method=method)
    except Exception as e:
        return {'error': str(e)}


def run_build_nonstandard_topology(pdb_path: str, topol_path: str, work_dir: str,
                                    ff: str = 'amber99sb-ildn',
                                    config: dict = None, **kw) -> dict:
    """Full pipeline: detect + parameterize + build topology + restraints."""
    try:
        from zhuangzhou.nonstandard_res import (
            detect_nonstandard, parameterize_system,
            build_topology, generate_restraints,
        )
        detection = detect_nonstandard(pdb_path)
        if detection.get('error'):
            return detection

        if not detection['summary']['needs_special_params']:
            return {'message': 'No non-standard residues found', 'detection': detection}

        params = parameterize_system(detection, pdb_path, work_dir, ff=ff)
        restraints = generate_restraints(detection, work_dir)
        topo = build_topology(detection, params, topol_path, work_dir,
                              restraint_result=restraints)

        return {
            'detection': detection['summary'],
            'parameterization': {
                'residues': len(params.get('residues', [])),
                'metals': len(params.get('metal_ions', [])),
            },
            'restraints': {
                'posre_metal': restraints.get('posre_metal'),
                'posre_cofactor': restraints.get('posre_cofactor'),
                'disre_coord': restraints.get('disre_coord'),
            },
            'topology': topo,
            'error': params.get('error') or topo.get('error'),
        }
    except Exception as e:
        return {'error': str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# Tool Registry - for API/CLI discovery
# ═══════════════════════════════════════════════════════════════════════════════

TOOL_REGISTRY = {
    'esm2_scan': {
        'fn': load_esm2_results,
        'description': 'Load and merge ESM-2 multi-model mutation scan results',
        'requires_gpu': False,
    },
    'mutate_protein': {
        'fn': mutate_protein,
        'description': 'Apply point mutations to a PDB structure',
        'requires_gpu': False,
    },
    'plip_analysis': {
        'fn': run_plip_analysis,
        'description': 'Analyze protein-ligand interactions (PLIP)',
        'requires_gpu': False,
    },
    'plip_visualize': {
        'fn': run_plip_visualize,
        'description': 'Publication-quality PLIP interaction figure (600 DPI, bond types, distances, legend)',
        'requires_gpu': False,
    },
    'vina_docking': {
        'fn': run_vina_docking,
        'description': 'Run AutoDock Vina docking',
        'requires_gpu': False,
    },
    'cofactor_detection': {
        'fn': detect_cofactors,
        'description': 'Detect cofactors and metal ions in PDB',
        'requires_gpu': False,
    },
    'mutation_classification': {
        'fn': classify_mutations,
        'description': 'Classify mutations by structural role',
        'requires_gpu': False,
    },
    'gromacs_md': {
        'fn': run_gromacs_md,
        'description': 'Run GROMACS MD simulation',
        'requires_gpu': True,
    },
    'smart_docking': {
        'fn': run_smart_docking,
        'description': 'PDB → preprocess → active site → Vina docking → report',
        'requires_gpu': False,
    },
    'batch_mutant_docking': {
        'fn': run_batch_mutant_docking,
        'description': 'Dock ligand against multiple mutant structures',
        'requires_gpu': False,
    },
    'mutate_minimize': {
        'fn': run_mutate_minimize,
        'description': 'Mutate protein + GROMACS energy minimization -> optimized PDB',
        'requires_gpu': False,
    },
    'minimize_structure': {
        'fn': run_minimize_structure,
        'description': 'GROMACS energy minimization on any protein PDB',
        'requires_gpu': False,
    },
    'ligbuilder_design': {
        'fn': run_ligbuilder,
        'description': 'De novo ligand design using LigBuilder V3 (WSL/Docker)',
        'requires_gpu': False,
    },
    'qm_cluster': {
        'fn': run_qm_cluster,
        'description': 'Extract QM cluster from PDB active site (PLIP-guided)',
        'requires_gpu': False,
    },
    'dft_optimization': {
        'fn': run_dft_optimization,
        'description': 'DFT geometry optimization (PySCF + geomeTRIC via WSL)',
        'requires_gpu': False,
    },
    'dft_single_point': {
        'fn': run_dft_single_point,
        'description': 'High-basis DFT single-point energy (PySCF via WSL)',
        'requires_gpu': False,
    },
    'ts_search': {
        'fn': run_ts_search,
        'description': 'Transition state search (PySCF + geomeTRIC RS-P-RFO via WSL)',
        'requires_gpu': False,
    },
    'frequency_analysis': {
        'fn': run_frequency_analysis,
        'description': 'Vibrational frequency analysis (verify TS, thermodynamics)',
        'requires_gpu': False,
    },
    'ts_pipeline': {
        'fn': run_ts_pipeline,
        'description': 'Full TS pipeline: PDB -> QM cluster -> DFT opt -> TS -> Freq -> SP',
        'requires_gpu': False,
    },
    'qm_energy_profile': {
        'fn': run_qm_energy_profile,
        'description': 'Plot reaction energy barrier diagram (publication-quality)',
        'requires_gpu': False,
    },
    'qm_reaction_analysis': {
        'fn': run_qm_reaction_analysis,
        'description': 'Analyze reaction energies and generate comprehensive QM report',
        'requires_gpu': False,
    },
    'detect_nonstandard': {
        'fn': run_detect_nonstandard,
        'description': 'Detect non-standard residues, metals, and coordination bonds in PDB',
        'requires_gpu': False,
    },
    'parameterize_nonstandard': {
        'fn': run_parameterize_nonstandard,
        'description': 'Parameterize non-standard residues for GROMACS (pre-built/SOBTOP/QM)',
        'requires_gpu': False,
    },
    'build_nonstandard_topology': {
        'fn': run_build_nonstandard_topology,
        'description': 'Build GROMACS topology with non-standard residues and restraints',
        'requires_gpu': False,
    },
}


def check_tool_availability(config: dict = None) -> dict:
    """Check which tools are available in the current environment."""
    _add_scripts_path(config)
    available = {}
    for name, info in TOOL_REGISTRY.items():
        try:
            # Try importing the underlying module
            info['fn'].__module__
            available[name] = True
        except Exception:
            available[name] = False

    # Check external binaries
    import shutil
    cfg = config or {}
    available['vina_binary'] = os.path.exists(
        cfg.get('vina_path', 'E:/computing bio/MDS-gmx/Docking/vina/vina.exe'))
    available['gmx_binary'] = os.path.exists(
        cfg.get('gmx_path', 'E:/gmx2025.1_GPU/bin/gmx.exe'))
    available['wsl'] = shutil.which('wsl') is not None

    return available

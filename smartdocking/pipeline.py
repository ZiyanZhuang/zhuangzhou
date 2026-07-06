"""
SmartDocking Pipeline
=====================
One-call docking: PDB → preprocess → site → Vina → report.

Usage:
    from zhuangzhou.smartdocking import SmartDocking

    sd = SmartDocking('E:/dockingtest/M6_Best.pdb')
    result = sd.run()             # dock ligand already in PDB
    result = sd.run(ligand_sdf='path/to/ligand.sdf')  # dock external ligand

All defaults come from config.py — edit that file to tune parameters.
"""

import json
import logging
import os
import time

from .config import VINA_DEFAULTS, PREPARE_DEFAULTS, SITE_DEFAULTS, check_availability
from .prepare import (
    preprocess_protein, detect_cofactors, detect_ligands,
    extract_ligand_sdf, pdb_to_pdbqt, sdf_to_pdbqt, generate_complexes,
)
from .site import define_active_site
from .vina import run_vina

logger = logging.getLogger(__name__)


def _auto_select_chain(pdb_file: str) -> "str | None":
    """Detect if PDB is a multimer; if so, pick the best chain for docking.

    Strategy: pick the chain with the most ATOM records (largest chain).
    If only one protein chain exists, return None (no filtering needed).
    """
    chain_atoms = {}
    with open(pdb_file, encoding='utf-8') as f:
        for line in f:
            if line.startswith('ATOM'):
                ch = line[21].strip()
                if ch:
                    chain_atoms[ch] = chain_atoms.get(ch, 0) + 1

    if len(chain_atoms) <= 1:
        return None  # monomer — no chain selection needed

    # Pick chain with most atoms
    best = max(chain_atoms, key=chain_atoms.get)
    logger.info("Multimer: %d chains detected %s → selecting chain %s (%d atoms)",
                len(chain_atoms), dict(chain_atoms), best, chain_atoms[best])
    return best


class SmartDocking:
    """
    Self-contained docking pipeline.

    All output goes to <pdb_dir>/<pdb_stem>_docking/.
    """

    def __init__(self, pdb_file: str, output_dir: str = None):
        self.pdb_file = os.path.abspath(pdb_file)
        self._init_error = None
        if not os.path.exists(self.pdb_file):
            # Store error instead of raising — follows project convention (error dict, never raise)
            self._init_error = f'PDB not found: {self.pdb_file}'
            self.stem = ''
            self.output_dir = ''
            return

        self.stem = os.path.splitext(os.path.basename(self.pdb_file))[0]
        # Convention: output defaults to CWD (project workspace), not input file dir
        self.output_dir = output_dir or os.path.join(os.getcwd(), f'{self.stem}_docking')
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def check() -> dict:
        """Check what's available before running. Returns status dict."""
        return check_availability()

    def run(self, ligand_sdf: str = None, ligand_resname: str = None,
            chain: str = None,
            remove_water: bool = PREPARE_DEFAULTS['remove_water'],
            remove_ions: bool = PREPARE_DEFAULTS['remove_ions'],
            keep_cofactors: bool = PREPARE_DEFAULTS['keep_cofactors'],
            exhaustiveness: int = VINA_DEFAULTS['exhaustiveness'],
            num_modes: int = VINA_DEFAULTS['num_modes'],
            padding: float = SITE_DEFAULTS['padding'],
            flex_pdbqt: str = None,
            timeout: int = VINA_DEFAULTS['timeout']) -> dict:
        """
        Run the full docking pipeline.

        Args:
            ligand_sdf: external ligand SDF file. If None, extracts from PDB.
            ligand_resname: which ligand to extract from PDB (auto-detected if None)
            chain: protein chain to keep (None = all)
            remove_water: remove water molecules
            keep_cofactors: keep HEME/FAD/NAD etc.
            exhaustiveness: Vina search thoroughness
            num_modes: max docking poses
            padding: extra space around ligand for docking box (Angstrom)
            flex_pdbqt: flexible residues PDBQT (optional)
            timeout: Vina timeout in seconds

        Returns:
            dict with full results and report_path
        """
        # Check for init errors (deferred from constructor)
        if self._init_error:
            return {'error': self._init_error, 'hint': 'Check file path'}

        t0 = time.time()
        result = {
            'pdb': self.pdb_file,
            'pdb_id': self.stem,
            'output_dir': self.output_dir,
            # protein_name and ligand_name populated below as data becomes available
        }

        # ── Step 0: Auto-detect multimer and select chain ──
        if chain is None:
            chain = _auto_select_chain(self.pdb_file)
            if chain:
                logger.info("[0/5] Multimer detected → using chain %s", chain)
                result['auto_chain'] = chain

        # ── Step 1: Preprocess protein ──
        logger.info("[1/5] Preprocessing protein...")
        prep = preprocess_protein(
            self.pdb_file, self.output_dir,
            chain=chain, remove_water=remove_water,
            remove_ions=remove_ions, keep_cofactors=keep_cofactors)
        protein_pdb = prep['pdb_path']
        result['preprocess'] = prep

        # ── Step 2: Detect cofactors & ligands ──
        # Use the ORIGINAL PDB (before preprocessing removes ligands), but filter
        # by selected chain so we don't anchor the box on a ligand from chain B
        # when we're docking to chain A.
        logger.info("[2/5] Detecting cofactors and ligands...")
        cofactors = detect_cofactors(self.pdb_file) if keep_cofactors else []
        ligands_all = detect_ligands(self.pdb_file)
        # Filter by chain if a specific chain was selected
        if chain:
            ligands = [l for l in ligands_all if not l['chain'] or l['chain'] == chain]
            cofactors = [c for c in cofactors if not c.get('chain') or c['chain'] == chain]
        else:
            ligands = ligands_all
        result['cofactors'] = [c['resname'] for c in cofactors]
        result['ligands_in_pdb'] = [
            {'resname': l['resname'], 'chain': l['chain'],
             'resid': l['resid'], 'n_atoms': l['n_atoms']}
            for l in ligands
        ]
        logger.info("  Cofactors: %s | Ligands: %s",
                     [c['resname'] for c in cofactors] or 'none',
                     [l['resname'] for l in ligands] or 'none')

        # ── Step 3: Prepare ligand ──
        logger.info("[3/5] Preparing ligand...")
        if ligand_sdf:
            # External ligand provided
            result['ligand_source'] = 'external'
        else:
            # Extract from PDB
            if not ligands:
                return {**result, 'error': 'No ligand found in PDB',
                        'hint': 'Provide ligand_sdf or ensure PDB contains a ligand'}

            # Pick target ligand
            if ligand_resname:
                target = next((l for l in ligands if l['resname'] == ligand_resname), None)
                if not target:
                    return {**result,
                            'error': f'Ligand {ligand_resname} not found in PDB',
                            'available': [l['resname'] for l in ligands]}
            else:
                target = max(ligands, key=lambda l: l['n_atoms'])
                ligand_resname = target['resname']

            ligand_sdf = extract_ligand_sdf(
                self.pdb_file, ligand_resname, self.output_dir)
            result['ligand_source'] = 'extracted'
            result['ligand_resname'] = ligand_resname

        result['ligand_sdf'] = ligand_sdf
        # Set ligand_name from filename stem or resname (used by report builder)
        if ligand_sdf:
            result['ligand_name'] = os.path.splitext(os.path.basename(ligand_sdf))[0]
        elif ligand_resname:
            result['ligand_name'] = ligand_resname

        # ── Step 4: Define active site ──
        logger.info("[4/5] Defining active site...")
        grid = define_active_site(
            protein_pdb, ligands=ligands, cofactors=cofactors, padding=padding)
        result['grid'] = grid

        # Save grid JSON
        grid_path = os.path.join(self.output_dir, 'docking_grid.json')
        with open(grid_path, 'w', encoding='utf-8') as f:
            json.dump(grid, f, indent=2)

        # ── Step 5: Convert & Dock ──
        logger.info("[5/5] Running Vina docking...")

        # Protein → PDBQT
        receptor_pdbqt = pdb_to_pdbqt(
            protein_pdb,
            os.path.join(self.output_dir, f'{self.stem}_receptor.pdbqt'))

        # Ligand → PDBQT
        ligand_pdbqt = sdf_to_pdbqt(
            ligand_sdf,
            os.path.join(self.output_dir, f'{ligand_resname or "ligand"}.pdbqt'))

        # Check for sdf_to_pdbqt error (returns dict on failure)
        if isinstance(ligand_pdbqt, dict) and 'error' in ligand_pdbqt:
            return {**result, **ligand_pdbqt}

        # Vina
        vina_result = run_vina(
            receptor_pdbqt, ligand_pdbqt, grid, self.output_dir,
            exhaustiveness=exhaustiveness, num_modes=num_modes,
            flex_pdbqt=flex_pdbqt, timeout=timeout)
        result['vina'] = vina_result
        # Expose best_score at top level for easy access
        result['best_score'] = vina_result.get('best_affinity')

        # ── Step 6: Generate complex PDBs (protein + each docked pose) ──
        if 'error' not in vina_result and vina_result.get('output_pdbqt'):
            logger.info("[6/6] Generating protein-ligand complexes...")
            lig_name = ligand_resname or 'LIG'
            complexes = generate_complexes(
                protein_pdb, vina_result['output_pdbqt'], self.output_dir,
                self.stem, ligand_resname=lig_name,
                affinities=vina_result.get('affinities', []))
            result['complexes'] = complexes

        elapsed = time.time() - t0
        result['elapsed_s'] = round(elapsed, 1)

        # ── Generate report ──
        report_path = self._generate_report(result)
        result['report_path'] = report_path

        if 'error' not in vina_result:
            n_complexes = len(result.get('complexes', []))
            logger.info("Done in %.1fs. Best: %s kcal/mol. Complexes: %d",
                        elapsed, vina_result.get('best_affinity'), n_complexes)
            logger.info("Report: %s", report_path)
            result['next_steps'] = [
                'plip_analyze: 分析蛋白-配体相互作用 (对最佳pose的complex PDB运行PLIP)',
                'pymol_render: 生成出版质量可视化图片 (渲染对接复合物)',
                'gromacs_md: 运行MD模拟验证对接pose稳定性',
            ]
        else:
            logger.error("Docking failed: %s", vina_result.get('error'))

        return result

    def _generate_report(self, result: dict) -> str:
        """Generate Markdown docking report."""
        lines = [
            f'# SmartDocking Report: {self.stem}',
            '',
            f'**PDB**: `{self.pdb_file}`  ',
            f'**Output**: `{self.output_dir}`  ',
            f'**Time**: {result.get("elapsed_s", "?")}s',
            '',
        ]

        # Preprocessing
        prep = result.get('preprocess', {})
        removed = prep.get('removed', {})
        lines.append('## Preprocessing')
        lines.append(f'- Kept atoms: {prep.get("kept_atoms", "?")}')
        if removed.get('water'):
            lines.append(f'- Removed water: {removed["water"]}')
        if removed.get('ions'):
            lines.append(f'- Removed ions: {removed["ions"]}')
        if removed.get('other_chains'):
            lines.append(f'- Removed other chains: {removed["other_chains"]}')
        lines.append('')

        # Cofactors & Ligands
        if result.get('cofactors'):
            lines.append(f'## Cofactors: {", ".join(result["cofactors"])}')
            lines.append('')
        lines.append(f'## Ligands in PDB')
        for lig in result.get('ligands_in_pdb', []):
            lines.append(f'- **{lig["resname"]}** chain {lig["chain"]} '
                         f'resid {lig["resid"]} ({lig["n_atoms"]} atoms)')
        lines.append(f'- Source: {result.get("ligand_source", "?")}')
        lines.append('')

        # Grid
        grid = result.get('grid', {})
        lines.append('## Docking Box')
        strategy = grid.get('site_strategy', 'unknown')
        detail = grid.get('site_detail', '')
        lines.append(f'- Strategy: **{strategy}**')
        if detail:
            lines.append(f'- {detail}')
        lines.append(f'- Center: ({grid.get("center_x", 0):.1f}, '
                     f'{grid.get("center_y", 0):.1f}, '
                     f'{grid.get("center_z", 0):.1f})')
        lines.append(f'- Size: ({grid.get("size_x", 0):.0f}, '
                     f'{grid.get("size_y", 0):.0f}, '
                     f'{grid.get("size_z", 0):.0f})')
        lines.append('')

        # Vina results
        vina = result.get('vina', {})
        lines.append('## Vina Docking Results')
        if 'error' in vina:
            lines.append(f'**Error**: {vina["error"]}')
        else:
            lines.append(f'- Engine: {vina.get("vina_exe", "?")}')
            lines.append(f'- Exhaustiveness: {vina.get("exhaustiveness", "?")}')
            lines.append(f'- Best affinity: **{vina.get("best_affinity", "?")} kcal/mol**')
            lines.append('')

            affinities = vina.get('affinities', [])
            if affinities:
                lines.append('| Rank | Affinity (kcal/mol) | RMSD l.b. | RMSD u.b. |')
                lines.append('|:----:|:-------------------:|:---------:|:---------:|')
                for mode, aff, rmsd_lb, rmsd_ub in affinities:
                    lines.append(f'| {mode} | {aff:.2f} | {rmsd_lb:.3f} | {rmsd_ub:.3f} |')
        lines.append('')

        # Complexes
        complexes = result.get('complexes', [])
        if complexes:
            lines.append('## Complexes (receptor + docked pose)')
            for cplx in complexes:
                aff = cplx.get('affinity', '?')
                rank = cplx.get('rank', '?')
                lines.append(f'- Rank {rank}: **{aff} kcal/mol** → `{cplx["path"]}`')
            lines.append('')

        report = '\n'.join(lines)
        report_path = os.path.join(self.output_dir, 'docking_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        # ── Also generate Word (.docx) ───────────────────────────────────────
        try:
            from zhuangzhou.report.templates import build_docking_report
            docx_path = os.path.join(self.output_dir, 'docking_report.docx')
            build_docking_report(result, docx_path)
            logger.info("Word report: %s", docx_path)
        except Exception as e:
            logger.warning("Word report skipped (%s) — markdown report available at %s",
                           e, report_path)

        return report_path

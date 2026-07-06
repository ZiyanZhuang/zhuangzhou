"""
UniMol Skills — deep learning molecular tools via sandboxed dplm2 env.

Skills:
  unimol_binding_affinity:  Protein-ligand affinity via UniMol-Docking V2
  unimol_conformer:         3D conformer generation
  unimol_mol_property:      Molecular property prediction
  unimol_check_gpu:         GPU availability check
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from .registry import Skill, SkillParam, SkillRegistry

logger = logging.getLogger(__name__)

BRIDGE_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              'envs', 'bridges', 'unimol_bridge.py')


def _get_sandbox():
    from zhuangzhou.envs import get_sandbox
    return get_sandbox('dplm2')


def _call_bridge(command: str, args: dict, timeout: int = 300) -> dict:
    """Call the unimol bridge script via dplm2 sandbox."""
    sb = _get_sandbox()
    if not sb.available:
        return {'status': 'unavailable', 'error': 'dplm2 conda env not found'}

    result = sb.run_json_bridge(BRIDGE_SCRIPT, {
        'command': command, 'args': args
    }, timeout=timeout)

    if result.success and isinstance(result.data, dict):
        return result.data.get('result', result.data)
    return {'status': 'error', 'error': result.error or 'Bridge call failed'}


def unimol_check_gpu(config: dict = None, **kw) -> dict:
    """Check GPU availability in the dplm2 environment."""
    return _call_bridge('check_gpu', {})


def unimol_binding_affinity(protein_pdb: str, ligand_sdf: str,
                             config: dict = None, **kw) -> dict:
    """
    Predict protein-ligand binding affinity using UniMol-Docking V2.
    Falls back to Vina if UniMol unavailable.
    """
    result = _call_bridge('binding_affinity', {
        'protein_pdb': protein_pdb,
        'ligand_sdf': ligand_sdf,
    }, timeout=600)

    if result.get('status') == 'unavailable':
        return {
            'error': 'UniMol docking unavailable',
            'hint': 'Ensure dplm2 conda env is active and UniMol-Docking V2 is installed. '
                    'Use smart_docking for Vina-based docking instead.',
        }

    return result


def unimol_conformer(smiles: str, n_conformers: int = 10,
                      config: dict = None, **kw) -> dict:
    """Generate 3D conformers from SMILES using UniMol (legacy, returns repr)."""
    return _call_bridge('conformer', {
        'smiles': smiles,
        'n_conformers': n_conformers,
    })


def conformer_generate(smiles: str, output_path: str = None,
                       num_conformers: int = 1, optimize: str = 'mmff',
                       remove_hs: bool = False, mode: str = 'auto',
                       compound_name: str = None,
                       config: dict = None, **kw) -> dict:
    """Generate 3D conformer(s) from SMILES and save as SDF.

    Modes:
      'auto': Try dplm2 bridge first, fallback to local RDKit
      'local': Use local RDKit only (no GPU needed)
      'bridge': Force dplm2 bridge (GPU env)

    If compound_name is provided, validates SMILES against PubChem first.
    """
    # Pre-validation: check SMILES syntax and optionally cross-validate with PubChem
    try:
        from zhuangzhou.molecularconformer import validate_smiles
        val = validate_smiles(smiles)
        if not val.get('valid', True):
            return {'error': f'Invalid SMILES: {smiles}', 'hint': val.get('error', ''),
                    'suggestion': 'Use search_pubchem_compound to get the correct SMILES'}

        # Cross-validate with PubChem if compound name is known
        if compound_name:
            try:
                from zhuangzhou.skills.biodbapi import validate_compound_smiles
                cross_val = validate_compound_smiles(compound_name, smiles)
                if cross_val.get('valid') is False and cross_val.get('correct_smiles'):
                    # Auto-correct: use PubChem's authoritative SMILES
                    correct = cross_val['correct_smiles']
                    logger.warning(
                        "SMILES mismatch for '%s': input=%s, PubChem=%s. Auto-correcting.",
                        compound_name, smiles, correct
                    )
                    smiles = correct
            except Exception as e:
                logger.debug("PubChem cross-validation skipped: %s", e)
    except ImportError:
        pass  # RDKit not available, skip validation

    if mode == 'local' or mode == 'auto':
        try:
            from zhuangzhou.molecularconformer import smiles_to_3d
            result = smiles_to_3d(
                smiles, output_path=output_path,
                num_conformers=num_conformers,
                optimize=optimize, remove_hs=remove_hs,
            )
            if result.get('success'):
                if compound_name:
                    result['compound_name'] = compound_name
                # Expose sdf_path/output_sdf as aliases for output_path
                if result.get('output_path'):
                    result.setdefault('sdf_path', result['output_path'])
                    result.setdefault('output_sdf', result['output_path'])
                return result
            if mode == 'local':
                return result
        except Exception as e:
            if mode == 'local':
                return {'error': str(e), 'hint': 'RDKit may not be installed'}

    # Try bridge (dplm2 env with better RDKit + GPU)
    return _call_bridge('conformer_to_sdf', {
        'smiles': smiles,
        'output_path': output_path or '',
        'num_conformers': num_conformers,
        'optimize': optimize,
        'remove_hs': remove_hs,
    })


def batch_conformer_generate(smiles_list: list, output_dir: str = None,
                              num_conformers: int = 1, optimize: str = 'mmff',
                              config: dict = None, **kw) -> dict:
    """Batch generate 3D conformers for multiple SMILES."""
    try:
        from zhuangzhou.molecularconformer import batch_smiles_to_sdf
        return batch_smiles_to_sdf(
            smiles_list, output_dir=output_dir,
            num_conformers=num_conformers, optimize=optimize,
        )
    except Exception as e:
        return {'error': str(e), 'hint': 'Check RDKit installation'}


def unimol_mol_property(smiles_list: list, properties: list = None,
                         config: dict = None, **kw) -> dict:
    """Predict molecular properties using unimol_tools."""
    return _call_bridge('mol_property', {
        'smiles_list': smiles_list,
        'properties': properties,
    })


def esm2_scan_gpu(fasta_path: str, output_dir: str,
                    model: str = 'esm2_t33_650M_UR50D',
                    config: dict = None, **kw) -> dict:
    """Run ESM-2 zero-shot mutation scan on GPU via dplm2 env."""
    return _call_bridge('esm2_scan', {
        'fasta_path': fasta_path,
        'output_dir': output_dir,
        'model': model,
    }, timeout=3600)


def register_skills(registry: SkillRegistry):
    registry.register(Skill(
        name='unimol_binding_affinity',
        description='Predict binding affinity (UniMol-Docking V2, GPU)',
        category='docking',
        fn=unimol_binding_affinity,
        requires_gpu=True,
        inputs=[
            SkillParam('protein_pdb', 'path', 'Protein PDB file'),
            SkillParam('ligand_sdf', 'path', 'Ligand SDF file'),
        ],
        outputs=[SkillParam('affinity', 'dict', 'Predicted binding affinity')],
        source='UniMol-Docking V2 (dplm2 env)',
    ))

    registry.register(Skill(
        name='unimol_conformer',
        description='Generate 3D conformers from SMILES (UniMol, legacy repr)',
        category='chemistry',
        fn=unimol_conformer,
        requires_gpu=True,
        inputs=[
            SkillParam('smiles', 'string', 'SMILES string'),
            SkillParam('n_conformers', 'int', 'Number of conformers', required=False, default=10),
        ],
        outputs=[SkillParam('conformers', 'dict', '3D conformer data')],
        source='UniMol (dplm2 env)',
    ))

    registry.register(Skill(
        name='conformer_generate',
        description='SMILES to 3D SDF conformer (RDKit+MMFF, UniMol-enhanced optional). '
                    'If compound_name is provided, auto-validates SMILES against PubChem '
                    'and corrects errors. For known compounds, prefer download_pubchem_sdf.',
        category='chemistry',
        fn=conformer_generate,
        requires_gpu=False,
        inputs=[
            SkillParam('smiles', 'string', 'SMILES string'),
            SkillParam('compound_name', 'string',
                       'Compound name for PubChem cross-validation (e.g. "asiatic acid")',
                       required=False),
            SkillParam('output_path', 'path', 'Output SDF path', required=False,
                       is_output_path=True),
            SkillParam('num_conformers', 'int', 'Number of conformers', required=False, default=1),
            SkillParam('optimize', 'string', 'Force field: mmff/uff/None', required=False, default='mmff'),
            SkillParam('remove_hs', 'bool', 'Remove hydrogens from output', required=False, default=False),
            SkillParam('mode', 'string', 'auto/local/bridge', required=False, default='auto'),
        ],
        outputs=[
            SkillParam('output_path', 'path', 'Generated SDF file'),
            SkillParam('energies', 'list', 'Conformer energies (kcal/mol)'),
        ],
        source='zhuangzhou/molecularconformer/',
    ))

    registry.register(Skill(
        name='batch_conformer_generate',
        description='Batch SMILES to 3D SDF conformers',
        category='chemistry',
        fn=batch_conformer_generate,
        requires_gpu=False,
        inputs=[
            SkillParam('smiles_list', 'list', 'List of SMILES strings'),
            SkillParam('output_dir', 'path', 'Output directory', required=False),
            SkillParam('num_conformers', 'int', 'Conformers per molecule', required=False, default=1),
        ],
        outputs=[
            SkillParam('results', 'list', 'Per-molecule results'),
            SkillParam('succeeded', 'int', 'Number succeeded'),
        ],
        source='zhuangzhou/molecularconformer/',
    ))

    registry.register(Skill(
        name='unimol_mol_property',
        description='Predict molecular properties (UniMol-Tools)',
        category='chemistry',
        fn=unimol_mol_property,
        requires_gpu=True,
        inputs=[
            SkillParam('smiles_list', 'list', 'List of SMILES strings'),
        ],
        outputs=[SkillParam('properties', 'dict', 'Predicted properties')],
        source='UniMol-Tools (dplm2 env)',
    ))

    registry.register(Skill(
        name='unimol_check_gpu',
        description='Check GPU availability in deep learning environment',
        category='simulation',
        fn=unimol_check_gpu,
        inputs=[],
        outputs=[SkillParam('gpu_info', 'dict', 'GPU status')],
        source='dplm2 env',
    ))

    registry.register(Skill(
        name='esm2_scan_gpu',
        description='Run ESM-2 mutation scan on GPU (dplm2 env)',
        category='plm',
        fn=esm2_scan_gpu,
        requires_gpu=True,
        inputs=[
            SkillParam('fasta_path', 'path', 'Input FASTA'),
            SkillParam('output_dir', 'path', 'Output directory'),
            SkillParam('model', 'string', 'ESM-2 model', required=False,
                       default='esm2_t33_650M_UR50D'),
        ],
        outputs=[SkillParam('result', 'dict', 'Scan status')],
        source='ESM-2 (dplm2 env)',
    ))

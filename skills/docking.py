"""Docking skills — Vina, SmartDocking, UniMol, batch docking."""

import os
from .registry import Skill, SkillParam, SkillRegistry
from .results import result_error


def vina_docking(protein_pdb: str, ligand_sdf: str,
                 center: tuple, box_size: tuple = (20, 20, 20),
                 exhaustiveness: int = 32, config: dict = None, **kw) -> dict:
    from zhuangzhou.tools import run_vina_docking
    return run_vina_docking(protein_pdb, ligand_sdf, center, box_size, exhaustiveness, config)


def smart_docking(protein_query: str, ligand_query: str,
                  output_dir: str = None, run_vina: bool = True,
                  run_plip: bool = True, config: dict = None, **kw) -> dict:
    if not run_vina:
        return {'error': 'Preparation-only smart_docking is not implemented; no docking was started',
                'hint': 'Use explicit preparation skills, or set run_vina=True'}
    import re as _re

    # Type safety: ensure string input
    if not isinstance(protein_query, str):
        return {'error': f'protein_query must be a string (got {type(protein_query).__name__})',
                'hint': 'Provide a local PDB file path or a 4-character PDB ID (e.g. "1Z10").'}

    pdb_file = protein_query

    # Auto-detect PDB ID (4-char alphanumeric) and download if not a local file
    if not os.path.exists(protein_query):
        pdb_id_match = _re.match(r'^([A-Za-z0-9]{4})$', protein_query.strip())
        if pdb_id_match:
            pdb_id = pdb_id_match.group(1).upper()
            # Download PDB from RCSB
            from .biodbapi import download_pdb_file
            dl_result = download_pdb_file(pdb_id, output_dir or os.getcwd())
            if isinstance(dl_result, dict) and 'error' in dl_result:
                return dl_result
            pdb_file = dl_result.get('pdb_path', protein_query)
        else:
            return {'error': f'Protein file not found: "{protein_query}"',
                    'hint': 'Provide a local PDB file path or a 4-character PDB ID (e.g. 1Z10). '
                            'Use download_pdb skill to download first, or search_pdb_structure to find PDB IDs.'}

    # Resolve ligand: auto-detect name/SMILES/CID and download/generate SDF
    ligand_sdf = ligand_query
    if not os.path.exists(ligand_query):
        lig_dir = output_dir or os.path.join(os.getcwd(), os.path.splitext(os.path.basename(pdb_file))[0] + '_docking')
        os.makedirs(lig_dir, exist_ok=True)

        # Detect SMILES: high ratio of chemical chars relative to total length
        _smiles_chars = set('CNOSPFIBrconsph=#@+\\/-[]()0123456789.')
        _is_smiles = (len(ligand_query) >= 4 and
                      sum(c in _smiles_chars for c in ligand_query) / len(ligand_query) > 0.85 and
                      any(c in ligand_query for c in '=#@[]()'))

        if _is_smiles:
            from .unified import generate_conformer
            conf_result = generate_conformer(smiles=ligand_query, output_dir=lig_dir, config=config)
            if isinstance(conf_result, dict) and 'error' in conf_result:
                return conf_result
            ligand_sdf = conf_result.get('output_path') or conf_result.get('sdf_path', '')
        else:
            # Treat as compound name or CID integer string → download from PubChem
            from .biodbapi import download_pubchem_sdf
            dl = download_pubchem_sdf(
                cid=int(ligand_query) if ligand_query.isdigit() else None,
                name=ligand_query if not ligand_query.isdigit() else None,
                output_dir=lig_dir,
            )
            if isinstance(dl, dict) and 'error' in dl:
                return dl
            sdf_path = dl.get('sdf_path', '')
            # PubChem 3D SDF may already be conformer-ready; generate 3D conformer if it's 2D
            if sdf_path and dl.get('is_3d', False):
                ligand_sdf = sdf_path
            elif sdf_path:
                from .unified import generate_conformer
                smiles = dl.get('isomeric_smiles') or dl.get('canonical_smiles', '')
                if smiles:
                    stem = ligand_query.replace(' ', '_')
                    conf_result = generate_conformer(smiles=smiles, output_dir=lig_dir,
                                                     compound_name=stem, config=config)
                    if isinstance(conf_result, dict) and 'error' not in conf_result:
                        ligand_sdf = conf_result.get('output_path') or conf_result.get('sdf_path', sdf_path)
                    else:
                        ligand_sdf = sdf_path
                else:
                    ligand_sdf = sdf_path
            else:
                return {'error': f'Could not resolve ligand: "{ligand_query}"',
                        'hint': 'Provide an SDF file path, a SMILES string, a compound name, or a PubChem CID.'}

        if not ligand_sdf or not os.path.exists(ligand_sdf):
            return {'error': f'Ligand SDF not found after resolution: "{ligand_sdf}"',
                    'hint': f'Original query was: "{ligand_query}". Check compound name or generate SDF manually.'}

    from zhuangzhou.tools import run_smart_docking
    result = run_smart_docking(pdb_file, ligand_sdf=ligand_sdf, output_dir=output_dir, config=config,
                              center=kw.get('center'), box_size=kw.get('box_size'))
    if result_error(result) or not run_plip:
        return result
    complex_pdb = result.get('complex_pdb')
    if not complex_pdb or not os.path.isfile(complex_pdb):
        return {**result, 'success': False, 'error': 'Docking completed without a readable docked complex for PLIP'}
    from zhuangzhou.tools import run_plip_analysis
    plip = run_plip_analysis(complex_pdb, config=config)
    result = {**result, 'plip': plip}
    if result_error(plip):
        return {**result, 'success': False, 'error': f'Docking completed, PLIP failed: {result_error(plip)}'}
    return result


def batch_mutant_docking(protein_pdb: str, ligand_sdf: str,
                         mutations: list, output_dir: str,
                         binding_site_center: tuple = None,
                         config: dict = None, **kw) -> dict:
    from zhuangzhou.tools import run_batch_mutant_docking
    return run_batch_mutant_docking(protein_pdb, ligand_sdf, mutations,
                                    output_dir, binding_site_center, config)


def generate_docking_grid(ligand_sdf: str, output_json: str = None,
                          add_size: float = 8.0, config: dict = None, **kw) -> dict:
    """Generate docking grid from ligand SDF."""
    import sys
    root = (config or {}).get('unimol_root', 'E:/Code/Uni-Mol-main')
    sd = os.path.join(root, 'scripts')
    if sd not in sys.path:
        sys.path.insert(0, sd)
    try:
        from generate_docking_grid import generate_grid
        return generate_grid(ligand_sdf, output_json, add_size)
    except ImportError:
        return {'error': 'generate_docking_grid not available',
                'hint': 'Requires Uni-Mol-main scripts at E:/Code/Uni-Mol-main'}


def generate_complex(protein_pdb: str, ligand_sdf: str,
                     output_pdb: str = None, config: dict = None, **kw) -> dict:
    """Merge protein PDB + ligand SDF into complex."""
    import sys
    root = (config or {}).get('unimol_root', 'E:/Code/Uni-Mol-main')
    sd = os.path.join(root, 'scripts')
    if sd not in sys.path:
        sys.path.insert(0, sd)
    try:
        from generate_complex import merge_complex
        out = output_pdb or protein_pdb.replace('.pdb', '_complex.pdb')
        merge_complex(protein_pdb, ligand_sdf, out)
        return {'success': True, 'complex_pdb': out}
    except ImportError:
        return {'error': 'generate_complex not available',
                'hint': 'Requires Uni-Mol-main scripts at E:/Code/Uni-Mol-main'}


def register_skills(registry: SkillRegistry):
    registry.register(Skill(
        name='vina_docking',
        description='AutoDock Vina molecular docking',
        category='docking',
        fn=vina_docking,
        inputs=[
            SkillParam('protein_pdb', 'path', 'Protein PDB'),
            SkillParam('ligand_sdf', 'path', 'Ligand SDF/MOL2'),
            SkillParam('center', 'list', 'Grid center (x,y,z)'),
            SkillParam('box_size', 'list', 'Box size', required=False, default=(20, 20, 20)),
            SkillParam('exhaustiveness', 'int', 'Search exhaustiveness', required=False, default=32),
        ],
        outputs=[SkillParam('result', 'dict', 'Docking result with affinity')],
        source='scripts/vina_docking.py',
    ))

    registry.register(Skill(
        name='smart_docking',
        description='Full SmartDocking pipeline: auto-downloads PDB if given a 4-char PDB ID, '
                    'auto-resolves ligand (compound name/SMILES/CID → PubChem SDF → conformer), '
                    'then preprocess→site detection→Vina docking→complex generation→report',
        category='docking',
        fn=smart_docking,
        inputs=[
            SkillParam('protein_query', 'string',
                       'Local PDB file path OR 4-char PDB ID (e.g. "1Z10" — auto-downloads from RCSB)'),
            SkillParam('ligand_query', 'string',
                       'Local SDF path, compound name (e.g. "nicotine"), SMILES string, or PubChem CID — auto-resolved'),
            SkillParam('output_dir', 'path', 'Output directory (default: CWD/<protein>_docking/)',
                       required=False, is_output_path=True),
            SkillParam('run_vina', 'bool', 'Run Vina docking', required=False, default=True),
            SkillParam('run_plip', 'bool', 'Run PLIP interaction analysis', required=False, default=True),
            SkillParam('center', 'list', 'Optional explicit grid center x,y,z in Angstrom', required=False),
            SkillParam('box_size', 'list', 'Optional box dimensions x,y,z in Angstrom', required=False),
        ],
        outputs=[
            SkillParam('result', 'dict', 'Docking results + interactions'),
            SkillParam('resolution_method', 'string',
                       'How ligand was resolved: local_file|smiles|pubchem_name|pubchem_cid'),
        ],
        source='scripts/smart_docking.py',
        output_type='docking',
    ))

    registry.register(Skill(
        name='batch_mutant_docking',
        description='Dock ligand against multiple mutant structures',
        category='docking',
        fn=batch_mutant_docking,
        inputs=[
            SkillParam('protein_pdb', 'path', 'Wild-type protein PDB'),
            SkillParam('ligand_sdf', 'path', 'Ligand SDF'),
            SkillParam('mutations', 'list', 'Mutation strings'),
            SkillParam('output_dir', 'path', 'Output directory', is_output_path=True),
        ],
        outputs=[SkillParam('results', 'dict', 'Mutation -> docking result map')],
        source='scripts/batch_mutant_docking.py',
        depends_on=['smart_docking'],
        output_type='docking',
    ))

    # Check if Uni-Mol-main scripts are available (avoid ghost skills)
    _unimol_available = os.path.isdir(os.path.expandvars('E:/Code/Uni-Mol-main/scripts'))

    registry.register(Skill(
        name='generate_docking_grid',
        description='Generate docking grid from ligand coordinates',
        category='docking',
        fn=generate_docking_grid,
        inputs=[SkillParam('ligand_sdf', 'path', 'Ligand SDF file')],
        outputs=[SkillParam('grid', 'dict', 'Grid center + size')],
        source='scripts/generate_docking_grid.py',
        available=_unimol_available,
    ))

    registry.register(Skill(
        name='generate_complex',
        description='Merge protein PDB + ligand SDF into complex',
        category='docking',
        fn=generate_complex,
        inputs=[
            SkillParam('protein_pdb', 'path', 'Protein PDB'),
            SkillParam('ligand_sdf', 'path', 'Ligand SDF'),
        ],
        outputs=[SkillParam('complex_pdb', 'path', 'Merged complex PDB')],
        source='scripts/generate_complex.py',
        available=_unimol_available,
    ))

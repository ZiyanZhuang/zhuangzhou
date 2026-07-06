"""
LigBuilder V3 Skills
=====================
De novo ligand design skills using LigBuilder V3.
"""

from .registry import Skill, SkillParam, SkillRegistry


def ligbuilder_cavity(protein_pdb: str, output_dir: str = None,
                      pocket: dict = None, config: dict = None) -> dict:
    """Detect binding cavities in a protein using LigBuilder V3."""
    from zhuangzhou.ligbuilder_gen import run_ligbuilder_cavity
    return run_ligbuilder_cavity(protein_pdb, output_dir, pocket, config)


def ligbuilder_design(protein_pdb: str, output_dir: str = None,
                      pocket: dict = None, design_params: dict = None,
                      config: dict = None) -> dict:
    """Generate de novo ligands using LigBuilder V3."""
    from zhuangzhou.ligbuilder_gen import run_ligbuilder
    return run_ligbuilder(protein_pdb, output_dir, pocket, design_params, config)


def ligbuilder_inhibitor(protein_pdb: str, output_dir: str = None,
                         config: dict = None) -> dict:
    """
    Auto-detect pocket and generate high-affinity inhibitors.
    Uses SmartDocking pocket definition → LigBuilder de novo design.
    """
    from zhuangzhou.smartdocking.prepare import PreProcessor
    from zhuangzhou.smartdocking.site import define_active_site
    from zhuangzhou.ligbuilder_gen import run_ligbuilder

    # Auto-detect pocket using SmartDocking
    try:
        pp = PreProcessor(protein_pdb)
        pp_result = pp.preprocess()
        ligands = pp_result.get('ligands', [])
        cofactors = pp_result.get('cofactors', [])
        pocket = define_active_site(protein_pdb, ligands, cofactors)
    except Exception as e:
        pocket = None  # Will use whole-protein cavity detection

    # Optimized design params for inhibitor generation
    inhibitor_params = {
        'mode': 'explore',
        'sessions': 10,
        'molecules': 1000,
        'min_pkd': 4.0,
        'max_mw': 500,
        'max_logp': 5.0,
        'max_hbd': 5,
        'max_hba': 10,
    }

    return run_ligbuilder(protein_pdb, output_dir, pocket, inhibitor_params, config)


def register_skills(registry: SkillRegistry):
    """Register LigBuilder skills."""
    registry.register(Skill(
        name='ligbuilder_cavity',
        description='Detect binding cavities using LigBuilder V3',
        category='docking',
        fn=ligbuilder_cavity,
        inputs=[
            SkillParam('protein_pdb', 'path', 'Protein PDB file'),
            SkillParam('output_dir', 'path', 'Output directory', required=False, is_output_path=True),
            SkillParam('pocket', 'dict', 'Pocket definition from SmartDocking', required=False),
        ],
        outputs=[
            SkillParam('pocket_file', 'path', 'Pocket atoms file'),
            SkillParam('grid_file', 'path', 'Grid file'),
        ],
        requires_wsl=True,
        source='zhuangzhou/ligbuilder_gen/pipeline.py',
    ))

    registry.register(Skill(
        name='ligbuilder_design',
        description='De novo ligand design using LigBuilder V3',
        category='docking',
        fn=ligbuilder_design,
        inputs=[
            SkillParam('protein_pdb', 'path', 'Protein PDB file'),
            SkillParam('output_dir', 'path', 'Output directory', required=False, is_output_path=True),
            SkillParam('pocket', 'dict', 'Pocket definition', required=False),
            SkillParam('design_params', 'dict', 'Design parameters', required=False),
        ],
        outputs=[
            SkillParam('sdf_files', 'list', 'Generated ligand SDF files'),
            SkillParam('report_html', 'path', 'HTML report'),
        ],
        requires_wsl=True,
        source='zhuangzhou/ligbuilder_gen/pipeline.py',
    ))

    registry.register(Skill(
        name='ligbuilder_inhibitor',
        description='Auto-detect pocket & generate high-affinity inhibitors (SmartDocking + LigBuilder)',
        category='docking',
        fn=ligbuilder_inhibitor,
        inputs=[
            SkillParam('protein_pdb', 'path', 'Protein PDB file'),
            SkillParam('output_dir', 'path', 'Output directory', required=False, is_output_path=True),
        ],
        outputs=[
            SkillParam('sdf_files', 'list', 'Generated inhibitor SDF files'),
            SkillParam('n_ligands', 'int', 'Number of ligands generated'),
        ],
        requires_wsl=True,
        source='zhuangzhou/ligbuilder_gen/pipeline.py',
    ))

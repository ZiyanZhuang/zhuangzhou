"""Chemical intuition skills — first-principles mutation analysis."""

from .registry import Skill, SkillParam, SkillRegistry


def analyze_mutation(mutation: str, role: str = 'unknown',
                     secondary_structure: str = 'loop',
                     is_conserved: bool = False,
                     distance_to_site: float = None, **kw) -> dict:
    """Analyze a single mutation with chemical intuition."""
    from zhuangzhou.chemical_intuition import analyze_mutation_chemistry

    wt_aa = mutation[0]
    mut_aa = mutation[-1]

    context = {
        'role': role,
        'secondary_structure': secondary_structure,
        'interactions': [],
        'neighbors': [],
        'distance_to_site': distance_to_site,
        'sasa': None,
        'in_conserved_motif': is_conserved,
    }
    return analyze_mutation_chemistry(wt_aa, mut_aa, context)


def run_pipeline(pdb_path: str, binding_site: set, esm2_dirs: list = None,
                 enzyme_class: str = '', ligand: str = '',
                 protein_name: str = '', conserved_positions: set = None,
                 output_dir: str = None, config: dict = None, **kw) -> dict:
    """Run the full 5-phase Zhuangzhou pipeline."""
    from zhuangzhou.core import ZhuangzhouFramework
    fw = ZhuangzhouFramework(config or {})
    return fw.run_enzyme_engineering(
        pdb_path=pdb_path,
        binding_site=binding_site,
        esm2_dirs=esm2_dirs or [],
        enzyme_class=enzyme_class,
        ligand=ligand,
        protein_name=protein_name,
        conserved_positions=conserved_positions or set(),
        output_dir=output_dir,
    )


def register_skills(registry: SkillRegistry):
    registry.register(Skill(
        name='chemical_analysis',
        description='Analyze mutation with chemical intuition (first-principles)',
        category='chemistry',
        fn=analyze_mutation,
        inputs=[
            SkillParam('mutation', 'string', 'Mutation string (e.g. V21L)'),
            SkillParam('role', 'string', 'Structural role', required=False, default='unknown'),
            SkillParam('secondary_structure', 'string', 'Secondary structure', required=False),
            SkillParam('is_conserved', 'bool', 'Is position conserved', required=False),
        ],
        outputs=[SkillParam('result', 'dict', 'Score, reasoning, warnings')],
        source='zhuangzhou/chemical_intuition.py',
    ))

    registry.register(Skill(
        name='zhuangzhou_pipeline',
        description='Full 5-phase enzyme engineering pipeline',
        category='chemistry',
        fn=run_pipeline,
        inputs=[
            SkillParam('pdb_path', 'path', 'Protein PDB file'),
            SkillParam('binding_site', 'set', 'Active site residue numbers'),
            SkillParam('esm2_dirs', 'list', 'ESM-2 result directories', required=False),
            SkillParam('enzyme_class', 'string', 'Enzyme class', required=False),
        ],
        outputs=[SkillParam('results', 'dict', 'Pipeline results + report')],
        source='zhuangzhou/core.py',
    ))

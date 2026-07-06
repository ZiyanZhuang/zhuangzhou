"""Mutagenesis skills — self-contained BioPython + PyMOL mutation tools."""

from .registry import Skill, SkillParam, SkillRegistry


def _mutate_skill(pdb_path: str, mutations: list,
                  output_path: str = None, config: dict = None, **kw) -> dict:
    """Skill wrapper for zhuangzhou.mutagenesis.mutate."""
    from zhuangzhou.mutagenesis import mutate
    chain = (config or {}).get('chain', kw.get('chain', 'A'))
    rebuild = (config or {}).get('rebuild', kw.get('rebuild', True))
    return mutate(pdb_path, mutations, chain=chain, output=output_path, rebuild=rebuild)


def register_skills(registry: SkillRegistry):
    registry.register(Skill(
        name='mutate_protein',
        description='Apply single/multi point mutations to PDB (PyMOL rotamer rebuild + BioPython fallback)',
        category='mutagenesis',
        fn=_mutate_skill,
        inputs=[
            SkillParam('pdb_path', 'path', 'Input PDB file'),
            SkillParam('mutations', 'list', 'Mutation strings, e.g. ["V9L"] or ["V9L","A5G"]'),
            SkillParam('output_path', 'path', 'Output PDB path (auto-generated if omitted)', required=False),
        ],
        outputs=[SkillParam('result', 'dict', 'Mutation result with output_pdb, details, method')],
        source='zhuangzhou/mutagenesis/mutator.py',
        output_type='pdb',
    ))

"""Non-standard residue parameterization skills for GROMACS MD."""

from .registry import Skill, SkillParam, SkillRegistry


def _detect_nonstandard(pdb_path: str, chain: str = None,
                        config: dict = None, **kw) -> dict:
    from zhuangzhou.nonstandard_res import detect_nonstandard
    return detect_nonstandard(pdb_path, chain=chain)


def _parameterize_nonstandard(pdb_path: str, work_dir: str,
                               method: str = 'auto', ff: str = 'amber99sb-ildn',
                               config: dict = None, **kw) -> dict:
    from zhuangzhou.nonstandard_res import detect_nonstandard, parameterize_system
    detection = detect_nonstandard(pdb_path)
    if detection.get('error'):
        return detection
    return parameterize_system(detection, pdb_path, work_dir, ff=ff, method=method)


def _build_nonstandard_topology(pdb_path: str, topol_path: str, work_dir: str,
                                 ff: str = 'amber99sb-ildn',
                                 config: dict = None, **kw) -> dict:
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
        'parameterization': params,
        'restraints': restraints,
        'topology': topo,
    }


def register_skills(registry: SkillRegistry):
    registry.register(Skill(
        name='detect_nonstandard',
        description='Detect non-standard residues, metal ions, and coordination bonds in PDB',
        category='simulation',
        fn=_detect_nonstandard,
        inputs=[
            SkillParam('pdb_path', 'path', 'Input PDB file'),
            SkillParam('chain', 'string', 'Chain filter (optional)', required=False),
        ],
        outputs=[SkillParam('result', 'dict',
                            'nonstandard_residues, metal_ions, coordination_bonds, summary')],
        source='zhuangzhou/nonstandard_res/detect.py',
    ))

    registry.register(Skill(
        name='parameterize_nonstandard',
        description='Parameterize non-standard residues for GROMACS (pre-built DB / SOBTOP / QM)',
        category='simulation',
        fn=_parameterize_nonstandard,
        inputs=[
            SkillParam('pdb_path', 'path', 'Input PDB with non-standard residues'),
            SkillParam('work_dir', 'path', 'Output directory for .itp/.gro files'),
            SkillParam('method', 'string', 'auto/prebuilt/sobtop/qm', required=False, default='auto',
                       choices=['auto', 'prebuilt', 'sobtop', 'qm']),
            SkillParam('ff', 'string', 'Force field name', required=False, default='amber99sb-ildn'),
        ],
        outputs=[SkillParam('result', 'dict', 'Per-residue itp/gro paths, method used')],
        source='zhuangzhou/nonstandard_res/parameterize.py',
    ))

    registry.register(Skill(
        name='build_nonstandard_topology',
        description='Full pipeline: detect + parameterize + topology + restraints for metalloprotein MD',
        category='simulation',
        fn=_build_nonstandard_topology,
        inputs=[
            SkillParam('pdb_path', 'path', 'Input PDB file'),
            SkillParam('topol_path', 'path', 'Existing topol.top from pdb2gmx'),
            SkillParam('work_dir', 'path', 'Working directory'),
            SkillParam('ff', 'string', 'Force field', required=False, default='amber99sb-ildn'),
        ],
        outputs=[SkillParam('result', 'dict', 'Detection, parameterization, restraints, topology')],
        source='zhuangzhou/nonstandard_res/',
    ))

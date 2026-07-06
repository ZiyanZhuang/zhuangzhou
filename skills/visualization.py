"""Visualization skills -- PyMOL + ChimeraX publication-quality figures."""

import os

from .registry import Skill, SkillParam, SkillRegistry

# ── Temporary skill disable list ──
# Add skill name here to stop it from being registered (and thus never called by agent).
# Reason: ChimeraX GUI mode unreliable on Windows headless; ChatMol LLM loop too slow.
# Remove from set to re-enable.
_DISABLED_VIZ_SKILLS: frozenset = frozenset({
    'chimerax_plip',
    'chimerax_render',
    'chimerax_trajectory',
    'chimerax_defattr',
    'chimerax_comparison',
    'chimerax_porcupine',
    'chimerax_morph',
    'chatmol_viz',
})


def pymol_visualize(pdb_path: str, selections: list = None,
                    output_png: str = None, config: dict = None, **kw) -> dict:
    """Open structure in PyMOL with optional selections highlighted."""
    import subprocess
    pymol_path = (config or {}).get('pymol_path', 'D:/Soft/Pymol3.03/PyMOLWin.exe')
    if not os.path.exists(pymol_path):
        return {'error': f'PyMOL not found at {pymol_path}'}
    subprocess.Popen([pymol_path, pdb_path])
    return {'success': True, 'note': 'PyMOL launched'}


def plip_visualize(complex_pdb: str, plip_result: dict = None,
                   output_dir: str = None, ligand_resname: str = None,
                   style: str = 'publication', render: bool = True,
                   global_view: bool = True, config: dict = None, **kw) -> dict:
    """
    Generate publication-quality interaction figures from PLIP results.

    Produces 600 DPI images with typed/colored interaction dashes,
    distance labels, residue annotations, and a color legend.
    Both detail (binding site) and global (full protein) views.
    """
    try:
        from zhuangzhou.pymol_viz import visualize_plip
        return visualize_plip(
            complex_pdb=complex_pdb,
            plip_result=plip_result,
            output_dir=output_dir,
            ligand_resname=ligand_resname,
            style=style,
            render=render,
            global_view=global_view,
        )
    except Exception as e:
        return {'error': str(e), 'hint': 'Check PyMOL path in pymol_viz/config.py'}


def chimerax_plip(complex_pdb: str, plip_result: dict = None,
                  output_dir: str = None, ligand_resname: str = None,
                  affinity: float = None, rank: int = None,
                  style: str = 'publication', overview: bool = True,
                  timeout: int = 90, config: dict = None, **kw) -> dict:
    """
    Fast publication-quality interaction figure using ChimeraX (5-15s, no LLM needed).
    Uses ChimeraX native hbonds + PLIP-derived interaction data.
    Produces both detail (binding site zoom) and global (full protein) views.
    """
    try:
        from zhuangzhou.chimerax_viz.interactions import (
            generate_interaction_overview, visualize_plip_interactions
        )
        if overview:
            return generate_interaction_overview(
                complex_pdb=complex_pdb,
                plip_result=plip_result or {},
                output_dir=output_dir,
                ligand_resname=ligand_resname,
                timeout=timeout,
            )
        return visualize_plip_interactions(
            complex_pdb=complex_pdb,
            plip_result=plip_result or {},
            output_dir=output_dir,
            ligand_resname=ligand_resname,
            affinity=affinity,
            rank=rank,
            style=style,
            timeout=timeout,
        )
    except Exception as e:
        return {'error': str(e), 'hint': 'Check ChimeraX path in chimerax_viz/config.py'}


def chatmol_visualize(complex_pdb: str, plip_result: dict = None,
                      output_dir: str = None, ligand_resname: str = None,
                      affinity: float = None, rank: int = None,
                      provider: str = 'deepseek', model: str = 'deepseek-chat',
                      timeout: int = 180, config: dict = None, **kw) -> dict:
    """
    Generate publication-quality protein-ligand interaction figure using
    the ChatMol v2 agentic PyMOL pipeline.

    Requires PyMOL python at D:/Soft/Pymol3.03/python.exe and an LLM API key
    in environment (DEEPSEEK_API_KEY, OPENROUTER_API_KEY, etc.).
    """
    try:
        from zhuangzhou.pymol_viz.chatmol_bridge import run_chatmol_viz
        return run_chatmol_viz(
            complex_pdb=complex_pdb,
            plip_result=plip_result or {},
            output_dir=output_dir,
            ligand_resname=ligand_resname,
            affinity=affinity,
            rank=rank,
            provider=provider,
            model=model,
            timeout=timeout,
        )
    except Exception as e:
        return {'error': str(e), 'hint': 'Check PyMOL path and LLM API key env vars'}


def pymol_render_verified(
    complex_pdb: str,
    plip_result: dict = None,
    output_dir: str = None,
    ligand_resname: str = None,
    backend=None,
    config: dict = None,
    **kw,
) -> dict:
    """
    Publication-quality PyMOL render with automatic vision verification.

    Runs fast_visualize() then sends the output PNG to the LLM backend
    (via capture_viewer_snapshot pattern from PyMolAI) to check for empty
    scenes, missing ligands, invisible interactions, and render failures.

    Returns the render result enriched with a 'verification' key.
    """
    try:
        from zhuangzhou.pymol_viz.fast_render import fast_visualize
        from zhuangzhou.pymol_viz.vision_verify import verify_render

        result = fast_visualize(
            complex_pdb=complex_pdb,
            plip_result=plip_result or {},
            output_dir=output_dir,
            ligand_resname=ligand_resname,
        )

        png_path = result.get('png_path') or result.get('output_png')
        if png_path and os.path.exists(png_path):
            context = f"Complex: {os.path.basename(complex_pdb)}"
            if ligand_resname:
                context += f", Ligand: {ligand_resname}"
            verify = verify_render(png_path, backend=backend, context=context)
            result['verification'] = verify
            if not verify.get('ok') and not verify.get('skipped'):
                result['warning'] = (
                    f"Render quality issues: {'; '.join(verify.get('issues', []))}"
                )

        return result
    except Exception as e:
        return {'error': str(e), 'hint': 'Check PyMOL path in pymol_viz/config.py'}


def register_skills(registry: SkillRegistry):
    def _reg(skill: Skill):
        if skill.name not in _DISABLED_VIZ_SKILLS:
            registry.register(skill)

    _reg(Skill(
        name='chimerax_plip',
        description='Fast protein-ligand interaction figure via ChimeraX (5-15s, no LLM). '
                    'Native hbond detection + PLIP data. Detail + global views, 3600x2700 PNG.',
        category='visualization',
        fn=chimerax_plip,
        inputs=[
            SkillParam('complex_pdb', 'path', 'Complex PDB (protein + ligand)'),
            SkillParam('plip_result', 'dict', 'PLIP analysis result dict', required=False),
            SkillParam('output_dir', 'path', 'Output directory', required=False),
            SkillParam('ligand_resname', 'string', 'Ligand 3-letter code (auto-detected if omitted)', required=False),
            SkillParam('affinity', 'float', 'Docking affinity kcal/mol (for metadata)', required=False),
            SkillParam('style', 'string', 'publication/presentation/minimal', required=False,
                       default='publication', choices=['publication', 'presentation', 'minimal']),
            SkillParam('overview', 'bool', 'Generate both detail + global views', required=False, default=True),
            SkillParam('timeout', 'int', 'ChimeraX timeout seconds', required=False, default=90),
        ],
        outputs=[
            SkillParam('detail_png', 'path', 'Binding site detail PNG'),
            SkillParam('global_png', 'path', 'Full protein global PNG'),
            SkillParam('cxc_path', 'path', 'ChimeraX .cxc script'),
        ],
        source='zhuangzhou/chimerax_viz/interactions.py',
        output_type='visualization',
        estimated_time='~5-15s',
        when_to_use=(
            "Default visualization choice after plip_analysis or smart_docking. "
            "User says 'ChimeraX', '快速', 'fast'. No API key needed. Fastest option (~5-15s)."
        ),
        when_not_to_use=(
            "User explicitly requests AI-generated/ChatMol style. "
            "User needs very artistic quality above speed."
        ),
        fallback_skills=["plip_visualize"],
        workflow_stage="visualize",
    ))

    _reg(Skill(
        name='chatmol_viz',
        description='Generate publication-quality protein-ligand interaction PNG using ChatMol v2 agentic PyMOL (Nature/JACS style, 600 DPI)',
        category='visualization',
        fn=chatmol_visualize,
        inputs=[
            SkillParam('complex_pdb', 'path', 'Complex PDB file (protein + ligand)'),
            SkillParam('plip_result', 'dict', 'PLIP analysis result (from plip_analysis skill)', required=False),
            SkillParam('output_dir', 'path', 'Output directory for PNG and logs', required=False),
            SkillParam('ligand_resname', 'string', 'Ligand 3-letter residue code', required=False),
            SkillParam('affinity', 'float', 'Docking affinity kcal/mol (for caption)', required=False),
            SkillParam('rank', 'int', 'Docking rank number (for caption)', required=False),
            SkillParam('provider', 'string', 'LLM provider: deepseek/openrouter/kimi', required=False, default='deepseek'),
            SkillParam('model', 'string', 'LLM model name', required=False, default='deepseek-chat'),
            SkillParam('timeout', 'int', 'Max seconds for agent loop', required=False, default=180),
        ],
        outputs=[
            SkillParam('png_path', 'path', 'Publication-quality PNG'),
            SkillParam('output_dir', 'path', 'Output directory'),
            SkillParam('chatmol_log', 'path', 'ChatMol agent log'),
        ],
        source='zhuangzhou/pymol_viz/chatmol_bridge.py',
        output_type='visualization',
        estimated_time='~2-3min',
        when_to_use=(
            "User explicitly says 'ChatMol', 'AI生成', 'best quality artistic'. "
            "API key available in DEEPSEEK_API_KEY env var."
        ),
        when_not_to_use=(
            "No API key. User wants fast results. Timeout < 3 minutes. Default visualization."
        ),
        fallback_skills=["chimerax_plip"],
        workflow_stage="visualize",
    ))

    _reg(Skill(
        name='pymol_render_verified',
        description=(
            'Publication-quality PyMOL PLIP render with automatic vision verification. '
            'Same as plip_visualize but sends output PNG to LLM vision to detect issues '
            '(empty scene, black image, missing ligand, invisible interactions). '
            'Reports quality score and fix hints. Preferred over plip_visualize when LLM backend available.'
        ),
        category='visualization',
        fn=pymol_render_verified,
        inputs=[
            SkillParam('complex_pdb', 'path', 'Complex PDB (protein + ligand)'),
            SkillParam('plip_result', 'dict', 'PLIP analysis result dict', required=False),
            SkillParam('output_dir', 'path', 'Output directory', required=False),
            SkillParam('ligand_resname', 'string', 'Ligand 3-letter code', required=False),
        ],
        outputs=[
            SkillParam('png_path', 'path', 'Rendered PNG'),
            SkillParam('verification', 'dict', 'Vision QC: {ok, quality, issues, fix_hints, summary}'),
        ],
        source='zhuangzhou/pymol_viz/vision_verify.py',
        output_type='visualization',
        estimated_time='~10-30s',
        when_to_use=(
            'After plip_analysis or smart_docking when LLM backend is active. '
            'Use to catch render failures before including in report. '
            'Default for interactive REPL sessions.'
        ),
        fallback_skills=['plip_visualize', 'chimerax_plip'],
        workflow_stage='visualize',
    ))

    _reg(Skill(
        name='pymol_view',
        description='Open structure in PyMOL for visualization',
        category='visualization',
        fn=pymol_visualize,
        inputs=[
            SkillParam('pdb_path', 'path', 'PDB file to visualize'),
            SkillParam('selections', 'list', 'Residue selections to highlight', required=False),
        ],
        outputs=[SkillParam('result', 'dict', 'Launch status')],
        source='PyMOL',
        output_type='visualization',
    ))

    _reg(Skill(
        name='plip_visualize',
        description='Publication-quality PLIP interaction figure (600 DPI, typed bonds, distance labels, legend)',
        category='visualization',
        fn=plip_visualize,
        inputs=[
            SkillParam('complex_pdb', 'path', 'Complex PDB (protein + ligand)'),
            SkillParam('plip_result', 'dict', 'PLIP analysis result dict', required=False),
            SkillParam('output_dir', 'path', 'Output directory', required=False),
            SkillParam('ligand_resname', 'string', 'Ligand 3-letter code', required=False),
            SkillParam('style', 'string', 'publication or presentation', required=False),
            SkillParam('render', 'bool', 'Run PyMOL headless to produce PNG', required=False),
            SkillParam('global_view', 'bool', 'Also render global overview', required=False),
        ],
        outputs=[
            SkillParam('png_path', 'path', 'Detail view PNG'),
            SkillParam('global_png', 'path', 'Global view PNG'),
            SkillParam('script_path', 'path', 'PyMOL .pml script'),
        ],
        source='zhuangzhou.pymol_viz',
        when_to_use="PyMOL-based PLIP visualization. Fallback when ChimeraX not installed.",
        fallback_skills=["chimerax_plip"],
        workflow_stage="visualize",
    ))

    # ── ChimeraX skills ──

    _reg(Skill(
        name='chimerax_render',
        description='ChimeraX structure rendering (cartoon/surface/stick, PNG + .cxc script)',
        category='visualization',
        fn=_chimerax_render,
        inputs=[
            SkillParam('pdb_path', 'path', 'PDB file to visualize'),
            SkillParam('output_dir', 'path', 'Output directory', required=False),
            SkillParam('style', 'string', 'cartoon/ribbon/surface/stick', required=False, default='cartoon',
                       choices=['cartoon', 'ribbon', 'surface', 'stick']),
            SkillParam('color_by', 'string', 'chain/secondary/bfactor/element', required=False, default='chain',
                       choices=['chain', 'secondary', 'bfactor', 'element']),
            SkillParam('highlight_residues', 'string', 'Comma-separated residues to show as sticks', required=False),
            SkillParam('ligand_resname', 'string', 'Ligand 3-letter code', required=False),
        ],
        outputs=[
            SkillParam('cxc_path', 'path', 'ChimeraX .cxc script'),
            SkillParam('png_path', 'path', 'Rendered PNG'),
        ],
        source='zhuangzhou/chimerax_viz/render.py',
        output_type='visualization',
    ))

    _reg(Skill(
        name='chimerax_trajectory',
        description='ChimeraX MD trajectory animation script (.cxc) with coordset playback',
        category='visualization',
        fn=_chimerax_trajectory,
        inputs=[
            SkillParam('pdb_path', 'path', 'Multi-model PDB trajectory'),
            SkillParam('output_dir', 'path', 'Output directory', required=False),
            SkillParam('color_by', 'string', 'secondary/chain/bfactor', required=False, default='secondary'),
        ],
        outputs=[SkillParam('cxc_path', 'path', 'ChimeraX .cxc script')],
        source='zhuangzhou/chimerax_viz/md_analysis.py',
    ))

    _reg(Skill(
        name='chimerax_defattr',
        description='Export per-residue metrics as ChimeraX .defattr for color mapping (RMSF, displacement, etc.)',
        category='visualization',
        fn=_chimerax_defattr,
        inputs=[
            SkillParam('work_dir', 'path', 'MD simulation directory'),
            SkillParam('metric', 'string', 'displacement/rmsf (default: displacement)', required=False, default='displacement',
                       choices=['displacement', 'rmsf']),
        ],
        outputs=[SkillParam('defattr_path', 'path', 'ChimeraX .defattr file')],
        source='zhuangzhou/chimerax_viz/md_analysis.py',
    ))

    _reg(Skill(
        name='chimerax_comparison',
        description='ChimeraX structure overlay (first vs last frame) with displacement coloring + mobile region labels',
        category='visualization',
        fn=_chimerax_comparison,
        inputs=[
            SkillParam('work_dir', 'path', 'MD simulation directory'),
        ],
        outputs=[
            SkillParam('cxc_path', 'path', 'ChimeraX .cxc script'),
            SkillParam('defattr_path', 'path', 'Displacement .defattr'),
        ],
        source='zhuangzhou/chimerax_viz/md_analysis.py',
    ))

    _reg(Skill(
        name='chimerax_porcupine',
        description='ChimeraX porcupine plot (BILD arrows) showing principal motion vectors',
        category='visualization',
        fn=_chimerax_porcupine,
        inputs=[
            SkillParam('work_dir', 'path', 'MD simulation directory'),
        ],
        outputs=[
            SkillParam('bild_path', 'path', 'BILD arrow file'),
            SkillParam('cxc_path', 'path', 'ChimeraX .cxc script'),
        ],
        source='zhuangzhou/chimerax_viz/md_analysis.py',
    ))

    _reg(Skill(
        name='chimerax_morph',
        description='ChimeraX morph animation between two structures (first/last MD frame)',
        category='visualization',
        fn=_chimerax_morph,
        inputs=[
            SkillParam('work_dir', 'path', 'MD simulation directory'),
            SkillParam('n_frames', 'int', 'Interpolation frames (default 50)', required=False, default=50),
        ],
        outputs=[SkillParam('cxc_path', 'path', 'ChimeraX .cxc morph script')],
        source='zhuangzhou/chimerax_viz/md_analysis.py',
    ))


# ── ChimeraX skill wrapper functions ──

def _chimerax_render(pdb_path: str, output_dir: str = None,
                     style: str = 'cartoon', color_by: str = 'chain',
                     highlight_residues: str = '', ligand_resname: str = '',
                     config: dict = None, **kw) -> dict:
    from zhuangzhou.chimerax_viz import render_structure
    res_list = [r.strip() for r in highlight_residues.split(',') if r.strip()] if highlight_residues else None
    return render_structure(
        pdb_path, output_dir=output_dir, style=style, color_by=color_by,
        highlight_residues=res_list, ligand_resname=ligand_resname or None,
    )


def _chimerax_trajectory(pdb_path: str, output_dir: str = None,
                         color_by: str = 'secondary',
                         config: dict = None, **kw) -> dict:
    from zhuangzhou.chimerax_viz import generate_trajectory_cxc
    return generate_trajectory_cxc(pdb_path, output_dir=output_dir, color_by=color_by)


def _chimerax_defattr(work_dir: str, metric: str = 'displacement',
                      config: dict = None, **kw) -> dict:
    from zhuangzhou.chimerax_viz import export_defattr
    import os

    out_dir = os.path.join(work_dir, 'chimerax_viz')
    os.makedirs(out_dir, exist_ok=True)
    defattr_path = os.path.join(out_dir, f'{metric}.defattr')

    if metric == 'displacement':
        from zhuangzhou.gromacs_md.analysis import compare_structures
        comp = compare_structures(work_dir)
        if 'error' in comp:
            return comp
        return export_defattr(comp['per_residue'], attr_name='displacement',
                              output_path=defattr_path)
    elif metric == 'rmsf':
        # Parse RMSF from existing xvg
        rmsf_xvg = os.path.join(work_dir, 'rmsf.xvg')
        if not os.path.exists(rmsf_xvg):
            from zhuangzhou.gromacs_md.analysis import compute_rmsf
            r = compute_rmsf(work_dir)
            rmsf_xvg = r.get('backbone', rmsf_xvg)
        if not os.path.exists(rmsf_xvg):
            return {'error': 'RMSF data not found. Run gromacs_analysis first.'}
        import numpy as np
        values = {}
        resid = 1
        with open(rmsf_xvg, encoding='utf-8', errors='replace') as f:
            for line in f:
                if line.startswith(('#', '@')):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    values[resid] = float(parts[1])
                    resid += 1
        return export_defattr(values, attr_name='rmsf', output_path=defattr_path)
    else:
        return {'error': f'Unknown metric: {metric}. Use displacement or rmsf.'}


def _chimerax_comparison(work_dir: str, config: dict = None, **kw) -> dict:
    from zhuangzhou.chimerax_viz import generate_comparison_cxc
    from zhuangzhou.gromacs_md.analysis import compare_structures
    import os

    # Ensure comparison data exists
    compare_dir = os.path.join(work_dir, 'structure_comparison')
    import glob
    frames = glob.glob(os.path.join(compare_dir, 'frame_*ps.pdb'))
    if len(frames) < 2:
        comp = compare_structures(work_dir)
        if 'error' in comp:
            return comp
    else:
        comp = compare_structures(work_dir)

    return generate_comparison_cxc(work_dir, comparison_result=comp)


def _chimerax_porcupine(work_dir: str, config: dict = None, **kw) -> dict:
    from zhuangzhou.chimerax_viz import generate_porcupine_cxc
    import os, glob

    # Ensure comparison frames exist
    compare_dir = os.path.join(work_dir, 'structure_comparison')
    frames = glob.glob(os.path.join(compare_dir, 'frame_*ps.pdb'))
    if len(frames) < 2:
        from zhuangzhou.gromacs_md.analysis import compare_structures
        comp = compare_structures(work_dir)
        if 'error' in comp:
            return comp

    return generate_porcupine_cxc(work_dir)


def _chimerax_morph(work_dir: str, n_frames: int = 50,
                    config: dict = None, **kw) -> dict:
    from zhuangzhou.chimerax_viz import generate_morph_cxc
    import os, glob

    compare_dir = os.path.join(work_dir, 'structure_comparison')
    frames = sorted(glob.glob(os.path.join(compare_dir, 'frame_*ps.pdb')))
    if len(frames) < 2:
        from zhuangzhou.gromacs_md.analysis import compare_structures
        comp = compare_structures(work_dir)
        if 'error' in comp:
            return comp
        frames = sorted(glob.glob(os.path.join(compare_dir, 'frame_*ps.pdb')))

    if len(frames) < 2:
        return {'error': 'Could not generate comparison frames'}

    return generate_morph_cxc(frames[0], frames[-1],
                              output_dir=os.path.join(work_dir, 'chimerax_viz'),
                              n_frames=n_frames)

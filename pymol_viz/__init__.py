"""
PyMOL Visualization Module
===========================
Publication-quality interaction figures from PLIP results.

Usage:
    from zhuangzhou.pymol_viz import fast_visualize

    result = fast_visualize(
        complex_pdb='path/to/complex.pdb',
        plip_result=plip_result_dict,
    )
    # result['png_path']         — rendered PNG
    # result['script_path']      — .pml for manual PyMOL
    # result['interactions']     — interaction count
    # result['render_time']      — e.g. "4.8s"

Components:
  - fast_render.py   — Direct cmd.* API renderer (< 10s, Nature/JACS style)
  - chatmol_bridge.py — ChatMol v2 LLM-driven renderer (slower, more flexible)
  - visualize.py      — Legacy .pml generator
"""

from .fast_render import fast_visualize
from .visualize import visualize_plip, visualize_docking, generate_pymol_script
from .chatmol_bridge import run_chatmol_viz, build_plip_prompt

__all__ = [
    'fast_visualize',
    'visualize_plip', 'visualize_docking', 'generate_pymol_script',
    'run_chatmol_viz', 'build_plip_prompt',
]

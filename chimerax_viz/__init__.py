"""
ChimeraX Visualization Module
===============================
Publication-quality structure & MD trajectory visualization via UCSF ChimeraX.

Usage:
    from zhuangzhou.chimerax_viz import (
        render_structure,       # Single structure rendering
        export_defattr,         # Per-residue attribute file (.defattr)
        export_bild_arrows,     # Porcupine BILD arrows
        generate_trajectory_cxc,  # Trajectory animation script
        generate_morph_cxc,     # Morph between structures
        generate_comparison_cxc,  # Structure comparison overlay
        generate_porcupine_cxc, # Porcupine plot (BILD + .cxc)
    )

Components:
  - config.py      — Auto-detecting paths, render defaults
  - render.py      — Structure rendering (PNG + .cxc script)
  - md_analysis.py — MD trajectory, defattr, BILD arrows, morph, comparison
"""

from .render import render_structure
from .md_analysis import (
    export_defattr,
    export_bild_arrows,
    generate_trajectory_cxc,
    generate_morph_cxc,
    generate_comparison_cxc,
    generate_porcupine_cxc,
)
from .interactions import visualize_plip_interactions, generate_interaction_overview

__all__ = [
    'render_structure',
    'export_defattr',
    'export_bild_arrows',
    'generate_trajectory_cxc',
    'generate_morph_cxc',
    'generate_comparison_cxc',
    'generate_porcupine_cxc',
    'visualize_plip_interactions',
    'generate_interaction_overview',
]

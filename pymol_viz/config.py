"""
PyMOL Visualization Configuration
===================================
Publication-quality settings for top-journal structural biology figures.
Color scheme inspired by Nature Structural Biology / Science / Cell.
"""

import os
import shutil as _shutil

_HERE = os.path.dirname(os.path.abspath(__file__))

# ── PyMOL paths ──
PYMOL_PYTHON = (
    os.environ.get('PYMOL_PYTHON')
    or (_shutil.which("pymol") or "")
)
PYMOL_EXE = (
    os.environ.get('PYMOL_EXE')
    or (_shutil.which("pymol") or _shutil.which("PyMOLWin") or "")
)

# Warn at import time if paths don't exist
import logging as _logging
_logger = _logging.getLogger(__name__)
for _p in [PYMOL_PYTHON, PYMOL_EXE]:
    if _p and not os.path.exists(_p):
        _logger.warning("PyMOL path not found: %s (set env var PYMOL_PYTHON / PYMOL_EXE)", _p)

# ── Bundled ChatMol plugins ──
CHATMOL_V1 = os.path.join(_HERE, "chatmol", "chatmol_plugin.py")
CHATMOL_V2 = os.path.join(_HERE, "chatmol", "chatmol_plugin_v2.py")
CHATMOL_PROVIDER = "deepseek"
CHATMOL_MODEL = "deepseek-chat"

# ── Render defaults (600 DPI, Nature single-column = 89mm ~ 3.5in) ──
RENDER_DEFAULTS = {
    "width": 3600,  # px at 600 DPI ≈ 6 inches
    "height": 2700,  # 4:3 aspect
    "dpi": 600,
    "ray": True,
    "bg_color": "white",
}

# ── Publication color palette ──
# Distinguishable, colorblind-safe, print-friendly
# H-bond: classic yellow (universally recognized in structural biology)
# Hydrophobic: warm gray (subtle, doesn't dominate)
# Salt bridge: orange-red (electrostatic = "hot")
# Pi-stacking: forest green (aromatic rings)
# Pi-cation: teal/cyan
# Halogen: cyan-blue
# Water bridge: steel blue
# Metal: orange-red
INTERACTION_COLORS = {
    "hydrogen_bonds": "0xFFCC00",  # golden yellow
    "hydrophobic": "0xA0A0A0",  # neutral gray
    "salt_bridges": "0xFF7A1A",  # orange-red
    "pi_stacking": "0x33AA33",  # forest green
    "pi_cation": "0x00AAAA",  # teal
    "halogen_bonds": "0x00B7EB",  # cyan-blue
    "water_bridges": "0x4488CC",  # steel blue
    "metal_complexes": "0xE65C00",  # orange
}

# Human-readable labels for legend
INTERACTION_LABELS = {
    "hydrogen_bonds": "H-bond",
    "hydrophobic": "Hydrophobic",
    "salt_bridges": "Salt bridge",
    "pi_stacking": "Pi-stacking",
    "pi_cation": "Pi-cation",
    "halogen_bonds": "Halogen bond",
    "water_bridges": "Water bridge",
    "metal_complexes": "Metal coord.",
}

# ── Dash styles per interaction type ──
DASH_STYLES = {
    "hydrogen_bonds": {
        "dash_gap": 0.15,
        "dash_length": 0.15,
        "dash_radius": 0.07,
    },
    "hydrophobic": {
        "dash_gap": 0.3,
        "dash_length": 0.1,
        "dash_radius": 0.04,
    },
    "salt_bridges": {
        "dash_gap": 0.1,
        "dash_length": 0.25,
        "dash_radius": 0.07,
    },
    "pi_stacking": {
        "dash_gap": 0.2,
        "dash_length": 0.15,
        "dash_radius": 0.06,
    },
    "pi_cation": {
        "dash_gap": 0.2,
        "dash_length": 0.15,
        "dash_radius": 0.06,
    },
    "halogen_bonds": {
        "dash_gap": 0.15,
        "dash_length": 0.15,
        "dash_radius": 0.07,
    },
    "water_bridges": {
        "dash_gap": 0.1,
        "dash_length": 0.1,
        "dash_radius": 0.05,
    },
    "metal_complexes": {
        "dash_gap": 0.0,
        "dash_length": 0.5,
        "dash_radius": 0.08,
    },
}

# Default dash settings (fallback)
DASH_SETTINGS = {
    "dash_gap": 0.15,
    "dash_length": 0.15,
    "dash_radius": 0.06,
    "dash_width": 2.5,
}

# ── Protein & ligand colors ──
PROTEIN_COLOR = "gray80"  # light gray cartoon
PROTEIN_CARBON = "0x7799BB"  # slate blue for binding-site carbons
LIGAND_CARBON = "0x44BB66"  # muted green for ligand carbons
CARTOON_TRANSPARENCY = 0.40

"""
ChimeraX Visualization Configuration
======================================
Auto-detecting paths + render defaults for publication figures.
"""

import os
import shutil

_HERE = os.path.dirname(os.path.abspath(__file__))

# ── ChimeraX path auto-detection ──
_COMMON_PATHS = [
    r"C:\Program Files\ChimeraX 1.9\bin\ChimeraX.exe",
    r"C:\Program Files\UCSF\ChimeraX 1.9\bin\ChimeraX.exe",
]


def _find_chimerax():
    """Find ChimeraX executable: env var -> common paths -> PATH."""
    env = os.environ.get("CHIMERAX_EXE")
    if env and os.path.isfile(env):
        return env
    for p in _COMMON_PATHS:
        if os.path.isfile(p):
            return p
    found = shutil.which("chimerax") or shutil.which("ChimeraX")
    return found or ""


CHIMERAX_EXE = _find_chimerax()

# ChimeraX on Windows without OSMesa can't render in --nogui.
# Use GUI mode with --cmd "...;exit" for headless-like rendering.
# Set to True to force --nogui (only works with OSMesa on Linux).
FORCE_NOGUI = False

# ── Render defaults ──
RENDER_DEFAULTS = {
    "width": 2400,
    "height": 1800,
    "supersample": 3,
    "bg_color": "white",
}

# ── Timeout (seconds) ──
TIMEOUT_RENDER = 120
TIMEOUT_SCRIPT = 30

# ── Publication color palette (hex, ChimeraX format) ──
COLORS = {
    "protein_cartoon": "#B0B0B0",  # light gray
    "protein_surface": "#D0D0D0",  # lighter gray
    "ligand": "#44BB66",  # muted green
    "heme": "#CC7777",  # salmon
    "heme_fe": "#FF8800",  # orange
    "mobile": "#E41A1C",  # red
    "rigid": "#377EB8",  # blue
    "hbond": "#FFCC00",  # golden yellow
    "salt_bridge": "#FF7A1A",  # orange-red
    "metal_coord": "#E65C00",  # orange
}

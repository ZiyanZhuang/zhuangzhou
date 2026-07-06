"""
GROMACS MD configuration — portable, auto-detecting.

Path resolution order for each binary:
  1. Environment variable (GMX_BIN, SOBTOP_DIR, OBABEL_EXE, PYMOL_PYTHON, FFMPEG)
  2. User config file: ~/.zhuangzhou/gromacs_md.ini
  3. Auto-detect from PATH / common install locations
  4. Bundled fallback (for gmx_copilot templates)

Users: copy gromacs_md.ini.example to ~/.zhuangzhou/gromacs_md.ini and edit.
"""

import configparser
import os
import shutil
from pathlib import Path

_HERE = Path(__file__).parent
TEMPLATE_DIR = _HERE / 'templates'

# ── User config file ────────────────────────────────────────────────────
_USER_CONFIG_DIR = Path.home() / '.zhuangzhou'
_USER_CONFIG = _USER_CONFIG_DIR / 'gromacs_md.ini'


def _load_user_config():
    """Load user config if it exists."""
    cfg = {}
    if _USER_CONFIG.exists():
        parser = configparser.ConfigParser()
        parser.read(str(_USER_CONFIG), encoding='utf-8')
        if 'paths' in parser:
            cfg.update(dict(parser['paths']))
        if 'simulation' in parser:
            cfg.update(dict(parser['simulation']))
        if 'gpu' in parser:
            cfg.update(dict(parser['gpu']))
    return cfg

_user_cfg = _load_user_config()


# ── Auto-detection helpers ──────────────────────────────────────────────

def _find_binary(name, env_var, config_key, search_paths=None):
    """Find a binary: env → config → PATH → search_paths."""
    # 1. Environment variable
    val = os.environ.get(env_var)
    if val and os.path.isfile(val):
        return val

    # 2. User config
    val = _user_cfg.get(config_key)
    if val and os.path.isfile(val):
        return val

    # 3. System PATH
    found = shutil.which(name)
    if found:
        return found

    # 4. Common locations
    for p in (search_paths or []):
        expanded = os.path.expandvars(p)
        if os.path.isfile(expanded):
            return expanded

    return None


def _find_dir(env_var, config_key, search_paths=None):
    """Find a directory: env → config → search_paths."""
    val = os.environ.get(env_var)
    if val and os.path.isdir(val):
        return val

    val = _user_cfg.get(config_key)
    if val and os.path.isdir(val):
        return val

    for p in (search_paths or []):
        expanded = os.path.expandvars(p)
        if os.path.isdir(expanded):
            return expanded

    return None


# ── Binary paths (auto-detected) ───────────────────────────────────────

GMX_BIN = _find_binary('gmx', 'GMX_BIN', 'gmx_bin', [
    r'C:\gromacs\bin\gmx.exe',
    '/usr/local/gromacs/bin/gmx',
    '/usr/bin/gmx',
])

# GMXLIB: topology directory (auto-detect from gmx location)
_gmx_root = Path(GMX_BIN).parent.parent if GMX_BIN else None
_gmxlib_candidate = (
    os.environ.get('GMXLIB')
    or _user_cfg.get('gmxlib')
    or (str(_gmx_root / 'share' / 'gromacs' / 'top') if _gmx_root else None)
)
GMXLIB = _gmxlib_candidate if _gmxlib_candidate and os.path.isdir(_gmxlib_candidate) else ''
if GMXLIB:
    os.environ['GMXLIB'] = GMXLIB

SOBTOP_DIR = _find_dir('SOBTOP_DIR', 'sobtop_dir', [
    r'C:\sobtop',
])
SOBTOP_EXE = os.path.join(SOBTOP_DIR, 'sobtop.exe') if SOBTOP_DIR else None

OBABEL_EXE = _find_binary('obabel', 'OBABEL_EXE', 'obabel_exe', [
    r'C:\Program Files\OpenBabel-3.1.1\obabel.exe',
])

PYMOL_PYTHON = _find_binary('pymol', 'PYMOL_PYTHON', 'pymol_python', [
    r'C:\PyMOL\python.exe',
])

FFMPEG = _find_binary('ffmpeg', 'FFMPEG', 'ffmpeg')


# ── Heme parameter files ──────────────────────────────────────────────

HEME_PARAMS_DIR = _find_dir('HEME_PARAMS_DIR', 'heme_params_dir')

# ── Metalloprotein force field ────────────────────────────────────────
# AMBER14SB + GAFF (Shahrokh 2012) for CYP450/heme systems
METALLOPROTEIN_FF = _user_cfg.get('metalloprotein_ff', 'amber14sb_parmbsc1')


# ── Default simulation parameters ──────────────────────────────────────

DEFAULT_FF = _user_cfg.get('ff', 'amber14sb_parmbsc1')
DEFAULT_WATER = _user_cfg.get('water', 'tip3p')
DEFAULT_BOX_DIST = float(_user_cfg.get('box_dist', '1.2'))
DEFAULT_BOX_TYPE = _user_cfg.get('box_type', 'dodecahedron')
DEFAULT_TEMP = float(_user_cfg.get('temp', '300'))
DEFAULT_PRESSURE = float(_user_cfg.get('pressure', '1.0'))


# ── GPU flags ──────────────────────────────────────────────────────────

GPU_FLAGS_EM = _user_cfg.get('gpu_flags_em', '-nb gpu')
GPU_FLAGS_FULL = _user_cfg.get('gpu_flags_full',
                               '-nb gpu -pme gpu -bonded gpu -update gpu')


# ── Timeouts (seconds) ────────────────────────────────────────────────

TIMEOUT_SETUP = int(_user_cfg.get('timeout_setup', '120'))
TIMEOUT_EM = int(_user_cfg.get('timeout_em', '600'))
TIMEOUT_EQUIL = int(_user_cfg.get('timeout_equil', '1200'))
TIMEOUT_MD = int(_user_cfg.get('timeout_md', '86400'))


# ── Status check ──────────────────────────────────────────────────────

def check_dependencies():
    """Check all dependencies and return status dict."""
    return {
        'gmx': {'path': GMX_BIN, 'ok': GMX_BIN and os.path.isfile(GMX_BIN)},
        'gmxlib': {'path': GMXLIB, 'ok': GMXLIB and os.path.isdir(GMXLIB)},
        'sobtop': {'path': SOBTOP_EXE, 'ok': SOBTOP_EXE and os.path.isfile(SOBTOP_EXE)},
        'obabel': {'path': OBABEL_EXE, 'ok': OBABEL_EXE and os.path.isfile(OBABEL_EXE)},
        'pymol': {'path': PYMOL_PYTHON, 'ok': PYMOL_PYTHON and os.path.isfile(PYMOL_PYTHON)},
        'ffmpeg': {'path': FFMPEG, 'ok': FFMPEG and os.path.isfile(FFMPEG)},
        'templates': {'path': str(TEMPLATE_DIR), 'ok': TEMPLATE_DIR.exists()},
        'heme_params': {'path': HEME_PARAMS_DIR, 'ok': HEME_PARAMS_DIR is not None and os.path.isdir(HEME_PARAMS_DIR)},
    }

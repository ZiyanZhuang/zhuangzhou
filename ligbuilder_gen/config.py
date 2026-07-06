"""
LigBuilder V3 Configuration
============================
Single source of truth for LigBuilder module settings.
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# LigBuilder V3 installation (Windows path, accessed via WSL /mnt/)
LIGBUILDER_ROOT = os.environ.get(
    'LIGBUILDER_ROOT',
    ''
)

# WSL mount path equivalent
LIGBUILDER_WSL_ROOT = os.environ.get('LIGBUILDER_WSL_ROOT', '')

# Docker
DOCKER_IMAGE = 'zhuangzhou-ligbuilder'
DOCKER_DIR = os.path.join(_HERE, 'docker')
DOCKER_TIMEOUT = 3600      # 1 hour for design jobs
DOCKER_BUILD_TIMEOUT = 600
WSL_TIMEOUT = 3600

# Design defaults
DESIGN_DEFAULTS = {
    'mode': 'explore',           # explore / grow / link
    'sessions': 5,               # number of design sessions
    'molecules': 500,            # target molecule count
    'min_pkd': 3.0,              # minimum pKd threshold
    'max_mw': 500,               # Lipinski MW
    'min_mw': 100,
    'max_logp': 5.0,
    'min_logp': -1.0,
    'max_hbd': 5,                # H-bond donors
    'max_hba': 10,               # H-bond acceptors
}

# Cavity detection defaults
CAVITY_DEFAULTS = {
    'detect_mode': 2,            # 2 = area mode (use pocket coordinates)
    'padding': 5.0,              # extra space around pocket (A)
    'atom_radius_adjust': 1.4,   # solvent accessible radius
    'radius_length': 15.0,       # detection sphere radius
}

# Database directories (relative to LIGBUILDER_ROOT)
DATABASE_DIRS = [
    'fragment.mdb',
    'derivation.mdb',
    'synthesize.mdb',
    'toxicity.mdb',
    'forbidden.mdb',
    'finalcheck.mdb',
    'filter.mdb',
]

"""
Google Colab Connection Configuration (v0.9.0 — Tunnel Mode)
=============================================================
Manages Colab GPU server connection via cloudflared tunnel.

v0.9.0 changes:
  - Primary mode: HTTP via cloudflared tunnel (single Notebook, zero abuse risk)
  - Deprecated: colab-mcp daemon mode (opens new browser tabs, triggers Google flags)

Configuration sources (priority order):
  1. Environment variables: ZZ_COLAB_TUNNEL, ZZ_COLAB_TIMEOUT
  2. User config file: ~/.zhuangzhou/colab_config.json
  3. Programmatic registration via set_config()

Setup:
  1. Open ONE Colab notebook, select T4 GPU runtime
  2. Run the colab_gpu_server.py script in a cell
  3. Copy the printed tunnel URL
  4. Set ZZ_COLAB_TUNNEL=<url> or add to colab_config.json
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_USER_CONFIG_DIR = Path.home() / '.zhuangzhou'
_COLAB_CONFIG_FILE = _USER_CONFIG_DIR / 'colab_config.json'


@dataclass
class ColabProfile:
    """Google Colab connection profile."""
    # Tunnel URL (primary connection method)
    tunnel_url: str = ''               # e.g. https://xxx.trycloudflare.com

    # Runtime settings
    runtime_type: str = 'T4'           # T4 (free), L4 (premium), TPU
    timeout: int = 600                 # Default execution timeout (seconds)
    install_timeout: int = 120         # pip install timeout
    max_output_size: int = 50_000      # Max stdout chars to capture
    connect_timeout: int = 30          # HTTP connection timeout (cloudflared needs ~10-20s)

    # Health check
    health_check_interval: int = 60    # Seconds between keepalive checks

    # ESM model defaults
    esm_default_model: str = 'esm2_t33_650M_UR50D'
    esm1v_default_model: str = 'esm1v_t33_650M_UR90S_1'

    # Pre-installed packages on Colab (skip install check)
    colab_preinstalled: list = field(default_factory=lambda: [
        'torch', 'torchvision', 'tensorflow', 'jax', 'numpy', 'pandas',
        'scipy', 'matplotlib', 'scikit-learn', 'transformers', 'datasets',
        'accelerate', 'huggingface_hub',
    ])

    # Legacy MCP settings (deprecated, kept for backward compatibility)
    mcp_command: str = 'uvx'
    mcp_args: list = field(default_factory=lambda: [
        'git+https://github.com/googlecolab/colab-mcp',
    ])
    mcp_timeout: int = 30_000


# ── Singleton Config ──────────────────────────────────────────────────────

_config: Optional[ColabProfile] = None


def _load_config() -> ColabProfile:
    """Load Colab config from file or defaults."""
    global _config
    if _config is not None:
        return _config

    profile = ColabProfile()

    # Environment variable overrides
    if os.environ.get('ZZ_COLAB_TUNNEL'):
        profile.tunnel_url = os.environ['ZZ_COLAB_TUNNEL'].rstrip('/')
    if os.environ.get('ZZ_COLAB_RUNTIME'):
        profile.runtime_type = os.environ['ZZ_COLAB_RUNTIME']
    if os.environ.get('ZZ_COLAB_TIMEOUT'):
        profile.timeout = int(os.environ['ZZ_COLAB_TIMEOUT'])

    # Config file
    if _COLAB_CONFIG_FILE.exists():
        try:
            with open(_COLAB_CONFIG_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for k, v in data.items():
                if hasattr(profile, k):
                    setattr(profile, k, v)
            # Normalize tunnel_url
            if profile.tunnel_url:
                profile.tunnel_url = profile.tunnel_url.rstrip('/')
            logger.info("Loaded Colab config from %s", _COLAB_CONFIG_FILE)
        except Exception as e:
            logger.warning("Failed to load Colab config: %s", e)

    _config = profile
    return _config


def get_config() -> ColabProfile:
    """Get the current Colab configuration."""
    return _load_config()


def set_config(profile: ColabProfile):
    """Programmatically set Colab configuration."""
    global _config
    _config = profile


def set_tunnel_url(url: str):
    """Set the tunnel URL and persist to config file."""
    config = get_config()
    config.tunnel_url = url.rstrip('/')
    save_config(config)


def save_config(profile: ColabProfile = None):
    """Save configuration to ~/.zhuangzhou/colab_config.json."""
    cfg = profile or get_config()
    _USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(_COLAB_CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)
    logger.info("Saved Colab config to %s", _COLAB_CONFIG_FILE)


def generate_example_config() -> str:
    """Generate example colab_config.json content."""
    return json.dumps(asdict(ColabProfile()), indent=2, ensure_ascii=False)

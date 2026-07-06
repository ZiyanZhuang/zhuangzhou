"""
SSH Compute Configuration
=========================
Manages SSH connection profiles for remote compute nodes.

Configuration sources (priority order):
  1. Environment variables: ZZ_SSH_HOST, ZZ_SSH_PORT, ZZ_SSH_USER, ZZ_SSH_KEY, ZZ_SSH_WORK_DIR
  2. User config file: ~/.zhuangzhou/ssh_compute.ini
  3. Programmatic registration via add_host()

Security:
  - Ed25519 key auth only (no plaintext passwords)
  - Dedicated unprivileged user per host (e.g. zz_runner)
  - All remote commands run inside tmux for disconnect resilience
"""

import configparser
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_USER_CONFIG_DIR = Path.home() / '.zhuangzhou'
_SSH_CONFIG_FILE = _USER_CONFIG_DIR / 'ssh_compute.ini'


@dataclass
class SSHHost:
    """A remote compute node profile."""
    name: str
    host: str
    port: int = 22
    user: str = 'zz_runner'
    key_path: str = ''  # path to Ed25519 private key
    remote_work_dir: str = '/tmp/zhuangzhou_jobs'
    conda_init: str = 'source /opt/conda/etc/profile.d/conda.sh'
    conda_env: str = ''  # e.g. 'zhuangzhou_env'
    capabilities: List[str] = field(default_factory=list)  # ['gpu', 'gromacs', 'qm', 'docking']
    use_tmux: bool = True
    timeout: int = 86400  # 24h default for long MD runs
    max_concurrent: int = 4

    @property
    def key_path_expanded(self) -> str:
        if self.key_path:
            return os.path.expanduser(os.path.expandvars(self.key_path))
        # Default: ~/.ssh/id_ed25519
        default = Path.home() / '.ssh' / 'id_ed25519'
        if default.exists():
            return str(default)
        # Fallback to RSA
        rsa = Path.home() / '.ssh' / 'id_rsa'
        if rsa.exists():
            return str(rsa)
        return ''

    @property
    def conda_activate(self) -> str:
        """Full conda activation command."""
        if not self.conda_env:
            return ''
        return f'{self.conda_init} && conda activate {self.conda_env}'

    def has_capability(self, cap: str) -> bool:
        return cap in self.capabilities or 'all' in self.capabilities


# ── Host Registry ────────────────────────────────────────────────────────

_hosts: Dict[str, SSHHost] = {}
_loaded = False


def _load_config():
    """Load SSH host profiles from ini file."""
    global _loaded
    if _loaded:
        return
    _loaded = True

    # 1. Environment variables (single default host)
    env_host = os.environ.get('ZZ_SSH_HOST')
    if env_host:
        _hosts['default'] = SSHHost(
            name='default',
            host=env_host,
            port=int(os.environ.get('ZZ_SSH_PORT', '22')),
            user=os.environ.get('ZZ_SSH_USER', 'zz_runner'),
            key_path=os.environ.get('ZZ_SSH_KEY', ''),
            remote_work_dir=os.environ.get('ZZ_SSH_WORK_DIR', '/tmp/zhuangzhou_jobs'),
            conda_env=os.environ.get('ZZ_SSH_CONDA_ENV', ''),
            capabilities=os.environ.get('ZZ_SSH_CAPABILITIES', 'all').split(','),
        )

    # 2. Config file (multiple hosts)
    if _SSH_CONFIG_FILE.exists():
        parser = configparser.ConfigParser()
        parser.read(str(_SSH_CONFIG_FILE), encoding='utf-8')

        for section in parser.sections():
            if section.startswith('host:'):
                name = section.split(':', 1)[1].strip()
                cfg = dict(parser[section])
                caps = [c.strip() for c in cfg.get('capabilities', 'all').split(',')]
                _hosts[name] = SSHHost(
                    name=name,
                    host=cfg.get('host', ''),
                    port=int(cfg.get('port', '22')),
                    user=cfg.get('user', 'zz_runner'),
                    key_path=cfg.get('key_path', ''),
                    remote_work_dir=cfg.get('remote_work_dir', '/tmp/zhuangzhou_jobs'),
                    conda_init=cfg.get('conda_init', 'source /opt/conda/etc/profile.d/conda.sh'),
                    conda_env=cfg.get('conda_env', ''),
                    capabilities=caps,
                    use_tmux=cfg.get('use_tmux', 'true').lower() in ('true', '1', 'yes'),
                    timeout=int(cfg.get('timeout', '86400')),
                    max_concurrent=int(cfg.get('max_concurrent', '4')),
                )

    if _hosts:
        logger.info("Loaded %d SSH compute hosts: %s",
                     len(_hosts), list(_hosts.keys()))


def add_host(host: SSHHost):
    """Programmatically register an SSH host."""
    _load_config()
    _hosts[host.name] = host


def get_host(name: str = 'default') -> Optional[SSHHost]:
    """Get an SSH host profile by name."""
    _load_config()
    return _hosts.get(name)


def list_hosts() -> List[SSHHost]:
    """List all configured SSH hosts."""
    _load_config()
    return list(_hosts.values())


def find_host_for_capability(capability: str) -> Optional[SSHHost]:
    """Find a host that supports the given capability."""
    _load_config()
    for host in _hosts.values():
        if host.has_capability(capability):
            return host
    return None


def has_ssh_compute() -> bool:
    """Check if any SSH compute hosts are configured."""
    _load_config()
    return len(_hosts) > 0


def generate_example_config() -> str:
    """Generate example ssh_compute.ini content."""
    return """\
; Zhuangzhou SSH Compute Configuration
; Copy to ~/.zhuangzhou/ssh_compute.ini and edit.
;
; Each [host:NAME] section defines a remote compute node.
; capabilities: comma-separated list of what this host can run
;   gpu, gromacs, qm, docking, plm, all

[host:gpu_server]
host = 192.168.1.100
port = 22
user = zz_runner
key_path = ~/.ssh/id_ed25519
remote_work_dir = /home/zz_runner/zhuangzhou_jobs
conda_init = source /opt/conda/etc/profile.d/conda.sh
conda_env = zhuangzhou_env
capabilities = gpu, gromacs, qm, docking, plm
use_tmux = true
timeout = 86400
max_concurrent = 4

[host:hpc_cluster]
host = hpc.example.edu
port = 22
user = myuser
key_path = ~/.ssh/id_ed25519_hpc
remote_work_dir = /scratch/myuser/zhuangzhou_jobs
conda_init = module load anaconda3 && source activate
conda_env = zz_compute
capabilities = gromacs, qm
use_tmux = true
timeout = 172800
max_concurrent = 8
"""

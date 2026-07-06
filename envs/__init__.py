"""
Zhuangzhou Environment Isolation Layer
=======================================
Each external tool lives in its own conda/WSL/SSH/Colab sandbox.
Zhuangzhou calls them via subprocess, never importing directly.

Environment map:
  graphrag      (conda)  -> AutoGraphRAG parquet queries, pandas
  py310_DPLM2   (conda)  -> UniMol, ESM-2, torch/GPU tools
  WSL           (system) -> PLIP, Linux-only bioinformatics
  native        (host)   -> Vina, GROMACS, basic Python
  ssh:<host>    (remote) -> GPU/HPC compute nodes via SSH
  colab         (cloud)  -> Google Colab T4/L4 GPU via cloudflared HTTP tunnel (v0.9.0)
"""

from .sandbox import (
    CondaSandbox, WSLSandbox, NativeSandbox, SSHSandbox,
    get_sandbox, get_ssh_sandbox,
)
from .colab_sandbox import ColabSandbox
from .colab_bootstrap import bootstrap_colab, is_bootstrap_needed

__all__ = [
    'CondaSandbox', 'WSLSandbox', 'NativeSandbox', 'SSHSandbox',
    'ColabSandbox',
    'get_sandbox', 'get_ssh_sandbox',
    'bootstrap_colab', 'is_bootstrap_needed',
]

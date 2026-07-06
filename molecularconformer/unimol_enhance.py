"""
UniMol-Enhanced Conformer Generation
======================================
Uses UniMol pretrained model (via dplm2 bridge) to generate
higher-quality conformers than RDKit alone.

This module is called via subprocess bridge to the py310_DPLM2 conda env
where unimol_tools + torch GPU are available.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def unimol_enhanced_conformer(
    smiles: str,
    output_path: str = None,
    num_conformers: int = 1,
    remove_hs: bool = False,
    timeout: int = 120,
    config: dict = None,
) -> dict:
    """
    Generate conformer via dplm2 sandbox (UniMol + RDKit in GPU env).

    Falls back to local RDKit generator if dplm2 unavailable.
    """
    try:
        from zhuangzhou.envs import get_sandbox

        sb = get_sandbox("dplm2")
        if not sb.available:
            logger.info("dplm2 env unavailable, falling back to RDKit")
            return _fallback_rdkit(smiles, output_path, num_conformers, remove_hs)

        bridge_script = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "envs", "bridges", "unimol_bridge.py",
        )

        result = sb.run_json_bridge(
            bridge_script,
            {
                "command": "conformer_to_sdf",
                "args": {
                    "smiles": smiles,
                    "output_path": output_path or "",
                    "num_conformers": num_conformers,
                    "remove_hs": remove_hs,
                },
            },
            timeout=timeout,
        )

        if result.success and isinstance(result.data, dict):
            data = result.data.get("result", result.data)
            if data.get("success"):
                return data

        # Bridge failed, fallback
        logger.warning("UniMol bridge failed: %s, falling back to RDKit", result.error)
        return _fallback_rdkit(smiles, output_path, num_conformers, remove_hs)

    except ImportError:
        return _fallback_rdkit(smiles, output_path, num_conformers, remove_hs)


def _fallback_rdkit(smiles, output_path, num_conformers, remove_hs):
    """Fallback to local RDKit generator."""
    from .generator import smiles_to_3d

    return smiles_to_3d(
        smiles,
        output_path=output_path,
        num_conformers=num_conformers,
        remove_hs=remove_hs,
    )

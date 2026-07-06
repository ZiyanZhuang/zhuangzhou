"""
Conformer Generator Configuration
===================================
Paths, defaults, and weight management.
"""

import os
import shutil
from pathlib import Path

# Module root
MODULE_DIR = Path(__file__).parent

# Local weights directory (ships with module)
LOCAL_WEIGHTS_DIR = MODULE_DIR / "weights"

# External UniMol weights (original source)
EXTERNAL_WEIGHTS_DIR = Path(
    os.environ.get("UNIMOL_WEIGHT_DIR", "")
)

# All UniMol weight files (bundled locally for plug-and-play)
WEIGHT_FILES = {
    "mol_dict": "mol.dict.txt",
    "poc_dict": "poc.dict.txt",
    "mol_model_no_h": "mol_pre_no_h_220816.pt",
    "mol_model_all_h": "mol_pre_all_h_220816.pt",
    "poc_model": "poc_pre_220816.pt",
    "binding_pose": "binding_pose_220908.pt",
    "drugs": "drugs_220908.pt",
    "qm9": "qm9_220908.pt",
}

# Default generation parameters
DEFAULTS = {
    "seed": 42,
    "mode": "fast",             # 'fast' (1 attempt) or 'heavy' (5000 attempts)
    "remove_hs": False,         # Keep hydrogens in output
    "optimize": "mmff",         # 'mmff', 'uff', or None
    "num_conformers": 1,        # Number of conformers to generate
    "max_atoms": 256,
    "prune_rms_thresh": 0.5,    # RMSD threshold for pruning similar conformers
    "energy_window": 50.0,      # kcal/mol above minimum to keep
}

# dplm2 conda env for UniMol-enhanced generation
DPLM2_PYTHON = os.environ.get("DPLM2_PYTHON") or os.environ.get("ZHUANGZHOU_DPLM2_PYTHON", "python")


def ensure_local_weights(force=False):
    """Copy essential weight files from external source to local weights dir.

    Returns dict of copied files or errors.
    """
    LOCAL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for key, filename in WEIGHT_FILES.items():
        local_path = LOCAL_WEIGHTS_DIR / filename
        external_path = EXTERNAL_WEIGHTS_DIR / filename

        if local_path.exists() and not force:
            results[key] = {"status": "exists", "path": str(local_path)}
            continue

        if not external_path.exists():
            results[key] = {"status": "missing_source", "path": str(external_path)}
            continue

        try:
            shutil.copy2(str(external_path), str(local_path))
            size_mb = local_path.stat().st_size / (1024 * 1024)
            results[key] = {
                "status": "copied",
                "path": str(local_path),
                "size_mb": round(size_mb, 1),
            }
        except Exception as e:
            results[key] = {"status": "error", "error": str(e)}

    return results


def get_weight_path(key: str) -> str:
    """Get weight file path, preferring local copy over external."""
    filename = WEIGHT_FILES.get(key)
    if not filename:
        return ""

    local = LOCAL_WEIGHTS_DIR / filename
    if local.exists():
        return str(local)

    external = EXTERNAL_WEIGHTS_DIR / filename
    if external.exists():
        return str(external)

    return ""

"""
AMBER MD Configuration — path resolution and constants.
"""

import configparser
import os
from pathlib import Path

# Default heme parameter base. Configure via AMBER_HEME_PARAMS or amber_md.ini.
DEFAULT_HEME_PARAMS_BASE = ""

# Heme states
VALID_STATES = ("CPDI", "DIOXY", "IC6")

# CYP450 conserved heme-binding motif: FxxGx[RH]xCxG
import re

CYP_MOTIF = re.compile(r"F..G.[RH].C.G")

# Heme anchor atom names
HEME_FE_NAMES = {"FE", "FE1", "FE2"}
HEME_N_NAMES_MAP = {
    "NA": "NA",
    "NB": "NB",
    "NC": "NC",
    "ND": "ND",
    "N A": "NA",
    "N B": "NB",
    "N C": "NC",
    "N D": "ND",
    "N1": "N1",
    "N2": "N2",
    "N3": "N3",
    "N4": "N4",
    "N_A": "NA",
    "N_B": "NB",
    "N_C": "NC",
    "N_D": "ND",
}

# 3-letter -> 1-letter amino acid mapping
AA3TO1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "HIE": "H",
    "HID": "H",
    "HIP": "H",
    "CYX": "C",
    "CYP": "C",
}

# Known heme residue names
HEME_RESNAMES = {
    "HEM",
    "HEC",
    "HEA",
    "HEB",
    "L01",
    "L02",
    "L03",
    "HME",
    "HAS",
    "HDD",
    "PP9",
    "COH",
    "FEH",
}


def get_heme_params_base(config=None):
    """Resolve heme parameters base directory from config or default."""
    if config and config.get("heme_params_base"):
        return config["heme_params_base"]

    # Try ini file
    ini_path = os.path.expanduser("~/.zhuangzhou/amber_md.ini")
    if os.path.exists(ini_path):
        cp = configparser.ConfigParser()
        cp.read(ini_path, encoding="utf-8")
        if cp.has_option("paths", "heme_params_base"):
            return cp.get("paths", "heme_params_base")

    # Environment variable
    env_val = os.environ.get("AMBER_HEME_PARAMS")
    if env_val:
        return env_val

    return DEFAULT_HEME_PARAMS_BASE


def check_dependencies(config=None):
    """Check AMBER tool availability (native)."""
    import shutil

    results = {}
    for tool in ["tleap", "pmemd.cuda", "pmemd", "cpptraj", "ambpdb"]:
        results[tool] = shutil.which(tool) is not None

    params_base = get_heme_params_base(config)
    results["heme_params"] = (
        os.path.isdir(params_base) if not params_base.startswith("/mnt/") else True
    )

    return results


def check_general_dependencies(via_wsl: bool = True) -> dict:
    """Check AMBER + helper tool availability for the general MD pipeline.

    Returns dict of {tool_name: bool}.  When *via_wsl* is True the check is
    performed inside WSL2 (the normal case on Windows).
    """
    import shutil
    import subprocess

    tools = [
        "tleap",
        "pmemd.cuda",
        "pmemd",
        "sander",
        "cpptraj",
        "antechamber",
        "parmchk2",
        "reduce",
        "obabel",
    ]
    optional = ["Multiwfn"]

    results = {}

    if via_wsl:
        for t in tools + optional:
            try:
                r = subprocess.run(
                    f'wsl -e bash -l -c "source ~/amber25/amber.sh && which {t}"',
                    shell=True,
                    capture_output=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=15,
                )
                results[t] = r.returncode == 0
            except Exception:
                results[t] = False
    else:
        for t in tools + optional:
            results[t] = shutil.which(t) is not None

    return results

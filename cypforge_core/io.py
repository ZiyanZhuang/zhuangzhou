from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


# ── file checksum ──────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ── structured write ───────────────────────────────────────────────────────

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


# ── platform path conversion ───────────────────────────────────────────────

def _win_to_wsl(path: str | Path) -> str:
    """Convert a Windows path to its WSL /mnt/<drive>/... equivalent."""
    resolved = Path(path).resolve()
    if resolved.drive:
        drive = resolved.drive.rstrip(":").lower()
        rest = resolved.as_posix().split(":", 1)[1]
        return f"/mnt/{drive}{rest}"
    return resolved.as_posix()


# ── external tool resolution ───────────────────────────────────────────────

def resolve_amber_sh(amber_sh: str | None = None) -> str:
    if amber_sh:
        return amber_sh
    env = os.environ.get("AMBER_SH")
    if env:
        return env
    amberhome = os.environ.get("AMBERHOME")
    if amberhome:
        return str(Path(amberhome) / "amber.sh")
    raise ValueError(
        "Amber environment not configured. Set AMBER_SH to the path of amber.sh, "
        "e.g. 'export AMBER_SH=/path/to/amber25/amber.sh', or set AMBERHOME."
    )


def resolve_multiwfn_bin(multiwfn_bin: str | None = None) -> str:
    if multiwfn_bin:
        return multiwfn_bin
    env = os.environ.get("MULTIWFN_BIN")
    if env:
        return env
    raise ValueError(
        "Multiwfn path not configured. Set MULTIWFN_BIN to the Multiwfn binary path, "
        "e.g. 'export MULTIWFN_BIN=/path/to/Multiwfn_noGUI'."
    )


# ── geometry utilities ─────────────────────────────────────────────────────

def _distance_dict(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Euclidean distance between two atoms represented as dicts with x/y/z."""
    return math.sqrt(
        (a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2
    )


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC in degrees."""
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    cosine = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def kabsch_transform(
    source_points: np.ndarray, target_points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimal rotation (Kabsch) aligning source onto target.

    Returns (rotation, source_centroid, target_centroid).
    """
    centroid_source = source_points.mean(axis=0)
    centroid_target = target_points.mean(axis=0)
    src_centered = source_points - centroid_source
    tgt_centered = target_points - centroid_target
    h = src_centered.T @ tgt_centered
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    return r, centroid_source, centroid_target


# ── frcmod parser ──────────────────────────────────────────────────────────

def parse_frcmod_sections(path: Path) -> dict[str, list[str]]:
    """Parse an Amber frcmod file into named sections."""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    valid = {"MASS", "BOND", "ANGLE", "DIHEDRAL", "DIHE", "IMPROPER", "NONBON"}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line in valid:
            current = "DIHEDRAL" if line == "DIHE" else line
            sections.setdefault(current, [])
            continue
        if current:
            sections[current].append(raw.rstrip())
    return sections

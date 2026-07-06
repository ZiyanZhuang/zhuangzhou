from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .ligand_pose_frame import check_ligand_pose_frame


PROTEIN_RESNAMES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "CYM", "CYX", "CYP", "GLN", "GLU", "GLY", "HIS", "HID", "HIE",
    "HIP", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
WATER_RESNAMES = {"HOH", "WAT", "H2O"}


def _run(cmd: list[str], cwd: Path | None = None, timeout: int = 7200) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, errors="replace", timeout=timeout)


def _guess_element(atom_name: str) -> str:
    letters = "".join(ch for ch in atom_name if ch.isalpha()).upper()
    if len(letters) >= 2 and letters[:2] in {"CL", "BR", "FE"}:
        return letters[:2]
    return letters[:1] or "X"


def _pdb_atoms(path: Path) -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        name = line[12:16].strip()
        atoms.append(
            {
                "line": line,
                "name": name,
                "resname": line[17:20].strip(),
                "chain": line[21].strip() or "A",
                "coord": np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float),
                "element": line[76:78].strip().upper() or _guess_element(name),
            }
        )
    return atoms


def extract_ligand_pdb(clean_pdb: str | Path, ligand_resname: str, ligand_chain: str, output_pdb: str | Path) -> str:
    src = Path(clean_pdb)
    out = Path(output_pdb)
    lines = [
        atom["line"]
        for atom in _pdb_atoms(src)
        if atom["resname"] == ligand_resname and atom["chain"] == ligand_chain
    ]
    if not lines:
        raise ValueError(f"No ligand atoms found for {ligand_resname} chain {ligand_chain} in {src}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")
    return str(out)


def prepare_receptor_pdb(current_receptor_pdb: str | Path, ligand_resname: str, output_pdb: str | Path) -> str:
    src = Path(current_receptor_pdb)
    out = Path(output_pdb)
    kept: list[str] = []
    for line in src.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith(("ATOM", "HETATM")):
            resname = line[17:20].strip()
            if resname in WATER_RESNAMES or resname == ligand_resname:
                continue
            kept.append(line)
        elif line.startswith(("TER", "END")):
            kept.append(line)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return str(out)


def obabel_convert(input_path: str | Path, input_format: str, output_path: str | Path, output_format: str, extra: list[str] | None = None) -> str:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["obabel", f"-i{input_format}", str(input_path), f"-o{output_format}", "-O", str(out)]
    if extra:
        cmd.extend(extra)
    result = _run(cmd)
    if result.returncode != 0 or not out.is_file():
        raise RuntimeError(f"OpenBabel failed: {' '.join(cmd)}\n{result.stderr or result.stdout}")
    return str(out)


def active_site_center(pdb_path: str | Path, heme_resname: str = "HEM") -> tuple[float, float, float]:
    for atom in _pdb_atoms(Path(pdb_path)):
        if atom["resname"] == heme_resname and atom["name"] == "FE":
            return tuple(float(x) for x in atom["coord"])
    raise ValueError(f"Cannot find {heme_resname} FE in {pdb_path}")


def parse_vina_poses(pdbqt_path: str | Path) -> list[str]:
    text = Path(pdbqt_path).read_text(encoding="utf-8", errors="ignore")
    poses: list[str] = []
    current: list[str] = []
    in_model = False
    for line in text.splitlines():
        if line.startswith("MODEL"):
            in_model = True
            current = [line]
        elif line.startswith("ENDMDL") and in_model:
            current.append(line)
            poses.append("\n".join(current) + "\n")
            in_model = False
        elif in_model:
            current.append(line)
    if not poses and text.strip():
        poses.append(text if text.endswith("\n") else text + "\n")
    return poses


def parse_pose_atoms(pose_text: str) -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    for line in pose_text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        name = line[12:16].strip()
        atoms.append(
            {
                "name": name,
                "element": line[76:78].strip().upper() or _guess_element(name),
                "coord": np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float),
            }
        )
    return atoms


def _heavy_coord_arrays(reference_atoms: list[dict[str, Any]], pose_atoms: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    ref = [atom for atom in reference_atoms if atom["element"] != "H"]
    pose = [atom for atom in pose_atoms if atom["element"] != "H"]
    pose_map = {atom["name"]: atom["coord"] for atom in pose}
    if len(pose_map) != len(pose):
        raise ValueError("Duplicate heavy atom names in docked pose; RMSD by atom identity is unsafe.")
    common = [atom["name"] for atom in ref if atom["name"] in pose_map]
    if len(common) < 3:
        raise ValueError("Fewer than 3 common heavy atom names between crystal ligand and docked pose.")
    ref_map = {atom["name"]: atom["coord"] for atom in ref}
    return np.array([ref_map[name] for name in common]), np.array([pose_map[name] for name in common])


def kabsch_rmsd(reference_coords: np.ndarray, pose_coords: np.ndarray) -> float:
    ref_centered = reference_coords - reference_coords.mean(axis=0)
    pose_centered = pose_coords - pose_coords.mean(axis=0)
    covariance = pose_centered.T @ ref_centered
    v, _, wt = np.linalg.svd(covariance)
    rotation = v @ wt
    if np.linalg.det(rotation) < 0:
        v[:, -1] *= -1
        rotation = v @ wt
    diff = pose_centered @ rotation - ref_centered
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _read_mol2_atoms(path: Path) -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    in_atoms = False
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("@<TRIPOS>ATOM"):
            in_atoms = True
            continue
        if line.startswith("@<TRIPOS>") and in_atoms:
            break
        if in_atoms and line.strip():
            parts = line.split()
            atoms.append(
                {
                    "name": parts[1],
                    "element": parts[5].split(".")[0].upper() if "." in parts[5] else _guess_element(parts[1]),
                    "coord": np.array([float(parts[2]), float(parts[3]), float(parts[4])], dtype=float),
                }
            )
    return atoms


def parse_pose_affinities(vina_text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in vina_text.splitlines():
        parts = line.replace("|", " ").split()
        if len(parts) >= 4 and parts[0].isdigit():
            try:
                rows.append({"mode": int(parts[0]), "affinity": float(parts[1]), "rmsd_lb": float(parts[2]), "rmsd_ub": float(parts[3])})
            except ValueError:
                pass
    return rows


def run_vina_crystal_rmsd_selection(
    *,
    current_receptor_pdb: str | Path,
    crystal_clean_pdb: str | Path,
    ligand_resname: str,
    ligand_chain: str,
    output_dir: str | Path,
    vina_bin: str | Path,
    exhaustiveness: int = 128,
    num_modes: int = 100,
    energy_range: float = 10.0,
    cpu: int = 8,
    seed: int = 20260531,
    box_size: float = 20.0,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    receptor_pdb = Path(prepare_receptor_pdb(current_receptor_pdb, ligand_resname, out / "docking_receptor_with_heme.pdb"))
    crystal_ligand_pdb = Path(extract_ligand_pdb(crystal_clean_pdb, ligand_resname, ligand_chain, out / f"crystal_{ligand_resname}.pdb"))
    receptor_pdbqt = Path(obabel_convert(receptor_pdb, "pdb", out / "receptor.pdbqt", "pdbqt", ["-xr"]))
    ligand_pdbqt = Path(obabel_convert(crystal_ligand_pdb, "pdb", out / "ligand_input.pdbqt", "pdbqt"))

    cx, cy, cz = active_site_center(receptor_pdb)
    docked = out / "docked_poses.pdbqt"
    vina_cmd = [
        str(vina_bin), "--receptor", str(receptor_pdbqt), "--ligand", str(ligand_pdbqt),
        "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
        "--size_x", str(box_size), "--size_y", str(box_size), "--size_z", str(box_size),
        "--exhaustiveness", str(exhaustiveness), "--num_modes", str(num_modes), "--energy_range", str(energy_range),
        "--seed", str(seed), "--cpu", str(cpu), "--out", str(docked),
    ]
    result = _run(vina_cmd, cwd=out, timeout=7200)
    (out / "vina.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (out / "vina.stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0 or not docked.is_file():
        raise RuntimeError(f"Vina failed: {result.stderr or result.stdout}")

    poses = parse_vina_poses(docked)
    affinities = {row["mode"]: row for row in parse_pose_affinities(result.stdout + "\n" + result.stderr)}
    reference_atoms = parse_pose_atoms(crystal_ligand_pdb.read_text(encoding="utf-8", errors="ignore"))
    pose_rows: list[dict[str, Any]] = []
    pose_dir = out / "poses"
    pose_dir.mkdir(exist_ok=True)
    for index, pose_text in enumerate(poses, start=1):
        pose_pdbqt = pose_dir / f"pose_{index}.pdbqt"
        pose_pdb = pose_dir / f"pose_{index}.pdb"
        pose_pdbqt.write_text(pose_text, encoding="utf-8")
        obabel_convert(pose_pdbqt, "pdbqt", pose_pdb, "pdb")
        pose_atoms = parse_pose_atoms(pose_text)
        ref_coords, pose_coords = _heavy_coord_arrays(reference_atoms, pose_atoms)
        row = {"mode": index, "heavy_atom_rmsd_a": round(kabsch_rmsd(ref_coords, pose_coords), 6), "pdbqt": str(pose_pdbqt), "pdb": str(pose_pdb)}
        row.update(affinities.get(index, {}))
        pose_rows.append(row)
    best = min(pose_rows, key=lambda row: row["heavy_atom_rmsd_a"])
    selected_pdbqt = out / f"{ligand_resname}_selected_min_rmsd_pose.pdbqt"
    selected_pdb = out / f"{ligand_resname}_selected_min_rmsd_pose.pdb"
    selected_mol2 = out / f"{ligand_resname}_selected_min_rmsd_pose.mol2"
    shutil.copyfile(best["pdbqt"], selected_pdbqt)
    obabel_convert(selected_pdbqt, "pdbqt", selected_pdb, "pdb")
    obabel_convert(selected_pdbqt, "pdbqt", selected_mol2, "mol2")
    frame = check_ligand_pose_frame(
        current_receptor_pdb=current_receptor_pdb,
        docking_receptor_pdb=receptor_pdb,
        ligand_mol2=selected_mol2,
        output_dir=out / "frame_check",
        ligand_resname=ligand_resname,
    )
    manifest = {
        "schema": "cypforge.vina_crystal_rmsd_pose_selection.v1",
        "status": "success" if frame["status"] == "success" else "failed",
        "current_receptor_pdb": str(current_receptor_pdb),
        "docking_receptor_pdb": str(receptor_pdb),
        "crystal_clean_pdb": str(crystal_clean_pdb),
        "ligand_resname": ligand_resname,
        "ligand_chain": ligand_chain,
        "heme_retained_in_docking_receptor": True,
        "vina_command": vina_cmd,
        "sampling": {"exhaustiveness": exhaustiveness, "num_modes": num_modes, "energy_range": energy_range, "cpu": cpu, "seed": seed},
        "all_pose_rmsd": pose_rows,
        "selected_pose": best,
        "selected_pose_pdbqt": str(selected_pdbqt),
        "selected_pose_pdb": str(selected_pdb),
        "selected_pose_mol2": str(selected_mol2),
        "frame_check": frame,
        "limitation": (
            "Vina is used only to locate a heavy-atom pose. It removes/changes hydrogens during PDBQT handling. "
            "The selected pose is not a final ligand-parameterization input until the user supplies a checked, "
            "hydrogen-complete protein-heme-ligand complex for ESP calculation."
        ),
    }
    (out / "pose_selection_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest

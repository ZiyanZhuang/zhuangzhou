from __future__ import annotations

import csv
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from .io import _win_to_wsl, resolve_amber_sh


def _read_mol2_atom_names(path: Path) -> list[str]:
    names: list[str] = []
    in_atoms = False
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("@<TRIPOS>ATOM"):
            in_atoms = True
            continue
        if line.startswith("@<TRIPOS>") and in_atoms:
            break
        if in_atoms and line.strip():
            parts = line.split()
            if len(parts) < 9:
                raise ValueError(f"Invalid MOL2 atom line: {line.strip()}")
            names.append(parts[1])
    if not names:
        raise ValueError(f"No atoms found in MOL2: {path}")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate MOL2 atom names prevent safe charge injection: {duplicates}")
    return names


def _read_charge_csv(path: Path) -> dict[str, float]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    required = {"name", "esp_charge"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Charge CSV must contain columns: {sorted(required)}")
    charges: dict[str, float] = {}
    for row in rows:
        name = row["name"].strip()
        if name in charges:
            raise ValueError(f"Duplicate atom name in charge CSV: {name}")
        charges[name] = float(row["esp_charge"])
    return charges


def _inject_charges(pose_mol2: Path, charges: dict[str, float], output_mol2: Path, resname: str | None) -> int:
    lines = pose_mol2.read_text(encoding="utf-8", errors="ignore").splitlines()
    output: list[str] = []
    in_atoms = False
    injected = 0
    for line in lines:
        if line.startswith("@<TRIPOS>ATOM"):
            in_atoms = True
            output.append(line)
            continue
        if line.startswith("@<TRIPOS>") and in_atoms:
            in_atoms = False
            output.append(line)
            continue
        if in_atoms and line.strip():
            parts = line.split()
            name = parts[1]
            if name not in charges:
                raise ValueError(f"Missing ESP charge for MOL2 atom: {name}")
            residue_name = resname or parts[7]
            output.append(
                f"{int(parts[0]):>7d} {name:<6s}{float(parts[2]):>10.4f}{float(parts[3]):>10.4f}{float(parts[4]):>10.4f} "
                f"{parts[5]:<8s}{parts[6]:>3s} {residue_name:<6s}{charges[name]:>10.6f}"
            )
            injected += 1
        else:
            output.append(line)
    output_mol2.parent.mkdir(parents=True, exist_ok=True)
    output_mol2.write_text("\n".join(output) + "\n", encoding="utf-8")
    return injected


def _run_parmchk2(output_dir: Path, mol2_name: str, frcmod_name: str, amber_sh: str) -> dict[str, Any]:
    wsl_dir = _win_to_wsl(output_dir)
    cmd = (
        f"source {shlex.quote(amber_sh)} && cd {shlex.quote(wsl_dir)} && "
        f"parmchk2 -i {shlex.quote(mol2_name)} -f mol2 -o {shlex.quote(frcmod_name)} -s gaff2"
    )
    result = subprocess.run(
        ["wsl", "bash", "-lc", cmd] if os.name == "nt" else ["bash", "-lc", cmd],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "command": "parmchk2 -i <mol2> -f mol2 -o <frcmod> -s gaff2",
    }


def parameterize_selected_ligand_pose(
    *,
    pose_mol2: str | Path,
    charge_csv: str | Path,
    formal_charge: int,
    output_dir: str | Path,
    resname: str | None = None,
    run_parmchk2: bool = True,
    amber_sh: str | None = None,
) -> dict[str, Any]:
    """Parameterize a user-selected ligand pose without changing its coordinates."""
    mol2 = Path(pose_mol2)
    csv_path = Path(charge_csv)
    out_dir = Path(output_dir)
    if not mol2.is_file():
        raise FileNotFoundError(f"Missing selected pose MOL2: {mol2}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing ESP charge CSV: {csv_path}")

    atom_names = _read_mol2_atom_names(mol2)
    charges = _read_charge_csv(csv_path)
    missing = [name for name in atom_names if name not in charges]
    extra = [name for name in charges if name not in set(atom_names)]
    if missing or extra:
        raise ValueError(f"MOL2/charge CSV atom mismatch: missing={missing}; extra={extra}")

    charge_sum = round(sum(charges[name] for name in atom_names), 8)
    if abs(charge_sum - float(formal_charge)) > 1.0e-4:
        raise ValueError(f"ESP charge sum {charge_sum} does not match formal charge {formal_charge}")

    ligand_name = resname or mol2.stem.split("_")[0]
    charged_mol2 = out_dir / f"{ligand_name}_selected_pose_esp.mol2"
    injected = _inject_charges(mol2, charges, charged_mol2, ligand_name)
    frcmod = out_dir / f"{ligand_name}.frcmod"
    parmchk2_result = None
    if run_parmchk2:
        resolved_amber_sh = resolve_amber_sh(amber_sh)
        parmchk2_result = _run_parmchk2(out_dir, charged_mol2.name, frcmod.name, resolved_amber_sh)
        if parmchk2_result["returncode"] != 0 or not frcmod.is_file():
            raise RuntimeError(f"parmchk2 failed: {parmchk2_result}")

    manifest = {
        "schema": "cypforge.selected_ligand_pose_parameterization.v1",
        "status": "success",
        "input_pose_mol2": str(mol2),
        "input_charge_csv": str(csv_path),
        "coordinate_policy": "coordinates are copied from user-selected pose MOL2 without geometry optimization",
        "charge_policy": "ESP charges are injected by unique atom name; row-order transfer is not used",
        "formal_charge": formal_charge,
        "amber_sh": amber_sh or os.environ.get("AMBER_SH") or os.environ.get("AMBERHOME", "not_set"),
        "partial_charge_sum": charge_sum,
        "atom_count": len(atom_names),
        "injected_atom_count": injected,
        "output_mol2": str(charged_mol2),
        "frcmod": str(frcmod) if frcmod.exists() else None,
        "parmchk2": parmchk2_result,
        "limitation": "This parameterizes a user-selected ligand pose; it does not choose the pose or prove binding correctness.",
    }
    manifest_path = out_dir / "ligand_pose_parameterization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest["manifest_json"] = str(manifest_path)
    return manifest

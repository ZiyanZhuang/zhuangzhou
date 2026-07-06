from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


HEME_ANCHORS = ("FE", "NA", "NB", "NC", "ND")


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _pdb_atoms(path: Path) -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        name = line[12:16].strip()
        element = line[76:78].strip().upper() or "".join(ch for ch in name if ch.isalpha()).upper()[:1] or "X"
        atoms.append(
            {
                "line": line,
                "record": line[:6].strip(),
                "name": name,
                "resname": line[17:20].strip(),
                "chain": line[21].strip() or "A",
                "resid": int(line[22:26]),
                "coord": (float(line[30:38]), float(line[38:46]), float(line[46:54])),
                "element": element,
            }
        )
    return atoms


def _mol2_atoms(path: Path) -> list[dict[str, Any]]:
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
            atom_type = parts[5]
            element = atom_type.split(".")[0].upper()
            if element not in {"C", "N", "O", "S", "P", "H", "F", "CL", "BR", "I"}:
                element = "".join(ch for ch in parts[1] if ch.isalpha()).upper()[:1] or "X"
            atoms.append(
                {
                    "name": parts[1],
                    "coord": (float(parts[2]), float(parts[3]), float(parts[4])),
                    "type": atom_type,
                    "element": element,
                }
            )
    if not atoms:
        raise ValueError(f"No atoms parsed from ligand MOL2: {path}")
    return atoms


def _heme_anchor_map(path: Path, heme_resname: str) -> dict[str, tuple[float, float, float]]:
    anchors: dict[str, tuple[float, float, float]] = {}
    for atom in _pdb_atoms(path):
        if atom["resname"] == heme_resname and atom["name"] in HEME_ANCHORS:
            anchors[atom["name"]] = atom["coord"]
    missing = [name for name in HEME_ANCHORS if name not in anchors]
    if missing:
        raise ValueError(f"Missing heme anchor atoms in {path}: {missing}")
    return anchors


def _centroid(coords: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    return tuple(sum(coord[i] for coord in coords) / len(coords) for i in range(3))  # type: ignore[return-value]


def write_protein_heme_ligand_check_pdb(
    *,
    protein_heme_pdb: str | Path,
    ligand_mol2: str | Path,
    output_pdb: str | Path,
    ligand_resname: str,
    ligand_chain: str = "L",
    ligand_resid: int = 1,
) -> str:
    protein_path = Path(protein_heme_pdb)
    ligand_path = Path(ligand_mol2)
    out = Path(output_pdb)
    base_lines = [
        line
        for line in protein_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if not line.startswith("END")
    ]
    serial = 1
    for line in base_lines:
        if line.startswith(("ATOM", "HETATM")):
            try:
                serial = max(serial, int(line[6:11]) + 1)
            except ValueError:
                pass
    lines = list(base_lines)
    for atom in _mol2_atoms(ligand_path):
        x, y, z = atom["coord"]
        element = atom["element"].title() if len(atom["element"]) > 1 else atom["element"]
        lines.append(
            f"HETATM{serial:5d} {atom['name'][:4]:>4s} {ligand_resname:>3s} {ligand_chain:1s}{ligand_resid:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}          {element:>2s}"
        )
        serial += 1
    lines.extend(["TER", "END"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(out)


def check_ligand_pose_frame(
    *,
    current_receptor_pdb: str | Path,
    docking_receptor_pdb: str | Path,
    ligand_mol2: str | Path,
    output_dir: str | Path,
    heme_resname: str = "HEM",
    ligand_resname: str = "LIG",
    anchor_rmsd_threshold: float = 0.25,
) -> dict[str, Any]:
    """Check whether a selected ligand pose is in the current receptor frame."""
    current = Path(current_receptor_pdb)
    docking = Path(docking_receptor_pdb)
    ligand = Path(ligand_mol2)
    out_dir = Path(output_dir)
    current_anchors = _heme_anchor_map(current, heme_resname)
    docking_anchors = _heme_anchor_map(docking, heme_resname)
    anchor_distances = {name: _distance(current_anchors[name], docking_anchors[name]) for name in HEME_ANCHORS}
    anchor_rmsd = math.sqrt(sum(value * value for value in anchor_distances.values()) / len(anchor_distances))

    ligand_atoms = _mol2_atoms(ligand)
    ligand_coords = [atom["coord"] for atom in ligand_atoms]
    ligand_centroid = _centroid(ligand_coords)
    fe = current_anchors["FE"]
    nearest = min((_distance(fe, atom["coord"]), atom["name"]) for atom in ligand_atoms)
    check_pdb = out_dir / "protein_heme_ligand_check.pdb"
    write_protein_heme_ligand_check_pdb(
        protein_heme_pdb=current,
        ligand_mol2=ligand,
        output_pdb=check_pdb,
        ligand_resname=ligand_resname,
    )

    status = "success" if anchor_rmsd <= anchor_rmsd_threshold else "failed"
    report = {
        "schema": "cypforge.ligand_pose_frame_check.v1",
        "status": status,
        "current_receptor_pdb": str(current),
        "docking_receptor_pdb": str(docking),
        "ligand_mol2": str(ligand),
        "protein_heme_ligand_check_pdb": str(check_pdb),
        "heme_anchor_atoms": list(HEME_ANCHORS),
        "heme_anchor_raw_distance_a": {name: round(value, 6) for name, value in anchor_distances.items()},
        "heme_anchor_raw_rmsd_a": round(anchor_rmsd, 6),
        "anchor_rmsd_threshold_a": anchor_rmsd_threshold,
        "ligand_atom_count": len(ligand_atoms),
        "ligand_centroid": [round(value, 6) for value in ligand_centroid],
        "ligand_centroid_to_current_fe_a": round(_distance(ligand_centroid, fe), 6),
        "nearest_ligand_atom_to_current_fe": {"atom": nearest[1], "distance_a": round(nearest[0], 6)},
        "interpretation": (
            "Pass means the selected ligand pose and current receptor share the same heme-anchor frame. "
            "It does not mean the pose is scientifically correct."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "ligand_pose_frame_check.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report["report_json"] = str(report_path)
    return report

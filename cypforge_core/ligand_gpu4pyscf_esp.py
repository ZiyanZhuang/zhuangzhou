from __future__ import annotations

import csv
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from .io import (
    _distance_dict as _distance,
    _win_to_wsl,
    kabsch_transform,
    resolve_amber_sh,
    resolve_multiwfn_bin,
    write_text,
)

VDW_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
}



COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "CL": 1.02,
    "BR": 1.20,
    "I": 1.39,
}

MAPPED_POSE_HEAVY_RMSD_TOLERANCE_A = 0.05
RESP_CHARGE_SUM_TOLERANCE_E = 1.0e-4


def _guess_element(name: str, atom_type: str = "") -> str:
    head = atom_type.split(".")[0].upper() if atom_type else ""
    if head in VDW_RADII or head == "FE":
        return head
    letters = "".join(ch for ch in name if ch.isalpha()).upper()
    if len(letters) >= 2 and letters[:2] in {"CL", "BR", "FE"}:
        return letters[:2]
    return letters[:1] or "X"


def read_ligand_atoms(path: str | Path, *, allow_duplicate_names: bool = False) -> list[dict[str, Any]]:
    src = Path(path)
    suffix = src.suffix.lower()
    atoms: list[dict[str, Any]] = []
    if suffix == ".mol2":
        in_atoms = False
        for line in src.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("@<TRIPOS>ATOM"):
                in_atoms = True
                continue
            if line.startswith("@<TRIPOS>") and in_atoms:
                break
            if in_atoms and line.strip():
                parts = line.split()
                if len(parts) < 9:
                    raise ValueError(f"Invalid MOL2 atom line: {line.strip()}")
                atoms.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "x": float(parts[2]),
                        "y": float(parts[3]),
                        "z": float(parts[4]),
                        "atom_type": parts[5],
                        "resid": parts[6],
                        "resname": parts[7],
                        "charge": float(parts[8]),
                        "element": _guess_element(parts[1], parts[5]),
                    }
                )
    elif suffix == ".pdb":
        for line in src.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.startswith(("ATOM", "HETATM")):
                continue
            name = line[12:16].strip()
            raw_element = line[76:78].strip().upper()
            element = raw_element if raw_element in VDW_RADII or raw_element == "FE" else _guess_element(name)
            atoms.append(
                {
                    "index": len(atoms) + 1,
                    "name": name,
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54]),
                    "atom_type": _guess_element(name, line[76:78].strip()),
                    "resid": str(int(line[22:26])),
                    "resname": line[17:20].strip(),
                    "charge": 0.0,
                    "element": element,
                }
            )
    else:
        raise ValueError(f"Unsupported ligand pose format for ESP core: {src.suffix}")
    if not atoms:
        raise ValueError(f"No ligand atoms parsed from {src}")
    names = [atom["name"] for atom in atoms]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates and not allow_duplicate_names:
        raise ValueError(f"Duplicate ligand atom names prevent identity-safe charge output: {duplicates}")
    return atoms


def extract_ligand_from_complex_pdb(
    *,
    complex_pdb: str | Path,
    ligand_resname: str,
    ligand_chain: str,
    output_pdb: str | Path,
) -> str:
    src = Path(complex_pdb)
    out = Path(output_pdb)
    lines: list[str] = []
    selected_serials: set[int] = set()
    all_lines = src.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in all_lines:
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if line[17:20].strip() != ligand_resname:
            continue
        if (line[21].strip() or "A") != ligand_chain:
            continue
        lines.append(line)
        try:
            selected_serials.add(int(line[6:11]))
        except ValueError:
            pass
    if not lines:
        raise ValueError(f"No {ligand_resname} chain {ligand_chain} atoms found in {src}")
    for line in all_lines:
        if not line.startswith("CONECT"):
            continue
        try:
            serials = [int(line[i : i + 5]) for i in range(6, len(line), 5) if line[i : i + 5].strip()]
        except ValueError:
            continue
        if serials and serials[0] in selected_serials:
            kept = [serial for serial in serials if serial in selected_serials]
            if len(kept) > 1:
                lines.append("CONECT" + "".join(f"{serial:5d}" for serial in kept))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")
    return str(out)


def _write_atom_table(atoms: list[dict[str, Any]], output_csv: Path) -> None:
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["idx", "name", "element", "x", "y", "z"])
        writer.writeheader()
        for i, atom in enumerate(atoms, start=1):
            writer.writerow(
                {
                    "idx": i,
                    "name": atom["name"],
                    "element": atom["element"],
                    "x": f"{atom['x']:.8f}",
                    "y": f"{atom['y']:.8f}",
                    "z": f"{atom['z']:.8f}",
                }
            )


def _write_xyz(atoms: list[dict[str, Any]], output_xyz: Path) -> None:
    lines = [str(len(atoms)), "CYPForge exact ligand pose for GPU4PySCF/PySCF ESP"]
    for atom in atoms:
        lines.append(f"{atom['element']} {atom['x']:.8f} {atom['y']:.8f} {atom['z']:.8f}")
    output_xyz.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _unit_vector(x: float, y: float, z: float) -> tuple[float, float, float]:
    length = math.sqrt(x * x + y * y + z * z)
    if length < 1.0e-8:
        return (1.0, 0.0, 0.0)
    return (x / length, y / length, z / length)


def _write_pose_with_updated_coordinates(source: Path, atoms: list[dict[str, Any]], output: Path) -> None:
    suffix = source.suffix.lower()
    lines = source.read_text(encoding="utf-8", errors="ignore").splitlines()
    updated: list[str] = []
    atom_idx = 0
    if suffix == ".mol2":
        in_atoms = False
        for line in lines:
            if line.startswith("@<TRIPOS>ATOM"):
                in_atoms = True
                updated.append(line)
                continue
            if line.startswith("@<TRIPOS>") and in_atoms:
                in_atoms = False
                updated.append(line)
                continue
            if in_atoms and line.strip():
                parts = line.split()
                atom = atoms[atom_idx]
                updated.append(
                    f"{int(parts[0]):>7d} {parts[1]:<6s}{atom['x']:>10.4f}{atom['y']:>10.4f}{atom['z']:>10.4f} "
                    f"{parts[5]:<8s}{parts[6]:>3s} {parts[7]:<6s}{float(parts[8]):>10.6f}"
                )
                atom_idx += 1
            else:
                updated.append(line)
    elif suffix == ".pdb":
        for line in lines:
            if line.startswith(("ATOM", "HETATM")):
                atom = atoms[atom_idx]
                updated.append(f"{line[:30]}{atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}{line[54:]}")
                atom_idx += 1
            else:
                updated.append(line)
    else:
        raise ValueError(f"Unsupported pose format for coordinate cleanup: {source.suffix}")
    if atom_idx != len(atoms):
        raise ValueError(f"Updated {atom_idx} pose atoms but expected {len(atoms)}")
    output.write_text("\n".join(updated) + "\n", encoding="utf-8")


def _prepare_resp_geometry(
    atoms: list[dict[str, Any]],
    *,
    mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if mode not in {"none", "h-only"}:
        raise ValueError(f"Unsupported RESP geometry cleanup mode: {mode}")
    cleaned = [dict(atom) for atom in atoms]
    hydrogens = [idx for idx, atom in enumerate(cleaned) if atom["element"] == "H"]
    if mode == "none" or not hydrogens:
        return cleaned, {
            "mode": mode,
            "status": "skipped" if mode == "none" else "no_hydrogens",
            "policy": "QM geometry is the supplied ligand pose without pre-RESP coordinate cleanup.",
        }

    heavy_indices = [idx for idx, atom in enumerate(cleaned) if atom["element"] != "H"]
    if not heavy_indices:
        raise ValueError("Hydrogen-only RESP geometry cleanup requires at least one heavy atom.")
    moved: list[float] = []
    warnings: list[str] = []
    for h_idx in hydrogens:
        h = cleaned[h_idx]
        nearest = min(heavy_indices, key=lambda idx: _distance(h, cleaned[idx]))
        heavy = cleaned[nearest]
        initial = {"x": h["x"], "y": h["y"], "z": h["z"]}
        start_dist = _distance(h, heavy)
        target = COVALENT_RADII.get("H", 0.31) + COVALENT_RADII.get(heavy["element"], 0.76)
        if start_dist > max(1.45, 1.35 * target):
            warnings.append(f"{h['name']} nearest heavy atom {heavy['name']} is {start_dist:.3f} A away before cleanup")
        ux, uy, uz = _unit_vector(h["x"] - heavy["x"], h["y"] - heavy["y"], h["z"] - heavy["z"])
        h["x"], h["y"], h["z"] = heavy["x"] + ux * target, heavy["y"] + uy * target, heavy["z"] + uz * target
        for _ in range(40):
            push_x = push_y = push_z = 0.0
            for other_idx, other in enumerate(cleaned):
                if other_idx in {h_idx, nearest}:
                    continue
                dx, dy, dz = h["x"] - other["x"], h["y"] - other["y"], h["z"] - other["z"]
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                min_dist = 0.55 * (VDW_RADII.get("H", 1.20) + VDW_RADII.get(other["element"], 1.70))
                if dist < min_dist:
                    ux, uy, uz = _unit_vector(dx, dy, dz)
                    push = min(0.08, 0.25 * (min_dist - dist))
                    push_x += ux * push
                    push_y += uy * push
                    push_z += uz * push
            if abs(push_x) + abs(push_y) + abs(push_z) < 1.0e-7:
                break
            h["x"] += push_x
            h["y"] += push_y
            h["z"] += push_z
            ux, uy, uz = _unit_vector(h["x"] - heavy["x"], h["y"] - heavy["y"], h["z"] - heavy["z"])
            h["x"], h["y"], h["z"] = heavy["x"] + ux * target, heavy["y"] + uy * target, heavy["z"] + uz * target
        moved.append(_distance(initial, h))
    return cleaned, {
        "mode": mode,
        "status": "success",
        "policy": (
            "RESP QM geometry keeps all heavy atoms fixed at the confirmed complex pose; only hydrogen coordinates "
            "are lightly regularized to covalent bond lengths with simple nonbonded clash relief. This avoids full "
            "gas-phase ligand optimization before RESP."
        ),
        "hydrogen_count": len(hydrogens),
        "fixed_heavy_atom_count": len(heavy_indices),
        "max_hydrogen_displacement_a": round(max(moved) if moved else 0.0, 6),
        "warnings": warnings,
    }


def read_sdf_template(path: str | Path) -> dict[str, Any]:
    src = Path(path)
    lines = src.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 4:
        raise ValueError(f"Invalid SDF file: {src}")
    counts = lines[3]
    atom_count = int(counts[0:3])
    bond_count = int(counts[3:6])
    atoms: list[dict[str, Any]] = []
    bonds: list[dict[str, Any]] = []
    charge_by_atom: dict[int, int] = {}
    charge_code_map = {1: 3, 2: 2, 3: 1, 5: -1, 6: -2, 7: -3}
    for idx, line in enumerate(lines[4 : 4 + atom_count], start=1):
        charge_code_text = line[36:39].strip() if len(line) >= 39 else ""
        charge = charge_code_map.get(int(charge_code_text), 0) if charge_code_text.isdigit() else 0
        if charge:
            charge_by_atom[idx] = charge
        atoms.append(
            {
                "index": idx,
                "name": f"{line[31:34].strip().upper()}{idx}",
                "element": line[31:34].strip().upper(),
                "x": float(line[0:10]),
                "y": float(line[10:20]),
                "z": float(line[20:30]),
            }
        )
    for line in lines[4 + atom_count : 4 + atom_count + bond_count]:
        a = int(line[0:3])
        b = int(line[3:6])
        order = int(line[6:9])
        bonds.append({"a": a, "b": b, "order": order})
    for line in lines[4 + atom_count + bond_count :]:
        if not line.startswith("M  CHG"):
            continue
        fields = line.split()
        if len(fields) < 4:
            continue
        pair_count = int(fields[2])
        for pair_idx in range(pair_count):
            atom_index = int(fields[3 + pair_idx * 2])
            charge_by_atom[atom_index] = int(fields[4 + pair_idx * 2])
    if not atoms or not bonds:
        raise ValueError(f"SDF did not provide atoms and bonds: {src}")
    return {
        "atoms": atoms,
        "bonds": bonds,
        "formal_charge": int(sum(charge_by_atom.values())),
        "formal_charge_source": "SDF M CHG/atom charge code" if charge_by_atom else "SDF no explicit charge records; inferred neutral",
        "aromatic_bond_count": sum(1 for bond in bonds if int(bond["order"]) == 4),
    }


def _bond_edges_from_sdf(sdf: dict[str, Any]) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for bond in sdf["bonds"]:
        a, b = int(bond["a"]) - 1, int(bond["b"]) - 1
        edges.add((min(a, b), max(a, b)))
    return edges


def _infer_bond_edges_from_coords(atoms: list[dict[str, Any]]) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for i, a in enumerate(atoms):
        for j in range(i + 1, len(atoms)):
            b = atoms[j]
            if a["element"] == "H" and b["element"] == "H":
                continue
            ra = COVALENT_RADII.get(a["element"], 0.76)
            rb = COVALENT_RADII.get(b["element"], 0.76)
            dist = _distance(a, b)
            if 0.35 <= dist <= 1.25 * (ra + rb):
                edges.add((i, j))
    return edges


def _degree_map(atom_count: int, edges: set[tuple[int, int]]) -> dict[int, int]:
    degrees = {i: 0 for i in range(atom_count)}
    for a, b in edges:
        degrees[a] += 1
        degrees[b] += 1
    return degrees


def _rmsd(distances: list[float]) -> float:
    if not distances:
        return 0.0
    return math.sqrt(sum(dist * dist for dist in distances) / len(distances))


def _map_sdf_atoms_to_complex_atoms_strict(
    *,
    sdf_template: str | Path,
    ligand_pdb: str | Path,
    max_mappings: int = 200,
) -> dict[str, Any]:
    sdf = read_sdf_template(sdf_template)
    template_atoms = sdf["atoms"]
    complex_atoms = read_ligand_atoms(ligand_pdb)
    template_hydrogen_count = sum(1 for atom in template_atoms if atom["element"] == "H")
    complex_hydrogen_count = sum(1 for atom in complex_atoms if atom["element"] == "H")
    heavy_only_complex = (
        complex_hydrogen_count == 0
        and template_hydrogen_count > 0
        and len([atom for atom in template_atoms if atom["element"] != "H"]) == len(complex_atoms)
    )
    if len(template_atoms) != len(complex_atoms) and not heavy_only_complex:
        raise ValueError(f"SDF atom count {len(template_atoms)} does not match complex ligand atom count {len(complex_atoms)}")
    matched_template_indices = [
        idx for idx, atom in enumerate(template_atoms) if not heavy_only_complex or atom["element"] != "H"
    ]
    matched_template_atoms = [template_atoms[idx] for idx in matched_template_indices]
    template_elements = sorted(atom["element"] for atom in matched_template_atoms)
    complex_elements = sorted(atom["element"] for atom in complex_atoms)
    if template_elements != complex_elements:
        raise ValueError(f"SDF/complex element counts differ: sdf={template_elements}; complex={complex_elements}")

    template_index_to_match_index = {template_idx: match_idx for match_idx, template_idx in enumerate(matched_template_indices)}
    all_t_edges = _bond_edges_from_sdf(sdf)
    t_edges = {
        (
            min(template_index_to_match_index[a], template_index_to_match_index[b]),
            max(template_index_to_match_index[a], template_index_to_match_index[b]),
        )
        for a, b in all_t_edges
        if a in template_index_to_match_index and b in template_index_to_match_index
    }
    c_edges = _infer_bond_edges_from_coords(complex_atoms)
    t_deg = _degree_map(len(matched_template_atoms), t_edges)
    c_deg = _degree_map(len(complex_atoms), c_edges)
    t_neighbors = {i: set() for i in range(len(matched_template_atoms))}
    c_neighbors = {i: set() for i in range(len(complex_atoms))}
    for a, b in t_edges:
        t_neighbors[a].add(b)
        t_neighbors[b].add(a)
    for a, b in c_edges:
        c_neighbors[a].add(b)
        c_neighbors[b].add(a)

    order = sorted(range(len(matched_template_atoms)), key=lambda i: (-t_deg[i], matched_template_atoms[i]["element"], i))
    candidates = {
        i: [
            j
            for j, atom in enumerate(complex_atoms)
            if atom["element"] == matched_template_atoms[i]["element"] and c_deg[j] == t_deg[i]
        ]
        for i in range(len(matched_template_atoms))
    }
    if any(not values for values in candidates.values()):
        empty = [i + 1 for i, values in candidates.items() if not values]
        raise ValueError(f"No element/degree candidates for SDF atoms: {empty}")

    mappings: list[dict[int, int]] = []

    def backtrack(pos: int, current: dict[int, int], used: set[int]) -> None:
        if len(mappings) >= max_mappings:
            return
        if pos == len(order):
            mappings.append(dict(current))
            return
        t_idx = order[pos]
        for c_idx in candidates[t_idx]:
            if c_idx in used:
                continue
            ok = True
            for mapped_t, mapped_c in current.items():
                t_has = mapped_t in t_neighbors[t_idx]
                c_has = mapped_c in c_neighbors[c_idx]
                if t_has != c_has:
                    ok = False
                    break
            if not ok:
                continue
            current[t_idx] = c_idx
            used.add(c_idx)
            backtrack(pos + 1, current, used)
            used.remove(c_idx)
            del current[t_idx]

    backtrack(0, {}, set())
    if not mappings:
        raise ValueError("Could not map SDF ligand graph to complex ligand coordinates.")

    heavy_ambiguous: list[dict[str, Any]] = []
    for match_idx, atom in enumerate(matched_template_atoms):
        t_idx = matched_template_indices[match_idx]
        mapped_values = sorted({mapping[match_idx] for mapping in mappings})
        if atom["element"] != "H" and len(mapped_values) > 1:
            heavy_ambiguous.append(
                {
                    "sdf_index": t_idx + 1,
                    "element": atom["element"],
                    "candidate_complex_atom_names": [complex_atoms[j]["name"] for j in mapped_values],
                }
            )
    if heavy_ambiguous:
        raise ValueError(
            "Ambiguous heavy-atom mapping prevents identity-safe coordinate and charge transfer: "
            f"{heavy_ambiguous[:10]}"
        )
    chosen = mappings[0]
    mapped_template_edges = {(min(chosen[a], chosen[b]), max(chosen[a], chosen[b])) for a, b in t_edges}
    if mapped_template_edges != c_edges:
        missing = sorted(mapped_template_edges - c_edges)
        extra = sorted(c_edges - mapped_template_edges)
        raise ValueError(
            "Mapped SDF connectivity does not match complex ligand coordinate-inferred connectivity: "
            f"missing_in_complex={missing[:10]}, extra_in_complex={extra[:10]}"
        )
    heavy_template_to_complex_distances = [
        _distance(template_atoms[template_idx], complex_atoms[chosen[match_idx]])
        for match_idx, template_idx in enumerate(matched_template_indices)
        if template_atoms[template_idx]["element"] != "H"
    ]
    rows = []
    explicit_map: dict[str, int] = {}
    for t_idx, atom in enumerate(template_atoms):
        if t_idx in template_index_to_match_index:
            c_idx = chosen[template_index_to_match_index[t_idx]]
            explicit_map[str(t_idx + 1)] = c_idx + 1
            rows.append(
                {
                    "sdf_index": t_idx + 1,
                    "sdf_element": atom["element"],
                    "complex_index": c_idx + 1,
                    "complex_atom_name": complex_atoms[c_idx]["name"],
                    "complex_element": complex_atoms[c_idx]["element"],
                }
            )
            continue
        bonded_heavy = [
            b if a == t_idx else a
            for a, b in all_t_edges
            if (a == t_idx or b == t_idx) and (b if a == t_idx else a) in template_index_to_match_index
        ]
        parent_template_idx = bonded_heavy[0] if bonded_heavy else None
        parent_complex_idx = (
            chosen[template_index_to_match_index[parent_template_idx]] if parent_template_idx is not None else None
        )
        rows.append(
            {
                "sdf_index": t_idx + 1,
                "sdf_element": atom["element"],
                "complex_index": None,
                "complex_atom_name": None,
                "complex_element": None,
                "coordinate_source": "sdf_hydrogen_rigid_fit_to_complex_heavy_pose",
                "parent_sdf_index": parent_template_idx + 1 if parent_template_idx is not None else None,
                "parent_complex_index": parent_complex_idx + 1 if parent_complex_idx is not None else None,
            }
        )
    return {
        "status": "success",
        "mapping_count": len(mappings),
        "heavy_atom_mapping_unique": True,
        "sdf_atom_count": len(template_atoms),
        "complex_atom_count": len(complex_atoms),
        "complex_hydrogen_policy": (
            "complex ligand is heavy-only; SDF hydrogens are retained and rigid-fitted into the complex heavy-atom frame"
            if heavy_only_complex
            else "complex ligand includes all SDF atoms"
        ),
        "sdf_bond_count": len(t_edges),
        "inferred_complex_bond_count": len(c_edges),
        "element_consistency": "passed",
        "connectivity_consistency": "passed",
        "formal_charge_from_sdf": sdf["formal_charge"],
        "formal_charge_source": sdf["formal_charge_source"],
        "aromaticity_status": "explicit_aromatic_bonds_present" if sdf["aromatic_bond_count"] else "not_explicit_or_kekulized",
        "aromatic_bond_count": sdf["aromatic_bond_count"],
        "raw_template_to_complex_heavy_atom_distance_rms_a": round(_rmsd(heavy_template_to_complex_distances), 6),
        "raw_template_to_complex_heavy_atom_max_distance_a": round(max(heavy_template_to_complex_distances) if heavy_template_to_complex_distances else 0.0, 6),
        "raw_template_to_complex_coordinate_distance_note": (
            "Pre-alignment SDF-template to complex-PDB coordinate-frame distance; "
            "not a mapped-pose QC metric. Final coordinate transfer is checked by "
            "written_vs_complex_heavy_atom_rmsd_a in the coordinate-transfer report."
        ),
        "mapping_policy": "graph_isomorphism_by_element_degree_and_connectivity; row-order transfer is forbidden",
        "ambiguity_policy": "fail if any heavy SDF atom maps to multiple possible complex heavy atoms",
        "mapping_source": "strict_graph_isomorphism",
        "sdf_atom_id_to_pdb_atom_index": explicit_map,
        "mapping": rows,
    }


def _map_sdf_atoms_to_complex_atoms_hypergraph_fallback(
    *,
    sdf_template: str | Path,
    ligand_pdb: str | Path,
    strict_error: Exception,
) -> dict[str, Any]:
    from .ligand_heavy_hypergraph_resolver import resolve_heavy_hypergraph_mapping

    sdf = read_sdf_template(sdf_template)
    template_atoms = sdf["atoms"]
    complex_atoms = read_ligand_atoms(ligand_pdb, allow_duplicate_names=True)
    result = resolve_heavy_hypergraph_mapping(sdf_template=sdf_template, ligand_pdb=ligand_pdb)
    if result["decision"] not in {"unique", "equivalent_ok"}:
        raise ValueError(
            "Heavy-hypergraph fallback did not resolve an identity-safe mapping: "
            f"decision={result['decision']}; strict_error={strict_error}"
        )

    explicit_map = {str(k): int(v) for k, v in result["sdf_atom_id_to_pdb_atom_index"].items()}
    all_t_edges = _bond_edges_from_sdf(sdf)
    rows = []
    for t_idx, atom in enumerate(template_atoms):
        mapped_index = explicit_map.get(str(t_idx + 1))
        if mapped_index is not None:
            c_atom = complex_atoms[mapped_index - 1]
            rows.append(
                {
                    "sdf_index": t_idx + 1,
                    "sdf_element": atom["element"],
                    "complex_index": mapped_index,
                    "complex_atom_name": c_atom["name"],
                    "complex_element": c_atom["element"],
                }
            )
            continue
        bonded_heavy = [
            b if a == t_idx else a
            for a, b in all_t_edges
            if (a == t_idx or b == t_idx) and str((b if a == t_idx else a) + 1) in explicit_map
        ]
        parent_template_idx = bonded_heavy[0] if bonded_heavy else None
        parent_complex_idx = explicit_map.get(str(parent_template_idx + 1)) if parent_template_idx is not None else None
        rows.append(
            {
                "sdf_index": t_idx + 1,
                "sdf_element": atom["element"],
                "complex_index": None,
                "complex_atom_name": None,
                "complex_element": None,
                "coordinate_source": "sdf_hydrogen_rigid_fit_to_complex_heavy_pose",
                "parent_sdf_index": parent_template_idx + 1 if parent_template_idx is not None else None,
                "parent_complex_index": parent_complex_idx,
            }
        )

    duplicates = sorted({atom["name"] for atom in complex_atoms if [x["name"] for x in complex_atoms].count(atom["name"]) > 1})
    metrics = result["best"]
    return {
        "status": "success",
        "mapping_count": 1 if result["decision"] == "unique" else result["counts"]["mappings"],
        "heavy_atom_mapping_unique": result["decision"] == "unique",
        "heavy_atom_mapping_decision": result["decision"],
        "sdf_atom_count": len(template_atoms),
        "complex_atom_count": len(complex_atoms),
        "complex_hydrogen_policy": "heavy-hypergraph fallback maps heavy atoms only; SDF hydrogens are retained and rigid-fitted into the complex heavy-atom frame",
        "sdf_bond_count": result["counts"]["sdf_edges"],
        "inferred_complex_bond_count": result["counts"]["pdb_inferred_edges"],
        "element_consistency": "passed",
        "connectivity_consistency": "passed",
        "formal_charge_from_sdf": sdf["formal_charge"],
        "formal_charge_source": sdf["formal_charge_source"],
        "aromaticity_status": "explicit_aromatic_bonds_present" if sdf["aromatic_bond_count"] else "not_explicit_or_kekulized",
        "aromatic_bond_count": sdf["aromatic_bond_count"],
        "raw_template_to_complex_heavy_atom_distance_rms_a": metrics["kabsch_rmsd_a"],
        "raw_template_to_complex_heavy_atom_max_distance_a": None,
        "raw_template_to_complex_coordinate_distance_note": "Fallback reports Kabsch-heavy RMSD after graph/hypergraph scoring, not raw unaligned coordinate distance.",
        "mapping_policy": "strict graph mapping failed; heavy_atom_graph_plus_local_geometry_hyperedges fallback accepted",
        "ambiguity_policy": "accept only unique mapping or proven equivalent heavy-atom exchanges",
        "mapping_source": "heavy_hypergraph_fallback",
        "fallback_reason": str(strict_error),
        "fallback_schema": result["schema"],
        "fallback_score": metrics,
        "fallback_score_gap": result["score_gap"],
        "fallback_equivalence_proof": result["equivalence_proof"],
        "duplicate_complex_atom_names": duplicates,
        "sdf_atom_id_to_pdb_atom_index": explicit_map,
        "mapping": rows,
    }


def map_sdf_atoms_to_complex_atoms(
    *,
    sdf_template: str | Path,
    ligand_pdb: str | Path,
    max_mappings: int = 200,
    enable_hypergraph_fallback: bool = True,
) -> dict[str, Any]:
    try:
        strict = _map_sdf_atoms_to_complex_atoms_strict(
            sdf_template=sdf_template,
            ligand_pdb=ligand_pdb,
            max_mappings=max_mappings,
        )
        if enable_hypergraph_fallback and int(strict.get("mapping_count", 1)) != 1:
            return _map_sdf_atoms_to_complex_atoms_hypergraph_fallback(
                sdf_template=sdf_template,
                ligand_pdb=ligand_pdb,
                strict_error=RuntimeError(f"strict mapper returned mapping_count={strict.get('mapping_count')}"),
            )
        return strict
    except Exception as exc:
        if not enable_hypergraph_fallback:
            raise
        return _map_sdf_atoms_to_complex_atoms_hypergraph_fallback(
            sdf_template=sdf_template,
            ligand_pdb=ligand_pdb,
            strict_error=exc,
        )


def _runner_script() -> str:
    return r'''
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from pyscf import gto, scf

VDW = {"H":1.20,"C":1.70,"N":1.55,"O":1.52,"F":1.47,"P":1.80,"S":1.80,"CL":1.75,"BR":1.85,"I":1.98}
Z = {"H":1,"C":6,"N":7,"O":8,"F":9,"P":15,"S":16,"CL":17,"BR":35,"I":53}


def fibonacci_sphere(n):
    pts = []
    phi = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2 if n > 1 else 0
        radius = math.sqrt(max(0.0, 1 - y * y))
        theta = phi * i
        pts.append(np.array([math.cos(theta) * radius, y, math.sin(theta) * radius], dtype=float))
    return pts


def read_atoms(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8", newline="")))
    atoms = []
    for row in rows:
        atoms.append({
            "idx": int(row["idx"]),
            "name": row["name"],
            "element": row["element"].upper(),
            "coord": np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=float),
        })
    return atoms


def make_grid(atoms, points_per_atom, scales):
    unit = fibonacci_sphere(points_per_atom)
    coords = [atom["coord"] for atom in atoms]
    elements = [atom["element"] for atom in atoms]
    grid = []
    for atom in atoms:
        radius = VDW.get(atom["element"], 1.70)
        for scale in scales:
            shell_r = radius * scale
            for direction in unit:
                point = atom["coord"] + shell_r * direction
                keep = True
                for other_coord, other_element in zip(coords, elements):
                    if np.linalg.norm(point - other_coord) < VDW.get(other_element, 1.70) * 1.25:
                        keep = False
                        break
                if keep:
                    grid.append(point)
    if len(grid) < len(atoms) * 4:
        raise RuntimeError(f"ESP grid too small after exclusions: {len(grid)} points")
    return np.array(grid, dtype=float)


def constrained_fit(atoms, grid, esp, formal_charge):
    matrix = np.zeros((len(grid), len(atoms)), dtype=float)
    for i, point in enumerate(grid):
        for j, atom in enumerate(atoms):
            r = np.linalg.norm(point - atom["coord"])
            matrix[i, j] = 1.0 / max(r, 1.0e-8)
    lhs = np.block([[matrix.T @ matrix, np.ones((len(atoms), 1))], [np.ones((1, len(atoms))), np.zeros((1, 1))]])
    rhs = np.concatenate([matrix.T @ esp, np.array([float(formal_charge)])])
    solution = np.linalg.solve(lhs, rhs)
    charges = solution[:len(atoms)]
    fitted = matrix @ charges
    residual = fitted - esp
    return charges, {
        "rmse_au": float(np.sqrt(np.mean(residual * residual))),
        "max_abs_error_au": float(np.max(np.abs(residual))),
        "charge_sum": float(np.sum(charges)),
    }


def main():
    cfg = json.loads(Path("esp_config.json").read_text(encoding="utf-8"))
    atoms = read_atoms("atom_table.csv")
    atom_str = "; ".join(f"{a['element']} {a['coord'][0]} {a['coord'][1]} {a['coord'][2]}" for a in atoms)
    mol = gto.Mole()
    mol.atom = atom_str
    mol.basis = cfg["basis"]
    mol.charge = int(cfg["formal_charge"])
    mol.spin = int(cfg["spin_multiplicity"]) - 1
    mol.unit = "Angstrom"
    mol.build()
    mf = scf.UHF(mol) if mol.spin else scf.RHF(mol)
    backend = "pyscf_cpu"
    if cfg.get("use_gpu", True):
        try:
            import gpu4pyscf  # noqa: F401
            mf = mf.to_gpu()
            backend = "gpu4pyscf"
        except Exception as exc:
            if cfg.get("require_gpu", False):
                raise
            backend = f"pyscf_cpu_gpu_unavailable:{exc}"
    energy = mf.kernel()
    if not mf.converged:
        raise RuntimeError("SCF did not converge")
    dm = mf.make_rdm1()
    if isinstance(dm, (tuple, list)) or getattr(dm, "ndim", 0) == 3:
        dm = dm[0] + dm[1]
    grid = make_grid(atoms, int(cfg["points_per_atom"]), list(cfg["vdw_scales"]))
    esp = []
    charges_z = mol.atom_charges()
    coords_bohr = mol.atom_coords()
    for point_ang in grid:
        point_bohr = point_ang / 0.529177210903
        mol.set_rinv_orig_(point_bohr)
        rinv = mol.intor("int1e_rinv")
        electronic = -float(np.einsum("ij,ji->", dm, rinv))
        nuclear = 0.0
        for z, coord_bohr in zip(charges_z, coords_bohr):
            nuclear += float(z) / max(float(np.linalg.norm(point_bohr - coord_bohr)), 1.0e-8)
        esp.append(nuclear + electronic)
    esp = np.array(esp, dtype=float)
    fitted_charges, fit = constrained_fit(atoms, grid, esp, int(cfg["formal_charge"]))
    with open("esp_grid.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y", "z", "esp_au"])
        for point, value in zip(grid, esp):
            writer.writerow([f"{point[0]:.8f}", f"{point[1]:.8f}", f"{point[2]:.8f}", f"{value:.12e}"])
    with open("esp_charges.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["idx", "name", "element", "esp_charge"])
        for atom, charge in zip(atoms, fitted_charges):
            writer.writerow([atom["idx"], atom["name"], atom["element"], f"{charge:.8f}"])
    report = {
        "schema": "cypforge.gpu4pyscf_esp_fit_result.v1",
        "status": "success",
        "backend": backend,
        "method": "HF",
        "basis": cfg["basis"],
        "formal_charge": int(cfg["formal_charge"]),
        "spin_multiplicity": int(cfg["spin_multiplicity"]),
        "scf_converged": bool(mf.converged),
        "scf_energy_hartree": float(energy),
        "grid_point_count": int(len(grid)),
        "fit": fit,
        "limitation": "ESP-fit charges are not a proof of binding correctness or chemical mechanism.",
    }
    Path("esp_fit_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
'''


def _molden_runner_script() -> str:
    return r'''
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from pyscf import gto, scf
from pyscf.tools import molden as molden_tool


def read_atoms(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8", newline="")))
    atoms = []
    for row in rows:
        atoms.append({
            "idx": int(row["idx"]),
            "name": row["name"],
            "element": row["element"].upper(),
            "coord": np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=float),
        })
    return atoms


def cpu_array(value):
    try:
        import cupy
        if isinstance(value, cupy.ndarray):
            return cupy.asnumpy(value)
    except Exception:
        pass
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_array(item) for item in value)
    return value


def main():
    cfg = json.loads(Path("esp_config.json").read_text(encoding="utf-8"))
    atoms = read_atoms("atom_table.csv")
    atom_str = "; ".join(f"{a['element']} {a['coord'][0]} {a['coord'][1]} {a['coord'][2]}" for a in atoms)
    mol = gto.Mole()
    mol.atom = atom_str
    mol.basis = cfg["basis"]
    mol.charge = int(cfg["formal_charge"])
    mol.spin = int(cfg["spin_multiplicity"]) - 1
    mol.unit = "Angstrom"
    mol.build()
    mf = scf.UHF(mol) if mol.spin else scf.RHF(mol)
    backend = "pyscf_cpu"
    if cfg.get("use_gpu", True):
        try:
            import gpu4pyscf  # noqa: F401
            mf = mf.to_gpu()
            backend = "gpu4pyscf"
        except Exception as exc:
            if cfg.get("require_gpu", False):
                raise
            backend = f"pyscf_cpu_gpu_unavailable:{exc}"
    energy = mf.kernel()
    if not mf.converged:
        raise RuntimeError("SCF did not converge")
    molden_name = cfg.get("molden_name", "ligand_hf631gstar.molden")
    mo_coeff = cpu_array(mf.mo_coeff)
    mo_energy = cpu_array(mf.mo_energy)
    mo_occ = cpu_array(mf.mo_occ)
    with open(molden_name, "w", encoding="utf-8") as handle:
        molden_tool.header(mol, handle)
        molden_tool.orbital_coeff(mol, handle, mo_coeff, ene=mo_energy, occ=mo_occ)
    report = {
        "schema": "cypforge.gpu4pyscf_molden_result.v1",
        "status": "success",
        "backend": backend,
        "method": "HF",
        "basis": cfg["basis"],
        "formal_charge": int(cfg["formal_charge"]),
        "spin_multiplicity": int(cfg["spin_multiplicity"]),
        "scf_converged": bool(mf.converged),
        "scf_energy_hartree": float(energy),
        "molden": molden_name,
        "limitation": "Molden generation is only the wavefunction step; charge fitting is delegated to Multiwfn.",
    }
    Path("molden_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
'''


def _inject_csv_charges_to_mol2(source_mol2: Path, charge_csv: Path, output_mol2: Path, resname: str | None) -> int:
    charges = {row["name"]: float(row["esp_charge"]) for row in csv.DictReader(charge_csv.open("r", encoding="utf-8", newline=""))}
    lines = source_mol2.read_text(encoding="utf-8", errors="ignore").splitlines()
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
                raise ValueError(f"Missing ESP charge for MOL2 atom {name}")
            final_resname = resname or parts[7]
            output.append(
                f"{int(parts[0]):>7d} {name:<6s}{float(parts[2]):>10.4f}{float(parts[3]):>10.4f}{float(parts[4]):>10.4f} "
                f"{parts[5]:<8s}{parts[6]:>3s} {final_resname:<6s}{charges[name]:>10.6f}"
            )
            injected += 1
        else:
            output.append(line)
    output_mol2.write_text("\n".join(output) + "\n", encoding="utf-8")
    return injected


def _read_multiwfn_chg(path: Path) -> list[float]:
    charges: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            charges.append(float(parts[-1]))
        except ValueError:
            continue
    if not charges:
        raise ValueError(f"No numeric charges parsed from Multiwfn .chg file: {path}")
    return charges


def _write_chg_charge_csv(atoms: list[dict[str, Any]], chg_file: Path, output_csv: Path) -> float:
    charges = _read_multiwfn_chg(chg_file)
    if len(charges) != len(atoms):
        raise ValueError(f"Multiwfn charge count {len(charges)} does not match atom count {len(atoms)}")
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["idx", "name", "element", "resp_charge"])
        for atom, charge in zip(atoms, charges):
            writer.writerow([atom["index"], atom["name"], atom["element"], f"{charge:.8f}"])
    return sum(charges)


def _inject_chg_charges_to_mol2(source_mol2: Path, chg_file: Path, output_mol2: Path, resname: str | None) -> int:
    charges = _read_multiwfn_chg(chg_file)
    lines = source_mol2.read_text(encoding="utf-8", errors="ignore").splitlines()
    output: list[str] = []
    in_atoms = False
    atom_idx = 0
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
            if atom_idx >= len(charges):
                raise ValueError("Multiwfn charge file has fewer charges than MOL2 atoms")
            parts = line.split()
            final_resname = resname or parts[7]
            output.append(
                f"{int(parts[0]):>7d} {parts[1]:<6s}{float(parts[2]):>10.4f}{float(parts[3]):>10.4f}{float(parts[4]):>10.4f} "
                f"{parts[5]:<8s}{parts[6]:>3s} {final_resname:<6s}{charges[atom_idx]:>10.6f}"
            )
            atom_idx += 1
        else:
            output.append(line)
    if atom_idx != len(charges):
        raise ValueError(f"Injected {atom_idx} MOL2 atoms but Multiwfn provided {len(charges)} charges")
    output_mol2.write_text("\n".join(output) + "\n", encoding="utf-8")
    return atom_idx


def prepare_gpu4pyscf_esp_job(
    *,
    ligand_pose: str | Path,
    formal_charge: int,
    output_dir: str | Path,
    resname: str = "LIG",
    spin_multiplicity: int = 1,
    basis: str = "6-31g*",
    points_per_atom: int = 24,
    vdw_scales: tuple[float, ...] = (1.4, 1.6, 1.8, 2.0),
    require_hydrogens: bool = True,
    resp_geometry_cleanup: str = "h-only",
    use_gpu: bool = True,
    require_gpu: bool = False,
) -> dict[str, Any]:
    src = Path(ligand_pose)
    out = Path(output_dir)
    atoms = read_ligand_atoms(src)
    hydrogen_count = sum(1 for atom in atoms if atom["element"] == "H")
    if require_hydrogens and hydrogen_count == 0:
        raise ValueError(
            "Ligand pose has no hydrogens. Vina/PDBQT heavy-atom output is not a valid GPU4PySCF ESP input; "
            "provide a user-confirmed hydrogen-complete pose."
        )
    out.mkdir(parents=True, exist_ok=True)
    qm_atoms, cleanup_report = _prepare_resp_geometry(atoms, mode=resp_geometry_cleanup)
    qm_pose = out / f"ligand_resp_geometry{src.suffix.lower()}"
    _write_pose_with_updated_coordinates(src, qm_atoms, qm_pose)
    atom_table = out / "atom_table.csv"
    xyz = out / "ligand_qm_input.xyz"
    _write_atom_table(qm_atoms, atom_table)
    _write_xyz(qm_atoms, xyz)
    config = {
        "schema": "cypforge.gpu4pyscf_esp_config.v1",
        "input_ligand_pose": str(src),
        "qm_ligand_pose": str(qm_pose),
        "resp_geometry_cleanup": cleanup_report,
        "formal_charge": formal_charge,
        "spin_multiplicity": spin_multiplicity,
        "basis": basis,
        "method": "HF",
        "points_per_atom": points_per_atom,
        "vdw_scales": list(vdw_scales),
        "use_gpu": use_gpu,
        "require_gpu": require_gpu,
        "resname": resname,
    }
    (out / "esp_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "run_gpu4pyscf_esp.py").write_text(_runner_script().lstrip(), encoding="utf-8")
    manifest = {
        "schema": "cypforge.gpu4pyscf_esp_job.v1",
        "status": "prepared",
        "input_ligand_pose": str(src),
        "qm_ligand_pose": str(qm_pose),
        "atom_count": len(atoms),
        "hydrogen_count": hydrogen_count,
        "resp_geometry_cleanup": cleanup_report,
        "formal_charge": formal_charge,
        "spin_multiplicity": spin_multiplicity,
        "qm_input_xyz": str(xyz),
        "atom_table_csv": str(atom_table),
        "runner": str(out / "run_gpu4pyscf_esp.py"),
        "config": str(out / "esp_config.json"),
        "limitation": "This prepares an ESP calculation for the confirmed ligand pose; it does not choose the binding pose or optimize heavy atoms.",
    }
    (out / "gpu4pyscf_esp_job_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def prepare_gpu4pyscf_molden_job(
    *,
    ligand_pose: str | Path,
    formal_charge: int,
    output_dir: str | Path,
    resname: str = "LIG",
    spin_multiplicity: int = 1,
    basis: str = "6-31g*",
    require_hydrogens: bool = True,
    resp_geometry_cleanup: str = "h-only",
    use_gpu: bool = True,
    require_gpu: bool = False,
) -> dict[str, Any]:
    manifest = prepare_gpu4pyscf_esp_job(
        ligand_pose=ligand_pose,
        formal_charge=formal_charge,
        output_dir=output_dir,
        resname=resname,
        spin_multiplicity=spin_multiplicity,
        basis=basis,
        require_hydrogens=require_hydrogens,
        resp_geometry_cleanup=resp_geometry_cleanup,
        use_gpu=use_gpu,
        require_gpu=require_gpu,
    )
    out = Path(output_dir)
    config_path = out / "esp_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["molden_name"] = f"{resname}_hf631gstar.molden"
    config["charge_fit_method"] = "Multiwfn RESP"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    runner = out / "run_gpu4pyscf_molden.py"
    runner.write_text(_molden_runner_script().lstrip(), encoding="utf-8")
    manifest.update(
        {
            "schema": "cypforge.gpu4pyscf_molden_job.v1",
            "runner": str(runner),
            "molden": str(out / config["molden_name"]),
            "charge_fit_method": "Multiwfn RESP",
            "limitation": "This prepares a PySCF/GPU4PySCF Molden wavefunction for Multiwfn RESP fitting.",
        }
    )
    (out / "gpu4pyscf_molden_job_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def _run_parmchk2(out: Path, mol2_name: str, frcmod_name: str, amber_sh: str) -> dict[str, Any]:
    wsl_out = _win_to_wsl(out)
    cmd = (
        f"source {shlex.quote(amber_sh)} && cd {shlex.quote(wsl_out)} && "
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
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def _run_antechamber_pdb_to_mol2(
    *,
    ligand_pdb: Path,
    output_dir: Path,
    output_name: str,
    resname: str,
    formal_charge: int,
    amber_sh: str,
) -> dict[str, Any]:
    wsl_out = _win_to_wsl(output_dir)
    wsl_ligand = _win_to_wsl(ligand_pdb)
    cmd = (
        f"source {shlex.quote(amber_sh)} && cd {shlex.quote(wsl_out)} && "
        f"antechamber -i {shlex.quote(wsl_ligand)} -fi pdb "
        f"-o {shlex.quote(output_name)} -fo mol2 -at gaff2 -c gas "
        f"-nc {formal_charge} -rn {shlex.quote(resname)} -pf y"
    )
    result = subprocess.run(
        ["wsl", "bash", "-lc", cmd] if os.name == "nt" else ["bash", "-lc", cmd],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def _run_antechamber_sdf_to_mol2(
    *,
    ligand_sdf: Path,
    output_dir: Path,
    output_name: str,
    resname: str,
    formal_charge: int,
    amber_sh: str,
) -> dict[str, Any]:
    wsl_out = _win_to_wsl(output_dir)
    wsl_ligand = _win_to_wsl(ligand_sdf)
    cmd = (
        f"source {shlex.quote(amber_sh)} && cd {shlex.quote(wsl_out)} && "
        f"antechamber -i {shlex.quote(wsl_ligand)} -fi sdf "
        f"-o {shlex.quote(output_name)} -fo mol2 -at gaff2 -c gas "
        f"-nc {formal_charge} -rn {shlex.quote(resname)} -pf y"
    )
    result = subprocess.run(
        ["wsl", "bash", "-lc", cmd] if os.name == "nt" else ["bash", "-lc", cmd],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def _pdb_has_ligand_conect(path: Path) -> bool:
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("CONECT"):
            return True
    return False


def _write_sdf_order_complex_pose_mol2(
    *,
    typed_sdf_mol2: Path,
    complex_ligand_pdb: Path,
    mapping: dict[str, Any],
    output_mol2: Path,
    resname: str,
) -> dict[str, Any]:
    typed_atoms = read_ligand_atoms(typed_sdf_mol2)
    complex_atoms = read_ligand_atoms(
        complex_ligand_pdb,
        allow_duplicate_names=mapping.get("mapping_source") == "heavy_hypergraph_fallback",
    )
    if len(typed_atoms) != len(mapping["mapping"]):
        raise ValueError("Typed SDF MOL2 atom count does not match SDF mapping count.")
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - numpy is required for this branch on server
        raise RuntimeError("SDF-to-complex coordinate transfer requires numpy for rigid fitting") from exc

    mapping_rows = sorted(mapping["mapping"], key=lambda row: row["sdf_index"])
    heavy_rows = [row for row in mapping_rows if row.get("complex_index") is not None and row["sdf_element"] != "H"]
    if len(heavy_rows) < 1:
        raise ValueError("At least one mapped heavy atom is required for coordinate transfer.")
    sdf_heavy = np.array(
        [[typed_atoms[int(row["sdf_index"]) - 1][axis] for axis in ("x", "y", "z")] for row in heavy_rows],
        dtype=float,
    )
    complex_heavy = np.array(
        [[complex_atoms[int(row["complex_index"]) - 1][axis] for axis in ("x", "y", "z")] for row in heavy_rows],
        dtype=float,
    )
    if len(heavy_rows) >= 3:
        rotation, sdf_centroid, complex_centroid = kabsch_transform(sdf_heavy, complex_heavy)
    else:
        rotation = np.eye(3)
        sdf_centroid = sdf_heavy[0]
        complex_centroid = complex_heavy[0]

    def transform_coord(x: float, y: float, z: float) -> tuple[float, float, float]:
        coord = (np.array([x, y, z], dtype=float) - sdf_centroid) @ rotation.T + complex_centroid
        return float(coord[0]), float(coord[1]), float(coord[2])

    output: list[str] = []
    in_atoms = False
    max_delta = 0.0
    lines = typed_sdf_mol2.read_text(encoding="utf-8", errors="ignore").splitlines()
    bond_neighbors: dict[int, set[int]] = {idx: set() for idx in range(len(typed_atoms))}
    in_bonds = False
    for line in lines:
        if line.startswith("@<TRIPOS>BOND"):
            in_bonds = True
            continue
        if line.startswith("@<TRIPOS>") and in_bonds:
            in_bonds = False
            continue
        if in_bonds and line.strip():
            parts = line.split()
            if len(parts) >= 3:
                a_idx, b_idx = int(parts[1]) - 1, int(parts[2]) - 1
                if a_idx in bond_neighbors and b_idx in bond_neighbors:
                    bond_neighbors[a_idx].add(b_idx)
                    bond_neighbors[b_idx].add(a_idx)

    def _target_heavy_coord(idx: int) -> np.ndarray:
        row = mapping_rows[idx]
        atom = typed_atoms[idx]
        if row.get("complex_index") is not None and row["sdf_element"] != "H":
            coord_atom = complex_atoms[int(row["complex_index"]) - 1]
            return np.array([coord_atom["x"], coord_atom["y"], coord_atom["z"]], dtype=float)
        return np.array(transform_coord(atom["x"], atom["y"], atom["z"]), dtype=float)

    def _rotation_from_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        source = source / np.linalg.norm(source)
        target = target / np.linalg.norm(target)
        cross = np.cross(source, target)
        dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
        if np.linalg.norm(cross) < 1.0e-10:
            return np.eye(3) if dot > 0 else -np.eye(3)
        skew = np.array(
            [[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]],
            dtype=float,
        )
        return np.eye(3) + skew + skew @ skew * ((1.0 - dot) / (np.linalg.norm(cross) ** 2))

    def _local_hydrogen_coord(idx: int) -> tuple[float, float, float]:
        atom = typed_atoms[idx]
        parent_indices = [n_idx for n_idx in bond_neighbors.get(idx, set()) if typed_atoms[n_idx]["element"] != "H"]
        if not parent_indices:
            return transform_coord(atom["x"], atom["y"], atom["z"])
        parent_idx = parent_indices[0]
        parent = typed_atoms[parent_idx]
        source_points = [np.array([parent["x"], parent["y"], parent["z"]], dtype=float)]
        target_points = [_target_heavy_coord(parent_idx)]
        for nbr_idx in sorted(bond_neighbors.get(parent_idx, set())):
            if nbr_idx == idx or typed_atoms[nbr_idx]["element"] == "H":
                continue
            source_points.append(np.array([typed_atoms[nbr_idx][axis] for axis in ("x", "y", "z")], dtype=float))
            target_points.append(_target_heavy_coord(nbr_idx))
        source_h = np.array([atom["x"], atom["y"], atom["z"]], dtype=float)
        if len(source_points) >= 3:
            local_rotation, local_source_centroid, local_target_centroid = kabsch_transform(
                np.array(source_points, dtype=float), np.array(target_points, dtype=float)
            )
            coord = (source_h - local_source_centroid) @ local_rotation.T + local_target_centroid
            return float(coord[0]), float(coord[1]), float(coord[2])
        if len(source_points) == 2:
            parent_source = source_points[0]
            parent_target = target_points[0]
            source_axis = source_points[1] - parent_source
            target_axis = target_points[1] - parent_target
            if np.linalg.norm(source_axis) > 1.0e-8 and np.linalg.norm(target_axis) > 1.0e-8:
                local_rotation = _rotation_from_vectors(source_axis, target_axis)
                coord = parent_target + (source_h - parent_source) @ local_rotation.T
                return float(coord[0]), float(coord[1]), float(coord[2])
        return transform_coord(atom["x"], atom["y"], atom["z"])

    atom_idx = 0
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
            row = mapping_rows[atom_idx]
            typed_element = _guess_element(parts[1], parts[5])
            if typed_element != row["sdf_element"]:
                raise ValueError(
                    f"Element mismatch while writing mapped MOL2 at SDF atom {row['sdf_index']}: "
                    f"typed={typed_element}, sdf={row['sdf_element']}"
                )
            original = {"x": float(parts[2]), "y": float(parts[3]), "z": float(parts[4])}
            use_complex_coord = row.get("complex_index") is not None
            if use_complex_coord:
                coord_atom = complex_atoms[int(row["complex_index"]) - 1]
                if coord_atom["element"] != row["sdf_element"]:
                    raise ValueError(
                        f"Element mismatch while writing mapped MOL2 at SDF atom {row['sdf_index']}: "
                        f"sdf={row['sdf_element']}, complex={coord_atom['element']}"
                    )
                if coord_atom["name"] != row["complex_atom_name"]:
                    raise ValueError(
                        f"Complex atom name changed during coordinate transfer at SDF atom {row['sdf_index']}: "
                        f"mapping={row['complex_atom_name']}, complex={coord_atom['name']}"
                    )
                atom_name = coord_atom["name"]
                x, y, z = coord_atom["x"], coord_atom["y"], coord_atom["z"]
                max_delta = max(max_delta, _distance(original, coord_atom))
            else:
                atom_name = parts[1]
                if row["sdf_element"] == "H":
                    x, y, z = _local_hydrogen_coord(atom_idx)
                else:
                    x, y, z = transform_coord(float(parts[2]), float(parts[3]), float(parts[4]))
            output.append(
                f"{int(parts[0]):>7d} {atom_name:<6s}{x:>10.4f}{y:>10.4f}{z:>10.4f} "
                f"{parts[5]:<8s}{parts[6]:>3s} {resname:<6s}{float(parts[8]):>10.6f}"
            )
            atom_idx += 1
        else:
            output.append(line)
    if atom_idx != len(mapping_rows):
        raise ValueError(f"Wrote {atom_idx} mapped MOL2 atoms but expected {len(mapping_rows)}")
    output_mol2.write_text("\n".join(output) + "\n", encoding="utf-8")
    written_atoms = read_ligand_atoms(output_mol2)
    written_vs_complex_distances: list[float] = []
    heavy_written_vs_complex_distances: list[float] = []
    mismatches: list[dict[str, Any]] = []
    for idx, row in enumerate(mapping_rows):
        if row.get("complex_index") is None or row["sdf_element"] == "H":
            continue
        written = written_atoms[idx]
        coord_atom = complex_atoms[int(row["complex_index"]) - 1]
        dist = _distance(written, coord_atom)
        written_vs_complex_distances.append(dist)
        if written["element"] != "H":
            heavy_written_vs_complex_distances.append(dist)
        if written["name"] != coord_atom["name"] or written["element"] != coord_atom["element"] or dist > 1.0e-3:
            mismatches.append(
                {
                    "sdf_index": row["sdf_index"],
                    "written_name": written["name"],
                    "complex_name": coord_atom["name"],
                    "written_element": written["element"],
                    "complex_element": coord_atom["element"],
                    "coordinate_delta_a": round(dist, 8),
                }
            )
    if mismatches:
        raise ValueError(f"Mapped MOL2 does not preserve complex PDB atom names/elements/coordinates: {mismatches[:10]}")
    heavy_rmsd = _rmsd(heavy_written_vs_complex_distances)
    if heavy_rmsd > MAPPED_POSE_HEAVY_RMSD_TOLERANCE_A:
        raise ValueError(
            f"Mapped MOL2 heavy-atom RMSD {heavy_rmsd:.6f} A exceeds "
            f"{MAPPED_POSE_HEAVY_RMSD_TOLERANCE_A:.3f} A tolerance"
        )
    return {
        "status": "success",
        "output_mol2": str(output_mol2),
        "atom_count": atom_idx,
        "max_template_to_complex_coordinate_delta_a": round(max_delta, 6),
        "written_vs_complex_heavy_atom_rmsd_a": round(heavy_rmsd, 8),
        "written_vs_complex_heavy_atom_rmsd_tolerance_a": MAPPED_POSE_HEAVY_RMSD_TOLERANCE_A,
        "written_vs_complex_heavy_atom_rmsd_check": "passed",
        "written_vs_complex_max_delta_a": round(max(written_vs_complex_distances) if written_vs_complex_distances else 0.0, 8),
        "atom_name_policy": "final MOL2 atom names are copied from the mapped complex PDB ligand atoms",
        "coordinate_source": str(complex_ligand_pdb),
        "chemistry_source": str(typed_sdf_mol2),
        "policy": (
            "Heavy-atom coordinates are overwritten from the confirmed complex pose. Hydrogen coordinates are "
            "locally rebuilt from the SDF/GAFF2 template using each hydrogen parent and nearby heavy-atom geometry "
            "so malformed complex-PDB hydrogens do not propagate into RESP. Bond graph, bond order, and GAFF2 atom "
            "types remain from the SDF/AmberTools chemistry source."
        ),
    }


def _run_pre_resp_qm_relaxation(
    *,
    input_mol2: Path,
    output_dir: Path,
    resname: str,
    formal_charge: int,
    spin_multiplicity: int,
    mode: str,
    use_gpu: bool,
    require_gpu: bool,
) -> tuple[Path, dict[str, Any]]:
    if mode not in {"none", "pbe-h-only"}:
        raise ValueError(f"Unsupported pre-RESP relaxation mode: {mode}")
    if mode == "none":
        return input_mol2, {
            "mode": mode,
            "status": "skipped",
            "input_mol2": str(input_mol2),
            "output_mol2": str(input_mol2),
            "policy": "No PBE pre-RESP coordinate relaxation was applied.",
        }

    script = Path(__file__).resolve().parents[2] / "scripts" / "qm_restrained_ligand_opt.py"
    if not script.is_file():
        raise FileNotFoundError(f"Pre-RESP relaxation script not found: {script}")
    relax_dir = output_dir / "pre_resp_qm_relaxation"
    relax_dir.mkdir(parents=True, exist_ok=True)
    output_mol2 = relax_dir / f"{resname}_h_only.mol2"
    manifest_path = relax_dir / f"{resname}_h_only_manifest.json"
    maxiter = 6

    if os.name == "nt":
        cmd = [
            "wsl",
            "bash",
            "-lc",
            " ".join(
                [
                    "python3",
                    shlex.quote(_win_to_wsl(script)),
                    "--input-mol2",
                    shlex.quote(_win_to_wsl(input_mol2)),
                    "--output-mol2",
                    shlex.quote(_win_to_wsl(output_mol2)),
                    "--manifest",
                    shlex.quote(_win_to_wsl(manifest_path)),
                    "--charge",
                    str(int(formal_charge)),
                    "--spin",
                    str(int(spin_multiplicity)),
                    "--basis",
                    shlex.quote("6-31g*"),
                    "--maxiter",
                    str(maxiter),
                    "--max-step",
                    "0.08",
                    "--grad-rms-target",
                    "3e-3",
                    *(["--cpu-only"] if not use_gpu else []),
                ]
            ),
        ]
    else:
        cmd = [
            sys.executable,
            str(script),
            "--input-mol2",
            str(input_mol2),
            "--output-mol2",
            str(output_mol2),
            "--manifest",
            str(manifest_path),
            "--charge",
            str(int(formal_charge)),
            "--spin",
            str(int(spin_multiplicity)),
            "--basis",
            "6-31g*",
            "--maxiter",
            str(maxiter),
            "--max-step",
            "0.08",
            "--grad-rms-target",
            "3e-3",
        ]
        if not use_gpu:
            cmd.append("--cpu-only")

    run = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=7200)
    (relax_dir / "h_only_relaxation.stdout.txt").write_text(run.stdout, encoding="utf-8")
    (relax_dir / "h_only_relaxation.stderr.txt").write_text(run.stderr, encoding="utf-8")
    if run.returncode != 0 or not output_mol2.is_file() or not manifest_path.is_file():
        return input_mol2, {
            "mode": mode,
            "status": "warning",
            "gate": "fallback_to_input_pose",
            "input_mol2": str(input_mol2),
            "output_mol2": str(input_mol2),
            "stdout": str(relax_dir / "h_only_relaxation.stdout.txt"),
            "stderr": str(relax_dir / "h_only_relaxation.stderr.txt"),
            "returncode": run.returncode,
            "policy": (
                "H-only PBE pre-RESP relaxation failed or did not write outputs; "
                "continuing with the confirmed complex pose and current hydrogens for RESP."
            ),
            "error_tail": (run.stderr or run.stdout)[-2000:],
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["mode"] = mode
    manifest["stdout"] = str(relax_dir / "h_only_relaxation.stdout.txt")
    manifest["stderr"] = str(relax_dir / "h_only_relaxation.stderr.txt")
    return output_mol2, manifest


def second_core_source_policy() -> dict[str, str]:
    return {
        "bond_graph_source": "ligand SDF/template; not inferred from complex PDB coordinates",
        "bond_order_source": "ligand SDF/template; not inferred from complex PDB coordinates",
        "gaff2_atom_type_source": "AmberTools antechamber using ligand SDF/template chemistry",
        "initial_coordinate_source": "confirmed protein-heme-ligand complex PDB",
        "qm_geometry_source": "confirmed protein-heme-ligand complex PDB ligand coordinates",
        "resp_charge_source": "GPU4PySCF/PySCF HF/6-31G* Molden followed by Multiwfn two-stage RESP",
        "force_field_parameter_source": "GAFF2 plus parmchk2 from SDF/template atom types and connectivity before RESP charge injection; not calculated from PDB bond lengths or angles",
        "pdb_geometry_use": "PDB bond lengths/angles define the starting geometry only, not equilibrium force constants or bonded parameters",
        "atom_mapping_policy": "SDF/template atoms are graph-matched to complex PDB ligand atoms; row-order coordinate transfer is forbidden",
    }


def _assert_atom_identity_and_coordinates(
    *,
    reference_pose: Path,
    candidate_mol2: Path,
    coordinate_tolerance: float = 1.0e-3,
) -> dict[str, Any]:
    reference_atoms = read_ligand_atoms(reference_pose)
    candidate_atoms = read_ligand_atoms(candidate_mol2)
    if len(reference_atoms) != len(candidate_atoms):
        raise ValueError(f"Typed MOL2 atom count {len(candidate_atoms)} does not match extracted ligand atom count {len(reference_atoms)}")
    mismatches: list[dict[str, Any]] = []
    distances: list[float] = []
    for idx, (ref, cand) in enumerate(zip(reference_atoms, candidate_atoms), start=1):
        dist = _distance(ref, cand)
        distances.append(dist)
        if ref["name"] != cand["name"] or ref["element"] != cand["element"] or dist > coordinate_tolerance:
            mismatches.append(
                {
                    "idx": idx,
                    "reference_name": ref["name"],
                    "candidate_name": cand["name"],
                    "reference_element": ref["element"],
                    "candidate_element": cand["element"],
                    "coordinate_distance_a": round(dist, 6),
                }
            )
    if mismatches:
        raise ValueError(f"Typed MOL2 does not preserve extracted ligand atom identity/coordinates: {mismatches[:10]}")
    return {
        "status": "success",
        "atom_count": len(reference_atoms),
        "max_coordinate_delta_a": round(max(distances) if distances else 0.0, 8),
        "coordinate_tolerance_a": coordinate_tolerance,
        "policy": "atom names, elements, order, and coordinates must match before RESP charge injection",
    }


def run_gpu4pyscf_multiwfn_resp_parameterization(
    *,
    ligand_pose: str | Path,
    formal_charge: int,
    output_dir: str | Path,
    resname: str = "LIG",
    spin_multiplicity: int = 1,
    basis: str = "6-31g*",
    use_gpu: bool = True,
    require_gpu: bool = False,
    multiwfn_bin: str | None = None,
    amber_sh: str | None = None,
    run_parmchk2: bool = True,
    resp_geometry_cleanup: str = "h-only",
) -> dict[str, Any]:
    src = Path(ligand_pose)
    if src.suffix.lower() != ".mol2":
        raise ValueError("Multiwfn RESP parameterization requires MOL2 input so charges can be injected safely.")
    resolved_multiwfn_bin = resolve_multiwfn_bin(multiwfn_bin)
    resolved_amber_sh = resolve_amber_sh(amber_sh)
    out = Path(output_dir)
    atoms = read_ligand_atoms(src)
    manifest = prepare_gpu4pyscf_molden_job(
        ligand_pose=ligand_pose,
        formal_charge=formal_charge,
        output_dir=out,
        resname=resname,
        spin_multiplicity=spin_multiplicity,
        basis=basis,
        resp_geometry_cleanup=resp_geometry_cleanup,
        use_gpu=use_gpu,
        require_gpu=require_gpu,
    )
    wsl_out = _win_to_wsl(out)
    python_exe = shlex.quote(sys.executable)
    molden_run = subprocess.run(
        ["wsl", "bash", "-lc", f"cd {shlex.quote(wsl_out)} && python3 run_gpu4pyscf_molden.py"]
        if os.name == "nt"
        else ["bash", "-lc", f"{python_exe} run_gpu4pyscf_molden.py"],
        cwd=None if os.name == "nt" else str(out),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=7200,
    )
    (out / "gpu4pyscf_molden.stdout.txt").write_text(molden_run.stdout, encoding="utf-8")
    (out / "gpu4pyscf_molden.stderr.txt").write_text(molden_run.stderr, encoding="utf-8")
    molden_path = out / f"{resname}_hf631gstar.molden"
    if molden_run.returncode != 0 or not molden_path.is_file():
        raise RuntimeError(f"GPU4PySCF/PySCF Molden generation failed: {molden_run.stderr or molden_run.stdout}")

    multiwfn_cmd = f"printf '7\\n18\\n1\\ny\\n0\\n0\\nq\\n' | {shlex.quote(resolved_multiwfn_bin)} {shlex.quote(molden_path.name)}"
    multiwfn = subprocess.run(
        ["wsl", "bash", "-lc", f"export OMP_STACKSIZE=4G && cd {shlex.quote(wsl_out)} && {multiwfn_cmd}"]
        if os.name == "nt"
        else ["bash", "-lc", multiwfn_cmd],
        cwd=None if os.name == "nt" else str(out),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=7200,
    )
    (out / "multiwfn_resp.stdout.txt").write_text(multiwfn.stdout, encoding="utf-8")
    (out / "multiwfn_resp.stderr.txt").write_text(multiwfn.stderr, encoding="utf-8")
    chg_file = out / f"{resname}_hf631gstar.chg"
    if not chg_file.is_file():
        chg_files = sorted(out.glob("*.chg"))
        if chg_files:
            chg_file = chg_files[0]
    if multiwfn.returncode != 0 or not chg_file.is_file():
        raise RuntimeError(f"Multiwfn RESP fitting failed or produced no .chg: {multiwfn.stderr or multiwfn.stdout}")

    charge_csv = out / "multiwfn_resp_charges.csv"
    charge_sum = _write_chg_charge_csv(atoms, chg_file, charge_csv)
    if abs(charge_sum - formal_charge) > RESP_CHARGE_SUM_TOLERANCE_E:
        raise RuntimeError(f"Multiwfn RESP charge sum {charge_sum:.8f} does not match formal charge {formal_charge}")
    charged_mol2 = out / f"{resname}_multiwfn_resp.mol2"
    injected = _inject_chg_charges_to_mol2(src, chg_file, charged_mol2, resname)
    final: dict[str, Any] = {
        **manifest,
        "status": "success",
        "charge_model": "RESP(HF/6-31G*) via Multiwfn",
        "molden": str(molden_path),
        "multiwfn_chg": str(chg_file),
        "resp_charges_csv": str(charge_csv),
        "partial_charge_sum": round(charge_sum, 8),
        "formal_charge": formal_charge,
        "multiwfn_bin": resolved_multiwfn_bin,
        "amber_sh": resolved_amber_sh,
        "charge_sum_tolerance_e": RESP_CHARGE_SUM_TOLERANCE_E,
        "charge_sum_check": "passed",
        "charged_mol2": str(charged_mol2),
        "injected_atom_count": injected,
        "multiwfn_command": "printf '7\\n18\\n1\\ny\\n0\\n0\\nq\\n' | Multiwfn_noGUI <molden>",
        "limitation": "RESP charges are fitted for the supplied hydrogen-complete ligand pose; this does not validate binding correctness.",
    }
    if run_parmchk2:
        parm = _run_parmchk2(out, charged_mol2.name, f"{resname}.frcmod", resolved_amber_sh)
        final["parmchk2"] = parm
        if parm["returncode"] != 0 or not (out / f"{resname}.frcmod").is_file():
            raise RuntimeError(f"parmchk2 failed after Multiwfn RESP charge injection: {parm['stderr'] or parm['stdout']}")
        final["frcmod"] = str(out / f"{resname}.frcmod")
    (out / "multiwfn_resp_parameterization_manifest.json").write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return final


def run_complex_ligand_multiwfn_resp_parameterization(
    *,
    complex_pdb: str | Path,
    ligand_resname: str,
    ligand_chain: str,
    formal_charge: int,
    output_dir: str | Path,
    spin_multiplicity: int = 1,
    basis: str = "6-31g*",
    use_gpu: bool = True,
    require_gpu: bool = False,
    multiwfn_bin: str | None = None,
    amber_sh: str | None = None,
    run_parmchk2: bool = True,
    resp_geometry_cleanup: str = "none",
    pre_resp_relax: str = "pbe-h-only",
) -> dict[str, Any]:
    """Second core: confirmed complex PDB -> extracted ligand -> GPU4PySCF + Multiwfn RESP parameters."""
    resolved_multiwfn_bin = resolve_multiwfn_bin(multiwfn_bin)
    resolved_amber_sh = resolve_amber_sh(amber_sh)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    extracted_pdb = out / f"{ligand_resname}_from_confirmed_complex.pdb"
    extract_ligand_from_complex_pdb(
        complex_pdb=complex_pdb,
        ligand_resname=ligand_resname,
        ligand_chain=ligand_chain,
        output_pdb=extracted_pdb,
    )
    extracted_atoms = read_ligand_atoms(extracted_pdb)
    if sum(1 for atom in extracted_atoms if atom["element"] == "H") == 0:
        raise ValueError("Extracted ligand has no hydrogens; confirmed complex is not valid for RESP parameterization.")

    typed_mol2 = out / f"{ligand_resname}_gaff2_from_complex.mol2"
    if not _pdb_has_ligand_conect(extracted_pdb):
        raise ValueError(
            "Extracted ligand PDB has no ligand CONECT records. A complex PDB with coordinates alone is insufficient "
            "for reliable GAFF2 atom typing; provide a confirmed complex PDB with ligand CONECT records or a typed ligand MOL2."
        )
    antechamber = _run_antechamber_pdb_to_mol2(
        ligand_pdb=extracted_pdb,
        output_dir=out,
        output_name=typed_mol2.name,
        resname=ligand_resname,
        formal_charge=formal_charge,
        amber_sh=resolved_amber_sh,
    )
    (out / "antechamber.stdout.txt").write_text(antechamber["stdout"], encoding="utf-8")
    (out / "antechamber.stderr.txt").write_text(antechamber["stderr"], encoding="utf-8")
    if antechamber["returncode"] != 0 or not typed_mol2.is_file():
        raise RuntimeError(f"antechamber failed to type extracted ligand: {antechamber['stderr'] or antechamber['stdout']}")
    identity = _assert_atom_identity_and_coordinates(reference_pose=extracted_pdb, candidate_mol2=typed_mol2)

    result = run_gpu4pyscf_multiwfn_resp_parameterization(
        ligand_pose=typed_mol2,
        formal_charge=formal_charge,
        output_dir=out,
        resname=ligand_resname,
        spin_multiplicity=spin_multiplicity,
        basis=basis,
        use_gpu=use_gpu,
        require_gpu=require_gpu,
        multiwfn_bin=resolved_multiwfn_bin,
        amber_sh=resolved_amber_sh,
        run_parmchk2=run_parmchk2,
    )
    final = {
        **result,
        "schema": "cypforge.complex_ligand_multiwfn_resp_parameterization.v1",
        "input_complex_pdb": str(complex_pdb),
        "ligand_resname": ligand_resname,
        "ligand_chain": ligand_chain,
        "extracted_ligand_pdb": str(extracted_pdb),
        "typed_ligand_mol2_before_resp": str(typed_mol2),
        "antechamber": antechamber,
        "multiwfn_bin": resolved_multiwfn_bin,
        "amber_sh": resolved_amber_sh,
        "atom_identity_coordinate_check": identity,
        "core_definition": "confirmed protein-heme-ligand complex -> ligand extraction -> GPU4PySCF/PySCF Molden -> Multiwfn RESP -> MOL2/frcmod",
    }
    (out / "complex_ligand_multiwfn_resp_manifest.json").write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return final


def run_complex_sdf_ligand_multiwfn_resp_parameterization(
    *,
    complex_pdb: str | Path,
    ligand_resname: str,
    ligand_chain: str,
    ligand_template_sdf: str | Path,
    formal_charge: int,
    output_dir: str | Path,
    spin_multiplicity: int = 1,
    basis: str = "6-31g*",
    use_gpu: bool = True,
    require_gpu: bool = False,
    multiwfn_bin: str | None = None,
    amber_sh: str | None = None,
    run_parmchk2: bool = True,
    resp_geometry_cleanup: str = "none",
    pre_resp_relax: str = "pbe-h-only",
) -> dict[str, Any]:
    """Second core with SDF chemistry source: complex coordinates + SDF bond orders -> RESP MOL2/frcmod."""
    resolved_multiwfn_bin = resolve_multiwfn_bin(multiwfn_bin)
    resolved_amber_sh = resolve_amber_sh(amber_sh)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sdf = Path(ligand_template_sdf)
    extracted_pdb = out / f"{ligand_resname}_from_confirmed_complex.pdb"
    extract_ligand_from_complex_pdb(
        complex_pdb=complex_pdb,
        ligand_resname=ligand_resname,
        ligand_chain=ligand_chain,
        output_pdb=extracted_pdb,
    )

    sdf_manifest = read_sdf_template(sdf)
    if int(sdf_manifest["formal_charge"]) != int(formal_charge):
        raise ValueError(
            f"SDF formal charge {sdf_manifest['formal_charge']} does not match user-provided formal charge {formal_charge}."
        )
    mapping = map_sdf_atoms_to_complex_atoms(sdf_template=sdf, ligand_pdb=extracted_pdb)
    mapping_path = out / "sdf_to_complex_atom_mapping.json"
    mapping_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    typed_sdf_mol2 = out / f"{ligand_resname}_gaff2_from_sdf.mol2"
    antechamber = _run_antechamber_sdf_to_mol2(
        ligand_sdf=sdf,
        output_dir=out,
        output_name=typed_sdf_mol2.name,
        resname=ligand_resname,
        formal_charge=formal_charge,
        amber_sh=resolved_amber_sh,
    )
    (out / "antechamber_sdf.stdout.txt").write_text(antechamber["stdout"], encoding="utf-8")
    (out / "antechamber_sdf.stderr.txt").write_text(antechamber["stderr"], encoding="utf-8")
    if antechamber["returncode"] != 0 or not typed_sdf_mol2.is_file():
        raise RuntimeError(f"antechamber failed to type ligand SDF: {antechamber['stderr'] or antechamber['stdout']}")

    parm: dict[str, Any] | None = None
    if run_parmchk2:
        parm = _run_parmchk2(out, typed_sdf_mol2.name, f"{ligand_resname}.frcmod", resolved_amber_sh)
        if parm["returncode"] != 0 or not (out / f"{ligand_resname}.frcmod").is_file():
            raise RuntimeError(f"parmchk2 failed on GAFF2 typed SDF MOL2 before RESP charge injection: {parm['stderr'] or parm['stdout']}")

    typed_pose_mol2 = out / f"{ligand_resname}_gaff2_sdf_chemistry_complex_pose.mol2"
    coordinate_transfer = _write_sdf_order_complex_pose_mol2(
        typed_sdf_mol2=typed_sdf_mol2,
        complex_ligand_pdb=extracted_pdb,
        mapping=mapping,
        output_mol2=typed_pose_mol2,
        resname=ligand_resname,
    )
    resp_input_mol2, pre_resp_relaxation = _run_pre_resp_qm_relaxation(
        input_mol2=typed_pose_mol2,
        output_dir=out,
        resname=ligand_resname,
        formal_charge=formal_charge,
        spin_multiplicity=spin_multiplicity,
        mode=pre_resp_relax,
        use_gpu=use_gpu,
        require_gpu=require_gpu,
    )

    result = run_gpu4pyscf_multiwfn_resp_parameterization(
        ligand_pose=resp_input_mol2,
        formal_charge=formal_charge,
        output_dir=out,
        resname=ligand_resname,
        spin_multiplicity=spin_multiplicity,
        basis=basis,
        use_gpu=use_gpu,
        require_gpu=require_gpu,
        multiwfn_bin=resolved_multiwfn_bin,
        amber_sh=resolved_amber_sh,
        run_parmchk2=False,
        resp_geometry_cleanup=resp_geometry_cleanup,
    )
    final = {
        **result,
        "schema": "cypforge.complex_sdf_ligand_multiwfn_resp_parameterization.v1",
        "source_policy": second_core_source_policy(),
        "input_complex_pdb": str(complex_pdb),
        "ligand_template_sdf": str(sdf),
        "ligand_resname": ligand_resname,
        "ligand_chain": ligand_chain,
        "extracted_ligand_pdb": str(extracted_pdb),
        "sdf_to_complex_atom_mapping": str(mapping_path),
        "sdf_formal_charge": sdf_manifest["formal_charge"],
        "sdf_formal_charge_source": sdf_manifest["formal_charge_source"],
        "sdf_aromaticity_status": "explicit_aromatic_bonds_present" if sdf_manifest["aromatic_bond_count"] else "not_explicit_or_kekulized",
        "typed_sdf_mol2_before_coordinate_transfer": str(typed_sdf_mol2),
        "typed_ligand_mol2_before_resp": str(typed_pose_mol2),
        "pre_resp_relaxation": pre_resp_relaxation,
        "resp_input_mol2": str(resp_input_mol2),
        "antechamber_sdf": antechamber,
        "multiwfn_bin": resolved_multiwfn_bin,
        "amber_sh": resolved_amber_sh,
        "parmchk2": parm,
        "frcmod": str(out / f"{ligand_resname}.frcmod") if parm else None,
        "frcmod_source_mol2": str(typed_sdf_mol2) if parm else None,
        "frcmod_policy": "parmchk2 is run on the GAFF2 typed SDF/template MOL2 before coordinate overwrite and before RESP charge injection; RESP charges do not define bonded/vdW parameters.",
        "coordinate_transfer": coordinate_transfer,
        "core_definition": "confirmed complex supplies exact coordinates; SDF supplies bond graph/bond order/GAFF2 chemistry; GPU4PySCF+Multiwfn RESP supplies charges",
    }
    (out / "complex_sdf_ligand_multiwfn_resp_manifest.json").write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return final


def run_gpu4pyscf_esp_parameterization(
    *,
    ligand_pose: str | Path,
    formal_charge: int,
    output_dir: str | Path,
    resname: str = "LIG",
    spin_multiplicity: int = 1,
    basis: str = "6-31g*",
    points_per_atom: int = 24,
    use_gpu: bool = True,
    require_gpu: bool = False,
    amber_sh: str | None = None,
    run_parmchk2: bool = True,
) -> dict[str, Any]:
    out = Path(output_dir)
    resolved_amber_sh = resolve_amber_sh(amber_sh)
    manifest = prepare_gpu4pyscf_esp_job(
        ligand_pose=ligand_pose,
        formal_charge=formal_charge,
        output_dir=out,
        resname=resname,
        spin_multiplicity=spin_multiplicity,
        basis=basis,
        points_per_atom=points_per_atom,
        use_gpu=use_gpu,
        require_gpu=require_gpu,
    )
    wsl_out = _win_to_wsl(out)
    python_exe = shlex.quote(sys.executable)
    result = subprocess.run(
        ["wsl", "bash", "-lc", f"cd {shlex.quote(wsl_out)} && python3 run_gpu4pyscf_esp.py"]
        if os.name == "nt"
        else ["bash", "-lc", f"{python_exe} run_gpu4pyscf_esp.py"],
        cwd=None if os.name == "nt" else str(out),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=7200,
    )
    (out / "gpu4pyscf_esp.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (out / "gpu4pyscf_esp.stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0 or not (out / "esp_charges.csv").is_file():
        raise RuntimeError(f"GPU4PySCF/PySCF ESP job failed: {result.stderr or result.stdout}")

    final: dict[str, Any] = {
        **manifest,
        "status": "success",
        "esp_charges_csv": str(out / "esp_charges.csv"),
        "esp_grid_csv": str(out / "esp_grid.csv"),
        "esp_fit_report": str(out / "esp_fit_report.json"),
        "amber_sh": resolved_amber_sh,
    }
    src = Path(ligand_pose)
    if src.suffix.lower() == ".mol2":
        charged_mol2 = out / f"{resname}_gpu4pyscf_esp.mol2"
        injected = _inject_csv_charges_to_mol2(src, out / "esp_charges.csv", charged_mol2, resname)
        final["charged_mol2"] = str(charged_mol2)
        final["injected_atom_count"] = injected
        if run_parmchk2:
            parm = _run_parmchk2(out, charged_mol2.name, f"{resname}.frcmod", resolved_amber_sh)
            final["parmchk2"] = parm
            if parm["returncode"] != 0 or not (out / f"{resname}.frcmod").is_file():
                raise RuntimeError(f"parmchk2 failed after ESP charge injection: {parm['stderr'] or parm['stdout']}")
            final["frcmod"] = str(out / f"{resname}.frcmod")
    else:
        final["charged_mol2"] = None
        final["limitation"] = "Input was not MOL2, so only ESP charge CSV/grid were generated; provide MOL2 for charge injection and frcmod."
    (out / "gpu4pyscf_esp_parameterization_manifest.json").write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return final

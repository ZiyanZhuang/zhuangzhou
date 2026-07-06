from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .io import kabsch_transform
from .ligand_gpu4pyscf_esp import (
    _bond_edges_from_sdf,
    _degree_map,
    _distance,
    _infer_bond_edges_from_coords,
    read_sdf_template,
)

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


def _guess_element_from_name(name: str, raw: str = "") -> str:
    letters = "".join(ch for ch in name if ch.isalpha()).upper()
    if letters[:2] in {"CL", "BR", "FE"}:
        return letters[:2]
    value = raw.strip().upper()
    if value in COVALENT_RADII or value == "FE":
        return value
    return letters[:1] or "X"


def _read_ligand_atoms_allow_duplicates(path: str | Path) -> list[dict[str, Any]]:
    src = Path(path)
    atoms: list[dict[str, Any]] = []
    for line in src.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        name = line[12:16].strip()
        atoms.append({
            "index": len(atoms) + 1,
            "name": name,
            "x": float(line[30:38]),
            "y": float(line[38:46]),
            "z": float(line[46:54]),
            "element": _guess_element_from_name(name, line[76:78] if len(line) >= 78 else ""),
        })
    if not atoms:
        raise ValueError(f"No ligand atoms parsed from {src}")
    return atoms


def _coord(atom: dict[str, Any]) -> np.ndarray:
    return np.array([float(atom["x"]), float(atom["y"]), float(atom["z"])], dtype=float)


def _pair_distance_matrix(atoms: list[dict[str, Any]]) -> np.ndarray:
    coords = np.array([_coord(atom) for atom in atoms], dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _kabsch_rmsd_for_mapping(
    template_atoms: list[dict[str, Any]],
    complex_atoms: list[dict[str, Any]],
    mapping: dict[int, int],
) -> float:
    template_points = []
    complex_points = []
    for t_idx, c_idx in sorted(mapping.items()):
        if template_atoms[t_idx]["element"] == "H":
            continue
        template_points.append([template_atoms[t_idx][axis] for axis in ("x", "y", "z")])
        complex_points.append([complex_atoms[c_idx][axis] for axis in ("x", "y", "z")])
    if len(template_points) < 3:
        distances = [
            _distance(template_atoms[t_idx], complex_atoms[c_idx])
            for t_idx, c_idx in mapping.items()
            if template_atoms[t_idx]["element"] != "H"
        ]
        return math.sqrt(sum(d * d for d in distances) / len(distances)) if distances else 0.0
    source = np.array(template_points, dtype=float)
    target = np.array(complex_points, dtype=float)
    rotation, source_centroid, target_centroid = kabsch_transform(source, target)
    fitted = (source - source_centroid) @ rotation.T + target_centroid
    diff = fitted - target
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _distance_matrix_error_for_mapping(
    template_dist: np.ndarray,
    complex_dist: np.ndarray,
    mapping: dict[int, int],
) -> dict[str, float]:
    deltas = []
    items = sorted(mapping.items())
    for pos, (ta, ca) in enumerate(items):
        for tb, cb in items[pos + 1 :]:
            deltas.append(abs(float(template_dist[ta, tb]) - float(complex_dist[ca, cb])))
    if not deltas:
        return {"rmsd": 0.0, "max": 0.0}
    return {
        "rmsd": math.sqrt(sum(d * d for d in deltas) / len(deltas)),
        "max": max(deltas),
    }


def _expected_bond_length(e1: str, e2: str, order: int) -> float:
    base = COVALENT_RADII.get(e1, 0.76) + COVALENT_RADII.get(e2, 0.76)
    if order in {2, 4}:
        return base * 0.90
    if order == 3:
        return base * 0.82
    return base


def _mapped_bond_error(
    sdf: dict[str, Any],
    complex_atoms: list[dict[str, Any]],
    matched_template_indices: list[int],
    mapping: dict[int, int],
) -> dict[str, Any]:
    template_to_match = {template_idx: match_idx for match_idx, template_idx in enumerate(matched_template_indices)}
    deltas = []
    worst = None
    for bond in sdf["bonds"]:
        a_idx = int(bond["a"]) - 1
        b_idx = int(bond["b"]) - 1
        if a_idx not in template_to_match or b_idx not in template_to_match:
            continue
        ca = complex_atoms[mapping[template_to_match[a_idx]]]
        cb = complex_atoms[mapping[template_to_match[b_idx]]]
        actual = _distance(ca, cb)
        expected = _expected_bond_length(ca["element"], cb["element"], int(bond["order"]))
        delta = abs(actual - expected)
        deltas.append(delta)
        row = {
            "sdf_a": a_idx + 1,
            "sdf_b": b_idx + 1,
            "complex_a": ca["index"],
            "complex_b": cb["index"],
            "complex_a_name": ca["name"],
            "complex_b_name": cb["name"],
            "distance_a": round(actual, 6),
            "expected_a": round(expected, 6),
            "abs_delta_a": round(delta, 6),
        }
        if worst is None or delta > worst["abs_delta_a"]:
            worst = row
    return {
        "rmsd": math.sqrt(sum(d * d for d in deltas) / len(deltas)) if deltas else 0.0,
        "max": max(deltas) if deltas else 0.0,
        "worst": worst,
    }


def _build_matched_graph(
    sdf: dict[str, Any],
    complex_atoms: list[dict[str, Any]],
) -> tuple[list[int], list[dict[str, Any]], set[tuple[int, int]], set[tuple[int, int]], bool]:
    template_atoms = sdf["atoms"]
    template_h_count = sum(1 for atom in template_atoms if atom["element"] == "H")
    complex_h_count = sum(1 for atom in complex_atoms if atom["element"] == "H")
    template_heavy_count = len([atom for atom in template_atoms if atom["element"] != "H"])
    complex_heavy_count = len([atom for atom in complex_atoms if atom["element"] != "H"])
    heavy_only_complex = (
        complex_h_count < template_h_count
        and template_h_count > 0
        and template_heavy_count == complex_heavy_count
    )
    if len(template_atoms) != len(complex_atoms) and not heavy_only_complex:
        raise ValueError(
            f"SDF atom count {len(template_atoms)} does not match complex ligand atom count {len(complex_atoms)}"
        )
    matched_template_indices = [
        idx for idx, atom in enumerate(template_atoms) if not heavy_only_complex or atom["element"] != "H"
    ]
    matched_template_atoms = [template_atoms[idx] for idx in matched_template_indices]
    if heavy_only_complex:
        complex_atoms[:] = [atom for atom in complex_atoms if atom["element"] != "H"]
    if sorted(atom["element"] for atom in matched_template_atoms) != sorted(atom["element"] for atom in complex_atoms):
        raise ValueError("SDF/complex element counts differ.")
    idx_to_match = {template_idx: match_idx for match_idx, template_idx in enumerate(matched_template_indices)}
    all_edges = _bond_edges_from_sdf(sdf)
    template_edges = {
        (min(idx_to_match[a], idx_to_match[b]), max(idx_to_match[a], idx_to_match[b]))
        for a, b in all_edges
        if a in idx_to_match and b in idx_to_match
    }
    complex_edges = _infer_bond_edges_from_coords(complex_atoms)
    return matched_template_indices, matched_template_atoms, template_edges, complex_edges, heavy_only_complex


def _enumerate_mappings(
    *,
    template_atoms: list[dict[str, Any]],
    complex_atoms: list[dict[str, Any]],
    template_edges: set[tuple[int, int]],
    complex_edges: set[tuple[int, int]],
    max_mappings: int,
    distance_prune_a: float | None,
) -> tuple[list[dict[int, int]], bool]:
    t_deg = _degree_map(len(template_atoms), template_edges)
    c_deg = _degree_map(len(complex_atoms), complex_edges)
    t_neighbors = {i: set() for i in range(len(template_atoms))}
    c_neighbors = {i: set() for i in range(len(complex_atoms))}
    for a, b in template_edges:
        t_neighbors[a].add(b)
        t_neighbors[b].add(a)
    for a, b in complex_edges:
        c_neighbors[a].add(b)
        c_neighbors[b].add(a)
    template_dist = _pair_distance_matrix(template_atoms)
    complex_dist = _pair_distance_matrix(complex_atoms)

    candidates = {
        i: [
            j
            for j, atom in enumerate(complex_atoms)
            if atom["element"] == template_atoms[i]["element"] and c_deg[j] == t_deg[i]
        ]
        for i in range(len(template_atoms))
    }
    empty = [i + 1 for i, values in candidates.items() if not values]
    if empty:
        raise ValueError(f"No element/degree candidates for SDF atoms: {empty}")

    order = sorted(range(len(template_atoms)), key=lambda i: (len(candidates[i]), -t_deg[i], template_atoms[i]["element"], i))
    mappings: list[dict[int, int]] = []
    truncated = False

    def backtrack(pos: int, current: dict[int, int], used: set[int]) -> None:
        nonlocal truncated
        if len(mappings) >= max_mappings:
            truncated = True
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
                if distance_prune_a is not None and abs(float(template_dist[t_idx, mapped_t]) - float(complex_dist[c_idx, mapped_c])) > distance_prune_a:
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
    return mappings, truncated


def resolve_sdf_to_complex_mapping(
    *,
    sdf_template: str | Path,
    ligand_pdb: str | Path,
    output_json: str | Path | None = None,
    max_mappings: int = 50000,
    distance_prune_a: float | None = None,
    rmsd_gap_unique_a: float = 0.25,
    equivalent_rmsd_tol_a: float = 0.02,
) -> dict[str, Any]:
    """Resolve SDF-to-complex ligand atom identity with graph and geometry gates.

    Decision states:
    - unique: one clearly best mapping.
    - equivalent_ok: multiple mappings are tied within tolerance. Treat as
      chemically equivalent only after charge/equivalence review downstream.
    - ambiguous_fail: multiple non-separated mappings remain.
    """
    sdf = read_sdf_template(sdf_template)
    complex_atoms = _read_ligand_atoms_allow_duplicates(ligand_pdb)
    matched_template_indices, matched_atoms, template_edges, complex_edges, heavy_only = _build_matched_graph(sdf, complex_atoms)
    mappings, truncated = _enumerate_mappings(
        template_atoms=matched_atoms,
        complex_atoms=complex_atoms,
        template_edges=template_edges,
        complex_edges=complex_edges,
        max_mappings=max_mappings,
        distance_prune_a=distance_prune_a,
    )
    if not mappings:
        raise ValueError("No graph- and distance-consistent mappings found.")
    template_dist = _pair_distance_matrix(matched_atoms)
    complex_dist = _pair_distance_matrix(complex_atoms)
    scored = []
    for mapping in mappings:
        kabsch = _kabsch_rmsd_for_mapping(matched_atoms, complex_atoms, mapping)
        dist_err = _distance_matrix_error_for_mapping(template_dist, complex_dist, mapping)
        bond_err = _mapped_bond_error(sdf, complex_atoms, matched_template_indices, mapping)
        # Bond geometry is conformer-independent and should dominate for flexible ligands.
        # Kabsch and distance-matrix terms are tie-breakers only.
        score = bond_err["rmsd"] + 0.05 * kabsch + 0.02 * dist_err["rmsd"]
        scored.append({
            "mapping": mapping,
            "kabsch_heavy_rmsd_a": round(kabsch, 6),
            "distance_matrix_rmsd_a": round(dist_err["rmsd"], 6),
            "distance_matrix_max_delta_a": round(dist_err["max"], 6),
            "bond_length_rmsd_a": round(bond_err["rmsd"], 6),
            "bond_length_max_delta_a": round(bond_err["max"], 6),
            "worst_bond": bond_err["worst"],
            "score": round(score, 6),
        })
    scored.sort(key=lambda row: (row["score"], row["kabsch_heavy_rmsd_a"], row["distance_matrix_rmsd_a"]))
    best = scored[0]
    second = scored[1] if len(scored) > 1 else None
    score_gap = round((second["score"] - best["score"]), 6) if second else None
    tied = [row for row in scored if abs(row["score"] - best["score"]) <= equivalent_rmsd_tol_a]
    if len(scored) == 1:
        decision = "unique"
    elif score_gap is not None and score_gap > rmsd_gap_unique_a:
        decision = "unique"
    elif len(tied) > 1:
        decision = "equivalent_ok"
    else:
        decision = "ambiguous_fail"

    chosen = best["mapping"]
    explicit_map: dict[str, int] = {}
    rows = []
    for match_idx, template_idx in enumerate(matched_template_indices):
        c_idx = chosen[match_idx]
        explicit_map[str(template_idx + 1)] = c_idx + 1
        rows.append({
            "sdf_index": template_idx + 1,
            "sdf_element": sdf["atoms"][template_idx]["element"],
            "complex_index": c_idx + 1,
            "complex_atom_name": complex_atoms[c_idx]["name"],
            "complex_element": complex_atoms[c_idx]["element"],
        })

    ambiguous_atoms = []
    for match_idx, template_idx in enumerate(matched_template_indices):
        values = sorted({row["mapping"][match_idx] for row in tied}) if len(tied) > 1 else []
        if len(values) > 1:
            ambiguous_atoms.append({
                "sdf_index": template_idx + 1,
                "element": sdf["atoms"][template_idx]["element"],
                "candidate_complex_indices": [v + 1 for v in values],
                "candidate_complex_atom_names": [complex_atoms[v]["name"] for v in values],
            })

    result = {
        "schema": "cypforge.ligand_mapping_resolver.v1",
        "status": "success",
        "decision": decision,
        "mapping_count_evaluated": len(scored),
        "mapping_count_truncated": truncated,
        "heavy_only_complex": heavy_only,
        "distance_prune_a": distance_prune_a,
        "rmsd_gap_unique_a": rmsd_gap_unique_a,
        "equivalent_rmsd_tol_a": equivalent_rmsd_tol_a,
        "best": {k: v for k, v in best.items() if k != "mapping"},
        "second_best": {k: v for k, v in second.items() if k != "mapping"} if second else None,
        "score_gap": score_gap,
        "ambiguous_atom_groups": ambiguous_atoms,
        "sdf_atom_id_to_pdb_atom_index": explicit_map,
        "mapping": rows,
        "duplicate_complex_atom_names": sorted({a["name"] for a in complex_atoms if [x["name"] for x in complex_atoms].count(a["name"]) > 1}),
        "policy": "Graph-consistent mappings are enumerated, then ranked mainly by mapped bond geometry with Kabsch/distance-matrix tie-breakers. Row-order transfer is forbidden.",
    }
    if output_json is not None:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_json).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result

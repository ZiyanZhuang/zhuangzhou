from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from .io import kabsch_transform
from .ligand_gpu4pyscf_esp import _bond_edges_from_sdf, _distance, _infer_bond_edges_from_coords, read_sdf_template
from .ligand_mapping_resolver import _read_ligand_atoms_allow_duplicates


def _coord(atom: dict[str, Any]) -> np.ndarray:
    return np.array((float(atom["x"]), float(atom["y"]), float(atom["z"])), dtype=float)


def _heavy_sdf_graph(sdf: dict[str, Any]) -> tuple[list[int], list[dict[str, Any]], set[tuple[int, int]]]:
    sdf_ids = [i for i, atom in enumerate(sdf["atoms"]) if atom["element"] != "H"]
    sdf_to_heavy = {sdf_i: heavy_i for heavy_i, sdf_i in enumerate(sdf_ids)}
    atoms = [sdf["atoms"][i] for i in sdf_ids]
    edges = {
        (min(sdf_to_heavy[a], sdf_to_heavy[b]), max(sdf_to_heavy[a], sdf_to_heavy[b]))
        for a, b in _bond_edges_from_sdf(sdf)
        if a in sdf_to_heavy and b in sdf_to_heavy
    }
    return sdf_ids, atoms, edges


def _heavy_pdb_atoms(path: str | Path) -> list[dict[str, Any]]:
    return [atom for atom in _read_ligand_atoms_allow_duplicates(path) if atom["element"] != "H"]


def _neighbors(n: int, edges: set[tuple[int, int]]) -> list[set[int]]:
    out = [set() for _ in range(n)]
    for a, b in edges:
        out[a].add(b)
        out[b].add(a)
    return out


def _graph_dist(neigh: list[set[int]]) -> np.ndarray:
    n = len(neigh)
    dist = np.full((n, n), np.inf, dtype=float)
    for start in range(n):
        dist[start, start] = 0.0
        q: deque[int] = deque([start])
        while q:
            i = q.popleft()
            for j in neigh[i]:
                if math.isinf(float(dist[start, j])):
                    dist[start, j] = dist[start, i] + 1.0
                    q.append(j)
    return dist


def _distance_matrix(atoms: list[dict[str, Any]]) -> np.ndarray:
    xyz = np.array([_coord(atom) for atom in atoms], dtype=float)
    diff = xyz[:, None, :] - xyz[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _local_signature(i: int, atoms: list[dict[str, Any]], neigh: list[set[int]], gdist: np.ndarray) -> tuple[Any, ...]:
    first_shell = tuple(sorted(atoms[j]["element"] for j in neigh[i]))
    second_shell = tuple(sorted(atoms[j]["element"] for j in range(len(atoms)) if int(gdist[i, j]) == 2))
    return atoms[i]["element"], len(neigh[i]), first_shell, second_shell


def _rms(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else 0.0


def _angle_rad(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    u = a - b
    v = c - b
    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
    if denom <= 1e-12:
        return 0.0
    return math.acos(max(-1.0, min(1.0, float(np.dot(u, v) / denom))))


def _angle_terms(atoms: list[dict[str, Any]], neigh: list[set[int]]) -> list[tuple[int, int, int, float]]:
    terms = []
    for center, values in enumerate(neigh):
        ordered = sorted(values)
        for pos, left in enumerate(ordered):
            for right in ordered[pos + 1 :]:
                terms.append((left, center, right, _angle_rad(_coord(atoms[left]), _coord(atoms[center]), _coord(atoms[right]))))
    return terms


def _score_mapping(
    *,
    mapping: dict[int, int],
    template_atoms: list[dict[str, Any]],
    complex_atoms: list[dict[str, Any]],
    template_edges: set[tuple[int, int]],
    template_angles: list[tuple[int, int, int, float]],
    template_dist: np.ndarray,
    complex_dist: np.ndarray,
    template_graph_dist: np.ndarray,
    tau: float,
) -> dict[str, float]:
    bond = _rms([
        abs(_distance(template_atoms[a], template_atoms[b]) - _distance(complex_atoms[mapping[a]], complex_atoms[mapping[b]]))
        for a, b in template_edges
    ])
    angle = _rms([
        abs(theta - _angle_rad(_coord(complex_atoms[mapping[a]]), _coord(complex_atoms[mapping[b]]), _coord(complex_atoms[mapping[c]])))
        for a, b, c, theta in template_angles
    ])
    numer = 0.0
    denom = 0.0
    items = sorted(mapping.items())
    for pos, (ta, ca) in enumerate(items):
        for tb, cb in items[pos + 1 :]:
            gd = float(template_graph_dist[ta, tb])
            if math.isinf(gd):
                continue
            weight = math.exp(-gd / tau)
            delta = float(template_dist[ta, tb] - complex_dist[ca, cb])
            numer += weight * delta * delta
            denom += weight
    weighted_distance = math.sqrt(numer / denom) if denom else 0.0

    source = np.array([_coord(template_atoms[i]) for i in sorted(mapping)], dtype=float)
    target = np.array([_coord(complex_atoms[mapping[i]]) for i in sorted(mapping)], dtype=float)
    if len(source) < 3:
        kabsch = _rms([float(np.linalg.norm(source[i] - target[i])) for i in range(len(source))])
    else:
        rotation, source_centroid, target_centroid = kabsch_transform(source, target)
        fitted = (source - source_centroid) @ rotation.T + target_centroid
        kabsch = float(np.sqrt(np.mean(np.sum((fitted - target) ** 2, axis=1))))

    # Bonds and angles are local graph hyperedges; weighted distances and Kabsch
    # only break ties because flexible torsions change long-range geometry.
    total = bond + 0.35 * angle + 0.25 * weighted_distance + 0.05 * kabsch
    return {
        "score": round(total, 6),
        "bond_rmsd_a": round(bond, 6),
        "angle_rmsd_rad": round(angle, 6),
        "weighted_distance_rmsd_a": round(weighted_distance, 6),
        "kabsch_rmsd_a": round(kabsch, 6),
    }


def _enumerate_isomorphisms(
    template_atoms: list[dict[str, Any]],
    complex_atoms: list[dict[str, Any]],
    template_edges: set[tuple[int, int]],
    complex_edges: set[tuple[int, int]],
    max_mappings: int,
) -> tuple[list[dict[int, int]], bool]:
    tn = _neighbors(len(template_atoms), template_edges)
    cn = _neighbors(len(complex_atoms), complex_edges)
    td = _graph_dist(tn)
    cd = _graph_dist(cn)
    t_sig = [_local_signature(i, template_atoms, tn, td) for i in range(len(template_atoms))]
    c_sig = [_local_signature(i, complex_atoms, cn, cd) for i in range(len(complex_atoms))]
    candidates = {
        i: [j for j in range(len(complex_atoms)) if t_sig[i] == c_sig[j]]
        for i in range(len(template_atoms))
    }
    if any(not v for v in candidates.values()):
        candidates = {
            i: [
                j
                for j in range(len(complex_atoms))
                if template_atoms[i]["element"] == complex_atoms[j]["element"] and len(tn[i]) == len(cn[j])
            ]
            for i in range(len(template_atoms))
        }
    missing = [i + 1 for i, values in candidates.items() if not values]
    if missing:
        raise ValueError(f"No heavy-atom candidates after local graph filtering: {missing}")

    order = sorted(range(len(template_atoms)), key=lambda i: (len(candidates[i]), -len(tn[i]), template_atoms[i]["element"], i))
    mappings: list[dict[int, int]] = []
    truncated = False

    def visit(pos: int, current: dict[int, int], used: set[int]) -> None:
        nonlocal truncated
        if len(mappings) >= max_mappings:
            truncated = True
            return
        if pos == len(order):
            mappings.append(dict(current))
            return
        i = order[pos]
        for j in candidates[i]:
            if j in used:
                continue
            if all((old_i in tn[i]) == (old_j in cn[j]) for old_i, old_j in current.items()):
                current[i] = j
                used.add(j)
                visit(pos + 1, current, used)
                used.remove(j)
                del current[i]

    visit(0, {}, set())
    return mappings, truncated


def _sdf_neighbors_and_orders(sdf: dict[str, Any]) -> tuple[list[set[int]], dict[tuple[int, int], int]]:
    neigh = [set() for _ in sdf["atoms"]]
    orders = {}
    for bond in sdf["bonds"]:
        a = int(bond["a"]) - 1
        b = int(bond["b"]) - 1
        neigh[a].add(b)
        neigh[b].add(a)
        orders[(min(a, b), max(a, b))] = int(bond["order"])
    return neigh, orders


def _exchange_class(sdf: dict[str, Any], a: int, b: int) -> str:
    atoms = sdf["atoms"]
    neigh, orders = _sdf_neighbors_and_orders(sdf)
    if atoms[a]["element"] != atoms[b]["element"]:
        return "non_equivalent"
    ha = [n for n in neigh[a] if atoms[n]["element"] != "H"]
    hb = [n for n in neigh[b] if atoms[n]["element"] != "H"]
    if len(ha) != 1 or len(hb) != 1 or ha[0] != hb[0]:
        return "non_equivalent"
    parent = ha[0]
    oa = orders[(min(a, parent), max(a, parent))]
    ob = orders[(min(b, parent), max(b, parent))]
    hca = sum(1 for n in neigh[a] if atoms[n]["element"] == "H")
    hcb = sum(1 for n in neigh[b] if atoms[n]["element"] == "H")
    if oa == ob and hca == hcb:
        return "same_parent_terminal_equivalent"
    if atoms[a]["element"] == "O" and atoms[parent]["element"] == "N" and sorted((oa, ob)) == [1, 2] and hca == hcb == 0:
        return "nitro_resonance_equivalent"
    return "non_equivalent"


def _equivalence_proof(
    sdf: dict[str, Any],
    heavy_sdf_ids: list[int],
    complex_atoms: list[dict[str, Any]],
    best: dict[int, int],
    second: dict[int, int] | None,
) -> dict[str, Any]:
    if second is None:
        return {"is_equivalent": False, "exchanges": []}
    diffs = [
        {
            "heavy_index": i,
            "sdf_index": heavy_sdf_ids[i] + 1,
            "element": sdf["atoms"][heavy_sdf_ids[i]]["element"],
            "best_pdb_index": int(complex_atoms[best[i]]["index"]),
            "best_name": complex_atoms[best[i]]["name"],
            "second_pdb_index": int(complex_atoms[second[i]]["index"]),
            "second_name": complex_atoms[second[i]]["name"],
        }
        for i in range(len(heavy_sdf_ids))
        if best[i] != second[i]
    ]
    by_swap = {(d["best_pdb_index"], d["second_pdb_index"]): d for d in diffs}
    exchanges = []
    seen: set[int] = set()
    ok = bool(diffs)
    for pos, row in enumerate(diffs):
        if pos in seen:
            continue
        partner = by_swap.get((row["second_pdb_index"], row["best_pdb_index"]))
        if partner is None:
            exchanges.append({"class": "unpaired_non_equivalent", "atoms": [row]})
            ok = False
            seen.add(pos)
            continue
        partner_pos = diffs.index(partner)
        seen.update({pos, partner_pos})
        klass = _exchange_class(sdf, int(row["sdf_index"]) - 1, int(partner["sdf_index"]) - 1)
        if klass == "non_equivalent":
            ok = False
        exchanges.append({
            "class": klass,
            "sdf_indices": [row["sdf_index"], partner["sdf_index"]],
            "best_pdb_indices": [row["best_pdb_index"], partner["best_pdb_index"]],
            "second_pdb_indices": [row["second_pdb_index"], partner["second_pdb_index"]],
        })
    return {"is_equivalent": ok, "exchanges": exchanges}


def resolve_heavy_hypergraph_mapping(
    *,
    sdf_template: str | Path,
    ligand_pdb: str | Path,
    output_json: str | Path | None = None,
    max_mappings: int = 200000,
    tau: float = 2.0,
    unique_gap: float = 0.05,
    equivalent_tol: float = 0.01,
) -> dict[str, Any]:
    """Resolve ligand heavy-atom identity by graph isomorphism plus local geometry.

    The mathematical invariant is not a 2D drawing. It is the heavy-atom graph
    augmented by geometric hyperedges. Full pairwise distances are rigid-body
    invariant; graph-weighting makes them robust to flexible torsions.
    """
    sdf = read_sdf_template(sdf_template)
    heavy_sdf_ids, template_atoms, template_edges = _heavy_sdf_graph(sdf)
    complex_atoms = _heavy_pdb_atoms(ligand_pdb)
    if sorted(a["element"] for a in template_atoms) != sorted(a["element"] for a in complex_atoms):
        raise ValueError("SDF and ligand PDB heavy-atom element counts differ.")

    complex_edges = _infer_bond_edges_from_coords(complex_atoms)
    mappings, truncated = _enumerate_isomorphisms(template_atoms, complex_atoms, template_edges, complex_edges, max_mappings)
    if not mappings:
        raise ValueError("No heavy-atom graph isomorphism found.")

    tn = _neighbors(len(template_atoms), template_edges)
    tdist = _distance_matrix(template_atoms)
    cdist = _distance_matrix(complex_atoms)
    tgdist = _graph_dist(tn)
    tangles = _angle_terms(template_atoms, tn)
    scored = []
    for mapping in mappings:
        metrics = _score_mapping(
            mapping=mapping,
            template_atoms=template_atoms,
            complex_atoms=complex_atoms,
            template_edges=template_edges,
            template_angles=tangles,
            template_dist=tdist,
            complex_dist=cdist,
            template_graph_dist=tgdist,
            tau=tau,
        )
        scored.append({"mapping": mapping, **metrics})
    scored.sort(key=lambda x: (x["score"], x["bond_rmsd_a"], x["weighted_distance_rmsd_a"], x["kabsch_rmsd_a"]))
    best = scored[0]
    second = scored[1] if len(scored) > 1 else None
    gap = round(second["score"] - best["score"], 6) if second else None
    proof = _equivalence_proof(sdf, heavy_sdf_ids, complex_atoms, best["mapping"], second["mapping"] if second else None)

    if len(scored) == 1 or (gap is not None and gap > unique_gap):
        decision = "unique"
    elif proof["is_equivalent"] or sum(abs(row["score"] - best["score"]) <= equivalent_tol for row in scored) > 1:
        decision = "equivalent_ok"
    else:
        decision = "ambiguous_fail"

    rows = []
    atom_map = {}
    for heavy_i, sdf_i in enumerate(heavy_sdf_ids):
        pdb_i = best["mapping"][heavy_i]
        atom_map[str(sdf_i + 1)] = int(complex_atoms[pdb_i]["index"])
        rows.append({
            "sdf_index": sdf_i + 1,
            "element": template_atoms[heavy_i]["element"],
            "pdb_index": int(complex_atoms[pdb_i]["index"]),
            "pdb_name": complex_atoms[pdb_i]["name"],
        })

    result = {
        "schema": "cypforge.heavy_hypergraph_mapping.v2",
        "status": "success",
        "decision": decision,
        "counts": {
            "heavy_atoms": len(template_atoms),
            "sdf_edges": len(template_edges),
            "pdb_inferred_edges": len(complex_edges),
            "mappings": len(scored),
            "truncated": truncated,
        },
        "thresholds": {"unique_gap": unique_gap, "equivalent_tol": equivalent_tol, "tau": tau},
        "best": {k: v for k, v in best.items() if k != "mapping"},
        "second": {k: v for k, v in second.items() if k != "mapping"} if second else None,
        "score_gap": gap,
        "equivalence_proof": proof,
        "sdf_atom_id_to_pdb_atom_index": atom_map,
        "mapping": rows,
        "math": {
            "object": "heavy_atom_graph_plus_local_geometry_hyperedges",
            "score": "bond + 0.35*angle + 0.25*graph_weighted_distance + 0.05*kabsch",
            "rigid_invariant": "pairwise distances are invariant to rotation and translation under a fixed atom permutation",
            "flexible_robustness": "long-range distances are downweighted by exp(-graph_distance/tau)",
        },
    }
    if output_json is not None:
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result

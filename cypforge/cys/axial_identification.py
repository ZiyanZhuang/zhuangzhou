from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..heme.mapping import (
    AtomRecord,
    build_source_frame,
    dedupe_heme_atoms,
    parse_pdb_atoms,
    parse_pdb_heme_atoms,
)


DEFAULT_ALLOWED_RESNAMES = ("CYS", "CYM", "CYX", "CYP")
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "CYM": "C", "CYX": "C", "CYP": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "HIE": "H", "HID": "H", "HIP": "H",
    "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
HEME_BINDING_MOTIF_RULES = (
    (-8, {"P"}, 2.0, "minus8_pro"),
    (-7, {"F", "Y", "W"}, 2.0, "minus7_aromatic"),
    (-4, {"G"}, 2.5, "minus4_gly"),
    (-2, {"R"}, 3.0, "minus2_arg"),
    (+1, {"I", "L", "V", "F", "M", "A", "C"}, 1.0, "plus1_hydrophobic"),
    (+2, {"G"}, 3.0, "plus2_gly"),
)


@dataclass
class AxialCysResult:
    heme_fe: np.ndarray
    heme_frame: object
    axial_atom: AtomRecord
    cys_atoms: Dict[str, AtomRecord]
    diagnostics: Dict[str, object]


def _vector_angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    cosine = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _group_residue_atoms(atoms: Sequence[AtomRecord]) -> Dict[Tuple[str, int, str], List[AtomRecord]]:
    grouped: Dict[Tuple[str, int, str], List[AtomRecord]] = {}
    for atom in atoms:
        key = (atom.chain, atom.resid, atom.resname)
        grouped.setdefault(key, []).append(atom)
    return grouped


def _extract_chain_sequences(
    pdb_path: str,
    chain_id: Optional[str] = None,
) -> Dict[str, List[Tuple[int, str, str]]]:
    sequences: Dict[str, List[Tuple[int, str, str]]] = {}
    seen = set()
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            chain = line[21].strip() or "A"
            if chain_id and chain != chain_id:
                continue
            resid = int(line[22:26])
            resname = line[17:20].strip()
            key = (chain, resid)
            if key in seen:
                continue
            seen.add(key)
            sequences.setdefault(chain, []).append((resid, resname, THREE_TO_ONE.get(resname, "X")))
    return sequences


def _sequence_context(
    chain_sequence: List[Tuple[int, str, str]],
    resid: int,
    window: int = 12,
) -> Optional[Dict[str, object]]:
    resids = [item[0] for item in chain_sequence]
    if resid not in resids:
        return None
    index = resids.index(resid)
    left = max(0, index - window)
    right = min(len(chain_sequence), index + window + 1)
    local = chain_sequence[left:right]
    seq = "".join(item[2] for item in local)
    return {
        "index": index,
        "window_start_resid": local[0][0],
        "window_end_resid": local[-1][0],
        "window_sequence": seq,
        "window_center_offset": index - left,
    }


def _sequence_motif_score(
    chain_sequence: List[Tuple[int, str, str]],
    resid: int,
) -> Dict[str, object]:
    resids = [item[0] for item in chain_sequence]
    aas = [item[2] for item in chain_sequence]
    if resid not in resids:
        return {
            "motif_score": 0.0,
            "motif_hits": [],
            "sequence_context": None,
        }
    index = resids.index(resid)
    hits: List[str] = []
    score = 0.0
    for offset, allowed_aas, weight, label in HEME_BINDING_MOTIF_RULES:
        pos = index + offset
        if pos < 0 or pos >= len(aas):
            continue
        if aas[pos] in allowed_aas:
            score += weight
            hits.append(label)
    context = _sequence_context(chain_sequence, resid)
    return {
        "motif_score": score,
        "motif_hits": hits,
        "sequence_context": context,
    }


def _score_candidate(
    sg_atom: AtomRecord,
    residue_atoms: Dict[str, AtomRecord],
    heme_fe: np.ndarray,
    heme_frame,
    target_fe_s: float,
) -> Tuple[float, Dict[str, float]]:
    fe_s_distance = float(np.linalg.norm(sg_atom.coord - heme_fe))
    local = heme_frame.to_local(sg_atom.coord)
    radial_offset = float(math.hypot(local[0], local[1]))
    out_of_plane = float(abs(local[2]))

    cb_atom = residue_atoms.get("CB")
    cb_angle = float("nan")
    cb_penalty = 2.0
    if cb_atom is not None:
        cb_angle = _vector_angle_deg(cb_atom.coord - sg_atom.coord, heme_fe - sg_atom.coord)
        cb_penalty = abs(cb_angle - 111.514) / 20.0

    score = (
        abs(fe_s_distance - target_fe_s)
        + 0.35 * radial_offset
        + 0.10 * abs(out_of_plane - target_fe_s)
        + cb_penalty
    )
    details = {
        "fe_s_distance": fe_s_distance,
        "radial_offset": radial_offset,
        "out_of_plane": out_of_plane,
        "cb_sg_fe_angle_deg": cb_angle,
        "score": score,
    }
    return score, details


def identify_axial_cys(
    pdb_path: str,
    heme_resname: str = "HEM",
    heme_chain: Optional[str] = None,
    protein_chain: Optional[str] = None,
    axial_cys_resid: Optional[int] = None,
    allowed_resnames: Sequence[str] = DEFAULT_ALLOWED_RESNAMES,
    target_fe_s: float = 2.660,
) -> AxialCysResult:
    raw_heme_atoms = parse_pdb_heme_atoms(pdb_path, resname=heme_resname, chain_id=heme_chain)
    if not raw_heme_atoms:
        raise ValueError(f"No {heme_resname} atoms found in {pdb_path}")

    heme_atoms, _ = dedupe_heme_atoms(raw_heme_atoms)
    heme_frame, heme_diag = build_source_frame(heme_atoms)
    heme_by_name = {atom.name: atom for atom in heme_atoms}
    if "FE" not in heme_by_name:
        raise ValueError("HEME does not contain FE.")
    heme_fe = heme_by_name["FE"].coord

    chain_sequences = _extract_chain_sequences(pdb_path, chain_id=protein_chain)
    all_atoms = parse_pdb_atoms(pdb_path, chain_id=protein_chain)
    grouped = _group_residue_atoms(all_atoms)

    candidates: List[Tuple[float, float, AtomRecord, Dict[str, AtomRecord], Dict[str, object]]] = []
    allowed = {name.upper() for name in allowed_resnames}
    for (chain, resid, resname), residue_atoms in grouped.items():
        if resname.upper() not in allowed:
            continue
        if axial_cys_resid is not None and resid != axial_cys_resid:
            continue
        residue_by_name = {atom.name: atom for atom in residue_atoms}
        sg_atom = residue_by_name.get("SG")
        if sg_atom is None:
            continue
        score, details = _score_candidate(
            sg_atom=sg_atom,
            residue_atoms=residue_by_name,
            heme_fe=heme_fe,
            heme_frame=heme_frame,
            target_fe_s=target_fe_s,
        )
        seq_info = _sequence_motif_score(chain_sequences.get(chain, []), resid)
        details = {
            **details,
            "motif_score": seq_info["motif_score"],
            "motif_hits": seq_info["motif_hits"],
            "sequence_context": seq_info["sequence_context"],
        }
        candidates.append((float(seq_info["motif_score"]), score, sg_atom, residue_by_name, details))

    if not candidates:
        raise ValueError("Could not find any candidate axial cysteine SG atoms.")

    candidates.sort(key=lambda item: (-item[0], item[1], item[2].chain, item[2].resid))
    best_motif_score, best_geom_score, best_sg, best_residue_atoms, best_details = candidates[0]
    top_candidates = []
    for motif_score, geom_score, sg_atom, residue_by_name, details in candidates[:5]:
        top_candidates.append(
            {
                "chain": sg_atom.chain,
                "resid": sg_atom.resid,
                "resname": sg_atom.resname,
                "atom": sg_atom.name,
                "motif_score": motif_score,
                "geometry_score": geom_score,
                **details,
            }
        )

    diagnostics = {
        "heme_fe_coord": [float(x) for x in heme_fe.tolist()],
        "selected_axial_cys": {
            "chain": best_sg.chain,
            "resid": best_sg.resid,
            "resname": best_sg.resname,
            "atom": best_sg.name,
            "motif_score": best_motif_score,
            "geometry_score": best_geom_score,
            **best_details,
        },
        "candidate_count": len(candidates),
        "top_candidates": top_candidates,
        "heme_frame": heme_diag,
        "target_fe_s": target_fe_s,
        "chain_sequences": {
            chain: {
                "length": len(sequence),
                "sequence": "".join(item[2] for item in sequence),
            }
            for chain, sequence in chain_sequences.items()
        },
    }

    return AxialCysResult(
        heme_fe=heme_fe,
        heme_frame=heme_frame,
        axial_atom=best_sg,
        cys_atoms=best_residue_atoms,
        diagnostics=diagnostics,
    )


def _json_ready(result: AxialCysResult) -> Dict[str, object]:
    residue_atoms = {
        name: {
            "coord": [float(x) for x in atom.coord.tolist()],
            "chain": atom.chain,
            "resid": atom.resid,
            "resname": atom.resname,
            "element": atom.element,
        }
        for name, atom in sorted(result.cys_atoms.items())
    }
    return {
        "heme_fe": [float(x) for x in result.heme_fe.tolist()],
        "selected_axial_atom": {
            "name": result.axial_atom.name,
            "chain": result.axial_atom.chain,
            "resid": result.axial_atom.resid,
            "resname": result.axial_atom.resname,
            "coord": [float(x) for x in result.axial_atom.coord.tolist()],
        },
        "cys_atoms": residue_atoms,
        "diagnostics": result.diagnostics,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Identify the axial cysteine coordinating heme iron.")
    parser.add_argument("pdb_path", help="Path to the full protein+heme PDB.")
    parser.add_argument("--heme-resname", default="HEM")
    parser.add_argument("--heme-chain", default=None)
    parser.add_argument("--protein-chain", default=None)
    parser.add_argument("--axial-cys-resid", type=int, default=None)
    parser.add_argument("--target-fe-s", type=float, default=2.660)
    parser.add_argument("--write-json", default=None)
    args = parser.parse_args(argv)

    result = identify_axial_cys(
        pdb_path=args.pdb_path,
        heme_resname=args.heme_resname,
        heme_chain=args.heme_chain,
        protein_chain=args.protein_chain,
        axial_cys_resid=args.axial_cys_resid,
        target_fe_s=args.target_fe_s,
    )
    payload = json.dumps(_json_ready(result), indent=2, ensure_ascii=False)
    if args.write_json:
        Path(args.write_json).write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

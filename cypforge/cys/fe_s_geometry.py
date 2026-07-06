from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from cypforge_core.io import _angle_deg, parse_frcmod_sections

from ..heme.mapping import AtomRecord, parse_pdb_atoms, parse_pdb_heme_atoms


def resolve_frcmod_path(frcmod_path: Optional[str] = None) -> Path:
    """
    Resolve frcmod path. Must be passed explicitly (state-specific: IC6 / CPDI / DIOXY).
    No default fallback — the caller must select the correct state frcmod.
    """
    if frcmod_path is None:
        raise ValueError(
            "frcmod_path is required. Pass the state-specific frcmod: "
            "params/IC6/IC6.frcmod, params/CPDI/CPDI.frcmod, or params/DIOXY/DIOXY.frcmod"
        )
    path = Path(frcmod_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing frcmod: {path}")
    return path


def extract_fe_s_parameters(frcmod_path: Path) -> Dict[str, object]:
    sections = parse_frcmod_sections(frcmod_path)
    bond = None
    angles: Dict[str, Tuple[float, float]] = {}
    dihedrals: Dict[str, str] = {}

    for line in sections.get("BOND", []):
        if line.strip().startswith("fe-SH"):
            parts = line.split()
            if len(parts) >= 3:
                bond = {"k": float(parts[1]), "r0": float(parts[2]), "raw": line}

    for line in sections.get("ANGLE", []):
        key = line.split()[0]
        if "fe" in key.lower() or "SH" in key:
            parts = line.split()
            if len(parts) >= 3:
                angles[key] = (float(parts[1]), float(parts[2]))

    for line in sections.get("DIHEDRAL", []):
        key = line.split()[0]
        if "fe" in key.lower() or "SH" in key:
            dihedrals[key] = line

    return {
        "bond": bond,
        "angles": angles,
        "dihedrals": dihedrals,
        "frcmod_path": str(frcmod_path),
    }


def _first_atom_by_name(atoms: Sequence[AtomRecord], name: str) -> Optional[AtomRecord]:
    for atom in atoms:
        if atom.name == name:
            return atom
    return None


def evaluate_fe_s_geometry(
    pdb_path: str,
    heme_resname: str = "HEM",
    heme_chain: Optional[str] = None,
    cys_chain: Optional[str] = None,
    cys_resid: Optional[int] = None,
    frcmod_path: Optional[str] = None,
) -> Dict[str, object]:
    heme_atoms = parse_pdb_heme_atoms(pdb_path, resname=heme_resname, chain_id=heme_chain)
    if not heme_atoms:
        raise ValueError(f"No {heme_resname} atoms found in {pdb_path}")

    fe_atom = _first_atom_by_name(heme_atoms, "FE")
    nc_atom = _first_atom_by_name(heme_atoms, "NC")
    nd_atom = _first_atom_by_name(heme_atoms, "ND")
    if fe_atom is None:
        raise ValueError("HEME does not contain FE.")

    protein_atoms = parse_pdb_atoms(pdb_path, chain_id=cys_chain)
    residue_atoms = [
        atom for atom in protein_atoms
        if cys_resid is None or atom.resid == cys_resid
    ]
    sg_atom = _first_atom_by_name(residue_atoms, "SG")
    cb_atom = _first_atom_by_name(residue_atoms, "CB")
    if sg_atom is None:
        raise ValueError("Could not locate SG for the target axial cysteine.")

    params = extract_fe_s_parameters(resolve_frcmod_path(frcmod_path))
    target_bond = params["bond"]["r0"] if params["bond"] else None
    measured_distance = float(np.linalg.norm(fe_atom.coord - sg_atom.coord))

    angles = {}
    if nc_atom is not None:
        angles["SH-fe-nc"] = _angle_deg(sg_atom.coord, fe_atom.coord, nc_atom.coord)
    if nd_atom is not None:
        angles["SH-fe-nd"] = _angle_deg(sg_atom.coord, fe_atom.coord, nd_atom.coord)
    if cb_atom is not None:
        angles["CT-SH-fe"] = _angle_deg(cb_atom.coord, sg_atom.coord, fe_atom.coord)

    angle_deltas = {}
    for key, measured in angles.items():
        if key in params["angles"]:
            angle_deltas[key] = measured - params["angles"][key][1]

    return {
        "frcmod": params,
        "selected_cys": {
            "chain": sg_atom.chain,
            "resid": sg_atom.resid,
            "resname": sg_atom.resname,
        },
        "measured": {
            "fe_s_distance": measured_distance,
            "distance_delta": (measured_distance - target_bond) if target_bond is not None else None,
            "angles_deg": angles,
            "angle_deltas_deg": angle_deltas,
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Fe-S geometry against frcmod parameters.")
    parser.add_argument("pdb_path", help="Path to the full prepared PDB.")
    parser.add_argument("--heme-resname", default="HEM")
    parser.add_argument("--heme-chain", default=None)
    parser.add_argument("--cys-chain", default=None)
    parser.add_argument("--cys-resid", type=int, default=None)
    parser.add_argument("--frcmod", required=True,
                        help="State-specific frcmod path: params/IC6/IC6.frcmod, params/CPDI/CPDI.frcmod, etc.")
    parser.add_argument("--write-json", default=None)
    args = parser.parse_args(argv)

    report = evaluate_fe_s_geometry(
        pdb_path=args.pdb_path,
        heme_resname=args.heme_resname,
        heme_chain=args.heme_chain,
        cys_chain=args.cys_chain,
        cys_resid=args.cys_resid,
        frcmod_path=args.frcmod,
    )
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.write_json:
        Path(args.write_json).write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

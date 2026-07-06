from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from cypforge_core.io import _angle_deg, kabsch_transform

from .axial_identification import AxialCysResult, identify_axial_cys
from .fe_s_geometry import extract_fe_s_parameters, resolve_frcmod_path
from ..heme.mapping import AtomRecord, infer_element


BACKBONE_ANCHORS = ("N", "CA", "C", "O")


def resolve_cyp_mol2_path(cyp_mol2_path: Optional[str] = None) -> Path:
    """
    Resolve CYP mol2 path. Must be passed explicitly with state prefix.
    e.g. params/IC6/IC6-CYP.mol2, params/CPDI/CPDI-CYP.mol2, params/DIOXY/DIOXY-CYP.mol2
    No generic fallback — state-specific charges must be selected by the caller.
    """
    if cyp_mol2_path is None:
        raise ValueError(
            "cyp_mol2_path is required. Pass the state-specific mol2: "
            "params/IC6/IC6-CYP.mol2, params/CPDI/CPDI-CYP.mol2, or params/DIOXY/DIOXY-CYP.mol2"
        )
    path = Path(cyp_mol2_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing CYP mol2: {path}")
    return path


@dataclass
class ProximalRewriteResult:
    residue_atoms: Dict[str, AtomRecord]
    rewritten_atoms: Dict[str, AtomRecord]
    diagnostics: Dict[str, object]


def load_cyp_template_atoms(cyp_mol2_path: Path) -> Dict[str, AtomRecord]:
    atoms: Dict[str, AtomRecord] = {}
    in_atoms = False
    for line in cyp_mol2_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("@<TRIPOS>ATOM"):
            in_atoms = True
            continue
        if line.startswith("@<TRIPOS>") and not line.startswith("@<TRIPOS>ATOM"):
            in_atoms = False
        if not in_atoms:
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        name = parts[1]
        coord = np.array([float(parts[2]), float(parts[3]), float(parts[4])], dtype=float)
        atoms[name] = AtomRecord(
            name=name,
            coord=coord,
            serial=int(parts[0]),
            element=infer_element(name),
            chain="T",
            resid=1,
            resname="CYP",
        )
    return atoms


def resolve_polymer_connect_atoms(cyp_mol2_path: str) -> Dict[str, object]:
    """
    Resolve the peptide-polymer connection atoms for a CYP template from atom names.

    We intentionally infer these from atom names instead of hardcoding serial numbers,
    so all states remain correct even if atom ordering changes in a future mol2.
    """
    template_atoms = load_cyp_template_atoms(resolve_cyp_mol2_path(cyp_mol2_path))
    if "N" not in template_atoms or "C" not in template_atoms:
        raise ValueError(
            f"CYP template {cyp_mol2_path} must contain backbone atoms N and C "
            "to be used as a polymer residue in tleap."
        )
    return {
        "head_name": "N",
        "head_index": int(template_atoms["N"].serial),
        "tail_name": "C",
        "tail_index": int(template_atoms["C"].serial),
    }


def apply_transform(
    coord: np.ndarray,
    rotation: np.ndarray,
    source_centroid: np.ndarray,
    target_centroid: np.ndarray,
) -> np.ndarray:
    return (coord - source_centroid) @ rotation + target_centroid


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Cannot normalize zero-length vector.")
    return vector / norm


def _rotate_vector_around_axis(vector: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = _normalize(axis)
    theta = math.radians(angle_deg)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    return (
        vector * cos_theta
        + np.cross(axis, vector) * sin_theta
        + axis * np.dot(axis, vector) * (1.0 - cos_theta)
    )



def _construct_sg_from_params(
    residue_heavy: Mapping[str, AtomRecord],
    template_atoms: Mapping[str, AtomRecord],
    heme_fe_coord: Sequence[float],
    heme_nc_coord: Sequence[float],
    heme_nd_coord: Sequence[float],
    frcmod_path: Optional[str],
) -> Tuple[np.ndarray, Dict[str, object]]:
    if "CA" not in residue_heavy or "CB" not in residue_heavy:
        raise ValueError("Residue is missing CA/CB required for SG construction.")
    if "SG" not in residue_heavy:
        raise ValueError("Residue is missing source SG required for SG construction.")
    if "CB" not in template_atoms or "SG" not in template_atoms:
        raise ValueError("CYP template is missing CB/SG atoms.")

    params = extract_fe_s_parameters(resolve_frcmod_path(frcmod_path))
    target_bond = params["bond"]["r0"] if params["bond"] else None
    target_cb_sg_fe = params["angles"].get("CT-SH-fe", (None, None))[1]
    target_sg_fe_nc = params["angles"].get("SH-fe-nc", (None, None))[1]
    target_sg_fe_nd = params["angles"].get("SH-fe-nd", (None, None))[1]
    k_bond = params["bond"]["k"] if params["bond"] else 0.0
    k_cb = params["angles"].get("CT-SH-fe", (0.0, None))[0]
    k_nc = params["angles"].get("SH-fe-nc", (0.0, None))[0]
    k_nd = params["angles"].get("SH-fe-nd", (0.0, None))[0]

    ca = residue_heavy["CA"].coord
    cb = residue_heavy["CB"].coord
    source_sg = residue_heavy["SG"].coord
    fe = np.asarray(heme_fe_coord, dtype=float)
    nc = np.asarray(heme_nc_coord, dtype=float)
    nd = np.asarray(heme_nd_coord, dtype=float)

    cb_s_length = float(np.linalg.norm(template_atoms["SG"].coord - template_atoms["CB"].coord))
    axis = cb - ca
    source_direction = _normalize(source_sg - cb)
    base_vector = source_direction * cb_s_length

    best_energy = None
    best_payload = None
    best_coord = None
    for angle_deg in range(0, 360):
        candidate = cb + _rotate_vector_around_axis(base_vector, axis, angle_deg)
        distance = float(np.linalg.norm(candidate - fe))
        angle_cb = _angle_deg(cb, candidate, fe)
        angle_nc = _angle_deg(candidate, fe, nc)
        angle_nd = _angle_deg(candidate, fe, nd)
        energy = 0.0
        if target_bond is not None:
            energy += k_bond * (distance - target_bond) ** 2
        if target_cb_sg_fe is not None:
            energy += k_cb * math.radians(angle_cb - target_cb_sg_fe) ** 2
        if target_sg_fe_nc is not None:
            energy += k_nc * math.radians(angle_nc - target_sg_fe_nc) ** 2
        if target_sg_fe_nd is not None:
            energy += k_nd * math.radians(angle_nd - target_sg_fe_nd) ** 2
        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_coord = candidate
            best_payload = {
                "selection_mode": "parameter-constructed-sg",
                "selected_rotation_deg": float(angle_deg),
                "target_fe_s": target_bond,
                "target_cb_sg_fe_angle": target_cb_sg_fe,
                "target_sg_fe_nc_angle": target_sg_fe_nc,
                "target_sg_fe_nd_angle": target_sg_fe_nd,
                "selected_fe_s_distance": distance,
                "selected_cb_sg_fe_angle": angle_cb,
                "selected_sg_fe_nc_angle": angle_nc,
                "selected_sg_fe_nd_angle": angle_nd,
                "energy": energy,
                "template_cb_s_length": cb_s_length,
                "source_cb_s_length": float(np.linalg.norm(source_sg - cb)),
            }
    return np.asarray(best_coord, dtype=float), best_payload


def standardize_proximal_cyp(
    residue_atoms: Mapping[str, AtomRecord],
    cyp_mol2_path: Optional[str] = None,
    heme_fe_coord: Optional[Sequence[float]] = None,
    heme_nc_coord: Optional[Sequence[float]] = None,
    heme_nd_coord: Optional[Sequence[float]] = None,
    frcmod_path: Optional[str] = None,
) -> ProximalRewriteResult:
    template_path = resolve_cyp_mol2_path(cyp_mol2_path)
    template_atoms = load_cyp_template_atoms(template_path)

    common_anchors = [name for name in BACKBONE_ANCHORS if name in residue_atoms and name in template_atoms]
    if len(common_anchors) < 3:
        raise ValueError(f"Need at least 3 common backbone anchors, got {common_anchors}")

    src_points = np.array([template_atoms[name].coord for name in common_anchors], dtype=float)
    tgt_points = np.array([residue_atoms[name].coord for name in common_anchors], dtype=float)
    rotation, source_centroid, target_centroid = kabsch_transform(src_points, tgt_points)

    transformed = {
        name: apply_transform(atom.coord, rotation, source_centroid, target_centroid)
        for name, atom in template_atoms.items()
        if atom.element != "H"
    }

    exemplar = next(iter(residue_atoms.values()))
    residue_heavy = {name: atom for name, atom in residue_atoms.items() if atom.element != "H"}
    sg_coord = None
    sg_selection = None
    if heme_fe_coord is not None and heme_nc_coord is not None and heme_nd_coord is not None:
        sg_coord, sg_selection = _construct_sg_from_params(
            residue_heavy=residue_heavy,
            template_atoms=template_atoms,
            heme_fe_coord=heme_fe_coord,
            heme_nc_coord=heme_nc_coord,
            heme_nd_coord=heme_nd_coord,
            frcmod_path=frcmod_path,
        )
    direct_mapped_atoms = []
    template_filled_atoms = []
    parameter_constructed_atoms = []
    rewritten: Dict[str, AtomRecord] = {}
    for name, template_atom in template_atoms.items():
        if template_atom.element == "H":
            continue
        if name == "SG" and sg_coord is not None:
            coord = sg_coord
            parameter_constructed_atoms.append(name)
        elif name in residue_heavy:
            coord = residue_heavy[name].coord.copy()
            direct_mapped_atoms.append(name)
        else:
            coord = transformed[name]
            template_filled_atoms.append(name)
        rewritten[name] = AtomRecord(
            name=name,
            coord=np.asarray(coord, dtype=float),
            serial=template_atom.serial,
            element=template_atom.element,
            chain=exemplar.chain,
            resid=exemplar.resid,
            resname="CYP",
        )

    anchor_rmsd = 0.0
    if common_anchors:
        transformed_anchors = np.array(
            [apply_transform(template_atoms[name].coord, rotation, source_centroid, target_centroid) for name in common_anchors],
            dtype=float,
        )
        anchor_rmsd = float(np.sqrt(np.mean(np.sum((transformed_anchors - tgt_points) ** 2, axis=1))))

    diagnostics = {
        "template_cyp_mol2_path": str(template_path),
        "anchor_atoms": common_anchors,
        "anchor_rmsd": anchor_rmsd,
        "mapping_mode": "backbone-anchored-cb-direct-sg-parameter-constructed",
        "rewritten_heavy_atoms": sorted(rewritten.keys()),
        "direct_mapped_heavy_atoms": sorted(direct_mapped_atoms),
        "template_filled_heavy_atoms": sorted(template_filled_atoms),
        "parameter_constructed_heavy_atoms": sorted(parameter_constructed_atoms),
        "direct_mapped_backbone_heavy_atoms": sorted([name for name in direct_mapped_atoms if name in BACKBONE_ANCHORS]),
        "direct_mapped_sidechain_heavy_atoms": sorted([name for name in direct_mapped_atoms if name not in BACKBONE_ANCHORS]),
        "parameter_constructed_sidechain_heavy_atoms": sorted([name for name in parameter_constructed_atoms if name not in BACKBONE_ANCHORS]),
        "sg_selection": sg_selection,
    }
    return ProximalRewriteResult(
        residue_atoms=dict(residue_atoms),
        rewritten_atoms=rewritten,
        diagnostics=diagnostics,
    )


def rewrite_from_full_pdb(
    pdb_path: str,
    heme_resname: str = "HEM",
    heme_chain: Optional[str] = None,
    protein_chain: Optional[str] = None,
    axial_cys_resid: Optional[int] = None,
    cyp_mol2_path: Optional[str] = None,
) -> Dict[str, object]:
    axial: AxialCysResult = identify_axial_cys(
        pdb_path=pdb_path,
        heme_resname=heme_resname,
        heme_chain=heme_chain,
        protein_chain=protein_chain,
        axial_cys_resid=axial_cys_resid,
    )
    rewritten = standardize_proximal_cyp(axial.cys_atoms, cyp_mol2_path=cyp_mol2_path)
    return {
        "axial_cys": axial,
        "rewritten": rewritten,
    }


def _json_ready(result: Dict[str, object]) -> Dict[str, object]:
    axial: AxialCysResult = result["axial_cys"]
    rewritten: ProximalRewriteResult = result["rewritten"]
    return {
        "selected_axial_cys": axial.diagnostics["selected_axial_cys"],
        "axial_candidates": axial.diagnostics["top_candidates"],
        "rewrite": rewritten.diagnostics,
        "rewritten_atoms": {
            name: [float(x) for x in atom.coord.tolist()]
            for name, atom in sorted(rewritten.rewritten_atoms.items())
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Rewrite proximal axial Cys as standardized CYP residue block.")
    parser.add_argument("pdb_path", help="Path to full protein+heme PDB.")
    parser.add_argument("--heme-resname", default="HEM")
    parser.add_argument("--heme-chain", default=None)
    parser.add_argument("--protein-chain", default=None)
    parser.add_argument("--axial-cys-resid", type=int, default=None)
    parser.add_argument("--cyp-mol2", required=True,
                        help="State-specific CYP mol2: params/IC6/IC6-CYP.mol2, params/CPDI/CPDI-CYP.mol2, etc.")
    parser.add_argument("--write-json", default=None)
    args = parser.parse_args(argv)

    result = rewrite_from_full_pdb(
        pdb_path=args.pdb_path,
        heme_resname=args.heme_resname,
        heme_chain=args.heme_chain,
        protein_chain=args.protein_chain,
        axial_cys_resid=args.axial_cys_resid,
        cyp_mol2_path=args.cyp_mol2,
    )
    payload = json.dumps(_json_ready(result), indent=2, ensure_ascii=False)
    if args.write_json:
        Path(args.write_json).write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

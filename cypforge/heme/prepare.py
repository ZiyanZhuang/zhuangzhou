from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from ..cys.axial_identification import identify_axial_cys
from ..cys.fe_s_geometry import evaluate_fe_s_geometry
from ..cys.interface_spec import generate_fe_s_interface_spec
from ..cys.proximal_rewrite import standardize_proximal_cyp
from .mapping import map_heme_template


_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_HEME_PARAMS_ROOT = _PACKAGE_ROOT / "data" / "heme_params"
_AMBER_PROTEIN_RESNAMES = {
    "ALA", "ARG", "ASH", "ASN", "ASP", "CYS", "CYM", "CYX", "GLH", "GLN",
    "GLU", "GLY", "HID", "HIE", "HIP", "HIS", "ILE", "LEU", "LYN", "LYS",
    "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}


def _resolve_state_specific_paths(
    heme_state: str,
    cyp_mol2_path: Optional[str],
    frcmod_path: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    if heme_state.upper() == "CUSTOM":
        if not cyp_mol2_path or not frcmod_path:
            raise ValueError(
                "CUSTOM heme state requires --custom-cyp-mol2 and --custom-frcmod; "
                "no built-in defaults exist for CUSTOM."
            )
        return cyp_mol2_path, frcmod_path
    resolved_cyp = Path(cyp_mol2_path) if cyp_mol2_path else _HEME_PARAMS_ROOT / heme_state / "CYP.mol2"
    resolved_frcmod = Path(frcmod_path) if frcmod_path else _HEME_PARAMS_ROOT / heme_state / f"{heme_state}.frcmod"
    return (
        str(resolved_cyp) if resolved_cyp.exists() else cyp_mol2_path,
        str(resolved_frcmod) if resolved_frcmod.exists() else frcmod_path,
    )


def _pdb_atom_line(
    record_type: str,
    serial: int,
    atom_name: str,
    resname: str,
    chain: str,
    resid: int,
    coord,
    element: str,
) -> str:
    return (
        f"{record_type:<6s}{serial:5d} {atom_name:>4s} {resname:>3s} {chain:1s}{resid:4d}    "
        f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
        f"{1.00:6.2f}{0.00:6.2f}          {element:>2s}\n"
    )


def _ordered_atoms(atom_map: Mapping[str, object]):
    return sorted(atom_map.values(), key=lambda atom: (atom.serial, atom.name))


def _passthrough_atom_priority(line: str) -> tuple[int, float, int]:
    altloc = line[16].strip()
    occupancy_text = line[54:60].strip()
    occupancy = float(occupancy_text) if occupancy_text else 0.0
    altloc_rank = 0 if altloc == "" else 1 if altloc == "A" else 2
    serial = int(line[6:11])
    return (altloc_rank, -occupancy, serial)


def _select_passthrough_atom_lines(
    input_pdb: str,
    heme_resname: str,
    heme_chain: str,
    heme_resid: int,
    cys_chain: str,
    cys_resid: int,
) -> List[tuple]:
    selected: Dict[tuple, tuple] = {}
    with open(input_pdb, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue

            resname = line[17:20].strip()
            chain = line[21].strip()
            resid = int(line[22:26])
            serial = int(line[6:11])
            atom_name = line[12:16].strip()

            if chain != cys_chain:
                continue
            if resname == heme_resname and resid == heme_resid:
                continue
            if chain == cys_chain and resid == cys_resid:
                continue
            if resname not in _AMBER_PROTEIN_RESNAMES:
                continue
            if atom_name == "OXT":
                continue

            key = (chain, resid, resname, atom_name)
            candidate = (_passthrough_atom_priority(line), chain, resid, serial, line.rstrip("\n"))
            incumbent = selected.get(key)
            if incumbent is None or candidate[0] < incumbent[0]:
                selected[key] = candidate

    passthrough = [(chain, resid, serial, line) for _, chain, resid, serial, line in selected.values()]
    passthrough.sort(key=lambda item: (item[0], item[1], item[2]))
    return passthrough


def _summarize_non_heme_hetatm_policy(
    input_pdb: str,
    heme_resname: str,
    heme_chain: str,
    heme_resid: int,
) -> Dict[str, object]:
    records: List[Dict[str, object]] = []
    seen = set()
    with open(input_pdb, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("HETATM"):
                continue
            resname = line[17:20].strip()
            chain = line[21].strip()
            resid = int(line[22:26])
            if resname == heme_resname and chain == heme_chain and resid == heme_resid:
                continue
            key = (chain, resid, resname)
            if key in seen:
                continue
            seen.add(key)
            records.append(
                {
                    "chain": chain,
                    "resid": resid,
                    "resname": resname,
                    "policy": "not_passed_through_by_heme_prepare",
                    "reason": (
                        "The heme preparation stage only rewrites the protein host, "
                        "proximal CYP residue, and selected heme. Non-heme HETATM residues "
                        "require an explicit ligand/cofactor policy before Amber assembly."
                    ),
                }
            )
    return {
        "default_policy": "do_not_silently_keep_non_heme_hetatm",
        "records": records,
        "record_count": len(records),
    }


def write_prepared_complex_pdb(
    input_pdb: str,
    output_pdb: str,
    heme_result,
    cyp_atoms,
    heme_resname: str = "HEM",
) -> None:
    source_heme = heme_result.source_atoms
    heme_chain = source_heme[0].chain or "A"
    heme_resid = source_heme[0].resid or 1

    exemplar_cyp = next(iter(cyp_atoms.values()))
    cys_chain = exemplar_cyp.chain
    cys_resid = exemplar_cyp.resid

    passthrough = _select_passthrough_atom_lines(
        input_pdb=input_pdb,
        heme_resname=heme_resname,
        heme_chain=heme_chain,
        heme_resid=heme_resid,
        cys_chain=cys_chain,
        cys_resid=cys_resid,
    )

    cyp_records: List[tuple] = []
    for atom in _ordered_atoms(cyp_atoms):
        cyp_records.append((atom.chain or cys_chain, atom.resid, atom.serial, atom))

    all_protein: List[tuple] = []
    for chain, resid, serial, line in passthrough:
        all_protein.append((chain, resid, serial, "line", line))
    for chain, resid, serial, atom in cyp_records:
        all_protein.append((chain, resid, serial, "atom", atom))
    all_protein.sort(key=lambda item: (item[0], item[1], item[2]))

    Path(output_pdb).parent.mkdir(parents=True, exist_ok=True)
    with open(output_pdb, "w", encoding="utf-8") as handle:
        serial = 1
        for entry in all_protein:
            if entry[3] == "line":
                line = entry[4]
                handle.write(line[:6] + f"{serial:5d}" + line[11:] + "\n")
            else:
                atom = entry[4]
                handle.write(
                    _pdb_atom_line(
                        record_type="ATOM",
                        serial=serial,
                        atom_name=atom.name,
                        resname=atom.resname,
                        chain=atom.chain or "A",
                        resid=atom.resid,
                        coord=atom.coord,
                        element=atom.element,
                    )
                )
            serial += 1

        handle.write("TER\n")
        for atom in heme_result.template_atoms:
            coord = heme_result.completed_heavy_atoms[atom.name]
            handle.write(
                _pdb_atom_line(
                    record_type="HETATM",
                    serial=serial,
                    atom_name=atom.name,
                    resname=heme_resname,
                    chain=heme_chain or "A",
                    resid=heme_resid,
                    coord=coord,
                    element=atom.element,
                )
            )
            serial += 1
        handle.write("TER\nEND\n")


def prepare_heme_complex(
    pdb_path: str,
    output_dir: str,
    heme_resname: str = "HEM",
    heme_chain: Optional[str] = None,
    protein_chain: Optional[str] = None,
    axial_cys_resid: Optional[int] = None,
    heme_state: Optional[str] = None,
    template_mol2_path: Optional[str] = None,
    cyp_mol2_path: Optional[str] = None,
    frcmod_path: Optional[str] = None,
) -> Dict[str, object]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    axial = identify_axial_cys(
        pdb_path=pdb_path,
        heme_resname=heme_resname,
        heme_chain=heme_chain,
        protein_chain=protein_chain,
        axial_cys_resid=axial_cys_resid,
    )
    resolved_heme_chain = heme_chain or axial.axial_atom.chain
    heme = map_heme_template(
        pdb_path=pdb_path,
        resname=heme_resname,
        chain_id=resolved_heme_chain,
        anchor_pdb_path=pdb_path,
        anchor_chain=axial.axial_atom.chain,
        axial_cys_resid=axial.axial_atom.resid,
        template_mol2_path=template_mol2_path,
        heme_state=heme_state,
    )
    cyp_mol2_path, frcmod_path = _resolve_state_specific_paths(
        heme.heme_state,
        cyp_mol2_path,
        frcmod_path,
    )
    proximal = standardize_proximal_cyp(
        residue_atoms=axial.cys_atoms,
        cyp_mol2_path=cyp_mol2_path,
        heme_fe_coord=heme.completed_heavy_atoms["FE"],
        heme_nc_coord=heme.completed_heavy_atoms["NC"],
        heme_nd_coord=heme.completed_heavy_atoms["ND"],
        frcmod_path=frcmod_path,
    )

    prepared_pdb = out_dir / "prepared_heme_complex.pdb"
    write_prepared_complex_pdb(
        input_pdb=pdb_path,
        output_pdb=str(prepared_pdb),
        heme_result=heme,
        cyp_atoms=proximal.rewritten_atoms,
        heme_resname=heme_resname,
    )

    prepared_heme_chain = heme.source_atoms[0].chain or "A"
    prepared_cys_chain = axial.axial_atom.chain or "A"

    fe_s = evaluate_fe_s_geometry(
        pdb_path=str(prepared_pdb),
        heme_resname=heme_resname,
        heme_chain=prepared_heme_chain,
        cys_chain=prepared_cys_chain,
        cys_resid=axial.axial_atom.resid,
        frcmod_path=frcmod_path,
    )
    interface_spec = generate_fe_s_interface_spec(
        pdb_path=str(prepared_pdb),
        output_dir=str(out_dir),
        heme_resname=heme_resname,
        heme_chain=prepared_heme_chain,
        cys_chain=prepared_cys_chain,
        cys_resid=axial.axial_atom.resid,
        frcmod_path=frcmod_path,
    )

    for stale_name in ("fe_s_constraint.json", "fe_s_distance.RST"):
        stale_path = out_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    report = {
        "workflow_name": "heme_prepare",
        "workflow_stage": "pre_tleap",
        "status": "success",
        "output_files": {
            "prepared_heme_complex_pdb": str(prepared_pdb),
            "report_json": str(out_dir / "prepare_report.json"),
            "fe_s_interface_json": interface_spec["interface_spec_json"],
        },
        "parameters": {
            "pdb_path": str(pdb_path),
            "heme_resname": heme_resname,
            "heme_chain": heme_chain,
            "protein_chain": protein_chain,
            "axial_cys_resid": axial_cys_resid,
            "heme_state": heme_state or "auto",
            "template_mol2_path": template_mol2_path,
            "cyp_mol2_path": cyp_mol2_path,
            "frcmod_path": frcmod_path,
        },
        "heme_mapping": heme.diagnostics,
        "input_hetatm_policy": _summarize_non_heme_hetatm_policy(
            input_pdb=pdb_path,
            heme_resname=heme_resname,
            heme_chain=heme.source_atoms[0].chain,
            heme_resid=heme.source_atoms[0].resid,
        ),
        "axial_cysteine": axial.diagnostics,
        "proximal_cysteine": proximal.diagnostics,
        "fe_s_quality_control": fe_s,
        "fe_s_interface": interface_spec["payload"],
    }
    (out_dir / "prepare_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def prepare_heme_system(
    pdb_path: str,
    output_dir: str,
    heme_resname: str = "HEM",
    heme_chain: Optional[str] = None,
    protein_chain: Optional[str] = None,
    axial_cys_resid: Optional[int] = None,
    heme_state: Optional[str] = None,
    template_mol2_path: Optional[str] = None,
    cyp_mol2_path: Optional[str] = None,
    frcmod_path: Optional[str] = None,
) -> Dict[str, object]:
    """Stable public wrapper for the current CYPForge heme/Cys workflow."""
    return prepare_heme_complex(
        pdb_path=pdb_path,
        output_dir=output_dir,
        heme_resname=heme_resname,
        heme_chain=heme_chain,
        protein_chain=protein_chain,
        axial_cys_resid=axial_cys_resid,
        heme_state=heme_state,
        template_mol2_path=template_mol2_path,
        cyp_mol2_path=cyp_mol2_path,
        frcmod_path=frcmod_path,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a CYP450 heme complex with the CYPForge heme/Cys workflow.")
    parser.add_argument("pdb_path", help="Path to full protein+heme PDB.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--heme-resname", default="HEM")
    parser.add_argument("--heme-chain", default=None)
    parser.add_argument("--protein-chain", default=None)
    parser.add_argument("--axial-cys-resid", type=int, default=None)
    parser.add_argument("--heme-state", choices=["IC6", "CPDI", "DIOXY"], default=None)
    parser.add_argument("--template-mol2", default=None)
    parser.add_argument("--cyp-mol2", default=None)
    parser.add_argument("--frcmod", default=None)
    args = parser.parse_args(argv)

    report = prepare_heme_system(
        pdb_path=args.pdb_path,
        output_dir=args.output_dir,
        heme_resname=args.heme_resname,
        heme_chain=args.heme_chain,
        protein_chain=args.protein_chain,
        axial_cys_resid=args.axial_cys_resid,
        heme_state=args.heme_state,
        template_mol2_path=args.template_mol2,
        cyp_mol2_path=args.cyp_mol2,
        frcmod_path=args.frcmod,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

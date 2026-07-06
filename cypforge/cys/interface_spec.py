from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

from .fe_s_geometry import evaluate_fe_s_geometry
from ..heme.mapping import parse_pdb_atoms


def _find_named_atom(
    pdb_path: str,
    resname: Optional[str],
    chain: Optional[str],
    resid: Optional[int],
    atom_name: str,
) -> Dict[str, object]:
    for atom in parse_pdb_atoms(pdb_path, chain_id=None):
        if atom.name != atom_name:
            continue
        if resname is not None and atom.resname != resname:
            continue
        if chain is not None and atom.chain != chain:
            continue
        if resid is not None and atom.resid != resid:
            continue
        return {
            "serial": atom.serial,
            "chain": atom.chain,
            "resid": atom.resid,
            "resname": atom.resname,
            "name": atom.name,
        }
    raise ValueError(f"Could not locate atom {atom_name} in prepared PDB.")


def generate_fe_s_interface_spec(
    pdb_path: str,
    output_dir: str,
    heme_resname: str = "HEM",
    heme_chain: Optional[str] = None,
    cys_chain: Optional[str] = None,
    cys_resid: Optional[int] = None,
    frcmod_path: Optional[str] = None,
) -> Dict[str, object]:
    geometry = evaluate_fe_s_geometry(
        pdb_path=pdb_path,
        heme_resname=heme_resname,
        heme_chain=heme_chain,
        cys_chain=cys_chain,
        cys_resid=cys_resid,
        frcmod_path=frcmod_path,
    )

    params = geometry["frcmod"]
    fe_atom = _find_named_atom(pdb_path, heme_resname, heme_chain, None, "FE")
    sg_atom = _find_named_atom(pdb_path, geometry["selected_cys"]["resname"], cys_chain, cys_resid, "SG")
    cb_atom = _find_named_atom(pdb_path, geometry["selected_cys"]["resname"], cys_chain, cys_resid, "CB")
    nc_atom = _find_named_atom(pdb_path, heme_resname, heme_chain, None, "NC")
    nd_atom = _find_named_atom(pdb_path, heme_resname, heme_chain, None, "ND")

    payload = {
        "mode": "fe-s-interface-spec",
        "prepared_pdb": str(pdb_path),
        "atoms": {
            "fe": fe_atom,
            "sg": sg_atom,
            "cb": cb_atom,
            "nc": nc_atom,
            "nd": nd_atom,
        },
        "targets": {
            "bond": params["bond"],
            "angles": {
                key: {"k": value[0], "theta0": value[1]}
                for key, value in params["angles"].items()
                if key in {"SH-fe-nc", "SH-fe-nd", "CT-SH-fe"}
            },
            "dihedrals": {
                key: value
                for key, value in params["dihedrals"].items()
                if key in {"cc-nc-fe-SH", "cd-nd-fe-SH", "X", "X-SH-fe-X"}
            },
        },
        "measured": geometry["measured"],
        "frcmod_path": params["frcmod_path"],
        "stage_note": "Stage 3 only records interface parameters and current geometry. No tleap or restraint files are produced here.",
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "fe_s_interface_spec.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "interface_spec_json": str(json_path),
        "payload": payload,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a neutral Fe-S interface spec for Stage 3.")
    parser.add_argument("pdb_path", help="Prepared PDB path.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--heme-resname", default="HEM")
    parser.add_argument("--heme-chain", default=None)
    parser.add_argument("--cys-chain", default=None)
    parser.add_argument("--cys-resid", type=int, default=None)
    parser.add_argument("--frcmod", required=True,
                        help="State-specific frcmod path: params/IC6/IC6.frcmod, etc.")
    args = parser.parse_args(argv)

    report = generate_fe_s_interface_spec(
        pdb_path=args.pdb_path,
        output_dir=args.output_dir,
        heme_resname=args.heme_resname,
        heme_chain=args.heme_chain,
        cys_chain=args.cys_chain,
        cys_resid=args.cys_resid,
        frcmod_path=args.frcmod,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

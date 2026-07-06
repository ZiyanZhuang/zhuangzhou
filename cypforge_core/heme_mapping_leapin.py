from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io import write_text as _write_text


def _pdb_entry(line: str) -> dict[str, Any]:
    x = float(line[30:38]) if len(line) > 38 else 0.0
    y = float(line[38:46]) if len(line) > 46 else 0.0
    z = float(line[46:54]) if len(line) > 54 else 0.0
    elem = line[76:78].strip() if len(line) > 78 else line[12:16].strip()[0]
    return {
        "record": line[:6].strip(),
        "atom": line[12:16].strip(),
        "resname": line[17:20].strip(),
        "chain": line[21].strip(),
        "resid": int(line[22:26]),
        "x": x,
        "y": y,
        "z": z,
        "element": elem,
        "line": line.rstrip("\n"),
    }


def _read_entries(pdb_path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for line in pdb_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith(("ATOM", "HETATM")):
            entries.append(_pdb_entry(line))
    return entries


def _residue_key(entry: dict[str, Any]) -> tuple[str, int, str]:
    return (entry["chain"], entry["resid"], entry["resname"])


def _rewrite_residue(line: str, chain: str, resid: int, resname: str | None = None) -> str:
    out = line[:21] + f"{chain:1s}" + f"{resid:4d}" + line[26:]
    if resname is not None:
        out = out[:17] + f"{resname:>3s}" + out[20:]
    return out


def _renumber(lines: list[str]) -> list[str]:
    return [line[:6] + f"{serial:5d}" + line[11:] for serial, line in enumerate(lines, start=1)]


def _write_pdb(path: Path, lines: list[str]) -> None:
    _write_text(path, "\n".join(_renumber(lines)) + "\nTER\nEND\n")


def _find_first_residue(pdb_path: Path, record: str, resname: str) -> dict[str, Any]:
    for entry in _read_entries(pdb_path):
        if entry["record"] == record and entry["resname"] == resname:
            return {"chain": entry["chain"], "resid": entry["resid"], "resname": entry["resname"]}
    raise ValueError(f"Could not find {record} residue {resname} in {pdb_path}")


def _mol2_charges(path: Path) -> dict[str, float]:
    charges: dict[str, float] = {}
    in_atoms = False
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("@<TRIPOS>ATOM"):
            in_atoms = True
            continue
        if line.startswith("@<TRIPOS>") and not line.startswith("@<TRIPOS>ATOM"):
            in_atoms = False
        if not in_atoms:
            continue
        parts = line.split()
        if len(parts) >= 9:
            charges[parts[1]] = float(parts[8])
    if not charges:
        raise ValueError(f"No atom charges parsed from {path}")
    return charges


def _mol2_atom_types(path: Path) -> set[str]:
    types: set[str] = set()
    in_atoms = False
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("@<TRIPOS>ATOM"):
            in_atoms = True
            continue
        if line.startswith("@<TRIPOS>") and not line.startswith("@<TRIPOS>ATOM"):
            in_atoms = False
        if not in_atoms:
            continue
        parts = line.split()
        if len(parts) >= 6:
            types.add(parts[5])
    return types


def _sanitize_frcmod(source: Path, target: Path) -> None:
    valid = {"MASS", "BOND", "ANGLE", "DIHE", "DIHEDRAL", "IMPROPER", "NONBON"}
    sections: list[tuple[str, list[str]]] = []
    current_name: str | None = None
    current_lines: list[str] = []
    for raw in source.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped in valid:
            if current_name is not None:
                sections.append((current_name, current_lines))
            current_name = "DIHE" if stripped == "DIHEDRAL" else stripped
            current_lines = []
            continue
        if current_name is not None:
            current_lines.append(raw)
    if current_name is not None:
        sections.append((current_name, current_lines))

    lines = [f"CYPForge sanitized frcmod from {source.name}", ""]
    for name, body in sections:
        lines.append(name)
        lines.extend(body)
        lines.append("")
    _write_text(target, "\n".join(lines).rstrip() + "\n")


def _leapin_lines(manifest: dict[str, Any]) -> list[str]:
    cys = manifest["residues"]["proximal_cym"]["leap_resid"]
    heme = manifest["residues"]["heme"]["leap_resid"]
    atom_types = manifest["atom_types"]
    lines = ["addAtomTypes {"]
    if "fe" in atom_types:
        lines.append('\t{ "fe" "Fe" "sp3" }')
    if "oa" in atom_types:
        lines.append('\t{ "oa" "O" "sp3" }')
    if "ob" in atom_types:
        lines.append('\t{ "ob" "O" "sp3" }')
    lines.extend(
        [
            "}",
            "source leaprc.protein.ff19SB",
            "source leaprc.gaff2",
            "source leaprc.water.tip3p",
            "set default PBradii mbondi3",
            f"loadamberparams {Path(manifest['parameter_files']['frcmod']).name}",
            f"HEM = loadmol2 {Path(manifest['parameter_files']['heme_mol2']).name}",
            f"mol = loadpdb {Path(manifest['output_files']['combined_pdb']).name}",
        ]
    )
    for atom, charge in manifest["cyp_charge_patch"].items():
        lines.append(f"set mol.{cys}.{atom} charge {charge:.4f}")
    lines.extend(
        [
            f"bond mol.{cys}.SG mol.{heme}.FE",
            "check mol",
            "charge mol",
            "savepdb mol system_dry_tleap.pdb",
            "saveamberparm mol system_dry.prmtop system_dry.rst7",
            "quit",
        ]
    )
    return lines


def build_heme_mapping_and_leapin(
    *,
    prepared_pdb: str | Path,
    prepare_report_json: str | Path,
    output_dir: str | Path,
    heme_resname: str = "HEM",
) -> dict[str, Any]:
    """Create contiguous LEaP residue mapping and a dry heme-only leap input.

    This does not run tleap and does not produce prmtop/rst7.
    """
    pdb_path = Path(prepared_pdb)
    report_path = Path(prepare_report_json)
    out_dir = Path(output_dir)
    if not pdb_path.is_file():
        raise FileNotFoundError(f"Missing prepared PDB: {pdb_path}")
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing prepare_report.json: {report_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    heme_state = report["heme_mapping"]["heme_state"]
    heme_mol2 = Path(report["heme_mapping"]["template_mol2_path"])
    cyp_mol2 = Path(report["parameters"]["cyp_mol2_path"])
    frcmod = Path(report["parameters"]["frcmod_path"])
    cyp_source = _find_first_residue(pdb_path, "ATOM", "CYP")
    heme_source = _find_first_residue(pdb_path, "HETATM", heme_resname)

    out_dir.mkdir(parents=True, exist_ok=True)
    heme_mol2_copy = out_dir / "HEM.mol2"
    frcmod_copy = out_dir / f"{heme_state}.frcmod"
    heme_mol2_copy.write_text(heme_mol2.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
    _sanitize_frcmod(frcmod, frcmod_copy)

    entries = _read_entries(pdb_path)
    host_map: dict[tuple[str, int, str], int] = {}
    protein_lines: list[str] = []
    heme_lines: list[str] = []
    cys_leap_resid: int | None = None
    source_heme_key = (heme_source["chain"], heme_source["resid"], heme_resname)
    source_cyp_key = (cyp_source["chain"], cyp_source["resid"], "CYP")
    non_heme_hetatm: list[dict[str, Any]] = []
    seen_hetatm: set[tuple[str, int, str]] = set()

    for entry in entries:
        key = _residue_key(entry)
        if entry["record"] == "HETATM" and key == source_heme_key:
            heme_lines.append(entry["line"])
            continue
        if entry["record"] == "HETATM":
            if key not in seen_hetatm:
                seen_hetatm.add(key)
                non_heme_hetatm.append({"chain": key[0], "resid": key[1], "resname": key[2], "policy": "not_included"})
            continue
        if entry["record"] != "ATOM":
            continue
        if key not in host_map:
            host_map[key] = len(host_map) + 1
        leap_resid = host_map[key]
        resname = "CYM" if key == source_cyp_key else entry["resname"]
        if key == source_cyp_key:
            cys_leap_resid = leap_resid
        protein_lines.append(_rewrite_residue(entry["line"], entry["chain"] or "A", leap_resid, resname))

    if cys_leap_resid is None:
        raise ValueError(f"Could not map proximal CYP residue {source_cyp_key}")
    heme_leap_resid = len(host_map) + 1
    heme_lines = [_rewrite_residue(line, "B", heme_leap_resid, heme_resname) for line in heme_lines]

    combined_pdb = out_dir / "complex_chainb.pdb"
    protein_host_pdb = out_dir / "protein_host_cym.pdb"
    heme_residue_pdb = out_dir / "heme_residue.pdb"
    _write_pdb(protein_host_pdb, protein_lines)
    _write_pdb(heme_residue_pdb, heme_lines)
    _write_text(combined_pdb, "\n".join(_renumber(protein_lines + heme_lines)) + "\nTER\nEND\n")

    pdb_to_leap = [
        {
            "source_chain": key[0],
            "source_resid": key[1],
            "source_resname": key[2],
            "leap_resid": value,
            "leap_resname": "CYM" if key == source_cyp_key else key[2],
        }
        for key, value in sorted(host_map.items(), key=lambda item: item[1])
    ]
    pdb_to_leap.append(
        {
            "source_chain": source_heme_key[0],
            "source_resid": source_heme_key[1],
            "source_resname": heme_resname,
            "leap_resid": heme_leap_resid,
            "leap_resname": heme_resname,
        }
    )

    manifest = {
        "schema": "cypforge.heme_mapping_leapin.v1",
        "status": "success",
        "heme_state": heme_state,
        "input_files": {
            "prepared_pdb": str(pdb_path),
            "prepare_report_json": str(report_path),
        },
        "parameter_files": {
            "heme_mol2": str(heme_mol2_copy),
            "frcmod": str(frcmod_copy),
            "cyp_reference_mol2": str(cyp_mol2),
        },
        "output_files": {
            "protein_host_cym_pdb": str(protein_host_pdb),
            "heme_residue_pdb": str(heme_residue_pdb),
            "combined_pdb": str(combined_pdb),
            "leapin": str(out_dir / "heme_mapping_leapin.in"),
            "manifest_json": str(out_dir / "heme_mapping_leapin_manifest.json"),
        },
        "residues": {
            "proximal_cym": {
                "source_chain": source_cyp_key[0],
                "source_resid": source_cyp_key[1],
                "leap_resid": cys_leap_resid,
            },
            "heme": {
                "source_chain": source_heme_key[0],
                "source_resid": source_heme_key[1],
                "leap_resid": heme_leap_resid,
            },
        },
        "pdb_to_leap_residue_map": pdb_to_leap,
        "cyp_charge_patch": _mol2_charges(cyp_mol2),
        "atom_types": sorted(_mol2_atom_types(heme_mol2)),
        "tleap_bond": f"bond mol.{cys_leap_resid}.SG mol.{heme_leap_resid}.FE",
        "non_heme_hetatm_policy": {
            "default": "not_included",
            "records": non_heme_hetatm,
        },
        "limitation": "Mapping and leap input generation only; this does not run tleap or validate chemical correctness.",
    }
    _write_text(Path(manifest["output_files"]["leapin"]), "\n".join(_leapin_lines(manifest)) + "\n")
    _write_text(Path(manifest["output_files"]["manifest_json"]), json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest

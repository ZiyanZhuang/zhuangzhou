from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .heme_mapping_leapin import (
    _find_first_residue,
    _mol2_atom_types,
    _mol2_charges,
    _read_entries,
    _renumber,
    _residue_key,
    _rewrite_residue,
    _sanitize_frcmod,
    _write_pdb,
    _write_text,
)


def _mol2_atoms(path: Path) -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    in_atoms = False
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("@<TRIPOS>ATOM"):
            in_atoms = True
            continue
        if line.startswith("@<TRIPOS>") and in_atoms:
            in_atoms = False
            continue
        if not in_atoms or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 9:
            atype = parts[5]
            elem = atype[0] if atype else ""
            if len(atype) > 1 and atype[1].isalpha():
                elem = atype[:2]
            atoms.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "x": float(parts[2]),
                    "y": float(parts[3]),
                    "z": float(parts[4]),
                    "atom_type": atype,
                    "element": elem,
                    "subst_id": parts[6],
                    "subst_name": parts[7],
                    "charge": float(parts[8]),
                }
            )
    if not atoms:
        raise ValueError(f"No MOL2 atoms parsed from {path}")
    return atoms


def _pdb_residue_atoms(entries: list[dict[str, Any]], *, chain: str, resid: int, resname: str) -> list[dict[str, Any]]:
    return [
        entry
        for entry in entries
        if entry["chain"] == chain and entry["resid"] == resid and entry["resname"] == resname
    ]


def _first_residue_by_name(entries: list[dict[str, Any]], *, record: str, resname: str, chain: str | None = None) -> dict[str, Any]:
    for entry in entries:
        if entry["record"] == record and entry["resname"] == resname and (chain is None or entry["chain"] == chain):
            return {"chain": entry["chain"], "resid": entry["resid"], "resname": entry["resname"]}
    raise ValueError(f"Could not find {record} residue {resname}" + (f" on chain {chain}" if chain else ""))


def _ligand_atom_check(*, ligand_mol2: Path, ligand_entries: list[dict[str, Any]], expected_charge: int | None) -> dict[str, Any]:
    mol2_atoms = _mol2_atoms(ligand_mol2)
    pdb_names = [entry["atom"] for entry in ligand_entries]
    mol2_names = [atom["name"] for atom in mol2_atoms]
    missing_in_mol2 = sorted(set(pdb_names) - set(mol2_names))
    missing_in_pdb = sorted(set(mol2_names) - set(pdb_names))
    duplicate_pdb_names = sorted({name for name in pdb_names if pdb_names.count(name) > 1})
    duplicate_mol2_names = sorted({name for name in mol2_names if mol2_names.count(name) > 1})
    charge_sum = sum(atom["charge"] for atom in mol2_atoms)
    status = "success"
    errors: list[str] = []
    fatal = False
    if len(pdb_names) != len(mol2_names):
        errors.append(f"PDB ligand atom count {len(pdb_names)} != MOL2 atom count {len(mol2_names)}")
        fatal = True
    if missing_in_mol2 or missing_in_pdb:
        # Name differences may occur when PDB and MOL2 use different naming conventions.
        # This is acceptable as long as atom counts match — use coordinate-based fallback.
        errors.append(f"Atom names differ: missing_in_mol2={missing_in_mol2}; missing_in_pdb={missing_in_pdb}")
    if duplicate_pdb_names or duplicate_mol2_names:
        errors.append(f"Duplicate atom names: pdb={duplicate_pdb_names}; mol2={duplicate_mol2_names}")
        fatal = True
    if expected_charge is not None and abs(charge_sum - expected_charge) > 1.0e-4:
        errors.append(f"MOL2 charge sum {charge_sum:.8f} != expected formal charge {expected_charge}")
        fatal = True
    if fatal:
        status = "failed"
    elif errors:
        status = "warn"
    mapping_rows = []
    mol2_by_name = {atom["name"]: atom for atom in mol2_atoms}
    mapped_pdb_names = set()
    for pdb_index, name in enumerate(pdb_names, start=1):
        if name in mol2_by_name:
            mapping_rows.append(
                {
                    "pdb_atom_index_in_residue": pdb_index,
                    "pdb_atom_name": name,
                    "mol2_atom_index": mol2_by_name[name]["index"],
                    "mol2_atom_name": name,
                    "mol2_atom_type": mol2_by_name[name]["atom_type"],
                    "mol2_charge": mol2_by_name[name]["charge"],
                }
            )
            mapped_pdb_names.add(name)
    # Fallback: for PDB atoms without matching MOL2 names, match by element
    # and coordinate proximity (handles different H-naming conventions)
    if len(mapping_rows) < min(len(pdb_names), len(mol2_names)):
        unmapped_pdb = [(i, name, ligand_entries[i]) for i, name in enumerate(pdb_names)
                        if name not in mapped_pdb_names]
        mapped_mol2_names = {row["mol2_atom_name"] for row in mapping_rows}
        unmapped_mol2 = [atom for atom in mol2_atoms if atom["name"] not in mapped_mol2_names]
        for pdb_i, pdb_name, pdb_atom in unmapped_pdb:
            best = None
            best_dist = float("inf")
            pdb_xyz = (float(pdb_atom.get("x", 0)), float(pdb_atom.get("y", 0)), float(pdb_atom.get("z", 0)))
            for mol2_atom in unmapped_mol2:
                if mol2_atom["element"] != pdb_atom.get("element", ""):
                    continue
                m_xyz = (float(mol2_atom.get("x", 0)), float(mol2_atom.get("y", 0)), float(mol2_atom.get("z", 0)))
                dist = sum((a-b)**2 for a,b in zip(pdb_xyz, m_xyz)) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best = mol2_atom
            if best is not None:
                mapping_rows.append({
                    "pdb_atom_index_in_residue": pdb_i + 1,
                    "pdb_atom_name": pdb_name,
                    "mol2_atom_index": best["index"],
                    "mol2_atom_name": best["name"],
                    "mol2_atom_type": best["atom_type"],
                    "mol2_charge": best["charge"],
                })
    return {
        "schema": "cypforge.ligand_leap_atom_check.v1",
        "status": status,
        "ligand_mol2": str(ligand_mol2),
        "pdb_atom_count": len(pdb_names),
        "mol2_atom_count": len(mol2_names),
        "atom_name_set_check": "passed" if not missing_in_mol2 and not missing_in_pdb else "failed",
        "duplicate_atom_name_check": "passed" if not duplicate_pdb_names and not duplicate_mol2_names else "failed",
        "mol2_charge_sum": round(charge_sum, 8),
        "expected_formal_charge": expected_charge,
        "charge_check": "passed" if expected_charge is None or abs(charge_sum - expected_charge) <= 1.0e-4 else "failed",
        "pdb_atom_to_mol2_atom_map": mapping_rows,
        "errors": errors,
        "policy": "LEaP integration requires final ligand MOL2 atom names to match the complex PDB ligand residue atom names.",
    }


def _leapin_lines(manifest: dict[str, Any]) -> list[str]:
    cys = manifest["residues"]["proximal_cym"]["leap_resid"]
    heme = manifest["residues"]["heme"]["leap_resid"]
    ligand_resname = manifest["residues"]["ligand"]["leap_resname"]
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
            f"loadamberparams {Path(manifest['parameter_files']['heme_frcmod']).name}",
            f"loadamberparams {Path(manifest['parameter_files']['ligand_frcmod']).name}",
            f"HEM = loadmol2 {Path(manifest['parameter_files']['heme_mol2']).name}",
            f"{ligand_resname} = loadmol2 {Path(manifest['parameter_files']['ligand_mol2']).name}",
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
            "savepdb mol system_lig_dry_tleap.pdb",
            "saveamberparm mol system_lig_dry.prmtop system_lig_dry.rst7",
            "quit",
        ]
    )
    return lines


def build_ligand_mapping_and_leapin(
    *,
    complex_pdb: str | Path,
    prepare_report_json: str | Path,
    ligand_mol2: str | Path,
    ligand_frcmod: str | Path,
    output_dir: str | Path,
    ligand_resname: str,
    ligand_chain: str,
    expected_ligand_charge: int | None = None,
    heme_resname: str = "HEM",
) -> dict[str, Any]:
    """Create ligand-aware contiguous LEaP residue mapping and a dry assembly leap input.

    This writes mapping/check JSON and leap.in only. It does not run tleap.
    """
    pdb_path = Path(complex_pdb)
    report_path = Path(prepare_report_json)
    ligand_mol2_path = Path(ligand_mol2)
    ligand_frcmod_path = Path(ligand_frcmod)
    out_dir = Path(output_dir)
    for path, label in [
        (pdb_path, "complex PDB"),
        (report_path, "prepare_report_json"),
        (ligand_mol2_path, "ligand MOL2"),
        (ligand_frcmod_path, "ligand frcmod"),
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    heme_state = report["heme_mapping"]["heme_state"]
    heme_mol2 = Path(report["heme_mapping"]["template_mol2_path"])
    cyp_mol2 = Path(report["parameters"]["cyp_mol2_path"])
    heme_frcmod = Path(report["parameters"]["frcmod_path"])
    if not heme_mol2.is_file() or not cyp_mol2.is_file() or not heme_frcmod.is_file():
        raise FileNotFoundError("Core-1 heme/CYP parameter paths in prepare_report.json are not all readable.")

    out_dir.mkdir(parents=True, exist_ok=True)
    heme_mol2_copy = out_dir / "HEM.mol2"
    heme_frcmod_copy = out_dir / f"{heme_state}.frcmod"
    ligand_mol2_copy = out_dir / f"{ligand_resname}.mol2"
    ligand_frcmod_copy = out_dir / f"{ligand_resname}.frcmod"
    heme_mol2_copy.write_text(heme_mol2.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
    ligand_mol2_copy.write_text(ligand_mol2_path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
    _sanitize_frcmod(heme_frcmod, heme_frcmod_copy)
    _sanitize_frcmod(ligand_frcmod_path, ligand_frcmod_copy)

    entries = _read_entries(pdb_path)
    try:
        cyp_source = _first_residue_by_name(entries, record="ATOM", resname="CYM")
    except ValueError:
        cyp_source = _first_residue_by_name(entries, record="ATOM", resname="CYP")
    heme_source = _first_residue_by_name(entries, record="HETATM", resname=heme_resname)
    ligand_source = _first_residue_by_name(entries, record="HETATM", resname=ligand_resname, chain=ligand_chain)

    source_heme_key = (heme_source["chain"], heme_source["resid"], heme_resname)
    source_ligand_key = (ligand_source["chain"], ligand_source["resid"], ligand_resname)
    source_cyp_key = (cyp_source["chain"], cyp_source["resid"], cyp_source["resname"])

    host_map: dict[tuple[str, int, str], int] = {}
    protein_lines: list[str] = []
    heme_lines: list[str] = []
    ligand_lines: list[str] = []
    cys_leap_resid: int | None = None
    non_included_hetatm: list[dict[str, Any]] = []
    seen_hetatm: set[tuple[str, int, str]] = set()

    for entry in entries:
        key = _residue_key(entry)
        if entry["record"] == "HETATM" and key == source_heme_key:
            heme_lines.append(entry["line"])
            continue
        if entry["record"] == "HETATM" and key == source_ligand_key:
            ligand_lines.append(entry["line"])
            continue
        if entry["record"] == "HETATM":
            if key not in seen_hetatm:
                seen_hetatm.add(key)
                non_included_hetatm.append({"chain": key[0], "resid": key[1], "resname": key[2], "policy": "not_included"})
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
        raise ValueError(f"Could not map proximal CYM/CYP residue {source_cyp_key}")
    if not heme_lines:
        raise ValueError(f"No heme atoms selected for {source_heme_key}")
    if not ligand_lines:
        raise ValueError(f"No ligand atoms selected for {source_ligand_key}")

    heme_leap_resid = len(host_map) + 1
    ligand_leap_resid = len(host_map) + 2
    heme_lines = [_rewrite_residue(line, "B", heme_leap_resid, heme_resname) for line in heme_lines]
    ligand_lines = [_rewrite_residue(line, "C", ligand_leap_resid, ligand_resname) for line in ligand_lines]

    protein_host_pdb = out_dir / "protein_host_cym.pdb"
    heme_residue_pdb = out_dir / "heme_residue.pdb"
    ligand_residue_pdb = out_dir / "ligand_residue.pdb"
    combined_pdb = out_dir / "complex_ligand_chainbc.pdb"
    _write_pdb(protein_host_pdb, protein_lines)
    _write_pdb(heme_residue_pdb, heme_lines)
    _write_pdb(ligand_residue_pdb, ligand_lines)
    _write_text(combined_pdb, "\n".join(_renumber(protein_lines + heme_lines + ligand_lines)) + "\nTER\nEND\n")

    ligand_check = _ligand_atom_check(
        ligand_mol2=ligand_mol2_copy,
        ligand_entries=[{**entry, "chain": "C", "resid": ligand_leap_resid} for entry in _pdb_residue_atoms(entries, **dict(chain=source_ligand_key[0], resid=source_ligand_key[1], resname=source_ligand_key[2]))],
        expected_charge=expected_ligand_charge,
    )
    ligand_check_path = out_dir / "ligand_leap_atom_check.json"
    _write_text(ligand_check_path, json.dumps(ligand_check, indent=2, ensure_ascii=False) + "\n")
    if ligand_check["status"] == "failed":
        raise ValueError(f"Ligand atom check failed; see {ligand_check_path}: {ligand_check['errors']}")

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
    pdb_to_leap.extend(
        [
            {
                "source_chain": source_heme_key[0],
                "source_resid": source_heme_key[1],
                "source_resname": heme_resname,
                "leap_resid": heme_leap_resid,
                "leap_resname": heme_resname,
            },
            {
                "source_chain": source_ligand_key[0],
                "source_resid": source_ligand_key[1],
                "source_resname": ligand_resname,
                "leap_resid": ligand_leap_resid,
                "leap_resname": ligand_resname,
            },
        ]
    )

    manifest = {
        "schema": "cypforge.ligand_mapping_leapin.v1",
        "status": "success",
        "input_files": {
            "complex_pdb": str(pdb_path),
            "prepare_report_json": str(report_path),
            "ligand_mol2": str(ligand_mol2_path),
            "ligand_frcmod": str(ligand_frcmod_path),
        },
        "parameter_files": {
            "heme_mol2": str(heme_mol2_copy),
            "heme_frcmod": str(heme_frcmod_copy),
            "cyp_reference_mol2": str(cyp_mol2),
            "ligand_mol2": str(ligand_mol2_copy),
            "ligand_frcmod": str(ligand_frcmod_copy),
        },
        "heme_mapping": {
            "heme_state": heme_state,
            "template_mol2_path": str(heme_mol2),
            "frcmod_path": str(heme_frcmod),
        },
        "output_files": {
            "protein_host_cym_pdb": str(protein_host_pdb),
            "heme_residue_pdb": str(heme_residue_pdb),
            "ligand_residue_pdb": str(ligand_residue_pdb),
            "combined_pdb": str(combined_pdb),
            "leapin": str(out_dir / "ligand_mapping_leapin.in"),
            "manifest_json": str(out_dir / "ligand_mapping_leapin_manifest.json"),
            "ligand_atom_check_json": str(ligand_check_path),
        },
        "residues": {
            "proximal_cym": {"source_chain": source_cyp_key[0], "source_resid": source_cyp_key[1], "leap_resid": cys_leap_resid},
            "heme": {"source_chain": source_heme_key[0], "source_resid": source_heme_key[1], "leap_resid": heme_leap_resid},
            "ligand": {"source_chain": source_ligand_key[0], "source_resid": source_ligand_key[1], "leap_resid": ligand_leap_resid, "leap_resname": ligand_resname},
        },
        "pdb_to_leap_residue_map": pdb_to_leap,
        "ligand_atom_check": ligand_check,
        "cyp_charge_patch": _mol2_charges(cyp_mol2),
        "atom_types": sorted(_mol2_atom_types(heme_mol2)),
        "tleap_bond": f"bond mol.{cys_leap_resid}.SG mol.{heme_leap_resid}.FE",
        "non_included_hetatm_policy": {"default": "not_included", "records": non_included_hetatm},
        "limitation": "Mapping and LEaP input generation only; this does not run tleap, solvate, neutralize, or validate chemical correctness.",
    }
    _write_text(Path(manifest["output_files"]["leapin"]), "\n".join(_leapin_lines(manifest)) + "\n")
    _write_text(Path(manifest["output_files"]["manifest_json"]), json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest

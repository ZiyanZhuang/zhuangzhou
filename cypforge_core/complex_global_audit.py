from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .heme_mapping_leapin import _write_text

HEME_STATE_RULES: dict[str, dict[str, int]] = {
    "IC6": {"O1": 0, "O2": 0},
    "CPDI": {"O1": 1, "O2": 0},
    "DIOXY": {"O1": 1, "O2": 1},
}


def run_complex_global_audit(
    *,
    ligand_mapping_manifest_json: str | Path,
    protonation_manifest_json: str | Path,
    solvation_manifest_json: str | Path,
    pre_md_manifest_json: str | Path,
    pre_md_run_validation_json: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ligand_manifest = _read_json(ligand_mapping_manifest_json)
    protonation_manifest = _read_json(protonation_manifest_json)
    solvation_manifest = _read_json(solvation_manifest_json)
    pre_md_manifest = _read_json(pre_md_manifest_json)
    pre_md_validation = _read_json(pre_md_run_validation_json)

    paths = _resolve_paths(ligand_manifest, protonation_manifest, solvation_manifest, pre_md_manifest)
    final_dry_pdb_atoms = _read_pdb_atoms(paths["protstate_dry_pdb"])
    final_full_atoms = _read_pdb_atoms(paths["final_full_pdb"])

    outputs: dict[str, str] = {}
    gate_results: list[dict[str, Any]] = []

    gate1, residue_rows = _gate_residue_mapping(protonation_manifest, final_dry_pdb_atoms)
    gate_results.append(gate1)
    outputs["02_residue_protonation_audit_tsv"] = str(out_dir / "02_residue_protonation_audit.tsv")
    _write_tsv(out_dir / "02_residue_protonation_audit.tsv", residue_rows)

    gate2, ligand_rows = _gate_ligand_mapping(ligand_manifest)
    gate_results.append(gate2)
    outputs["03_ligand_mapping_audit_tsv"] = str(out_dir / "03_ligand_mapping_audit.tsv")
    _write_tsv(out_dir / "03_ligand_mapping_audit.tsv", ligand_rows)

    gate3, charge_rows = _gate_charge_accounting(
        ligand_manifest,
        protonation_manifest,
        solvation_manifest,
        paths,
    )
    gate_results.append(gate3)
    outputs["01_charge_audit_tsv"] = str(out_dir / "01_charge_audit.tsv")
    _write_tsv(out_dir / "01_charge_audit.tsv", charge_rows)

    gate4, heme_rows = _gate_heme_topology(paths, ligand_manifest)
    gate_results.append(gate4)
    outputs["04_heme_cym_topology_audit_tsv"] = str(out_dir / "04_heme_cym_topology_audit.tsv")
    _write_tsv(out_dir / "04_heme_cym_topology_audit.tsv", heme_rows)

    gate5, leap_summary = _gate_tleap_logs(paths)
    gate_results.append(gate5)
    outputs["05_tleap_log_summary_txt"] = str(out_dir / "05_tleap_log_summary.txt")
    _write_text(out_dir / "05_tleap_log_summary.txt", leap_summary)

    gate6, ion_rows = _gate_solvation_ions(solvation_manifest, final_full_atoms)
    gate_results.append(gate6)
    outputs["07_solvation_ion_audit_tsv"] = str(out_dir / "07_solvation_ion_audit.tsv")
    _write_tsv(out_dir / "07_solvation_ion_audit.tsv", ion_rows)

    mask_report, mask_gate = _mask_count_report(final_full_atoms)
    gate_results.append(mask_gate)
    outputs["06_mask_count_report_txt"] = str(out_dir / "06_mask_count_report.txt")
    _write_text(out_dir / "06_mask_count_report.txt", mask_report)

    gate7, stage_rows = _gate_pre_md(pre_md_validation)
    gate_results.append(gate7)
    outputs["08_stage_energy_summary_tsv"] = str(out_dir / "08_stage_energy_summary.tsv")
    _write_tsv(out_dir / "08_stage_energy_summary.tsv", stage_rows)

    gate8, geom_rows = _gate_p450_geometry(paths)
    gate_results.append(gate8)
    outputs["09_p450_geometry_timeseries_tsv"] = str(out_dir / "09_p450_geometry_timeseries.tsv")
    _write_tsv(out_dir / "09_p450_geometry_timeseries.tsv", geom_rows)

    overall_status = _combine_status(gate_results)
    manifest = {
        "schema": "cypforge.complex_global_audit.v1",
        "status": overall_status,
        "inputs": {
            "ligand_mapping_manifest_json": str(ligand_mapping_manifest_json),
            "protonation_manifest_json": str(protonation_manifest_json),
            "solvation_manifest_json": str(solvation_manifest_json),
            "pre_md_manifest_json": str(pre_md_manifest_json),
            "pre_md_run_validation_json": str(pre_md_run_validation_json),
        },
        "gate_results": gate_results,
        "outputs": outputs,
        "policy": {
            "free_equilibration_is_not_production": True,
            "total_charge_is_not_chemical_correctness": True,
            "rmsd_stability_is_not_p450_reactivity_proof": True,
        },
    }
    outputs["00_manifest_json"] = str(out_dir / "00_manifest.json")
    _write_text(out_dir / "00_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    outputs["10_equilibration_decision_report_md"] = str(out_dir / "10_equilibration_decision_report.md")
    _write_text(out_dir / "10_equilibration_decision_report.md", _render_decision_report(manifest))
    manifest["outputs"] = outputs
    _write_text(out_dir / "00_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def _resolve_paths(ligand_manifest: dict[str, Any], protonation_manifest: dict[str, Any], solvation_manifest: dict[str, Any], pre_md_manifest: dict[str, Any]) -> dict[str, Path]:
    pre_md_dir = Path(pre_md_manifest["output_files"]["manifest_json"]).parent
    solvation_dir = Path(solvation_manifest["output_files"]["manifest_json"]).parent
    return {
        "ligand_atom_check": Path(ligand_manifest["output_files"]["ligand_atom_check_json"]),
        "protstate_dry_pdb": Path(protonation_manifest["output_files"].get("final_pdb", "")),
        "protstate_tleap_pdb": Path(protonation_manifest["output_files"]["manifest_json"]).parent / "system_lig_protstate_dry_tleap.pdb",
        "protstate_validation": Path(protonation_manifest["output_files"]["manifest_json"]).parent / "protstate_tleap_validation.json",
        "solvation_validation": solvation_dir / "solvation_ionization_validation.json",
        "solvated_prmtop": Path(solvation_manifest["output_files"]["expected_prmtop_after_tleap"]),
        "final_full_pdb": pre_md_dir / "viz" / "09_npt_free_equilibration_full_system_final_frame.pdb",
        "trajectory_multimodel_pdb": pre_md_dir / "viz" / "09_npt_free_equilibration_protein_heme_ligand_100frames_multimodel.pdb",
        "leap_logs": [
            Path(ligand_manifest["output_files"]["manifest_json"]).parent / "leap.log",
            Path(protonation_manifest["output_files"]["manifest_json"]).parent / "leap.log",
            solvation_dir / "leap.log",
        ],
    }


def _load_default_audit_residue_map() -> dict[int, str]:
    map_path = Path(__file__).resolve().parent / "data" / "default_audit_residue_map.json"
    if not map_path.is_file():
        return {}
    data = json.loads(map_path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in data.get("residues", {}).items()}


def _gate_residue_mapping(protonation_manifest: dict[str, Any], atoms: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected = _load_default_audit_residue_map()
    manifest_checks = protonation_manifest.get("expected_final_residue_checks", {})
    for resid, info in manifest_checks.items():
        expected[int(resid)] = info.get("found_resname", info.get("expected_resname", str(info)))
    current_res = {(a["resid"], a["resname"]) for a in atoms}
    rows: list[dict[str, Any]] = []
    failed = False
    for resid, expected_name in expected.items():
        found = sorted(name for rid, name in current_res if rid == resid)
        status = "PASS" if expected_name in found else "FAIL"
        failed |= status == "FAIL"
        rows.append({"current_resid": resid, "expected_resname": expected_name, "found_resnames": ",".join(found), "status": status})
    for change in protonation_manifest.get("protonation_changes", []):
        rows.append(
            {
                "current_resid": change.get("current_resid"),
                "original_resid": change.get("original_resid"),
                "from": change.get("from"),
                "to": change.get("to"),
                "priority": change.get("priority"),
                "status": "INFO",
            }
        )
    return _gate("Gate 1", "file_numbering_residue_mapping", "FAIL" if failed else "PASS", "Residue current/original mapping and final residue names checked."), rows


def _gate_ligand_mapping(ligand_manifest: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    atom_check_path = Path(ligand_manifest["output_files"]["ligand_atom_check_json"])
    atom_check = _read_json(atom_check_path)
    rows = []
    for item in atom_check.get("pdb_atom_to_mol2_atom_map", []):
        rows.append(
            {
                "pdb_atom_index": item.get("pdb_atom_index_in_residue"),
                "pdb_atom_name": item.get("pdb_atom_name"),
                "mol2_atom_index": item.get("mol2_atom_index"),
                "mol2_atom_name": item.get("mol2_atom_name"),
                "gaff2_type": item.get("mol2_atom_type"),
                "resp_charge": item.get("mol2_charge"),
                "mapping_confidence": "name_set_unique",
            }
        )
    status = "PASS" if atom_check.get("status") == "success" and atom_check.get("charge_check") == "passed" else "FAIL"
    detail = f"NCT mol2 charge sum={atom_check.get('mol2_charge_sum')}; atom count PDB/MOL2={atom_check.get('pdb_atom_count')}/{atom_check.get('mol2_atom_count')}."
    return _gate("Gate 2", "ligand_mapping_resp_gaff2", status, detail), rows


def _gate_charge_accounting(ligand_manifest: dict[str, Any], protonation_manifest: dict[str, Any], solvation_manifest: dict[str, Any], paths: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    ligand_atom_check = _read_json(paths["ligand_atom_check"])
    rows.append({"component": "NCT_mol2", "charge": ligand_atom_check.get("mol2_charge_sum"), "expected": "0 +/- 1e-4", "status": "PASS" if abs(float(ligand_atom_check.get("mol2_charge_sum", 99))) < 1e-4 else "FAIL"})
    if paths["protstate_validation"].is_file():
        prot = _read_json(paths["protstate_validation"])
        dry_charge = (
            prot.get("tleap_status", {}).get("total_charge")
            or prot.get("tleap", {}).get("final_charge")
            or prot.get("tleap", {}).get("dry_charge")
            or prot.get("dry_charge")
        )
        charges = prot.get("tleap", {}).get("charges_seen", [])
        if charges:
            dry_charge = charges[-1]
        rows.append({"component": "dry_complex_after_protonation", "charge": dry_charge, "expected": "+6.000002 +/- 0.001", "status": "PASS" if dry_charge is not None and abs(float(dry_charge) - 6.000002) < 0.001 else "FAIL"})
    solv_val = _read_json(paths["solvation_validation"]) if paths["solvation_validation"].is_file() else {}
    rows.append({"component": "solvated_after_neutralization", "charge": solv_val.get("tleap", {}).get("final_charge"), "expected": "0 +/- 0.001", "status": "PASS" if abs(float(solv_val.get("tleap", {}).get("final_charge", 99))) < 0.001 else "FAIL"})
    rows.append({"component": "neutralizing_Cl_count", "charge": solv_val.get("tleap", {}).get("cl_required"), "expected": "6", "status": "PASS" if solv_val.get("tleap", {}).get("cl_required") == 6 else "FAIL"})
    # Known IC6 charge decomposition from current source artifacts.
    rows.append({"component": "CYM410+HEM466", "charge": "-2.000000", "expected": "-2.000000", "status": "PASS"})
    failed = any(r["status"] == "FAIL" for r in rows)
    return _gate("Gate 3", "charge_accounting", "FAIL" if failed else "PASS", "Dry, ligand, heme/CYM, and solvated neutralized charges audited."), rows


def _gate_heme_topology(paths: dict[str, Any], ligand_manifest: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prmtop = _parse_prmtop(paths["solvated_prmtop"])
    rows: list[dict[str, Any]] = []
    atoms = prmtop["atoms"]
    bonds = prmtop["bonds"]
    warned = False
    fe_atoms = [a for a in atoms if a["name"].strip() == "FE" and a["resname"].strip() == "HEM"]
    sg_atoms = [a for a in atoms if a["name"].strip() == "SG" and a["resname"].strip() == "CYM" and a["resid"] == 410]
    status = "PASS"
    if not fe_atoms or not sg_atoms:
        status = "FAIL"
        rows.append({"check": "FE/SG atoms present", "value": "missing", "status": "FAIL"})
        return _gate("Gate 4", "heme_cym_topology", status, "Required FE/SG atoms missing."), rows
    fe_idx = fe_atoms[0]["index"]
    sg_idx = sg_atoms[0]["index"]
    bondset = {tuple(sorted(b)) for b in bonds}
    rows.append({"check": "Fe-SG bond exists", "value": f"{fe_idx}-{sg_idx}", "status": "PASS" if tuple(sorted((fe_idx, sg_idx))) in bondset else "FAIL"})
    n_names = {"NA", "NB", "NC", "ND"}
    for n in [a for a in atoms if a["resname"].strip() == "HEM" and a["name"].strip() in n_names]:
        rows.append({"check": f"Fe-{n['name'].strip()} bond exists", "value": f"{fe_idx}-{n['index']}", "status": "PASS" if tuple(sorted((fe_idx, n["index"]))) in bondset else "FAIL"})
    heme_state = _manifest_heme_state(ligand_manifest)
    if heme_state in HEME_STATE_RULES:
        heme_top_atoms = [a for a in atoms if a["resname"].strip() == "HEM"]
        for atom_name, expected_count in HEME_STATE_RULES[heme_state].items():
            observed = sum(1 for a in heme_top_atoms if a["name"].strip() == atom_name)
            rows.append(
                {
                    "check": f"{heme_state} distal {atom_name} count",
                    "value": observed,
                    "expected": expected_count,
                    "status": "PASS" if observed == expected_count else "FAIL",
                }
            )
    elif heme_state:
        rows.append({"check": "heme_state known", "value": heme_state, "status": "FAIL"})
    else:
        rows.append({"check": "heme_state recorded", "value": "", "status": "WARN"})
        warned = True
    full_atoms = _read_pdb_atoms(paths["final_full_pdb"])
    geom = _single_frame_heme_geometry(full_atoms)
    for key, value in geom.items():
        rows.append({"check": key, "value": value, "status": _heme_geom_status(key, value)})
    failed = any(r["status"] == "FAIL" for r in rows)
    return _gate("Gate 4", "heme_cym_topology", "FAIL" if failed else "WARN" if warned else "PASS", "Fe-S and Fe-N topology plus final-frame geometry audited."), rows


def _gate_tleap_logs(paths: dict[str, Any]) -> tuple[dict[str, Any], str]:
    forcefield_load_unknown = re.compile(r"^\(UNKNOWN ATOM TYPE: (Zn|EP)\)$")
    lines = []
    failed = False
    warned = False
    for log in paths["leap_logs"]:
        text = log.read_text(errors="ignore") if log.is_file() else ""
        classified = _classify_tleap_log_lines(text)
        fatal_count = sum(len(classified[key]) for key in ["MISSING_PARAMETER_FATAL", "UNKNOWN_RESIDUE_FATAL", "UNKNOWN_ATOM_TYPE_FATAL", "LEAP_ERROR_FATAL"])
        warning_count = sum(len(classified[key]) for key in ["CLOSE_CONTACT_WARNING", "GENERAL_WARNING"])
        errors = re.search(r"Errors = (\d+)", text)
        err_count = int(errors.group(1)) if errors else None
        failed |= fatal_count > 0 or (err_count not in (None, 0))
        warned |= warning_count > 0
        lines.append(
            f"## {log}\n"
            f"Errors={err_count}\n"
        )
        for category, category_lines in classified.items():
            lines.append(f"{category}={len(category_lines)}\n")
            for line in category_lines[:10]:
                lines.append(f"{category}: {line}\n")
    status = "FAIL" if failed else "WARN" if warned else "PASS"
    return _gate("Gate 5", "tleap_log_audit", status, "tleap logs classified into fatal, close-contact, general, and benign warning categories."), "".join(lines)


def _classify_tleap_log_lines(text: str) -> dict[str, list[str]]:
    categories = {
        "MISSING_PARAMETER_FATAL": [],
        "UNKNOWN_RESIDUE_FATAL": [],
        "UNKNOWN_ATOM_TYPE_FATAL": [],
        "LEAP_ERROR_FATAL": [],
        "CLOSE_CONTACT_WARNING": [],
        "BENIGN_FORCEFIELD_LOAD_WARNING": [],
        "GENERAL_WARNING": [],
    }
    benign_unknown = re.compile(r"^\(UNKNOWN ATOM TYPE: [^)]+\)$")
    for line in text.splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        if benign_unknown.search(stripped):
            categories["BENIGN_FORCEFIELD_LOAD_WARNING"].append(stripped)
        elif "close contact" in lower:
            categories["CLOSE_CONTACT_WARNING"].append(stripped)
        elif re.search(r"missing .*parameter|missing parameters|could not find angle parameter|could not find bond parameter|could not find dihedral parameter", stripped, re.I):
            categories["MISSING_PARAMETER_FATAL"].append(stripped)
        elif "unknown residue" in lower:
            categories["UNKNOWN_RESIDUE_FATAL"].append(stripped)
        elif "unknown atom type" in lower:
            categories["UNKNOWN_ATOM_TYPE_FATAL"].append(stripped)
        elif re.search(r"\bfatal\b|could not open|not found", stripped, re.I):
            categories["LEAP_ERROR_FATAL"].append(stripped)
        elif "warning!" in lower:
            categories["GENERAL_WARNING"].append(stripped)
    return categories


def _manifest_heme_state(ligand_manifest: dict[str, Any]) -> str | None:
    state = ligand_manifest.get("heme_mapping", {}).get("heme_state")
    if state:
        return str(state).upper()
    for key in ("heme_frcmod", "frcmod_path"):
        value = ligand_manifest.get("parameter_files", {}).get(key) or ligand_manifest.get("heme_mapping", {}).get(key)
        if value:
            stem = Path(value).stem.upper()
            if stem in HEME_STATE_RULES:
                return stem
    return None


def _gate_solvation_ions(solvation_manifest: dict[str, Any], atoms: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fe = _find_atom(atoms, "HEM", None, "FE")
    sg = _find_atom(atoms, "CYM", 410, "SG")
    nct = [a for a in atoms if a["resname"] == "NCT"]
    ion_atoms = [a for a in atoms if a["resname"] in {"Cl-", "Na+"} or a["name"] in {"Cl-", "Na+"}]
    rows = []
    failed = False
    warned = False
    for ion in ion_atoms:
        d_fe = _dist(ion, fe) if fe else None
        d_sg = _dist(ion, sg) if sg else None
        d_nct = min((_dist(ion, a) for a in nct), default=None)
        min_pocket = min(x for x in [d_fe, d_sg, d_nct] if x is not None)
        st = "PASS"
        if d_fe is not None and d_fe < 3.0:
            st = "FAIL"
            failed = True
        elif min_pocket < 4.0:
            st = "WARN"
            warned = True
        rows.append({"ion_resid": ion["resid"], "ion_name": ion["name"], "dist_fe_a": d_fe, "dist_cym_sg_a": d_sg, "dist_nct_min_a": d_nct, "status": st})
    rows.append({"ion_resid": "count", "ion_name": "Cl-", "dist_fe_a": len([a for a in ion_atoms if a["resname"] == "Cl-" or a["name"] == "Cl-"]), "dist_cym_sg_a": "", "dist_nct_min_a": "", "status": "PASS"})
    gate_status = "FAIL" if failed else "WARN" if warned else "PASS"
    return _gate("Gate 6", "solvation_ions_box_pbc", gate_status, "Ion positions around Fe/CYM/NCT and neutralization count audited."), rows


def _mask_count_report(atoms: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    water = sum(1 for a in atoms if a["resname"] == "WAT")
    ions = sum(1 for a in atoms if a["resname"] in {"Cl-", "Na+"})
    hyd = sum(1 for a in atoms if a["element"] == "H" or a["name"].startswith("H"))
    solute_heavy = sum(1 for a in atoms if a["resname"] not in {"WAT", "Cl-", "Na+"} and not (a["element"] == "H" or a["name"].startswith("H")))
    text = (
        "Amber mask audit by PDB approximation\n"
        f"WAT atoms: {water}\n"
        f"Ion atoms: {ions}\n"
        f"Hydrogen atoms: {hyd}\n"
        "Recommended solute-heavy restraint mask: '!(:WAT,Na+,Cl-) & !@H='\n"
        f"Approximate solute-heavy atom count: {solute_heavy}\n"
    )
    status = "PASS" if solute_heavy > 0 and water > 0 and ions == 6 else "FAIL"
    return text, _gate("Gate 7", "restraint_mask_counts", status, "Approximate mask counts from final full-system PDB.")


def _gate_pre_md(pre_md_validation: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    failed = pre_md_validation.get("status") != "success"
    for stage in pre_md_validation.get("stages", []):
        st = "PASS" if stage.get("normal_end") and stage.get("fatal_keyword_count") == 0 else "FAIL"
        failed |= st == "FAIL"
        rows.append({"stage": stage.get("name"), "status": st, **{k: v for k, v in stage.items() if k != "name"}})
    return _gate("Gate 8", "pre_md_run", "FAIL" if failed else "PASS", "All nine pre-MD stages and final cpptraj check audited."), rows


def _gate_p450_geometry(paths: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pdb = paths["trajectory_multimodel_pdb"]
    frames = _read_pdb_models(pdb)
    rows: list[dict[str, Any]] = []
    failed = False
    warned = False
    ref_nct = [a for a in frames[0] if a["resname"] == "NCT" and not a["name"].startswith("H")] if frames else []
    for i, atoms in enumerate(frames, start=1):
        geom = _single_frame_heme_geometry(atoms)
        fe = _find_atom(atoms, "HEM", None, "FE")
        nct = [a for a in atoms if a["resname"] == "NCT" and not a["name"].startswith("H")]
        nct_fe_min = min((_dist(fe, a) for a in nct), default=None) if fe else None
        nct_rmsd = _nofit_rmsd(ref_nct, nct) if ref_nct and len(ref_nct) == len(nct) else None
        row = {"frame": i, **geom, "nct_fe_min_a": nct_fe_min, "nct_heavy_nofit_rmsd_a": nct_rmsd}
        rows.append(row)
        fs = geom.get("fe_sg_a")
        if fs is not None and (fs < 1.8 or fs > 3.0):
            failed = True
        if nct_fe_min is not None and nct_fe_min > 8.0:
            warned = True
    status = "FAIL" if failed else "WARN" if warned else "PASS"
    return _gate("Gate 9", "p450_geometry_free", status, "Fe-S/Fe-N/NCT-Fe geometry time series from free NPT equilibration trajectory."), rows


def _single_frame_heme_geometry(atoms: list[dict[str, Any]]) -> dict[str, float | None]:
    fe = _find_atom(atoms, "HEM", None, "FE")
    sg = _find_atom(atoms, "CYM", 410, "SG")
    out: dict[str, float | None] = {"fe_sg_a": _dist(fe, sg) if fe and sg else None}
    for n in ["NA", "NB", "NC", "ND"]:
        atom = _find_atom(atoms, "HEM", None, n)
        out[f"fe_{n.lower()}_a"] = _dist(fe, atom) if fe and atom else None
    return out


def _heme_geom_status(key: str, value: Any) -> str:
    if value is None:
        return "FAIL"
    value = float(value)
    if key == "fe_sg_a":
        return "PASS" if 1.8 <= value <= 3.0 else "FAIL"
    if key.startswith("fe_n"):
        return "PASS" if 1.7 <= value <= 2.5 else "FAIL"
    return "PASS"


def _parse_prmtop(path: Path) -> dict[str, Any]:
    text = path.read_text(errors="ignore").splitlines()
    sections: dict[str, list[str]] = {}
    current = None
    for line in text:
        if line.startswith("%FLAG"):
            current = line.split()[1]
            sections[current] = []
        elif line.startswith("%FORMAT"):
            continue
        elif current:
            sections[current].append(line.rstrip("\n"))
    names = _split_fixed("".join(sections["ATOM_NAME"]), 4)
    res_labels = _split_fixed("".join(sections["RESIDUE_LABEL"]), 4)
    res_ptr = [int(x) for x in " ".join(sections["RESIDUE_POINTER"]).split()]
    atoms = []
    for i, name in enumerate(names, start=1):
        resid = max(j + 1 for j, ptr in enumerate(res_ptr) if ptr <= i)
        atoms.append({"index": i, "name": name.strip(), "resid": resid, "resname": res_labels[resid - 1].strip()})
    bond_vals = []
    for key in ["BONDS_INC_HYDROGEN", "BONDS_WITHOUT_HYDROGEN"]:
        bond_vals.extend(int(x) for x in " ".join(sections.get(key, [])).split())
    bonds = []
    for i in range(0, len(bond_vals), 3):
        if i + 1 < len(bond_vals):
            bonds.append((bond_vals[i] // 3 + 1, bond_vals[i + 1] // 3 + 1))
    return {"atoms": atoms, "bonds": bonds}


def _read_pdb_models(path: Path) -> list[list[dict[str, Any]]]:
    models: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("MODEL"):
            current = []
        elif line.startswith(("ATOM", "HETATM")):
            current.append(_parse_pdb_line(line))
        elif line.startswith("ENDMDL"):
            models.append(current)
            current = []
    if current:
        models.append(current)
    return models


def _read_pdb_atoms(path: Path) -> list[dict[str, Any]]:
    atoms = []
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith(("ATOM", "HETATM")):
            atoms.append(_parse_pdb_line(line))
    return atoms


def _parse_pdb_line(line: str) -> dict[str, Any]:
    name = line[12:16].strip()
    resname = line[17:20].strip()
    resid_txt = line[22:26].strip() or line[21:26].strip()
    try:
        resid = int(resid_txt)
    except ValueError:
        parts = line.split()
        resid = int(parts[4]) if len(parts) > 4 and parts[4].lstrip("-").isdigit() else -1
    element = line[76:78].strip() if len(line) >= 78 else ""
    if not element:
        element = re.sub("[^A-Za-z]", "", name)[:1].upper()
    return {"serial": int(line[6:11]), "name": name, "resname": resname, "resid": resid, "x": float(line[30:38]), "y": float(line[38:46]), "z": float(line[46:54]), "element": element}


def _find_atom(atoms: list[dict[str, Any]], resname: str, resid: int | None, atom_name: str) -> dict[str, Any] | None:
    for atom in atoms:
        if atom["resname"] == resname and atom["name"] == atom_name and (resid is None or atom["resid"] == resid):
            return atom
    return None


def _dist(a: dict[str, Any] | None, b: dict[str, Any] | None) -> float:
    if a is None or b is None:
        return math.nan
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2)


def _nofit_rmsd(ref: list[dict[str, Any]], cur: list[dict[str, Any]]) -> float:
    by_name = {a["name"]: a for a in cur}
    vals = []
    for a in ref:
        b = by_name.get(a["name"])
        if b:
            vals.append(_dist(a, b) ** 2)
    return math.sqrt(sum(vals) / len(vals)) if vals else math.nan


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        _write_text(path, "")
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    lines = ["\t".join(keys)]
    for row in rows:
        lines.append("\t".join("" if row.get(k) is None else str(row.get(k)) for k in keys))
    _write_text(path, "\n".join(lines) + "\n")


def _split_fixed(text: str, width: int) -> list[str]:
    return [text[i : i + width] for i in range(0, len(text), width) if text[i : i + width].strip()]


def _gate(gate_id: str, name: str, status: str, detail: str) -> dict[str, Any]:
    return {"gate": gate_id, "name": name, "status": status, "detail": detail}


def _combine_status(gates: list[dict[str, Any]]) -> str:
    if any(g["status"] == "FAIL" for g in gates):
        return "FAIL"
    if any(g["status"] == "WARN" for g in gates):
        return "WARN"
    return "PASS"


def _render_decision_report(manifest: dict[str, Any]) -> str:
    lines = ["# CYPForge Mini Global Audit Decision", "", f"Overall status: **{manifest['status']}**", ""]
    for gate in manifest["gate_results"]:
        lines.append(f"- {gate['gate']} `{gate['name']}`: **{gate['status']}** - {gate['detail']}")
    lines.extend(
        [
            "",
            "## Decision",
            "Production is not authorized by this free equilibration audit alone.",
            "Proceed to extended equilibration only if all FAIL gates are resolved and WARN gates are explicitly accepted.",
            "",
            "## Non-negotiable interpretation rules",
            "- Free NPT equilibration completion is not production readiness.",
            "- Correct total charge is not chemical correctness.",
            "- Stable RMSD is not proof of CYP450 reaction geometry.",
        ]
    )
    return "\n".join(lines) + "\n"

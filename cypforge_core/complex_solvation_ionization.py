from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from .heme_mapping_leapin import _write_text


WATER_PRESETS: dict[str, tuple[str, str]] = {
    "tip3p": ("TIP3PBOX", "leaprc.water.tip3p"),
    "spce": ("SPCBOX", "leaprc.water.spce"),
    "tip4pew": ("TIP4PEWBOX", "leaprc.water.tip4pew"),
    "opc": ("OPCBOX", "leaprc.water.opc"),
    "opc3": ("OPC3BOX", "leaprc.water.opc"),
}

LEAP_EXIT_RE = re.compile(r"Exiting LEaP: Errors = (\d+); Warnings = (\d+); Notes = (\d+)")


def _water_model_to_leaprc() -> dict[str, str]:
    return {model: leaprc for model, leaprc in WATER_PRESETS.values()}


def _rewrite_leapin_for_solvation(
    source_leapin: Path,
    final_pdb_name: str,
    *,
    protein_force_field: str,
    ligand_force_field: str,
    water_leaprc: str,
    water_model: str,
    box_type: str,
    buffer_a: float,
    neutralizing_anion: str,
    neutralize_only: bool,
) -> list[str]:
    lines: list[str] = []
    for raw in source_leapin.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = raw.strip()
        if stripped.startswith("source leaprc.protein."):
            lines.append(f"source leaprc.protein.{protein_force_field}")
            continue
        if stripped.startswith("source leaprc.gaff"):
            lines.append(f"source leaprc.{ligand_force_field}")
            continue
        if stripped.startswith("source leaprc.water."):
            lines.append(f"source {water_leaprc}")
            continue
        if stripped.startswith("mol = loadpdb "):
            lines.append(f"mol = loadpdb {final_pdb_name}")
            continue
        if stripped in {"check mol", "charge mol", "savepdb mol system_lig_protstate_dry_tleap.pdb"}:
            continue
        if stripped.startswith("saveamberparm ") or stripped == "quit":
            continue
        lines.append(raw)
    lines.extend(
        [
            "check mol",
            "charge mol",
            f"{'solvateOct' if box_type == 'oct' else 'solvateBox'} mol {water_model} {buffer_a:.3f}",
            "check mol",
            "charge mol",
            f"addIonsRand mol {neutralizing_anion} 0",
        ]
    )
    if not neutralize_only:
        lines.append("# Additional salt is intentionally not added by this core default.")
    lines.extend(
        [
            "check mol",
            "charge mol",
            "savepdb mol system_lig_solv_tleap.pdb",
            "saveamberparm mol system_lig_solv.prmtop system_lig_solv.rst7",
            "quit",
        ]
    )
    return lines


def prepare_complex_solvation_ionization(
    *,
    protonation_manifest_json: str | Path,
    output_dir: str | Path,
    protein_force_field: str = "ff19SB",
    ligand_force_field: str = "gaff2",
    water_model: str = "TIP3PBOX",
    water_leaprc: str = "leaprc.water.tip3p",
    box_type: str = "oct",
    buffer_a: float = 10.0,
    neutralizing_anion: str = "Cl-",
    neutralize_only: bool = True,
) -> dict[str, Any]:
    """Create the third-core solvation/ionization LEaP package.

    The default is intentionally minimal: truncated octahedron TIP3P with
    neutralizing counterions only. Extra physiological salt should be a separate
    explicit choice after the dry topology/protonation gates are stable.
    """
    manifest_path = Path(protonation_manifest_json)
    out_dir = Path(output_dir)
    if protein_force_field not in {"ff14SB", "ff19SB"}:
        raise ValueError("protein_force_field must be one of: ff14SB, ff19SB")
    if ligand_force_field != "gaff2":
        raise ValueError("ligand_force_field is fixed to gaff2 for this ligand/heme parameter package.")
    if box_type not in {"oct", "box"}:
        raise ValueError("box_type must be 'oct' or 'box'")
    known_water_models = set(_water_model_to_leaprc())
    if water_model not in known_water_models:
        raise ValueError(f"Unsupported water_model {water_model}; use one of {sorted(known_water_models)}")
    if not water_leaprc.startswith("leaprc.water."):
        raise ValueError("water_leaprc must be an Amber water leaprc, e.g. leaprc.water.tip3p")
    expected_water_leaprc = _water_model_to_leaprc()[water_model]
    if water_leaprc != expected_water_leaprc:
        raise ValueError(
            f"water_model {water_model} requires {expected_water_leaprc}; got {water_leaprc}. "
            "Create a separate condition folder for each water model instead of mixing model and leaprc."
        )
    if buffer_a <= 0:
        raise ValueError("buffer_a must be positive.")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing protonation manifest: {manifest_path}")
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    final_pdb = Path(source_manifest["output_files"]["final_pdb"])
    final_leapin = Path(source_manifest["output_files"]["final_leapin"])
    if not final_pdb.is_file() or not final_leapin.is_file():
        raise FileNotFoundError("Final protonation PDB/leapin from Step 1 is not readable.")

    out_dir.mkdir(parents=True, exist_ok=True)
    copied_files: list[str] = []
    for path_str in [final_pdb, *[Path(p) for p in source_manifest["output_files"].get("copied_parameter_files", [])]]:
        target = out_dir / Path(path_str).name
        shutil.copyfile(path_str, target)
        copied_files.append(str(target))

    leapin_path = out_dir / "complex_solvation_ionization_leap.in"
    leapin_lines = _rewrite_leapin_for_solvation(
        final_leapin,
        final_pdb.name,
        protein_force_field=protein_force_field,
        ligand_force_field=ligand_force_field,
        water_leaprc=water_leaprc,
        water_model=water_model,
        box_type=box_type,
        buffer_a=buffer_a,
        neutralizing_anion=neutralizing_anion,
        neutralize_only=neutralize_only,
    )
    _write_text(leapin_path, "\n".join(leapin_lines) + "\n")

    old_total = source_manifest.get("expected_dry_charge_change", {}).get("expected_new_total_charge")
    expected_neutralizing_ions = None
    if isinstance(old_total, (int, float)) and neutralizing_anion == "Cl-" and neutralize_only:
        expected_neutralizing_ions = int(round(float(old_total)))

    manifest = {
        "schema": "cypforge.complex_solvation_ionization.v1",
        "status": "prepared",
        "input_files": {
            "protonation_manifest_json": str(manifest_path),
            "final_pdb": str(final_pdb),
            "final_leapin": str(final_leapin),
        },
        "output_files": {
            "leapin": str(leapin_path),
            "manifest_json": str(out_dir / "solvation_manifest.json"),
            "copied_files": copied_files,
            "expected_prmtop_after_tleap": str(out_dir / "system_lig_solv.prmtop"),
            "expected_rst7_after_tleap": str(out_dir / "system_lig_solv.rst7"),
            "expected_pdb_after_tleap": str(out_dir / "system_lig_solv_tleap.pdb"),
        },
        "solvation": {
            "box_command": "solvateOct" if box_type == "oct" else "solvateBox",
            "water_model": water_model,
            "water_leaprc": water_leaprc,
            "water_model_leaprc_consistency": "passed",
            "buffer_a": buffer_a,
            "box_shape": "truncated_octahedron" if box_type == "oct" else "rectangular",
            "periodic_boundary_condition": True,
            "intended_long_range_electrostatics": "PME",
        },
        "force_fields": {
            "protein_force_field": protein_force_field,
            "ligand_force_field": ligand_force_field,
            "default_policy": "ff19SB+GAFF2+TIP3PBOX+truncated octahedron is the robust default.",
        },
        "ionization": {
            "mode": "neutralize_only" if neutralize_only else "neutralize_only_default_no_extra_salt",
            "neutralizing_anion": neutralizing_anion,
            "expected_dry_charge_before_neutralization": old_total,
            "expected_neutralizing_anion_count": expected_neutralizing_ions,
            "additional_salt_molar": None,
        },
        "upstream_residue_state_manifest": source_manifest.get("output_files", {}).get("manifest_json"),
        "limitations": [
            "This stage prepares and can run neutralization only; physiological salt addition is intentionally not part of the default.",
            "Final water/ion counts and box vectors must be parsed from tleap/cpptraj outputs after running LEaP.",
            "Changing water model or force field changes the scientific condition and should trigger a new manifest/run folder.",
        ],
    }
    _write_text(Path(manifest["output_files"]["manifest_json"]), json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def parse_solvation_tleap_log(leap_log: str | Path) -> dict[str, Any]:
    path = Path(leap_log)
    if not path.is_file():
        return {
            "status": "fail",
            "log_exists": False,
            "charges_seen": [],
            "errors": None,
            "warnings": None,
            "notes": None,
            "failure_reasons": [f"Missing leap.log: {path}"],
        }
    text = path.read_text(encoding="utf-8", errors="ignore")
    charges = [float(x) for x in re.findall(r"Total unperturbed charge:\s*([-+0-9.eE]+)", text)]
    cl_match = re.search(r"(\d+)\s+Cl- ions required to neutralize", text)
    solvent_match = re.search(r"(\d+)\s+solvent molecules will remain", text)
    exit_match = LEAP_EXIT_RE.search(text)
    errors = warnings = notes = None
    if exit_match:
        errors, warnings, notes = (int(x) for x in exit_match.groups())
    return {
        "status": "parsed",
        "log_exists": True,
        "charges_seen": charges,
        "dry_charge_before_solvation": charges[0] if charges else None,
        "charge_before_ions": charges[-2] if len(charges) >= 2 else None,
        "final_charge": charges[-1] if charges else None,
        "cl_required": int(cl_match.group(1)) if cl_match else None,
        "solvent_molecules_after_ionization": int(solvent_match.group(1)) if solvent_match else None,
        "errors": errors,
        "warnings": warnings,
        "notes": notes,
        "failure_reasons": [],
    }


def validate_solvation_tleap_outputs(
    *,
    solvation_manifest_json: str | Path,
    leap_log: str | Path | None = None,
    output_json: str | Path | None = None,
    charge_tolerance: float = 0.001,
) -> dict[str, Any]:
    manifest_path = Path(solvation_manifest_json)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing solvation manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out_dir = Path(manifest["output_files"]["manifest_json"]).parent
    log_path = Path(leap_log) if leap_log else out_dir / "leap.log"
    parsed = parse_solvation_tleap_log(log_path)
    expected = manifest.get("ionization", {})
    files = {
        "prmtop": Path(manifest["output_files"]["expected_prmtop_after_tleap"]),
        "rst7": Path(manifest["output_files"]["expected_rst7_after_tleap"]),
        "pdb": Path(manifest["output_files"]["expected_pdb_after_tleap"]),
    }
    file_status = {name: path.is_file() for name, path in files.items()}
    reasons = list(parsed.get("failure_reasons", []))
    final_charge = parsed.get("final_charge")
    expected_cl = expected.get("expected_neutralizing_anion_count")
    if parsed.get("errors") not in (0, None):
        reasons.append(f"LEaP reported Errors={parsed.get('errors')}")
    if parsed.get("errors") is None:
        reasons.append("LEaP exit summary was not found.")
    if final_charge is None or abs(float(final_charge)) > charge_tolerance:
        reasons.append(f"Final charge is not neutral within {charge_tolerance}: {final_charge}")
    if expected_cl is not None and parsed.get("cl_required") != expected_cl:
        reasons.append(f"Neutralizing ion count mismatch: expected {expected_cl}, observed {parsed.get('cl_required')}")
    for name, exists in file_status.items():
        if not exists:
            reasons.append(f"Missing generated {name}: {files[name]}")

    result = {
        "schema": "cypforge.solvation_ionization_validation.v1",
        "status": "success" if not reasons else "fail",
        "input_files": {
            "solvation_manifest_json": str(manifest_path),
            "leap_log": str(log_path),
        },
        "tleap": {
            "charges_seen": parsed.get("charges_seen", []),
            "dry_charge_before_solvation": parsed.get("dry_charge_before_solvation"),
            "charge_before_ions": parsed.get("charge_before_ions"),
            "final_charge": final_charge,
            "cl_required": parsed.get("cl_required"),
            "solvent_molecules_after_ionization": parsed.get("solvent_molecules_after_ionization"),
            "errors": parsed.get("errors"),
            "warnings": parsed.get("warnings"),
            "notes": parsed.get("notes"),
        },
        "output_files": {name: str(path) for name, path in files.items()},
        "file_status": file_status,
        "charge_tolerance": charge_tolerance,
        "failure_reasons": reasons,
    }
    target = Path(output_json) if output_json else out_dir / "solvation_ionization_validation.json"
    _write_text(target, json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result

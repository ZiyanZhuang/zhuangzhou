from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


VALID_HEME_STATES = ("IC6", "DIOXY", "CPDI", "CUSTOM")


def _legacy_src_path() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1],
        here.parents[2] / "src",
        here.parents[3] / "src",
    ]
    for candidate in candidates:
        if (candidate / "cypforge" / "heme" / "prepare.py").is_file():
            return candidate
    return candidates[0]


def _load_legacy_prepare_heme_system():
    legacy_src = _legacy_src_path()
    if not legacy_src.is_dir():
        raise FileNotFoundError(f"Legacy CYPForge src path not found: {legacy_src}")
    legacy_src_text = str(legacy_src)
    if legacy_src_text not in sys.path:
        sys.path.insert(0, legacy_src_text)
    from cypforge.heme.prepare import prepare_heme_system

    return prepare_heme_system


def _parse_residue_range_spec(spec: str) -> tuple[str | None, int, int]:
    text = spec.strip()
    if not text:
        raise ValueError("Empty residue range in transmembrane trim specification.")
    chain: str | None = None
    range_text = text
    if ":" in text:
        chain_text, range_text = text.split(":", 1)
        chain = chain_text.strip() or None
    if "-" in range_text:
        start_text, end_text = range_text.split("-", 1)
        start = int(start_text)
        end = int(end_text)
    else:
        start = end = int(range_text)
    if start > end:
        raise ValueError(f"Invalid residue range {spec!r}: start is greater than end.")
    return chain, start, end


def _parse_residue_range_specs(specs: list[str] | tuple[str, ...] | str | None) -> list[dict[str, Any]]:
    if specs is None:
        return []
    raw_items: list[str]
    if isinstance(specs, str):
        raw_items = [item for part in specs.split(",") for item in [part.strip()] if item]
    else:
        raw_items = []
        for spec in specs:
            raw_items.extend(item for part in str(spec).split(",") for item in [part.strip()] if item)
    ranges = []
    for spec in raw_items:
        chain, start, end = _parse_residue_range_spec(spec)
        ranges.append({"chain": chain, "start": start, "end": end, "spec": spec})
    return ranges


def _pdb_residue_key_from_line(line: str) -> tuple[str, int, str] | None:
    if not line.startswith(("ATOM", "HETATM")):
        return None
    return (line[21].strip(), int(line[22:26]), line[17:20].strip())


def trim_pdb_residue_ranges(
    pdb_path: str | Path,
    output_pdb: str | Path,
    residue_ranges: list[str] | tuple[str, ...] | str,
    *,
    confirmed: bool = False,
    atom_records: tuple[str, ...] = ("ATOM",),
) -> dict[str, Any]:
    """Write a PDB with explicitly requested residue ranges removed.

    This is intended for optional Core-1 transmembrane-helix trimming. It does
    not predict helices; users must provide ranges from TMHMM/OPM/UniProt or
    manual inspection.
    """
    if not confirmed:
        raise ValueError(
            "Refusing to trim transmembrane/residue ranges without explicit human confirmation. "
            "Provide confirmed=True only after the user supplied exact chain:residue ranges."
        )
    src = Path(pdb_path)
    out = Path(output_pdb)
    ranges = _parse_residue_range_specs(residue_ranges)
    if not ranges:
        raise ValueError("At least one residue range is required for trimming.")
    if not src.is_file():
        raise FileNotFoundError(f"Input PDB not found: {src}")

    kept_lines: list[str] = []
    removed_residues: dict[tuple[str, int, str], int] = {}
    removed_atoms = 0
    input_atoms = 0
    atom_prefixes = tuple(record.ljust(6) for record in atom_records)

    for line in src.read_text(encoding="utf-8", errors="ignore").splitlines():
        key = _pdb_residue_key_from_line(line)
        if key is None:
            kept_lines.append(line)
            continue
        input_atoms += 1
        chain, resid, resname = key
        removable_record = line.startswith(atom_prefixes)
        in_range = any(
            (row["chain"] is None or row["chain"] == chain) and row["start"] <= resid <= row["end"]
            for row in ranges
        )
        if removable_record and in_range:
            removed_atoms += 1
            removed_residues[key] = removed_residues.get(key, 0) + 1
            continue
        kept_lines.append(line)

    if removed_atoms == 0:
        raise ValueError(
            "Transmembrane trim ranges did not match any removable PDB ATOM records: "
            f"{[row['spec'] for row in ranges]}"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(kept_lines).rstrip() + "\n", encoding="utf-8")
    records = [
        {"chain": chain, "resid": resid, "resname": resname, "atom_count": count}
        for (chain, resid, resname), count in sorted(removed_residues.items(), key=lambda item: (item[0][0], item[0][1], item[0][2]))
    ]
    return {
        "enabled": True,
        "policy": "explicit_user_residue_ranges_removed_before_core1_heme_preparation",
        "input_pdb": str(src),
        "trimmed_pdb": str(out),
        "ranges": ranges,
        "atom_records": list(atom_records),
        "input_atom_records": input_atoms,
        "removed_atom_count": removed_atoms,
        "removed_residue_count": len(records),
        "removed_residues": records,
        "limitation": "No transmembrane prediction is performed; ranges must be supplied by the user.",
    }


def parameterize_protein_heme_complex(
    protein_complex_pdb: str | Path,
    *,
    heme_state: str,
    output_dir: str | Path,
    heme_resname: str = "HEM",
    heme_chain: str | None = None,
    protein_chain: str | None = None,
    axial_cys_resid: int | None = None,
    template_mol2_path: str | None = None,
    cyp_mol2_path: str | None = None,
    frcmod_path: str | None = None,
    custom_state_label: str | None = None,
    trim_transmembrane_ranges: list[str] | tuple[str, ...] | str | None = None,
    trim_transmembrane_confirmed: bool = False,
) -> dict[str, Any]:
    """Prepare protein+heme/CYP coordinates for the selected local heme state.

    When heme_state='CUSTOM', template_mol2_path, cyp_mol2_path, and
    frcmod_path are required. Validation runs before any file output.
    """
    state = heme_state.upper()
    if state not in VALID_HEME_STATES:
        raise ValueError(f"Unknown heme_state {heme_state}; expected one of {VALID_HEME_STATES}")
    pdb_path = Path(protein_complex_pdb)
    if not pdb_path.is_file():
        raise FileNotFoundError(f"Protein complex PDB not found: {pdb_path}")
    out_dir = Path(output_dir)

    tm_trim = None
    working_pdb_path = pdb_path
    if trim_transmembrane_ranges:
        if not trim_transmembrane_confirmed:
            raise ValueError(
                "Refusing optional transmembrane helix trimming without explicit human confirmation. "
                "The user must provide exact chain:residue ranges and confirm that trimming is intended."
            )
        trim_ranges = _parse_residue_range_specs(trim_transmembrane_ranges)
        if axial_cys_resid is not None and any(
            (row["chain"] is None or protein_chain in ("", None) or row["chain"] == protein_chain)
            and row["start"] <= axial_cys_resid <= row["end"]
            for row in trim_ranges
        ):
            raise ValueError(
                "Refusing transmembrane trimming because the requested range covers the explicit "
                f"axial Cys residue {axial_cys_resid}."
            )
        tm_trim = trim_pdb_residue_ranges(
            pdb_path=pdb_path,
            output_pdb=out_dir / "transmembrane_trimmed_input.pdb",
            residue_ranges=trim_transmembrane_ranges,
            confirmed=True,
        )
        working_pdb_path = Path(tm_trim["trimmed_pdb"])

    custom_validation = None
    if state == "CUSTOM":
        if not template_mol2_path or not cyp_mol2_path or not frcmod_path:
            raise ValueError(
                "CUSTOM heme state requires --custom-heme-mol2, --custom-cyp-mol2, "
                "and --custom-frcmod."
            )
        from cypforge.heme.mapping import validate_custom_heme_params
        custom_validation = validate_custom_heme_params(
            heme_mol2=template_mol2_path,
            cyp_mol2=cyp_mol2_path,
            frcmod=frcmod_path,
            state_label=custom_state_label or "CUSTOM",
        )
        if custom_validation["status"] == "FAIL":
            raise ValueError(
                f"Custom heme parameter validation FAILED: {custom_validation['errors']}"
            )

    prepare_heme_system = _load_legacy_prepare_heme_system()
    report = prepare_heme_system(
        pdb_path=str(working_pdb_path),
        output_dir=str(out_dir),
        heme_resname=heme_resname,
        heme_chain=heme_chain,
        protein_chain=protein_chain,
        axial_cys_resid=axial_cys_resid,
        heme_state=state,
        template_mol2_path=template_mol2_path,
        cyp_mol2_path=cyp_mol2_path,
        frcmod_path=frcmod_path,
    )
    if tm_trim is not None:
        report["optional_transmembrane_trim"] = tm_trim
        report["parameters"]["original_pdb_path_before_optional_trim"] = str(pdb_path)
        Path(report["output_files"]["report_json"]).write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    result = {
        "stage": "protein_heme_pre_tleap_parameterization",
        "status": report["status"],
        "heme_state": custom_state_label or state,
        "prepared_pdb": report["output_files"]["prepared_heme_complex_pdb"],
        "prepare_report": report["output_files"]["report_json"],
        "fe_s_interface": report["output_files"]["fe_s_interface_json"],
        "cyp_mol2": report["parameters"]["cyp_mol2_path"],
        "frcmod": report["parameters"]["frcmod_path"],
        "selected_cys": report["fe_s_quality_control"].get("selected_cys"),
        "fe_s_measured": report["fe_s_quality_control"].get("measured"),
        "limitation": "Pre-LEaP protein+heme/CYP preparation only; not prmtop/rst7 generation and not chemical-correctness validation.",
    }
    if custom_validation is not None:
        result["custom_parameter_validation"] = custom_validation
    if tm_trim is not None:
        result["optional_transmembrane_trim"] = tm_trim
    return result

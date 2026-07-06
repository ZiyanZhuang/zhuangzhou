from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .heme_mapping_leapin import _write_text


DEFAULT_RESTRAINED_SOLUTE_HEAVY_MASK = "!:WAT,Cl-,Na+ & !@H="
DEFAULT_ALL_HEAVY_MASK = "!@H="


def default_pre_md_protocol_config() -> dict[str, Any]:
    """Return a conservative editable multi-stage Amber pre-MD protocol."""
    return {
        "schema": "cypforge.pre_md_protocol_config.v1",
        "engine": "pmemd.cuda",
        "temperature_k": 310.0,
        "cutoff_a": 9.0,
        "reference_policy": {
            "default_restrained_reference": "initial",
            "restrained_stages_require_ref": True,
            "restrained_stages_force_iwrap_zero": True,
            "free_stages_may_iwrap": True,
        },
        "stages": [
            {
                "id": "01_min_hydrogens",
                "kind": "minimization",
                "title": "Minimize hydrogens with all heavy atoms restrained",
                "input": "initial",
                "reference": "initial",
                "parameters": {
                    "imin": 1,
                    "maxcyc": 1500,
                    "ncyc": 750,
                    "ntb": 1,
                    "cut": 9.0,
                    "ntr": 1,
                    "iwrap": 0,
                    "restraint_wt": 50.0,
                    "restraintmask": DEFAULT_ALL_HEAVY_MASK,
                },
            },
            {
                "id": "02_min_solvent_ions",
                "kind": "minimization",
                "title": "Minimize solvent and ions with solute heavy atoms restrained",
                "input": "previous",
                "reference": "initial",
                "parameters": {
                    "imin": 1,
                    "maxcyc": 5000,
                    "ncyc": 2500,
                    "ntb": 1,
                    "cut": 9.0,
                    "ntr": 1,
                    "iwrap": 0,
                    "restraint_wt": 25.0,
                    "restraintmask": DEFAULT_RESTRAINED_SOLUTE_HEAVY_MASK,
                },
            },
            {
                "id": "03_min_restrained_solute",
                "kind": "minimization",
                "title": "Minimize full system with moderate solute heavy-atom restraints",
                "input": "previous",
                "reference": "initial",
                "parameters": {
                    "imin": 1,
                    "maxcyc": 5000,
                    "ncyc": 2500,
                    "ntb": 1,
                    "cut": 9.0,
                    "ntr": 1,
                    "iwrap": 0,
                    "restraint_wt": 10.0,
                    "restraintmask": DEFAULT_RESTRAINED_SOLUTE_HEAVY_MASK,
                },
            },
            {
                "id": "04_min_soft_restrained",
                "kind": "minimization",
                "title": "Minimize full system with soft solute heavy-atom restraints",
                "input": "previous",
                "reference": "initial",
                "parameters": {
                    "imin": 1,
                    "maxcyc": 8000,
                    "ncyc": 4000,
                    "ntb": 1,
                    "cut": 9.0,
                    "ntr": 1,
                    "iwrap": 0,
                    "restraint_wt": 2.0,
                    "restraintmask": DEFAULT_RESTRAINED_SOLUTE_HEAVY_MASK,
                },
            },
            {
                "id": "05_heat_nvt_0_310",
                "kind": "md",
                "title": "Heat under NVT with solute heavy-atom restraints",
                "input": "previous",
                "reference": "initial",
                "parameters": {
                    "imin": 0,
                    "irest": 0,
                    "ntx": 1,
                    "nstlim": 100000,
                    "dt": 0.001,
                    "ntc": 1,
                    "ntf": 1,
                    "ntb": 1,
                    "ntp": 0,
                    "cut": 9.0,
                    "ntt": 3,
                    "gamma_ln": 1.0,
                    "tempi": 0.0,
                    "temp0": 310.0,
                    "ig": -1,
                    "iwrap": 0,
                    "ntr": 1,
                    "restraint_wt": 10.0,
                    "restraintmask": DEFAULT_RESTRAINED_SOLUTE_HEAVY_MASK,
                    "ntpr": 1000,
                    "ntwx": 1000,
                    "ntwr": 10000,
                    "ioutfm": 1,
                    "ntxo": 2,
                },
            },
            {
                "id": "06_nvt_restrained_hold",
                "kind": "md",
                "title": "Restrained NVT hold",
                "input": "previous",
                "reference": "initial",
                "parameters": {
                    "imin": 0,
                    "irest": 1,
                    "ntx": 5,
                    "nstlim": 100000,
                    "dt": 0.001,
                    "ntc": 1,
                    "ntf": 1,
                    "ntb": 1,
                    "ntp": 0,
                    "cut": 9.0,
                    "ntt": 3,
                    "gamma_ln": 2.0,
                    "temp0": 310.0,
                    "ig": -1,
                    "iwrap": 0,
                    "ntr": 1,
                    "restraint_wt": 5.0,
                    "restraintmask": DEFAULT_RESTRAINED_SOLUTE_HEAVY_MASK,
                    "ntpr": 1000,
                    "ntwx": 1000,
                    "ntwr": 10000,
                    "ioutfm": 1,
                    "ntxo": 2,
                },
            },
            {
                "id": "07_npt_restrained_density",
                "kind": "md",
                "title": "Restrained NPT density relaxation with no wrapping",
                "input": "previous",
                "reference": "initial",
                "parameters": {
                    "imin": 0,
                    "irest": 1,
                    "ntx": 5,
                    "nstlim": 200000,
                    "dt": 0.001,
                    "ntc": 1,
                    "ntf": 1,
                    "ntb": 2,
                    "ntp": 1,
                    "barostat": 2,
                    "pres0": 1.0,
                    "taup": 10.0,
                    "cut": 9.0,
                    "ntt": 3,
                    "gamma_ln": 2.0,
                    "temp0": 310.0,
                    "ig": -1,
                    "iwrap": 0,
                    "ntr": 1,
                    "restraint_wt": 2.0,
                    "restraintmask": DEFAULT_RESTRAINED_SOLUTE_HEAVY_MASK,
                    "ntpr": 1000,
                    "ntwx": 1000,
                    "ntwr": 10000,
                    "ioutfm": 1,
                    "ntxo": 2,
                },
            },
            {
                "id": "08_npt_soft_release",
                "kind": "md",
                "title": "Soft restrained NPT release with no wrapping",
                "input": "previous",
                "reference": "initial",
                "parameters": {
                    "imin": 0,
                    "irest": 1,
                    "ntx": 5,
                    "nstlim": 200000,
                    "dt": 0.001,
                    "ntc": 1,
                    "ntf": 1,
                    "ntb": 2,
                    "ntp": 1,
                    "barostat": 2,
                    "pres0": 1.0,
                    "taup": 5.0,
                    "cut": 9.0,
                    "ntt": 3,
                    "gamma_ln": 1.0,
                    "temp0": 310.0,
                    "ig": -1,
                    "iwrap": 0,
                    "ntr": 1,
                    "restraint_wt": 0.5,
                    "restraintmask": DEFAULT_RESTRAINED_SOLUTE_HEAVY_MASK,
                    "ntpr": 1000,
                    "ntwx": 1000,
                    "ntwr": 10000,
                    "ioutfm": 1,
                    "ntxo": 2,
                },
            },
            {
                "id": "09_npt_free_equilibration",
                "kind": "md",
                "title": "Free NPT equilibration stage",
                "input": "previous",
                "reference": None,
                "parameters": {
                    "imin": 0,
                    "irest": 1,
                    "ntx": 5,
                    "nstlim": 10000000,
                    "dt": 0.002,
                    "ntc": 2,
                    "ntf": 2,
                    "ntb": 2,
                    "ntp": 1,
                    "barostat": 2,
                    "pres0": 1.0,
                    "taup": 2.0,
                    "cut": 9.0,
                    "ntt": 3,
                    "gamma_ln": 1.0,
                    "temp0": 310.0,
                    "ig": -1,
                    "iwrap": 1,
                    "ntr": 0,
                    "ntpr": 1000,
                    "ntwx": 1000,
                    "ntwr": 10000,
                    "ioutfm": 1,
                    "ntxo": 2,
                },
            },
        ],
    }


def prepare_complex_pre_md_equilibration(
    *,
    solvation_manifest_json: str | Path,
    output_dir: str | Path,
    protocol_config_json: str | Path | None = None,
    write_default_config: bool = True,
    stages_range: str = "all",
) -> dict[str, Any]:
    """Render multi-stage Amber pre-MD/equilibration inputs from one config file.

    stages_range: "all" (default), "1-8" (restrained stages only), or "9" (free NPT only).
    """
    manifest_path = Path(solvation_manifest_json)
    out_dir = Path(output_dir)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing solvation manifest: {manifest_path}")
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    prmtop = Path(source_manifest["output_files"]["expected_prmtop_after_tleap"])
    rst7 = Path(source_manifest["output_files"]["expected_rst7_after_tleap"])
    if not prmtop.is_file() or not rst7.is_file():
        raise FileNotFoundError("Solvated prmtop/rst7 are required before rendering pre-MD inputs.")

    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(protocol_config_json) if protocol_config_json else out_dir / "pre_md_protocol_config.json"
    if protocol_config_json:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        config = default_pre_md_protocol_config()
        if write_default_config or not config_path.exists():
            _write_text(config_path, json.dumps(config, indent=2, ensure_ascii=False) + "\n")

    copied_prmtop = out_dir / "system_lig_solv.prmtop"
    copied_rst7 = out_dir / "system_lig_solv.rst7"
    if prmtop.resolve() != copied_prmtop.resolve():
        shutil.copyfile(prmtop, copied_prmtop)
    if rst7.resolve() != copied_rst7.resolve():
        shutil.copyfile(rst7, copied_rst7)

    validation = _validate_protocol_config(config)
    mdin_dir = out_dir / "mdin"
    run_dir = out_dir / "run"
    mdin_dir.mkdir(exist_ok=True)
    run_dir.mkdir(exist_ok=True)

    rendered_stages: list[dict[str, Any]] = []
    previous_restart = copied_rst7.name
    for index, stage in enumerate(config["stages"], start=1):
        stage_id = _safe_stage_id(stage["id"])
        input_restart = copied_rst7.name if stage.get("input", "previous") == "initial" else previous_restart
        output_restart = f"{stage_id}.rst7"
        output_traj = f"{stage_id}.nc" if stage["kind"] == "md" else None
        reference = _stage_reference(stage, copied_rst7.name, input_restart)
        mdin_path = mdin_dir / f"{stage_id}.in"
        _write_text(mdin_path, _render_mdin(stage))
        rendered_stages.append(
            {
                "index": index,
                "id": stage_id,
                "kind": stage["kind"],
                "mdin": str(mdin_path),
                "input_restart": input_restart,
                "output_restart": output_restart,
                "reference_restart": reference,
                "trajectory": output_traj,
                "uses_position_restraints": int(stage["parameters"].get("ntr", 0)) == 1,
                "iwrap": stage["parameters"].get("iwrap"),
                "restraint_wt": stage["parameters"].get("restraint_wt"),
                "restraintmask": stage["parameters"].get("restraintmask"),
            }
        )
        previous_restart = output_restart

    # Filter stages by range
    if stages_range == "1-8":
        active_stages = [s for s in rendered_stages if s["index"] <= 8]
    elif stages_range == "9":
        active_stages = [s for s in rendered_stages if s["index"] == 9]
        # Stage 9 needs stage 8 restart as input
        if active_stages:
            active_stages[0]["input_restart"] = "08_npt_soft_release.rst7"
    else:
        active_stages = rendered_stages  # "all"

    # Write run script(s)
    script_files: dict[str, str] = {}
    if stages_range in ("all", "1-8"):
        run_script = out_dir / "run_pre_md_1_8.sh" if stages_range == "1-8" else out_dir / "run_pre_md.sh"
        _write_text(run_script, _render_run_script(config, active_stages))
        script_files["run_script"] = str(run_script)
    if stages_range in ("all", "9"):
        run_script_9 = out_dir / "run_pre_md_9.sh"
        _write_text(run_script_9, _render_run_script(config, active_stages))
        script_files["run_script_9"] = str(run_script_9)
    if stages_range == "all":
        # Also generate split scripts for convenience
        stages_1_8 = [s for s in rendered_stages if s["index"] <= 8]
        _write_text(out_dir / "run_pre_md_1_8.sh", _render_run_script(config, stages_1_8))
        stages_9_only = [s for s in rendered_stages if s["index"] == 9]
        if stages_9_only:
            stages_9_only[0]["input_restart"] = "08_npt_soft_release.rst7"
        _write_text(out_dir / "run_pre_md_9.sh", _render_run_script(config, stages_9_only))
        script_files["run_script_1_8"] = str(out_dir / "run_pre_md_1_8.sh")
        script_files["run_script_9"] = str(out_dir / "run_pre_md_9.sh")

    windows_run_script = out_dir / "run_pre_md_windows.ps1"
    _write_text(windows_run_script, _render_windows_run_script())

    manifest = {
        "schema": "cypforge.complex_pre_md_equilibration.v1",
        "status": "prepared",
        "stages_range": stages_range,
        "input_files": {
            "solvation_manifest_json": str(manifest_path),
            "source_prmtop": str(prmtop),
            "source_rst7": str(rst7),
            "protocol_config_json": str(config_path),
        },
        "output_files": {
            "manifest_json": str(out_dir / "complex_pre_md_equilibration_manifest.json"),
            "copied_prmtop": str(copied_prmtop),
            "copied_initial_rst7": str(copied_rst7),
            "run_script": script_files.get("run_script", ""),
            "run_script_1_8": script_files.get("run_script_1_8", ""),
            "run_script_9": script_files.get("run_script_9", ""),
            "windows_wsl_launcher": str(windows_run_script),
            "mdin_dir": str(mdin_dir),
            "run_dir": str(run_dir),
        },
        "engine": config.get("engine", "pmemd.cuda"),
        "stage_count": len(rendered_stages),
        "stages": rendered_stages,
        "safety_gates": {
            "topology_restart_pairing_probe_required_before_execution": True,
            "restrained_stages_require_ref": True,
            "restrained_stages_iwrap_zero": True,
            "ntr_zero_stages_do_not_pass_ref": True,
            "reference_coordinates_atom_order_source": "copied_initial_rst7_or_stage_input_with_same_prmtop",
            "validation": validation,
        },
        "limitations": [
            "This stage renders Amber inputs and run scripts; it does not prove equilibration unless the generated stages are executed and QC is parsed.",
            "Users may edit pre_md_protocol_config.json, but any ntr=1 stage must keep iwrap=0 and must pass a same-topology reference restart.",
            "The copied initial restart and all generated stage restarts must be paired with the copied prmtop; mixing files from other runs is a hard failure.",
        ],
    }
    _write_text(Path(manifest["output_files"]["manifest_json"]), json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def _validate_protocol_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema") != "cypforge.pre_md_protocol_config.v1":
        raise ValueError("Unsupported pre-MD protocol config schema.")
    stages = config.get("stages")
    if not isinstance(stages, list) or len(stages) < 1:
        raise ValueError("Protocol config must contain at least one stage.")
    seen: set[str] = set()
    restrained_count = 0
    for stage in stages:
        stage_id = _safe_stage_id(stage.get("id", ""))
        if stage_id in seen:
            raise ValueError(f"Duplicate stage id: {stage_id}")
        seen.add(stage_id)
        if stage.get("kind") not in {"minimization", "md"}:
            raise ValueError(f"Stage {stage_id} kind must be minimization or md.")
        params = stage.get("parameters")
        if not isinstance(params, dict):
            raise ValueError(f"Stage {stage_id} is missing parameters.")
        ntr = int(params.get("ntr", 0))
        if ntr == 1:
            restrained_count += 1
            if not stage.get("reference"):
                raise ValueError(f"Stage {stage_id} uses ntr=1 but has no reference restart policy.")
            if int(params.get("iwrap", 0)) != 0:
                raise ValueError(f"Stage {stage_id} uses ntr=1 and must keep iwrap=0.")
            if "restraint_wt" not in params or "restraintmask" not in params:
                raise ValueError(f"Stage {stage_id} uses ntr=1 and must define restraint_wt/restraintmask.")
        elif ntr != 0:
            raise ValueError(f"Stage {stage_id} has unsupported ntr={ntr}; use 0 or 1.")
    return {
        "status": "passed",
        "stage_ids_unique": True,
        "restrained_stage_count": restrained_count,
        "free_stage_count": len(stages) - restrained_count,
    }


def _safe_stage_id(stage_id: str) -> str:
    if not stage_id or any(ch in stage_id for ch in "\\/:*?\"<>| "):
        raise ValueError(f"Invalid stage id: {stage_id!r}")
    return stage_id


def _stage_reference(stage: dict[str, Any], initial_rst7: str, input_restart: str) -> str | None:
    if int(stage["parameters"].get("ntr", 0)) == 0:
        return None
    reference = stage.get("reference", "initial")
    if reference == "initial":
        return initial_rst7
    if reference == "input":
        return input_restart
    if isinstance(reference, str) and reference.endswith(".rst7"):
        return reference
    raise ValueError(f"Unsupported reference policy for stage {stage['id']}: {reference!r}")


def _render_mdin(stage: dict[str, Any]) -> str:
    params = dict(stage["parameters"])
    lines = [f"{stage['id']}: {stage.get('title', stage['kind'])}", "&cntrl"]
    for key, value in params.items():
        if isinstance(value, str):
            rendered = f"'{value}'"
        elif isinstance(value, bool):
            rendered = int(value)
        else:
            rendered = value
        lines.append(f"  {key}={rendered},")
    lines.extend(["/", ""])
    return "\n".join(lines)


def _render_run_script(config: dict[str, Any], stages: list[dict[str, Any]]) -> str:
    engine = config.get("engine", "pmemd.cuda")
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'AMBER_ENGINE="${AMBER_ENGINE:-' + engine + '}"',
        'cd "$(dirname "$0")"',
        "mkdir -p run",
        "date -Is > run/run_pre_md.started_at.txt",
        "printf 'stage\\tmdout\\trestart\\texit_code\\tnormal_end\\n' > run/stage_status.tsv",
        "",
        "if ! command -v cpptraj >/dev/null 2>&1; then",
        "  echo 'ERROR: cpptraj is required for topology/restart pairing probe.' >&2",
        "  echo 2 > run/run_pre_md.exit_code.txt",
        "  date -Is > run/run_pre_md.finished_at.txt",
        "  exit 2",
        "fi",
        "set +e",
        "cpptraj -p system_lig_solv.prmtop <<'EOF' > pre_md_pairing_probe.log",
        "trajin system_lig_solv.rst7",
        "check",
        "go",
        "EOF",
        "probe_rc=$?",
        "set -e",
        'if [ "$probe_rc" -ne 0 ]; then',
        "  echo 'ERROR: cpptraj topology/restart pairing probe failed.' >&2",
        '  echo "$probe_rc" > run/run_pre_md.exit_code.txt',
        "  date -Is > run/run_pre_md.finished_at.txt",
        '  exit "$probe_rc"',
        "fi",
        "grep -Eq '1 frames|1 frame' pre_md_pairing_probe.log || { echo 'ERROR: cpptraj did not process the initial restart.' >&2; echo 3 > run/run_pre_md.exit_code.txt; date -Is > run/run_pre_md.finished_at.txt; exit 3; }",
        "grep -Eiq 'bad bond|Unusual bond length' pre_md_pairing_probe.log && { echo 'ERROR: cpptraj reported bad/unusual bond-length warnings.' >&2; echo 4 > run/run_pre_md.exit_code.txt; date -Is > run/run_pre_md.finished_at.txt; exit 4; }",
        "",
    ]
    for stage in stages:
        args = [
            '"$AMBER_ENGINE"',
            "-O",
            f"-i mdin/{Path(stage['mdin']).name}",
            f"-o run/{stage['id']}.out",
            "-p system_lig_solv.prmtop",
            f"-c {stage['input_restart']}",
            f"-r {stage['output_restart']}",
            f"-inf run/{stage['id']}.info",
        ]
        if stage["trajectory"]:
            args.append(f"-x run/{stage['trajectory']}")
        if stage["reference_restart"]:
            args.append(f"-ref {stage['reference_restart']}")
        lines.extend(
            [
                "set +e",
                " ".join(args),
                "rc=$?",
                "set -e",
                "normal_end=0",
                f"grep -Eq 'Total wall time|Final Performance Info' run/{stage['id']}.out 2>/dev/null && normal_end=1",
                f"printf '{stage['id']}\\trun/{stage['id']}.out\\t{stage['output_restart']}\\t%s\\t%s\\n' \"$rc\" \"$normal_end\" >> run/stage_status.tsv",
                'if [ "$rc" -ne 0 ]; then',
                '  echo "$rc" > run/run_pre_md.exit_code.txt',
                "  date -Is > run/run_pre_md.finished_at.txt",
                '  exit "$rc"',
                "fi",
            ]
        )
    lines.extend(["echo 0 > run/run_pre_md.exit_code.txt", "date -Is > run/run_pre_md.finished_at.txt", ""])
    return "\n".join(lines)


def _render_windows_run_script() -> str:
    return (
        "$ErrorActionPreference = 'Stop'\n"
        "$here = Split-Path -Parent $MyInvocation.MyCommand.Path\n"
        "$wslPath = (wsl -e wslpath -a \"$here\").Trim()\n"
        "if (-not $env:AMBER_SH -and -not $env:AMBERHOME) { Write-Error 'Set AMBER_SH or AMBERHOME'; exit 1 }\n"
        "$amberSetup = if ($env:AMBER_SH) { $env:AMBER_SH } else { \"$($env:AMBERHOME)/amber.sh\" }\n"
        "wsl -u john -e bash -lc \"source '$amberSetup' && cd '$wslPath' && bash run_pre_md.sh\"\n"
    )

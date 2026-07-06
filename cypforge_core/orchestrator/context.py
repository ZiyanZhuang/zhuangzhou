"""Context builder — aggregate JSON reports for LLM agent decision-making."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# Non-negotiable scientific rules from SKILL.md — surfaced as policy reminders
POLICY_REMINDERS = [
    "SDF is the ligand chemistry source: graph, bond order, aromaticity, formal charge, GAFF2 typing.",
    "PDB is the protein/HEME/ligand pose source: coordinates, residue names, atom names, chain identifiers.",
    "GAFF2/parmchk2 define ligand bonded and van der Waals parameters. GAFF is not a silent drop-in replacement.",
    "GPU4PySCF/Multiwfn or ESP-LSQ defines RESP/ESP charges.",
    "Amber/tLeap/pmemd.cuda/cpptraj are the final construction, charge, MD execution, and trajectory audit tools.",
    "Successful tleap execution is not proof of chemical correctness.",
    "Correct total charge is not proof of correct protonation or per-atom charge mapping.",
    "A 20 ns free NPT equilibration is not production readiness.",
    "A hard gate failure stops the workflow. Do not continue to the next core.",
    "Do not auto-correct residue names unless an explicit decision JSON requires it.",
    "Do not infer ligand atom mapping from atom order.",
    "Do not use PDB bond order as ligand chemistry truth.",
    "Do not hide warnings from tleap, pmemd.cuda, cpptraj, parmchk2, PROPKA, reduce, or RESP fitting.",
    "Do not run production MD from this skill shell.",
    "Do not perform transmembrane helix trimming unless the human user supplied exact chain:residue ranges and explicitly confirmed deletion.",
]


def build_agent_context(
    manifest: dict[str, Any],
    max_warnings: int = 20,
) -> dict[str, Any]:
    """Build a structured context dict for LLM agent decision-making.

    Aggregates all completed module JSON manifests into a single dict that
    provides the structured chemical context, gate results, and evidence for
    the LLM agent to make informed decisions within constrained boundaries.

    Args:
        manifest: The RunManifest dict (from run_manifest.json).
        max_warnings: Maximum number of warnings to include per module.

    Returns:
        A dict with sections: workflow, modules, summary, next_allowed_actions,
        policy_reminders.
    """
    modules_data: dict[str, dict[str, Any]] = {}
    run_root = Path(manifest.get("run_root", "."))
    completed_ids: list[str] = []
    failed_ids: list[str] = []
    pending_ids: list[str] = []
    total_gates_passed = 0
    total_gates_warned = 0
    total_gates_failed = 0
    critical_findings: list[str] = []
    warnings: list[str] = []

    for skill_id, record in manifest.get("modules", {}).items():
        status = record.get("status", "PENDING")

        if status in ("PASS", "WARN"):
            completed_ids.append(skill_id)
        elif status == "FAIL":
            failed_ids.append(skill_id)
        else:
            pending_ids.append(skill_id)

        # Load manifest content for completed or failed modules
        if status in ("PASS", "WARN", "FAIL"):
            module_entry: dict[str, Any] = {
                "skill_id": skill_id,
                "status": status,
                "gate_result": record.get("gate_result", ""),
                "errors": record.get("errors", []),
                "warnings": record.get("warnings", [])[:max_warnings],
                "summary": record.get("summary", ""),
                "manifests": {},
            }

            for manifest_name, manifest_rel_path in record.get("manifest_paths", {}).items():
                full_path = run_root / manifest_rel_path
                if full_path.is_file():
                    try:
                        data = json.loads(full_path.read_text(encoding="utf-8"))
                        module_entry["manifests"][manifest_name] = _compact_manifest(data)
                    except (json.JSONDecodeError, OSError):
                        module_entry["manifests"][manifest_name] = {
                            "_error": f"Cannot read {full_path}"
                        }

            modules_data[skill_id] = module_entry

            # Count gates
            gate = record.get("gate_result", "")
            if gate == "PASS":
                total_gates_passed += 1
            elif gate == "WARN":
                total_gates_warned += 1
            elif gate == "FAIL":
                total_gates_failed += 1

            # Collect critical findings
            for err in record.get("errors", []):
                critical_findings.append(f"[{skill_id}] {err}")
            for warn in record.get("warnings", [])[:max_warnings]:
                warnings.append(f"[{skill_id}] {warn}")

    # Determine next allowed actions
    next_actions = _determine_next_actions(manifest, completed_ids, failed_ids)

    context: dict[str, Any] = {
        "workflow": {
            "run_name": manifest.get("run_name", ""),
            "run_root": manifest.get("run_root", ""),
            "overall_status": manifest.get("workflow_status", "UNKNOWN"),
            "completed_modules": completed_ids,
            "failed_modules": failed_ids,
            "pending_modules": pending_ids,
            "current_module": _find_current_module(manifest),
        },
        "modules": modules_data,
        "summary": {
            "total_modules": len(manifest.get("modules", {})),
            "modules_completed": len(completed_ids),
            "modules_failed": len(failed_ids),
            "modules_pending": len(pending_ids),
            "gates_passed": total_gates_passed,
            "gates_warned": total_gates_warned,
            "gates_failed": total_gates_failed,
            "critical_findings": critical_findings,
            "warnings": warnings[:max_warnings],
        },
        "next_allowed_actions": next_actions,
        "policy_reminders": POLICY_REMINDERS,
    }

    return context


def _compact_manifest(data: dict[str, Any], max_depth: int = 3) -> dict[str, Any]:
    """Trim a manifest to essential fields for context window efficiency.

    Removes large arrays (coordinates, per-atom data) while keeping gate results,
    status, charge sums, distances, and other decision-relevant metrics.
    """
    keep_keys = {
        "status", "schema", "inputs", "outputs", "output_files",
        "gate_results", "gates", "policy",
        "heme_state", "heme_charge", "fe_charge",
        "charge", "charge_sum", "dry_charge", "final_charge",
        "mol2_charge_sum", "total_charge",
        "fe_sg_a", "fe_na_a", "fe_nb_a", "fe_nc_a", "fe_nd_a",
        "heme_resname", "protonation_changes", "expected_final_residue_checks",
        "neutralizing_ion_count", "water_count", "atom_count",
        "status_summary", "error_summary", "warning_summary",
    }

    compact: dict[str, Any] = {}
    for key, value in data.items():
        if key in keep_keys:
            if isinstance(value, list) and len(value) > 50:
                compact[key] = f"[{len(value)} items]"
            elif isinstance(value, dict):
                compact[key] = _compact_manifest(value, max_depth - 1) if max_depth > 0 else "{...}"
            else:
                compact[key] = value
    return compact


def _find_current_module(manifest: dict[str, Any]) -> str | None:
    """Identify the module the workflow is currently at."""
    ordered = [
        "cypforge.environment_check",
        "cypforge.core1_prepare_heme_cym",
        "cypforge.core2_prepare_ligand_resp_gaff2",
        "cypforge.core3_finalize_protonation",
        "cypforge.core3_solvate_ionize",
        "cypforge.core3_render_pre_md",
        "cypforge.core3_run_pre_md",
        "cypforge.global_audit",
        "cypforge.equilibration_decision",
        "cypforge.production_readiness_check",
    ]
    for skill_id in ordered:
        record = manifest.get("modules", {}).get(skill_id, {})
        status = record.get("status", "PENDING")
        if status in ("PENDING", "RUNNING", "FAIL"):
            return skill_id
    return None


def _determine_next_actions(
    manifest: dict[str, Any],
    completed_ids: list[str],
    failed_ids: list[str],
) -> list[str]:
    """Determine the list of next allowed actions based on workflow state."""
    actions: list[str] = []

    if failed_ids:
        actions.append("STOP_FIX_UPSTREAM_MODULE")
        actions.append(f"Failed modules: {', '.join(failed_ids)}")
        return actions

    wf_status = manifest.get("workflow_status", "")
    if wf_status == "PAUSED":
        actions.append("WARN_HUMAN_REVIEW")
        actions.append("Review warnings in completed modules before continuing.")
        return actions

    all_modules = [
        "cypforge.environment_check",
        "cypforge.core1_prepare_heme_cym",
        "cypforge.core2_prepare_ligand_resp_gaff2",
        "cypforge.core3_finalize_protonation",
        "cypforge.core3_solvate_ionize",
        "cypforge.core3_render_pre_md",
        "cypforge.core3_run_pre_md",
        "cypforge.global_audit",
        "cypforge.equilibration_decision",
        "cypforge.production_readiness_check",
    ]

    remaining = [m for m in all_modules if m not in completed_ids]

    if not remaining:
        # Check the equilibration decision for the final allowed action
        modules = manifest.get("modules", {})
        eq_decision = modules.get("cypforge.equilibration_decision", {})
        if eq_decision.get("status") == "PASS":
            actions.append("ALLOW_1_5_NS_FREE_EQUILIBRATION")
            actions.append("ALLOW_PRODUCTION_SETUP_ONLY_AFTER_EXTENDED_EQUILIBRATION")
        else:
            actions.append("WARN_HUMAN_REVIEW")
    elif "cypforge.global_audit" in completed_ids:
        actions.append("ALLOW_1_5_NS_FREE_EQUILIBRATION")
    elif "cypforge.core3_run_pre_md" in completed_ids:
        actions.append("RUN_GLOBAL_AUDIT")
    else:
        actions.append(f"Next module: {remaining[0]}")

    return actions

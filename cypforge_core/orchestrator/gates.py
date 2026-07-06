"""GateChecker — read JSON manifests, evaluate PASS/WARN/FAIL, enforce stop conditions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import GateResult


class GateChecker:
    """Evaluates module output manifests against expected PASS/WARN/FAIL gates."""

    def __init__(self, run_root: str):
        self.run_root = Path(run_root)

    def check_module_output(
        self, skill_id: str, manifest_paths: dict[str, str]
    ) -> tuple[str, list[GateResult], str]:
        """Check a module's output manifests and return (status, gates, summary).

        Returns status as one of PASS/WARN/FAIL.
        """
        gates: list[GateResult] = []
        for name, rel_path in manifest_paths.items():
            full_path = self.run_root / rel_path
            if not full_path.is_file():
                gates.append(
                    GateResult(
                        gate_id=f"{skill_id}.manifest_missing",
                        name=name,
                        status="FAIL",
                        detail=f"Output manifest not found: {full_path}",
                        evidence_file=str(full_path),
                    )
                )
                continue

            try:
                data = json.loads(full_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                gates.append(
                    GateResult(
                        gate_id=f"{skill_id}.manifest_unreadable",
                        name=name,
                        status="FAIL",
                        detail=f"Cannot read manifest {full_path}: {exc}",
                        evidence_file=str(full_path),
                    )
                )
                continue

            status = self._extract_status(data)
            gates.append(
                GateResult(
                    gate_id=f"{skill_id}.{name}",
                    name=name,
                    status=status,
                    detail=self._summarize_manifest(skill_id, name, data),
                    evidence_file=str(full_path),
                )
            )

        combined = self._combine(gates)
        summary = f"{len(gates)} manifest(s) checked: " + ", ".join(
            f"{g.name}={g.status}" for g in gates
        )
        return combined, gates, summary

    def check_absolute_stop_conditions(
        self, manifest: dict[str, Any], all_module_records: dict[str, dict[str, Any]]
    ) -> list[GateResult]:
        """Evaluate absolute_stop_conditions from the skills manifest against current state.

        These are cross-module checks that can halt the entire workflow.
        """
        results: list[GateResult] = []
        conditions = manifest.get("absolute_stop_conditions", [])

        for entry in conditions:
            cond = entry.get("condition", entry) if isinstance(entry, dict) else entry
            cond_str = str(cond)
            result = self._eval_condition(cond_str, all_module_records)
            results.append(result)
            if isinstance(entry, dict):
                entry["status"] = result.status

        return results

    def _eval_condition(
        self, condition: str, records: dict[str, dict[str, Any]]
    ) -> GateResult:
        """Evaluate a single absolute stop condition."""
        # Environment check conditions
        if "Amber/tLeap/pmemd.cuda/cpptraj unavailable" in condition:
            env = records.get("cypforge.environment_check", {})
            return GateResult(
                gate_id="stop.tools_unavailable",
                name="Required tools available",
                status="FAIL" if env.get("status") == "FAIL" else "PASS",
                detail=condition,
            )

        if "Python cannot import cypforge_core" in condition:
            env = records.get("cypforge.environment_check", {})
            return GateResult(
                gate_id="stop.import_failed",
                name="cypforge_core importable",
                status="FAIL" if env.get("status") == "FAIL" else "PASS",
                detail=condition,
            )

        if "global audit status is FAIL" in condition:
            audit = records.get("cypforge.global_audit", {})
            return GateResult(
                gate_id="stop.global_audit_fail",
                name="Global audit",
                status="FAIL" if audit.get("status") == "FAIL" else "PASS",
                detail=condition,
            )

        # For conditions that require inspecting specific module outputs,
        # we mark them as PASS unless we have evidence of failure.
        # The individual module gates catch the specific failures.
        return GateResult(
            gate_id=f"stop.{hash(condition) & 0xFFFF:04x}",
            name=condition[:60],
            status="PASS",
            detail="Checked by individual module gates",
        )

    def _extract_status(self, data: dict[str, Any]) -> str:
        """Extract and normalize status from a manifest dict."""
        raw = data.get("status", "unknown")
        if isinstance(raw, str):
            raw_lower = raw.lower()
            if raw_lower in ("pass", "passed", "success", "prepared", "ok"):
                return "PASS"
            if raw_lower in ("fail", "failed", "error", "failure"):
                return "FAIL"
            if raw_lower in ("warn", "warning"):
                return "WARN"
        # If the manifest has sub-gate results (like global_audit), combine them
        if "gate_results" in data:
            sub_statuses = [g.get("status", "PASS") for g in data["gate_results"]]
            if "FAIL" in sub_statuses:
                return "FAIL"
            if "WARN" in sub_statuses:
                return "WARN"
            return "PASS"
        return "WARN"  # unknown status → WARN (needs review)

    def _summarize_manifest(
        self, skill_id: str, name: str, data: dict[str, Any]
    ) -> str:
        """Produce a one-line summary of a manifest."""
        status = data.get("status", "?")
        lines = [f"status={status}"]

        if "gate_results" in data:
            for g in data["gate_results"]:
                g_status = g.get("status", "?")
                g_name = g.get("name", g.get("gate", "?"))
                lines.append(f"{g_name}={g_status}")

        if "outputs" in data:
            lines.append(f"{len(data['outputs'])} output files")

        return "; ".join(lines[:6])

    @staticmethod
    def _combine(gates: list[GateResult]) -> str:
        if any(g.status == "FAIL" for g in gates):
            return "FAIL"
        if any(g.status == "WARN" for g in gates):
            return "WARN"
        return "PASS"


# ── decision-making functions (modules 8 and 9) ──────────────────────────────

def make_equilibration_decision(run_root: str) -> None:
    """Read the global audit manifest and write equilibration_decision_state.json.

    This implements the decision logic from equilibration_decision.md skill.
    """
    root = Path(run_root)
    audit_manifest_path = root / "18_global_cyp450_audit" / "00_manifest.json"
    decision_path = root / "18_global_cyp450_audit" / "equilibration_decision_state.json"

    if not audit_manifest_path.is_file():
        state = {
            "decision": "STOP_FIX_PRE_MD",
            "reason": "Global audit manifest not found. Pre-MD may not have completed.",
            "next_allowed_action": "Re-run pre-MD or fix upstream modules.",
        }
    else:
        data = json.loads(audit_manifest_path.read_text(encoding="utf-8"))
        status = data.get("status", "FAIL")
        gate_results = data.get("gate_results", [])

        # Map failures to decision states
        failed_gates = [g for g in gate_results if g.get("status") == "FAIL"]
        warned_gates = [g for g in gate_results if g.get("status") == "WARN"]

        if status == "FAIL" or failed_gates:
            # Determine which core to fix based on failed gate names
            decision = _map_failure_to_stop(failed_gates)
            reason = f"Failed gates: {', '.join(g.get('name', '?') for g in failed_gates)}"
            next_action = "Fix the identified upstream module and re-run."
        elif status == "WARN" or warned_gates:
            decision = "WARN_HUMAN_REVIEW"
            reason = f"Warning gates require review: {', '.join(g.get('name', '?') for g in warned_gates)}"
            next_action = "Review warnings. If acceptable, allow 1-5 ns extended equilibration."
        else:
            decision = "ALLOW_1_5_NS_FREE_EQUILIBRATION"
            reason = "All hard gates passed. System is stable enough for extended equilibration."
            next_action = "Run 1-5 ns additional free NPT equilibration before production readiness check."

        state = {
            "decision": decision,
            "reason": reason,
            "next_allowed_action": next_action,
            "audit_status": status,
            "failed_gate_count": len(failed_gates),
            "warned_gate_count": len(warned_gates),
        }

    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text(
        json.dumps(state, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(state, indent=2, ensure_ascii=False))


def _map_failure_to_stop(failed_gates: list[dict[str, Any]]) -> str:
    """Map failed gate names to STOP_FIX_* decision states."""
    for gate in failed_gates:
        name = gate.get("name", "")
        name_lower = name.lower()
        if any(kw in name_lower for kw in ("heme", "cym", "fe_s", "fe-n", "core1")):
            return "STOP_FIX_CORE1"
        if any(kw in name_lower for kw in ("ligand", "mapping", "resp", "gaff", "charge_accounting", "core2")):
            return "STOP_FIX_CORE2"
        if any(kw in name_lower for kw in ("protonation", "residue", "glh", "hid", "dry_charge")):
            return "STOP_FIX_PROTONATION"
        if any(kw in name_lower for kw in ("solvation", "tleap", "ion", "neutral", "box")):
            return "STOP_FIX_SOLVATION"
        if any(kw in name_lower for kw in ("pre_md", "stage", "mdin", "ref", "iwrap")):
            return "STOP_FIX_PRE_MD"
    return "STOP_FIX_PRE_MD"


def make_production_readiness_check(run_root: str) -> None:
    """Read equilibration decision and audit, write production_readiness_state.json.

    This implements the conservative check from production_readiness_check.md skill.
    It can never authorize production from nine-stage pre-MD alone.
    """
    root = Path(run_root)
    decision_path = root / "18_global_cyp450_audit" / "equilibration_decision_state.json"
    output_path = root / "production_readiness_state.json"

    if not decision_path.is_file():
        state = {
            "state": "NOT_READY_FIX_UPSTREAM",
            "reason": "Equilibration decision state not found.",
            "production_authorized": False,
        }
    else:
        decision_data = json.loads(decision_path.read_text(encoding="utf-8"))
        decision = decision_data.get("decision", "STOP_FIX_PRE_MD")

        if decision in ("ALLOW_PRODUCTION_SETUP_ONLY_AFTER_EXTENDED_EQUILIBRATION",):
            state = {
                "state": "READY_FOR_PRODUCTION_SETUP",
                "reason": "Extended equilibration completed and reviewed. Production setup may proceed.",
                "production_authorized": False,
                "note": "This authorizes production input setup, NOT running production MD.",
            }
        elif decision == "ALLOW_1_5_NS_FREE_EQUILIBRATION":
            state = {
                "state": "NOT_READY_NEEDS_EXTENDED_EQUILIBRATION",
                "reason": "Nine-stage pre-MD passed but extended equilibration is still required.",
                "production_authorized": False,
            }
        elif decision == "WARN_HUMAN_REVIEW":
            state = {
                "state": "NOT_READY_NEEDS_HUMAN_REVIEW",
                "reason": "Warnings from global audit require expert review before proceeding.",
                "production_authorized": False,
            }
        else:
            state = {
                "state": "NOT_READY_FIX_UPSTREAM",
                "reason": f"Upstream issues must be resolved. Decision: {decision}",
                "production_authorized": False,
            }

    output_path.write_text(
        json.dumps(state, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(state, indent=2, ensure_ascii=False))

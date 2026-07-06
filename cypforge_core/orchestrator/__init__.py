"""CYPForge Outer Shell — workflow orchestration layer."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from .models import (
    RunConfig,
    RunManifest,
    StepDef,
    ModuleDef,
    StepRecord,
    ModuleRecord,
    GateResult,
    build_module_definitions,
    get_module_by_index,
    get_module_by_skill_id,
)
from .runner import ModuleRunner
from .workflow import WorkflowManager
from .gates import GateChecker
from .context import build_agent_context


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


class CYPForgeOrchestrator:
    """Main workflow controller for CYPForge.

    Coordinates the full Core1 → Core2 → Core3 pipeline:
    - init: creates run_root and run_manifest.json
    - run: executes all outstanding modules from the start or resume point
    - status: prints a summary table of module states
    - context: builds aggregated context for LLM agent decision-making
    """

    def __init__(self, config: RunConfig):
        self.config = config
        self.runner = ModuleRunner(config)
        self.workflow = WorkflowManager(config)
        self.gates = GateChecker(config.run_root)

    # ── public API ──────────────────────────────────────────────────────

    def init(self) -> RunManifest:
        """Initialize a new run: create directories, write config, return manifest."""
        return self.workflow.init_run()

    def run(self) -> RunManifest:
        """Execute the pipeline from the first pending module.

        If a run_manifest.json exists, resumes from the checkpoint.
        """
        manifest = self.workflow.load_manifest()
        if manifest is None:
            manifest = self.init()

        resume_point = self.workflow.find_resume_point(manifest)
        if resume_point is None:
            manifest.workflow_status = "COMPLETED"
            manifest.completed_at = _ts()
            self.workflow.save_manifest(manifest)
            return manifest

        manifest.workflow_status = "RUNNING"
        self.workflow.save_manifest(manifest)

        modules = self.workflow.get_ordered_modules()
        started = False
        for mod_def in modules:
            if not started and mod_def.skill_id != resume_point.skill_id:
                continue
            started = True

            print(f"\n{'='*60}")
            print(f"Module {mod_def.index}: {mod_def.skill_id}")
            print(f"  {mod_def.description}")
            print(f"{'='*60}")

            record = self._execute_module(mod_def, manifest)
            manifest = self.workflow.load_manifest()  # reload fresh

            if record.status == "FAIL":
                manifest.workflow_status = "STOPPED_ON_FAIL"
                self.workflow.save_manifest(manifest)
                print(f"\nSTOP: Module {mod_def.skill_id} FAILED.")
                print(f"  Errors: {record.errors}")
                return manifest

            if record.status == "WARN" and not self.config.auto_accept_warn:
                manifest.workflow_status = "PAUSED"
                self.workflow.save_manifest(manifest)
                print(f"\nPAUSED: Module {mod_def.skill_id} returned WARN.")
                print(f"  Warnings: {record.warnings}")
                print(f"  Review before continuing. Use 'resume' after review.")
                return manifest

        manifest.workflow_status = "COMPLETED"
        manifest.completed_at = _ts()
        self.workflow.save_manifest(manifest)
        print(f"\nWorkflow COMPLETED: {manifest.run_name}")
        return manifest

    def resume(self) -> RunManifest:
        """Resume a paused or failed run from the checkpoint."""
        manifest = self.workflow.load_manifest()
        if manifest is None:
            print("[ERROR] No run_manifest.json found. Use 'init' first.")
            return RunManifest()

        if manifest.workflow_status not in ("PAUSED", "STOPPED_ON_FAIL", "RUNNING", "INITIALIZING"):
            print(f"[ERROR] Workflow status is {manifest.workflow_status}. Nothing to resume.")
            return manifest

        print(f"Resuming {manifest.run_name} from status: {manifest.workflow_status}")
        return self.run()

    def status(self) -> str:
        """Return a formatted status table string."""
        manifest = self.workflow.load_manifest()
        if manifest is None:
            return "[ERROR] No run found. Use 'init' first."

        lines = [
            f"Run: {manifest.run_name}",
            f"Status: {manifest.workflow_status}",
            f"Root: {manifest.run_root}",
            "",
            f"{'#':<3} {'Module':<40} {'Status':<12} {'Gate':<8}",
            f"{'-'*3} {'-'*40} {'-'*12} {'-'*8}",
        ]

        modules = self.workflow.get_ordered_modules()
        for mod_def in modules:
            record = manifest.modules.get(mod_def.skill_id, {})
            status = record.get("status", "PENDING")
            gate = record.get("gate_result", "-")
            marker = " ←" if status in ("WARN", "FAIL", "RUNNING") and manifest.workflow_status != "COMPLETED" else ""
            lines.append(
                f"{mod_def.index:<3} {mod_def.skill_id:<40} {status:<12} {gate:<8}{marker}"
            )

        if manifest.workflow_status == "COMPLETED":
            lines.append(f"\nCompleted at: {manifest.completed_at}")
        elif manifest.workflow_status == "PAUSED":
            lines.append("\nWorkflow paused — review WARN modules before resuming.")
        elif manifest.workflow_status == "STOPPED_ON_FAIL":
            lines.append("\nWorkflow stopped — fix the FAILed module before resuming.")

        return "\n".join(lines)

    def get_context(self) -> dict:
        """Build aggregated context for LLM agent consumption."""
        manifest = self.workflow.load_manifest()
        if manifest is None:
            return {"error": "No run found."}
        raw_manifest = {
            "run_name": manifest.run_name,
            "run_root": manifest.run_root,
            "workflow_status": manifest.workflow_status,
            "modules": manifest.modules,
        }
        return build_agent_context(raw_manifest)

    # ── internal ────────────────────────────────────────────────────────

    def _execute_module(
        self, mod_def: ModuleDef, manifest: RunManifest
    ) -> ModuleRecord:
        """Execute all steps of a single module and return its record."""
        now = _ts()
        log_dir = Path(self.config.run_root) / "logs" / mod_def.skill_id.replace("cypforge.", "").replace(".", "_")
        log_dir.mkdir(parents=True, exist_ok=True)
        output_dir = Path(self.config.run_root) / mod_def.output_dir_rel if mod_def.output_dir_rel else Path(self.config.run_root)
        output_dir.mkdir(parents=True, exist_ok=True)

        record = ModuleRecord(
            skill_id=mod_def.skill_id,
            status="RUNNING",
            output_dir=str(output_dir),
            log_dir=str(log_dir),
            started_at=now,
        )
        self.workflow.update_module_record(manifest, mod_def.skill_id, record)

        # Check dependencies
        ready, reason = self.workflow.is_module_ready(mod_def, manifest)
        if not ready:
            record.status = "FAIL"
            record.errors.append(reason)
            record.completed_at = _ts()
            self.workflow.update_module_record(manifest, mod_def.skill_id, record)
            return record

        # Execute steps
        all_passed = True
        for step in mod_def.steps:
            print(f"  Step: {step.name} ... ", end="", flush=True)
            step_record = self.runner.run_step(step, log_dir)
            record.steps.append(step_record)

            if step_record.status == "FAIL":
                print("FAIL")
                all_passed = False
                record.errors.append(f"Step '{step.name}' failed (exit code {step_record.exit_code}).")
                if step_record.error_message:
                    record.errors.append(step_record.error_message)
                break
            print("OK")

        # Gate checking from output manifests
        manifest_paths_found: dict[str, str] = {}
        for rel_path in mod_def.output_manifests:
            full = Path(self.config.run_root) / rel_path
            if full.is_file():
                manifest_paths_found[rel_path] = rel_path

        if manifest_paths_found:
            gate_status, gate_results, gate_summary = self.gates.check_module_output(
                mod_def.skill_id, manifest_paths_found
            )
            record.gate_result = gate_status
            record.manifest_paths = manifest_paths_found
            record.summary = gate_summary

            for g in gate_results:
                if g.status == "FAIL":
                    record.errors.append(f"Gate {g.name}: {g.detail}")
                elif g.status == "WARN":
                    record.warnings.append(f"Gate {g.name}: {g.detail}")

            if gate_status == "FAIL":
                all_passed = False
        elif mod_def.output_manifests:
            record.warnings.append("No output manifests found for gate checking.")
            record.gate_result = "WARN"

        # Cross-module absolute stop conditions
        module_records = manifest.modules.copy()
        module_records[mod_def.skill_id] = {
            "skill_id": record.skill_id,
            "status": record.status,
            "gate_result": record.gate_result,
            "errors": record.errors,
            "warnings": record.warnings,
        }
        stop_checks = self.gates.check_absolute_stop_conditions(
            {
                "absolute_stop_conditions": manifest.absolute_stop_conditions,
            },
            module_records,
        )
        for sc in stop_checks:
            if sc.status == "FAIL":
                all_passed = False
                record.errors.append(f"ABSOLUTE STOP: {sc.name} — {sc.detail}")

        # Final status
        if not all_passed:
            record.status = "FAIL"
        elif record.warnings:
            record.status = "WARN"
        else:
            record.status = "PASS"

        record.completed_at = _ts()
        self.workflow.update_module_record(manifest, mod_def.skill_id, record)

        if record.status == "PASS":
            print(f"  Result: PASS")
        elif record.status == "WARN":
            print(f"  Result: WARN — {len(record.warnings)} warning(s)")
        else:
            print(f"  Result: FAIL — {len(record.errors)} error(s)")

        return record


__all__ = [
    "CYPForgeOrchestrator",
    "RunConfig",
    "RunManifest",
    "StepDef",
    "ModuleDef",
    "StepRecord",
    "ModuleRecord",
    "GateResult",
    "build_module_definitions",
    "get_module_by_index",
    "get_module_by_skill_id",
    "ModuleRunner",
    "WorkflowManager",
    "GateChecker",
    "build_agent_context",
]

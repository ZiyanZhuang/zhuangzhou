"""WorkflowManager — state machine, checkpoint persistence, resume logic."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import (
    RunConfig,
    RunManifest,
    ModuleDef,
    ModuleRecord,
    build_module_definitions,
    WORKFLOW_STATUS,
)


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_skills_manifest(project_root: str) -> dict[str, Any]:
    """Load skills_manifest.json from the project skills directory."""
    manifest_path = Path(project_root) / "skills" / "cypforge" / "skills_manifest.json"
    if not manifest_path.is_file():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


class WorkflowManager:
    """Manages the workflow state machine and checkpoint persistence."""

    def __init__(self, config: RunConfig):
        self.config = config
        self.run_root = Path(config.run_root)
        self._module_defs: list[ModuleDef] = build_module_definitions()
        self._skills_manifest = _load_skills_manifest(config.project_root)

    # ── manifest path helpers ─────────────────────────────────────────────

    @property
    def manifest_path(self) -> Path:
        return self.run_root / "run_manifest.json"

    @property
    def config_path(self) -> Path:
        return self.run_root / "run_config.json"

    # ── init ──────────────────────────────────────────────────────────────

    def init_run(self) -> RunManifest:
        """Create run_root, write run_config.json, and return a fresh RunManifest."""
        self.run_root.mkdir(parents=True, exist_ok=True)
        # Ensure standard subdirectories exist
        (self.run_root / "logs").mkdir(exist_ok=True)
        (self.run_root / "decisions").mkdir(exist_ok=True)

        # Persist config
        config_dict = {
            "run_name": self.config.run_name,
            "run_root": self.config.run_root,
            "project_root": self.config.project_root,
            "raw_protein_heme_pdb": self.config.raw_protein_heme_pdb,
            "ligand_template_sdf": self.config.ligand_template_sdf,
            "heme_state": self.config.heme_state,
            "heme_resname": self.config.heme_resname,
            "heme_chain": self.config.heme_chain,
            "protein_chain": self.config.protein_chain,
            "axial_cys_resid": self.config.axial_cys_resid,
            "ligand_resname": self.config.ligand_resname,
            "ligand_chain": self.config.ligand_chain,
            "formal_charge": self.config.formal_charge,
            "spin": self.config.spin,
            "basis": self.config.basis,
            "points_per_atom": self.config.points_per_atom,
            "fit_method": self.config.fit_method,
            "pre_resp_relax": self.config.pre_resp_relax,
            "protonation_decision_json": self.config.protonation_decision_json,
            "protein_force_field": self.config.protein_force_field,
            "water_leaprc": self.config.water_leaprc,
            "water_model": self.config.water_model,
            "box_type": self.config.box_type,
            "buffer_a": self.config.buffer_a,
            "neutralizing_anion": self.config.neutralizing_anion,
            "wsl_user": self.config.wsl_user,
            "amber_sh": self.config.amber_sh,
            "multiwfn_bin": self.config.multiwfn_bin,
            "auto_accept_warn": self.config.auto_accept_warn,
            "max_retries": self.config.max_retries,
        }
        self.config_path.write_text(
            json.dumps(config_dict, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        # Build absolute stop conditions from skills manifest
        stop_conditions = []
        for cond_text in self._skills_manifest.get("absolute_stop_conditions", []):
            stop_conditions.append({"condition": cond_text, "status": "PENDING"})

        now = _ts()
        manifest = RunManifest(
            run_name=self.config.run_name,
            run_root=str(self.run_root),
            project_root=self.config.project_root,
            workflow_status="INITIALIZING",
            modules={},
            run_config=config_dict,
            absolute_stop_conditions=stop_conditions,
            started_at=now,
            updated_at=now,
        )

        # Initialize all module records as PENDING
        for mod_def in self._module_defs:
            manifest.modules[mod_def.skill_id] = {
                "skill_id": mod_def.skill_id,
                "status": "PENDING",
                "steps": [],
                "gate_result": "",
                "manifest_paths": {},
                "output_dir": str(self.run_root / mod_def.output_dir_rel) if mod_def.output_dir_rel else "",
                "log_dir": str(self.run_root / "logs" / mod_def.skill_id.replace("cypforge.", "").replace(".", "_")),
                "errors": [],
                "warnings": [],
                "summary": "",
                "started_at": "",
                "completed_at": "",
            }

        self._write_manifest(manifest)
        return manifest

    # ── checkpoint persistence ────────────────────────────────────────────

    def load_manifest(self) -> RunManifest | None:
        """Load an existing run_manifest.json. Returns None if absent."""
        if not self.manifest_path.is_file():
            return None
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return RunManifest(
            schema=data.get("schema", ""),
            run_name=data.get("run_name", ""),
            run_root=data.get("run_root", ""),
            project_root=data.get("project_root", ""),
            workflow_status=data.get("workflow_status", "INITIALIZING"),
            modules=data.get("modules", {}),
            run_config=data.get("run_config", {}),
            absolute_stop_conditions=data.get("absolute_stop_conditions", []),
            started_at=data.get("started_at", ""),
            updated_at=data.get("updated_at", ""),
            completed_at=data.get("completed_at", ""),
        )

    def save_manifest(self, manifest: RunManifest) -> None:
        """Persist the manifest to run_manifest.json."""
        manifest.updated_at = _ts()
        self._write_manifest(manifest)

    def _write_manifest(self, manifest: RunManifest) -> None:
        payload = {
            "schema": manifest.schema,
            "run_name": manifest.run_name,
            "run_root": manifest.run_root,
            "project_root": manifest.project_root,
            "workflow_status": manifest.workflow_status,
            "modules": manifest.modules,
            "run_config": manifest.run_config,
            "absolute_stop_conditions": manifest.absolute_stop_conditions,
            "started_at": manifest.started_at,
            "updated_at": manifest.updated_at,
            "completed_at": manifest.completed_at,
        }
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    # ── state queries ─────────────────────────────────────────────────────

    def find_resume_point(self, manifest: RunManifest) -> ModuleDef | None:
        """Return the first module that is PENDING, FAIL, or needs WARN review.

        Returns None if all modules have completed (PASS, WARN-accepted, or SKIPPED).
        """
        auto_accept = manifest.run_config.get("auto_accept_warn", False)
        for mod_def in self._module_defs:
            record = manifest.modules.get(mod_def.skill_id)
            if record is None:
                return mod_def  # never started
            status = record.get("status", "PENDING")
            if status == "PENDING":
                return mod_def
            if status == "RUNNING":
                return mod_def  # was interrupted mid-module
            if status == "FAIL":
                return mod_def  # re-run failed module
            if status == "WARN" and not auto_accept:
                return mod_def  # needs human review
            if status in ("PASS", "SKIPPED"):
                continue  # done, move to next
        return None  # all complete

    def is_module_ready(self, module_def: ModuleDef, manifest: RunManifest) -> tuple[bool, str]:
        """Check if a module's input dependencies are satisfied.

        Returns (ready, reason).
        """
        for rel_path_template in module_def.required_input_manifests:
            path = Path(self.run_root) / rel_path_template
            if not path.is_file():
                return False, f"Required manifest missing: {path}"

        for rel_path_template in module_def.required_input_files:
            path_str = rel_path_template.format(
                run_root=str(self.run_root),
                raw_protein_heme_pdb=self.config.raw_protein_heme_pdb,
                ligand_template_sdf=self.config.ligand_template_sdf,
                protonation_decision_json=self.config.protonation_decision_json,
            )
            path = Path(self.run_root) / path_str
            if not path.is_file():
                return False, f"Required input file missing: {path}"

        return True, "All dependencies satisfied"

    def update_module_record(
        self,
        manifest: RunManifest,
        skill_id: str,
        record: ModuleRecord,
    ) -> None:
        """Update a module's record in the manifest and persist."""
        manifest.modules[skill_id] = {
            "skill_id": record.skill_id,
            "status": record.status,
            "steps": [
                {
                    "name": s.name,
                    "status": s.status,
                    "command": s.command,
                    "working_dir": s.working_dir,
                    "stdout_path": s.stdout_path,
                    "stderr_path": s.stderr_path,
                    "exit_code": s.exit_code,
                    "started_at": s.started_at,
                    "completed_at": s.completed_at,
                    "error_message": s.error_message,
                }
                for s in record.steps
            ],
            "gate_result": record.gate_result,
            "manifest_paths": record.manifest_paths,
            "output_dir": record.output_dir,
            "log_dir": record.log_dir,
            "errors": record.errors,
            "warnings": record.warnings,
            "summary": record.summary,
            "started_at": record.started_at,
            "completed_at": record.completed_at,
        }
        self.save_manifest(manifest)

    def get_ordered_modules(self) -> list[ModuleDef]:
        return list(self._module_defs)

    def get_module_def(self, skill_id: str) -> ModuleDef | None:
        for m in self._module_defs:
            if m.skill_id == skill_id:
                return m
        return None

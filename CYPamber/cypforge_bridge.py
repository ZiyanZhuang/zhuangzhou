"""Zhuangzhou bridge for the integrated CYPForge core.

The bridge exposes lightweight, non-running helpers so the AI DBTL layer can
inspect CYPForge capabilities and prepare a run configuration without launching
Amber, WSL, docking, or QM jobs implicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def cypforge_available() -> dict[str, Any]:
    try:
        import cypforge
        import cypforge_core
        from cypforge_core.orchestrator import build_module_definitions

        modules = build_module_definitions()
        return {
            "available": True,
            "cypforge_version": getattr(cypforge, "__version__", "unknown"),
            "module_count": len(modules),
            "modules": [
                {
                    "index": module.index,
                    "skill_id": module.skill_id,
                    "description": module.description,
                    "output_dir": module.output_dir_rel,
                }
                for module in modules
            ],
            "note": "CYPForge core is importable. Execution requires explicit run setup and external Amber/WSL tools.",
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def make_cypforge_run_config(
    run_name: str,
    run_root: str,
    raw_protein_heme_pdb: str,
    ligand_template_sdf: str = "",
    heme_state: str = "IC6",
    ligand_resname: str = "LIG",
    formal_charge: int = 0,
    project_root: str | None = None,
    **overrides: Any,
):
    """Create a CYPForge RunConfig object without executing the workflow."""
    from cypforge_core.orchestrator import RunConfig

    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[1]
    config = RunConfig(
        run_name=run_name,
        run_root=str(Path(run_root)),
        project_root=str(root),
        raw_protein_heme_pdb=str(Path(raw_protein_heme_pdb)),
        ligand_template_sdf=str(Path(ligand_template_sdf)) if ligand_template_sdf else "",
        heme_state=heme_state,
        ligand_resname=ligand_resname,
        formal_charge=formal_charge,
    )
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def initialize_cypforge_run(**kwargs: Any) -> dict[str, Any]:
    """Initialize CYPForge manifest files without running scientific modules."""
    from cypforge_core.orchestrator import CYPForgeOrchestrator

    config = make_cypforge_run_config(**kwargs)
    orchestrator = CYPForgeOrchestrator(config)
    manifest = orchestrator.init()
    return {
        "run_name": manifest.run_name,
        "run_root": manifest.run_root,
        "workflow_status": manifest.workflow_status,
        "manifest_path": str(Path(manifest.run_root) / "run_manifest.json"),
        "module_count": len(manifest.modules),
    }

"""
Tool Router — Routes tool calls to appropriate specialist agents.
Extracts lessons from execution results automatically.

The ToolRouter sits between AgentExecutor._execute_skill_with_ui() and the
actual skill call. It:
  1. Detects which domain a tool belongs to.
  2. Provides specialist context (learned lessons) before execution.
  3. Records outcome lessons after execution — silently and automatically.

This is intentionally a thin, side-effect-only layer: it never blocks
tool execution and never raises exceptions to the caller.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from zhuangzhou.nlp.specialist import SpecialistAgent, CATEGORY_TO_DOMAIN

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword-based domain detection (fast path; no LLM cost)
# ---------------------------------------------------------------------------

CATEGORY_KEYWORDS: Dict[str, list] = {
    "docking": [
        "docking", "dock", "vina", "binding", "affinity", "grid", "对接",
        "plip", "pose", "ligbuilder", "smartdocking", "batch_dock",
    ],
    "md_simulation": [
        "md", "gromacs", "amber", "simulation", "trajectory", "模拟",
        "equilibr", "nvt", "npt", "force_field", "cgenff", "topol",
        "metalloprotein", "xtc", "trr",
    ],
    "visualization": [
        "pymol", "chimerax", "render", "visualize", "图", "可视化",
        "view", "png", "ray", "cartoon",
    ],
    "mutation": [
        "mutation", "mutate", "mutagenesis", "突变", "engineering",
        "esm2", "variant", "saturation", "scan",
    ],
    "analysis": [
        "analysis", "rmsd", "rmsf", "plip", "分析", "analyze",
        "comprehensive", "trajectory_process", "energy",
    ],
    "chemistry": [
        "smiles", "rdkit", "conformer", "pubchem", "sdf", "mol2",
        "compound", "化合物", "qm", "dft", "pyscf",
    ],
    "structure": [
        "structure", "pdb", "prep", "mutagenesis", "missing_residue",
        "compare", "align", "alphafold",
    ],
    "literature": [
        "pubmed", "uniprot", "literature", "文献", "search_knowledge",
        "graphrag", "interpro", "ncbi",
    ],
}

# Skill-name prefix → domain (higher priority than keywords)
_SKILL_PREFIX_MAP: Dict[str, str] = {
    "smart_docking": "docking",
    "batch_dock": "docking",
    "plip_analyze": "docking",
    "run_vina": "docking",
    "run_protein_md": "md_simulation",
    "run_metalloprotein": "md_simulation",
    "gromacs": "md_simulation",
    "amber": "md_simulation",
    "auto_md": "md_simulation",
    "pymol": "visualization",
    "chimerax": "visualization",
    "generate_report": "visualization",
    "mutate_protein": "mutation",
    "esm2": "mutation",
    "generate_conformer": "chemistry",
    "download_pubchem": "chemistry",
    "qm_": "chemistry",
    "pyscf": "chemistry",
    "search_pubmed": "literature",
    "get_protein_profile": "literature",
    "search_knowledge": "literature",
    "graphrag_query": "literature",
    "get_pdb_entry": "structure",
    "search_pdb": "structure",
    "compare_structures": "structure",
    "codon_": "structure",
}


def _detect_domain_from_tool(tool_name: str, task: str = "") -> str:
    """Detect domain from tool name (prefix map) then task keywords.

    Returns a domain string or 'analysis' as safe default.
    """
    tool_lower = tool_name.lower()

    # 1. Exact prefix match on skill name (highest priority)
    for prefix, domain in _SKILL_PREFIX_MAP.items():
        if tool_lower.startswith(prefix):
            return domain

    # 2. Keyword match on tool name itself
    for domain, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in tool_lower:
                return domain

    # 3. Keyword match on task description (lower priority)
    if task:
        task_lower = task.lower()
        scores: Dict[str, int] = {}
        for domain, keywords in CATEGORY_KEYWORDS.items():
            for kw in keywords:
                if kw in task_lower:
                    scores[domain] = scores.get(domain, 0) + 1
        if scores:
            return max(scores, key=scores.__getitem__)

    return "analysis"  # safe default


# ---------------------------------------------------------------------------
# ToolRouter
# ---------------------------------------------------------------------------


class ToolRouter:
    """Routes tool invocations to specialist agents and accumulates lessons.

    Usage in AgentExecutor:
        self.router = ToolRouter(backend=self.backend, registry=self.registry)

        # Before execution — get enriched context for the LLM:
        context_hint = self.router.get_enriched_context(tool_name, user_task)

        # After execution — record outcome silently:
        self.router.record_tool_outcome(tool_name, result, user_task)
    """

    def __init__(self, backend=None, registry=None):
        self.backend = backend
        self.registry = registry
        self._specialists: Dict[str, SpecialistAgent] = {}

    # ------------------------------------------------------------------
    # Specialist access
    # ------------------------------------------------------------------

    def get_specialist(self, domain: str) -> SpecialistAgent:
        """Get or create specialist for domain (lazy-init)."""
        if domain not in self._specialists:
            self._specialists[domain] = SpecialistAgent(
                domain=domain,
                backend=self.backend,
                registry=self.registry,
            )
        return self._specialists[domain]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_domain(self, tool_name: str, task: str = "") -> str:
        """Detect which domain a tool call belongs to."""
        return _detect_domain_from_tool(tool_name, task)

    def get_enriched_context(self, tool_name: str, task: str = "") -> str:
        """Get specialist context (domain rules + lessons) to inject before a tool call.

        Returns an empty string if no lessons are available, so callers can
        safely skip injection when the specialist is fresh.
        """
        try:
            domain = self.detect_domain(tool_name, task)
            specialist = self.get_specialist(domain)
            return specialist._build_specialist_system_prompt()
        except Exception as e:
            logger.debug("get_enriched_context failed: %s", e)
            return ""

    def record_tool_outcome(
        self, tool_name: str, result: Any, task: str = ""
    ):
        """After tool execution, extract and record lessons automatically.

        Never raises — all errors are silently logged.
        """
        try:
            domain = self.detect_domain(tool_name, task)
            specialist = self.get_specialist(domain)

            if not isinstance(result, dict):
                return

            if "error" in result:
                err_msg = str(result["error"])[:80]
                hint = str(result.get("hint", "check params"))[:60]
                lesson = (
                    f"Tool {tool_name} failed: {err_msg}. Hint: {hint}"
                )
                specialist.record_lesson(tool_name, "failure", lesson)

            elif result.get("success") or any(
                k in result for k in (
                    "png_path", "docx_path", "pdb_path", "output_dir",
                    "sdf_path", "report_path", "trajectory", "mol2_path",
                )
            ):
                output_keys = [k for k in result if "path" in k or "dir" in k]
                outputs_str = ", ".join(output_keys) if output_keys else "completed"
                lesson = (
                    f"Tool {tool_name} succeeded; outputs: {outputs_str}"
                )
                specialist.record_lesson(tool_name, "success", lesson)

        except Exception as e:
            logger.debug("record_tool_outcome failed silently: %s", e)

    def all_summaries(self) -> Dict[str, Dict]:
        """Return lesson summaries for all loaded specialists."""
        return {
            domain: spec.summary()
            for domain, spec in self._specialists.items()
        }

    def get_all_lessons_text(self) -> str:
        """Return a human-readable multi-domain lesson report."""
        parts = []
        for domain, spec in self._specialists.items():
            if spec.lessons:
                parts.append(f"### {domain} ({spec.call_count} calls)")
                for l in spec.lessons[-5:]:
                    tag = "[OK]" if l.get("outcome") == "success" else "[FAIL]"
                    parts.append(f"  {tag} {l['lesson']}")
        return "\n".join(parts) if parts else "(no lessons recorded yet)"

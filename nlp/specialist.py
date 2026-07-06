"""
Specialist Agents — Domain-specific ReAct agents that accumulate expertise.

Each specialist manages one tool category and maintains a 'lesson log' of
what worked / what failed, injected as system prompt additions on next call.

This module provides the lesson-learning layer on top of the existing
zhuangzhou/agents/ multi-agent system. While agents/specialists.py defines
domain prompts, this module adds persistent per-call learning so each domain
agent improves from its own execution history.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# Domain Expertise Prompts
# =============================================================================

SPECIALIST_PROMPTS: Dict[str, str] = {
    "docking": """You are a molecular docking specialist.
Always: 1) Verify receptor and ligand files exist 2) Check for protonation state \
3) Define grid box from co-crystallized ligand or known binding site \
4) Run with exhaustiveness=16 for publication quality.
Common pitfalls: missing hydrogens on receptor, wrong ligand charge state, box too small.""",

    "md_simulation": """You are a molecular dynamics specialist.
Always: 1) Use C-rescale barostat for NPT 2) Wrap trajectory after EM \
3) Check RMSD convergence before analysis 4) For metalloproteins, use bonded metal models.
Key lessons: PBC artifacts appear as discontinuous RMSD; check with trjconv -pbc whole.""",

    "visualization": """You are a structural visualization specialist.
Always: 1) Remove solvent before rendering 2) For H-bonds use mode=2 with cutoff=4.2 \
3) Set ray_trace_mode=1 for publication 4) Save both detail and global views.""",

    "mutation": """You are a protein engineering specialist.
Always: 1) Check active site proximity before recommending mutations \
2) Avoid proline in loops 3) Check thermostability impact \
4) Verify no disulfide bond disruption.""",

    "analysis": """You are a computational biology analysis specialist.
Always: 1) Report mean +/- std for all metrics 2) Use convergence criteria before \
concluding stability 3) Cross-reference with experimental data when available.""",

    "chemistry": """You are a computational chemistry specialist.
Always: 1) Embed hydrogens before exporting structure 2) Use canonical SMILES for storage \
3) Verify drug-likeness (Lipinski RO5) 4) Check PAINS/toxicophores before synthesis recommendation.""",

    "structure": """You are a protein structure preparation specialist.
Always: 1) Check missing residues and chain breaks 2) Handle insertion codes \
3) Verify HETATM/ATOM labeling for PLIP vs GROMACS 4) Minimize after mutagenesis.""",

    "literature": """You are a bioinformatics literature and database specialist.
Always: 1) Confirm organism/species before reporting IDs \
2) CYP2A1=rat, CYP2A6=human — never conflate 3) Report PMID+DOI for key citations \
4) Cross-check GraphRAG KG for local curated knowledge.""",
}

# Map skill registry categories to specialist domain names
CATEGORY_TO_DOMAIN: Dict[str, str] = {
    "docking": "docking",
    "simulation": "md_simulation",
    "visualization": "visualization",
    "mutagenesis": "mutation",
    "plm": "mutation",
    "analysis": "analysis",
    "chemistry": "chemistry",
    "qm": "chemistry",
    "structure": "structure",
    "research": "literature",
    "brain": "literature",
}

# =============================================================================
# SpecialistAgent — learns from execution patterns
# =============================================================================


class SpecialistAgent:
    """A domain specialist that learns from execution patterns.

    Maintains a persistent lesson log under ~/.zhuangzhou/specialist_lessons/<domain>.json.
    Lessons are injected into the system prompt on subsequent calls so the LLM
    benefits from what worked and what failed in previous runs.
    """

    def __init__(
        self,
        domain: str,
        backend=None,
        registry=None,
        max_lessons: int = 10,
    ):
        self.domain = domain
        self.backend = backend
        self.registry = registry
        self.max_lessons = max_lessons
        self.lessons: List[Dict] = []
        self.call_count: int = 0

        self._lesson_file = (
            Path.home() / ".zhuangzhou" / "specialist_lessons" / f"{domain}.json"
        )
        self._load_lessons()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_lessons(self):
        """Load persisted lessons from disk."""
        try:
            if self._lesson_file.exists():
                with open(self._lesson_file, encoding="utf-8") as f:
                    data = json.load(f)
                self.lessons = data.get("lessons", [])
                self.call_count = data.get("call_count", 0)
        except Exception as e:
            logger.debug("Could not load lessons for domain '%s': %s", self.domain, e)
            self.lessons = []

    def _save_lessons(self):
        """Persist lessons to disk."""
        try:
            self._lesson_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "domain": self.domain,
                "call_count": self.call_count,
                "lessons": self.lessons,
            }
            with open(self._lesson_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.debug("Could not save lessons for domain '%s': %s", self.domain, e)

    # ------------------------------------------------------------------
    # Prompt Construction
    # ------------------------------------------------------------------

    def _build_specialist_system_prompt(self) -> str:
        """Build system prompt with domain expertise + learned lessons."""
        base = SPECIALIST_PROMPTS.get(self.domain, "")
        if not base:
            base = f"You are a specialist for the '{self.domain}' domain."

        if self.lessons:
            recent = self.lessons[-5:]
            lesson_lines = []
            for l in recent:
                outcome_tag = "[OK]" if l.get("outcome") == "success" else "[FAIL]"
                lesson_lines.append(f"  {outcome_tag} {l['lesson']}")
            lesson_text = "\n".join(lesson_lines)
            base += (
                f"\n\nLearned patterns from previous {self.domain} runs "
                f"({self.call_count} total calls):\n{lesson_text}"
            )
        return base

    # ------------------------------------------------------------------
    # Lesson Recording
    # ------------------------------------------------------------------

    def record_lesson(self, tool: str, outcome: str, lesson: str):
        """Record what worked/failed for this domain.

        Args:
            tool: Name of the tool/skill that was called.
            outcome: 'success' or 'failure'.
            lesson: Human-readable description of the pattern observed.
        """
        self.call_count += 1
        entry = {
            "tool": tool,
            "outcome": outcome,  # 'success' | 'failure'
            "lesson": lesson,
            "count": self.call_count,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.lessons.append(entry)
        # Keep only last max_lessons entries
        self.lessons = self.lessons[-self.max_lessons :]
        self._save_lessons()

    # ------------------------------------------------------------------
    # Summary / Stats
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        """Return a compact summary of this specialist's lesson log."""
        success_count = sum(1 for l in self.lessons if l.get("outcome") == "success")
        failure_count = len(self.lessons) - success_count
        return {
            "domain": self.domain,
            "call_count": self.call_count,
            "lesson_count": len(self.lessons),
            "success": success_count,
            "failure": failure_count,
            "lesson_file": str(self._lesson_file),
        }

    def clear_lessons(self):
        """Remove all persisted lessons for this domain."""
        self.lessons = []
        self.call_count = 0
        if self._lesson_file.exists():
            self._lesson_file.unlink()

    def __repr__(self) -> str:
        return (
            f"SpecialistAgent(domain={self.domain!r}, "
            f"lessons={len(self.lessons)}, calls={self.call_count})"
        )

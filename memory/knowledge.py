"""
Knowledge Store — semantic memory
===================================
Learns from experience: successful workflows, mutation insights,
common error patterns, and user preferences.

Structure:
  ~/.zhuangzhou/knowledge/
    mutation_insights.json    - learned mutation patterns
    workflow_history.json     - successful workflow configs
    error_patterns.json       - common errors + solutions
    preferences.json          - inferred user preferences
"""

import json
import os
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional


KNOWLEDGE_DIR = os.path.expanduser("~/.zhuangzhou/knowledge")


class KnowledgeStore:
    """Persistent learned knowledge — gets smarter with use."""

    def __init__(self, knowledge_dir: str = None):
        self.knowledge_dir = knowledge_dir or KNOWLEDGE_DIR
        os.makedirs(self.knowledge_dir, exist_ok=True)
        self._mutation_insights = self._load("mutation_insights.json")
        self._workflow_history = self._load("workflow_history.json")
        self._error_patterns = self._load("error_patterns.json")
        self._preferences = self._load("preferences.json")
        self._skill_patterns = self._load("skill_patterns.json")
        # Auto-decay stale chain entries (runs at most once per day via mtime check)
        self._auto_decay()

    # ── Mutation Insights ──────────────────────────────────────────────

    def record_mutation_result(
        self,
        mutation: str,
        protein: str,
        score: float,
        outcome: str,
        details: dict = None,
    ):
        """Record a mutation analysis outcome for learning."""
        key = f"{protein}:{mutation}"
        if "mutations" not in self._mutation_insights:
            self._mutation_insights["mutations"] = {}
        self._mutation_insights["mutations"][key] = {
            "mutation": mutation,
            "protein": protein,
            "score": score,
            "outcome": outcome,
            "details": details or {},
        }
        self._save("mutation_insights.json", self._mutation_insights)

    def get_mutation_history(self, protein: str = None) -> List[dict]:
        """Get historical mutation results, optionally filtered by protein."""
        all_muts = self._mutation_insights.get("mutations", {})
        results = list(all_muts.values())
        if protein:
            results = [m for m in results if m.get("protein") == protein]
        return results

    def get_similar_mutations(self, mutation: str) -> List[dict]:
        """Find historically similar mutations (same position type, same AA change)."""
        if len(mutation) < 3:
            return []
        # Only handle standard single-letter format (e.g. V21L, A5G)
        # Non-standard formats (3-letter, deletions, insertions) return empty safely
        if not (mutation[0].isalpha() and mutation[-1].isalpha() and mutation[1:-1].isdigit()):
            return []
        wt, mut = mutation[0].upper(), mutation[-1].upper()
        change = f"{wt}->{mut}"
        all_muts = self._mutation_insights.get("mutations", {})
        similar = []
        for m in all_muts.values():
            m_change = f"{m['mutation'][0]}->{m['mutation'][-1]}"
            if m_change == change:
                similar.append(m)
        return similar

    # ── Workflow History ───────────────────────────────────────────────

    def record_workflow(
        self, name: str, inputs: dict, success: bool, elapsed: float, notes: str = ""
    ):
        """Record a workflow execution for learning."""
        if "runs" not in self._workflow_history:
            self._workflow_history["runs"] = []
        self._workflow_history["runs"].append(
            {
                "name": name,
                "inputs_summary": {k: str(v)[:100] for k, v in inputs.items()},
                "success": success,
                "elapsed": elapsed,
                "notes": notes,
            }
        )
        # Keep last 100
        self._workflow_history["runs"] = self._workflow_history["runs"][-100:]
        self._save("workflow_history.json", self._workflow_history)

    def get_workflow_stats(self) -> dict:
        """Get workflow success/failure statistics."""
        runs = self._workflow_history.get("runs", [])
        by_name = Counter()
        successes = Counter()
        for r in runs:
            by_name[r["name"]] += 1
            if r["success"]:
                successes[r["name"]] += 1
        stats = {}
        for name in by_name:
            stats[name] = {
                "total": by_name[name],
                "success": successes[name],
                "rate": successes[name] / by_name[name] if by_name[name] else 0,
            }
        return stats

    # ── Error Patterns ─────────────────────────────────────────────────

    def record_error(self, error_type: str, context: str, solution: str = ""):
        """Record an error pattern and its solution."""
        if "errors" not in self._error_patterns:
            self._error_patterns["errors"] = {}
        key = error_type
        if key not in self._error_patterns["errors"]:
            self._error_patterns["errors"][key] = {
                "count": 0,
                "contexts": [],
                "solutions": [],
            }
        entry = self._error_patterns["errors"][key]
        entry["count"] += 1
        ctx = context[:200]
        if ctx not in entry["contexts"]:
            entry["contexts"].append(ctx)
            entry["contexts"] = entry["contexts"][-5:]  # keep last 5
        if solution and solution not in entry["solutions"]:
            entry["solutions"].append(solution)
        self._save("error_patterns.json", self._error_patterns)

    def suggest_solution(self, error_type: str) -> Optional[str]:
        """Suggest a solution based on past errors."""
        errors = self._error_patterns.get("errors", {})
        entry = errors.get(error_type)
        if entry and entry.get("solutions"):
            return entry["solutions"][-1]  # most recent solution
        return None

    # ── Skill Chain Patterns (Dynamic Memory) ─────────────────────────

    def record_skill_chain(
        self, chain: List[str], context: str, success: bool, total_elapsed: float = 0
    ):
        """Record a skill execution chain and its outcome.

        Tracks patterns like:
          "CYP + CHARMM27 + metalloprotein_md → 95% success"
          "mutate_minimize → smart_docking → plip_analyze → 80% success"

        Args:
            chain: ordered list of skill names that were executed
            context: brief description of the target (e.g. "CYP3A4 V21L")
            success: whether the chain completed successfully
            total_elapsed: total time in seconds
        """
        if "chains" not in self._skill_patterns:
            self._skill_patterns["chains"] = {}

        key = " → ".join(chain)
        if key not in self._skill_patterns["chains"]:
            self._skill_patterns["chains"][key] = {
                "count": 0,
                "successes": 0,
                "contexts": [],
                "avg_elapsed": 0,
            }
        entry = self._skill_patterns["chains"][key]
        entry["count"] += 1
        entry["last_accessed"] = datetime.now().isoformat()
        if success:
            entry["successes"] += 1
        if context not in entry["contexts"]:
            entry["contexts"].append(context[:100])
            entry["contexts"] = entry["contexts"][-5:]
        # Running average
        if total_elapsed > 0:
            entry["avg_elapsed"] = (
                entry["avg_elapsed"] * (entry["count"] - 1) + total_elapsed
            ) / entry["count"]
        self._save("skill_patterns.json", self._skill_patterns)

    def record_skill_failure(
        self,
        skill_name: str,
        context: str,
        error: str,
        params_summary: str = "",
        traceback_str: str = "",
        input_files: list = None,
        root_cause: str = "",
    ):
        """Record a skill failure with full context for root cause analysis.

        Enhanced from v0.6: now captures traceback, input files, and optional
        root_cause classification for reverse failure analysis.

        Root cause categories (auto or manual):
          - 'input_error': bad file path, wrong format, missing file
          - 'param_error': invalid parameter combination
          - 'env_error': missing dependency, sandbox unavailable
          - 'compute_error': OOM, timeout, numerical instability
          - 'data_error': corrupt PDB, invalid SMILES, wrong species
          - 'workflow_error': wrong step order, missing prerequisite
        """
        if "failures" not in self._skill_patterns:
            self._skill_patterns["failures"] = {}

        if skill_name not in self._skill_patterns["failures"]:
            self._skill_patterns["failures"][skill_name] = []

        # Auto-classify root cause if not provided
        if not root_cause:
            root_cause = self._classify_failure(error, traceback_str)

        self._skill_patterns["failures"][skill_name].append(
            {
                "timestamp": __import__("datetime").datetime.now().isoformat(),
                "context": context[:300],
                "error": error[:500],
                "params": params_summary[:300],
                "traceback": traceback_str[:500] if traceback_str else "",
                "input_files": (input_files or [])[:5],
                "root_cause": root_cause,
            }
        )
        # Keep last 15 per skill (was 10)
        self._skill_patterns["failures"][skill_name] = self._skill_patterns["failures"][
            skill_name
        ][-15:]
        self._save("skill_patterns.json", self._skill_patterns)

    @staticmethod
    def _classify_failure(error: str, traceback_str: str = "") -> str:
        """Auto-classify failure root cause from error/traceback text."""
        text = (error + " " + traceback_str).lower()
        if any(
            kw in text
            for kw in (
                "no such file",
                "not found",
                "does not exist",
                "filenotfounderror",
            )
        ):
            return "input_error"
        # env_error before data_error ('import rdkit' is env, not data)
        if any(
            kw in text
            for kw in (
                "importerror",
                "modulenotfounderror",
                "not installed",
                "no module named",
                "unavailable",
                "not available",
                "conda",
                "sandbox",
            )
        ):
            return "env_error"
        # data_error before param_error (both match 'invalid')
        if any(
            kw in text
            for kw in (
                "smiles",
                "corrupt",
                "invalid pdb",
                "rdkit",
                "kekulize",
                "kekulization",
                "mol from",
                "sanitize",
            )
        ):
            return "data_error"
        if any(
            kw in text
            for kw in ("timeout", "oom", "memory", "cuda", "nan", "lincs", "blowup")
        ):
            return "compute_error"
        if any(
            kw in text
            for kw in (
                "prerequisite",
                "before",
                "first run",
                "depends",
                "missing output",
            )
        ):
            return "workflow_error"
        if any(
            kw in text
            for kw in ("invalid", "unexpected", "typeerror", "valueerror", "parameter")
        ):
            return "param_error"
        return "unknown"

    def get_chain_success_rate(self, chain: List[str]) -> Optional[float]:
        """Get historical success rate for a skill chain."""
        key = " → ".join(chain)
        chains = self._skill_patterns.get("chains", {})
        entry = chains.get(key)
        if entry and entry["count"] > 0:
            return entry["successes"] / entry["count"]
        return None

    def get_skill_failure_hints(self, skill_name: str) -> List[str]:
        """Get recent failure patterns for a skill to inject as warnings."""
        failures = self._skill_patterns.get("failures", {})
        entries = failures.get(skill_name, [])
        return [f"{e['context']}: {e['error']}" for e in entries[-3:]]

    def get_recommended_chains(self, context: str = "") -> List[dict]:
        """Get historically successful chains, optionally filtered by context.

        v0.7.5: Respects _weight from memory decay — old unused chains rank lower.
        """
        chains = self._skill_patterns.get("chains", {})
        results = []
        for key, entry in chains.items():
            if entry["count"] < 2:
                continue
            rate = entry["successes"] / entry["count"]
            if rate < 0.5:
                continue
            if context:
                if not any(context.lower() in c.lower() for c in entry["contexts"]):
                    continue
            weight = entry.get("_weight", 1.0)
            results.append(
                {
                    "chain": key,
                    "success_rate": round(rate * weight, 2),
                    "count": entry["count"],
                    "avg_elapsed": round(entry["avg_elapsed"], 1),
                }
            )
        results.sort(key=lambda x: (-x["success_rate"], -x["count"]))
        return results[:5]

    def _auto_decay(self):
        """Run decay_old_entries() at most once per day (based on file mtime)."""
        marker = os.path.join(self.knowledge_dir, ".last_decay")
        try:
            if os.path.exists(marker):
                age_hours = (datetime.now().timestamp() - os.path.getmtime(marker)) / 3600
                if age_hours < 24:
                    return
            self.decay_old_entries()
            # Touch the marker file
            with open(marker, "w", encoding="utf-8") as f:
                f.write(datetime.now().isoformat())
        except OSError:
            pass  # non-fatal

    def decay_old_entries(self, max_age_days: int = 90):
        """Decay entries not accessed within max_age_days.

        Reduces _weight of stale chain entries so they rank lower
        in get_recommended_chains(). Does not delete anything.
        """
        now = datetime.now()
        chains = self._skill_patterns.get("chains", {})
        changed = False
        for key, entry in chains.items():
            last = entry.get("last_accessed") or entry.get("_last_used")
            if not last:
                continue
            try:
                last_dt = datetime.fromisoformat(last)
                age_days = (now - last_dt).days
                if age_days > max_age_days:
                    new_weight = max(0.1, 1.0 - age_days / 365)
                    if entry.get("_weight", 1.0) != new_weight:
                        entry["_weight"] = round(new_weight, 3)
                        changed = True
            except (ValueError, TypeError):
                continue
        if changed:
            self._save("skill_patterns.json", self._skill_patterns)

    # ── User Preferences ───────────────────────────────────────────────

    def track_preference(self, key: str, value: Any):
        """Track a user preference (inferred or explicit)."""
        self._preferences[key] = value
        self._save("preferences.json", self._preferences)

    def get_preference(self, key: str, default: Any = None) -> Any:
        return self._preferences.get(key, default)

    def get_skill_stats(self) -> dict:
        """Merged skill stats view combining workflow history and execution history."""
        stats = self.get_workflow_stats()
        try:
            from zhuangzhou.memory.execution_history import ExecutionHistory

            recent = ExecutionHistory().get_recent(limit=1000)
            exec_totals = Counter()
            exec_success = Counter()
            for entry in recent:
                name = entry.get("skill", "")
                if not name:
                    continue
                exec_totals[name] += 1
                if entry.get("success"):
                    exec_success[name] += 1
            for name in set(list(stats.keys()) + list(exec_totals.keys())):
                item = stats.setdefault(name, {"total": 0, "success": 0, "rate": 0})
                item["exec_total"] = exec_totals.get(name, 0)
                item["exec_success"] = exec_success.get(name, 0)
                item["exec_rate"] = (
                    exec_success.get(name, 0) / exec_totals[name]
                    if exec_totals.get(name, 0)
                    else 0
                )
        except Exception:
            pass
        return stats

    def get_execution_summary(self) -> dict:
        """Expose persistent execution summary via KnowledgeStore."""
        try:
            from zhuangzhou.memory.execution_history import ExecutionHistory

            return ExecutionHistory().summary()
        except Exception:
            return {}

    def query_local_graph(self, entity: str, limit: int = 5) -> List[dict]:
        """Query local KG triples via KnowledgeStore unified interface."""
        try:
            from zhuangzhou.memory.kg_engine import LocalKGEngine

            return LocalKGEngine.query(entity, limit=limit)
        except Exception:
            return []

    # ── I/O ────────────────────────────────────────────────────────────

    def _load(self, filename: str) -> dict:
        path = os.path.join(self.knowledge_dir, filename)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save(self, filename: str, data: dict):
        path = os.path.join(self.knowledge_dir, filename)
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            os.replace(tmp, path)  # atomic on same filesystem
        except OSError:
            pass  # disk full / read-only mount — non-fatal, memory just won't persist

    def summary(self) -> dict:
        """Get knowledge store summary."""
        return {
            "mutations_recorded": len(self._mutation_insights.get("mutations", {})),
            "workflows_recorded": len(self._workflow_history.get("runs", [])),
            "error_patterns": len(self._error_patterns.get("errors", {})),
            "skill_chains": len(self._skill_patterns.get("chains", {})),
            "skill_failures": sum(
                len(v) for v in self._skill_patterns.get("failures", {}).values()
            ),
            "preferences": len(self._preferences),
            "knowledge_dir": self.knowledge_dir,
        }

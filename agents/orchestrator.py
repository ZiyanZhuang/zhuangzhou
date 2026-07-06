"""
OrchestratorAgent — Central Routing + Aggregation Hub
======================================================
Pattern: AutoGen-inspired hub-spoke.
  - One orchestrator receives user requests
  - Routes to one or more domain specialists
  - Aggregates results into a coherent response
  - Maintains shared state (last PDB path, active session, etc.)

Routing logic is LLM-assisted with rule-based fast path:
  keywords → domain → specialist agent (lazy-initialized on first use)

Multi-key API: each specialist can share a key pool or use its own key.
Since DeepSeek rate limits are account-level, distributing across specialists
doesn't increase total throughput — but it does provide fault isolation.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Routing keywords → specialist domain
_ROUTING_RULES: List[tuple] = [
    # (keywords_list, domain_name)
    (["md", "gromacs", "动力学", "trajectory", "xtc", "trr", "topol", "mdp",
      "equilibr", "平衡", "模拟", "simulation", "force field", "力场",
      "metalloprotein", "heme", "cgenff", "gaff"], "md"),

    (["dock", "vina", "对接", "binding", "结合", "plip", "interaction",
      "相互作用", "active site", "活性位点", "ligbuilder", "pose"], "docking"),

    (["pubmed", "literature", "文献", "uniprot", "pdb search", "alphafold",
      "protein profile", "interpro", "domain", "功能", "function",
      "ncbi", "gene", "sequence search", "homolog"], "literature"),

    (["smiles", "rdkit", "conformer", "构象", "pubchem", "sdf", "mol2",
      "molecule", "化合物", "compound", "drug-like", "lipinski",
      "qm", "dft", "pyscf", "quantum", "量子化学", "b3lyp"], "chemistry"),

    (["mutate", "mutation", "突变", "esm2", "scan", "saturation",
      "mutagenesis", "variant", "变异", "missing residue", "pdb prep",
      "structure prep", "结构准备", "compare structure", "align"], "structure"),

    (["memory", "知识库", "kg", "graph", "记忆", "recall", "history",
      "pattern", "project memory", "context"], "memory"),
]


def _route_to_domain(query: str) -> Optional[str]:
    """Fast keyword routing. Returns domain name or None (→ main agent)."""
    q = query.lower()
    scores: Dict[str, int] = {}
    for keywords, domain in _ROUTING_RULES:
        for kw in keywords:
            if kw in q:
                scores[domain] = scores.get(domain, 0) + 1
    if not scores:
        return None
    return max(scores, key=scores.__getitem__)


class SharedState:
    """Shared mutable state accessible by all specialists via orchestrator."""

    def __init__(self):
        self.active_pdb: Optional[str] = None
        self.active_ligand: Optional[str] = None
        self.active_work_dir: Optional[str] = None
        self.last_skill_result: Optional[Dict] = None
        self.session_id: str = f"orch_{int(time.time())}"

    def update_from_result(self, result: Any):
        """Extract paths from skill results and update shared state."""
        if not isinstance(result, dict):
            return
        for key in ("pdb_path", "output_pdb", "prepared_pdb"):
            if key in result and isinstance(result[key], str):
                self.active_pdb = result[key]
                break
        for key in ("ligand_path", "output_sdf", "sdf_path"):
            if key in result and isinstance(result[key], str):
                self.active_ligand = result[key]
                break
        for key in ("output_dir", "work_dir"):
            if key in result and isinstance(result[key], str):
                self.active_work_dir = result[key]
                break
        self.last_skill_result = result

    def to_context_str(self) -> str:
        """Compact context string for injection into specialist prompts."""
        parts = []
        if self.active_pdb:
            parts.append(f"活跃PDB: {self.active_pdb}")
        if self.active_ligand:
            parts.append(f"活跃配体: {self.active_ligand}")
        if self.active_work_dir:
            parts.append(f"工作目录: {self.active_work_dir}")
        return " | ".join(parts) if parts else ""


class OrchestratorAgent:
    """Hub-spoke orchestrator for Zhuangzhou multi-agent system.

    Routing strategy:
      1. Keyword-based fast path (no LLM cost)
      2. Route to specialist (lazy-initialized)
      3. If no specialist matches, fall back to main ZhuangzhouAgent

    Specialists are lazy-initialized on first use to avoid paying startup
    cost for domains the user doesn't touch.
    """

    def __init__(
        self,
        backend=None,
        api_key: Optional[str] = None,
        verbose: bool = False,
        use_specialists: bool = True,
    ):
        self.verbose = verbose
        self.use_specialists = use_specialists
        self.shared_state = SharedState()
        self._backend = backend
        self._api_key = api_key
        self._specialists: Dict[str, Any] = {}  # lazy cache
        self._main_agent = None  # lazy

    def _get_specialist(self, domain: str):
        """Lazy-initialize specialist on first access."""
        if domain not in self._specialists:
            try:
                from .specialists import create_specialist
                agent = create_specialist(domain, backend=self._backend, api_key=self._api_key)
                self._specialists[domain] = agent
                logger.debug("Initialized specialist: %s", domain)
            except Exception as e:
                logger.warning("Could not init specialist '%s': %s", domain, e)
                self._specialists[domain] = None
        return self._specialists[domain]

    def _get_main_agent(self):
        """Lazy-initialize main ZhuangzhouAgent as fallback."""
        if self._main_agent is None:
            try:
                from zhuangzhou.nlp.agent import ZhuangzhouAgent
                self._main_agent = ZhuangzhouAgent(
                    backend=self._backend,
                    api_key=self._api_key,
                    verbose=self.verbose,
                )
            except Exception as e:
                logger.error("Could not init main agent: %s", e)
        return self._main_agent

    def route(self, query: str, status_monitor=None) -> str:
        """Route a query to the best specialist or main agent.

        Returns the response text.
        """
        # Inject shared state context into query if available
        ctx = self.shared_state.to_context_str()
        enriched_query = f"[上下文: {ctx}]\n{query}" if ctx else query

        if self.use_specialists:
            domain = _route_to_domain(query)
            if domain:
                specialist = self._get_specialist(domain)
                if specialist is not None:
                    if self.verbose:
                        logger.info("Routing to specialist: %s", domain)
                    try:
                        result = specialist.chat(enriched_query, status_monitor=status_monitor)
                        return result
                    except Exception as e:
                        logger.warning("Specialist %s failed, falling back: %s", domain, e)

        # Fallback: main agent handles everything
        main = self._get_main_agent()
        if main is None:
            return f"[error] Failed to initialize any agent for query: {query}"
        return main.chat(enriched_query, status_monitor=status_monitor)

    def reset_all(self):
        """Reset conversation history for all agents."""
        for spec in self._specialists.values():
            if spec is not None:
                spec.reset()
        if self._main_agent:
            self._main_agent.reset()
        self.shared_state = SharedState()

    def status(self) -> Dict:
        """Return current orchestrator status."""
        initialized = [k for k, v in self._specialists.items() if v is not None]
        return {
            "session_id": self.shared_state.session_id,
            "specialists_loaded": initialized,
            "main_agent_loaded": self._main_agent is not None,
            "shared_state": self.shared_state.to_context_str(),
            "use_specialists": self.use_specialists,
        }

    def toggle_specialists(self) -> bool:
        """Enable/disable specialist routing."""
        self.use_specialists = not self.use_specialists
        return self.use_specialists


# Global singleton factory
_global_orchestrator: Optional[OrchestratorAgent] = None


def get_orchestrator(
    backend=None,
    api_key: Optional[str] = None,
    verbose: bool = False,
    use_specialists: bool = True,
    reset: bool = False,
) -> OrchestratorAgent:
    """Get or create the global orchestrator singleton."""
    global _global_orchestrator
    if _global_orchestrator is None or reset:
        _global_orchestrator = OrchestratorAgent(
            backend=backend,
            api_key=api_key,
            verbose=verbose,
            use_specialists=use_specialists,
        )
    return _global_orchestrator

"""
SpecialistAgent — Domain-tuned thin wrapper over AgentExecutor
==============================================================
Each specialist carries:
  1. A domain-specific system prompt addendum (tool paths, constraints, best practices)
  2. A curated skill category subset (faster schema loading, better focus)
  3. Its own conversation history (isolated from main agent)

Design: thin wrapper — we reuse AgentExecutor internals rather than reimplementing.
The specialist overrides the system prompt section and pre-loads only its categories.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SpecialistAgent:
    """Base class for domain specialist agents.

    Subclass and override:
      - name: str               — agent identifier
      - description: str        — one-line role description
      - skill_categories: list  — which skill categories to pre-load
      - system_addendum: str    — domain-specific rules injected into system prompt
    """

    name: str = "specialist"
    description: str = "Domain specialist"
    skill_categories: List[str] = []
    system_addendum: str = ""

    def __init__(self, backend=None, api_key: Optional[str] = None, verbose: bool = False):
        """Initialize specialist with its own AgentExecutor instance."""
        self.verbose = verbose
        self._executor: Optional[Any] = None
        self._backend = backend
        self._api_key = api_key
        self._initialized = False

    def _lazy_init(self):
        """Defer heavy initialization until first use."""
        if self._initialized:
            return
        try:
            from zhuangzhou.nlp.agent import AgentExecutor
            from zhuangzhou.nlp.backends import get_backend

            backend = self._backend
            if backend is None or isinstance(backend, str):
                backend = get_backend(backend or "auto", api_key=self._api_key)

            self._executor = AgentExecutor(backend=backend)

            # Pre-load domain categories for faster first response
            if self.skill_categories:
                for cat in self.skill_categories:
                    try:
                        self._executor.load_category(cat)
                    except Exception as e:
                        logger.debug("Could not pre-load category %s: %s", cat, e)

            self._initialized = True
            logger.debug("SpecialistAgent '%s' initialized", self.name)
        except Exception as e:
            logger.warning("SpecialistAgent '%s' init failed: %s", self.name, e)
            raise

    def chat(self, message: str, status_monitor=None) -> str:
        """Run a single turn. Returns response text."""
        self._lazy_init()
        if self._executor is None:
            return f"[{self.name}] initialization failed"
        try:
            return self._executor.run(message, status_monitor=status_monitor)
        except Exception as e:
            logger.error("SpecialistAgent '%s' chat error: %s", self.name, e)
            return f"[{self.name}] error: {e}"

    def get_system_addendum(self) -> str:
        """Return the domain-specific system prompt addition."""
        return self.system_addendum

    def reset(self):
        """Clear conversation history."""
        if self._executor:
            self._executor.messages.clear()

    @property
    def initialized(self) -> bool:
        return self._initialized

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, categories={self.skill_categories})"

"""
Memory Registry — Global singleton for all memory layers
==========================================================
Eliminates duplicate instantiation across conversation.py, nlp/agent.py, etc.

Usage:
    from zhuangzhou.memory.registry import MemoryRegistry
    reg = MemoryRegistry.get()
    reg.knowledge.get_similar_mutations('V21L')
    reg.kg_query('CYP3A4', limit=5)
"""

import threading


class MemoryRegistry:
    """Global singleton — all memory layers share one entry point."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._knowledge = None
        self._context = None
        self._project_memory = None
        self._exec_history = None
        self._checkpoint = None
        self._workspace = None
        self._workspace_intel = None
        self._journal = None
        self._session_archive = None

    @classmethod
    def get(cls) -> "MemoryRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    # -- Lazy properties (import on first access to avoid circular imports) --

    @property
    def knowledge(self):
        if self._knowledge is None:
            from zhuangzhou.memory.knowledge import KnowledgeStore

            self._knowledge = KnowledgeStore()
        return self._knowledge

    @property
    def context(self):
        if self._context is None:
            from zhuangzhou.memory.context import SessionContext

            self._context = SessionContext()
        return self._context

    @property
    def project_memory(self):
        if self._project_memory is None:
            from zhuangzhou.memory.project_memory import ProjectMemory

            self._project_memory = ProjectMemory()
        return self._project_memory

    @property
    def exec_history(self):
        if self._exec_history is None:
            from zhuangzhou.memory.execution_history import ExecutionHistory

            self._exec_history = ExecutionHistory()
        return self._exec_history

    @property
    def checkpoint(self):
        if self._checkpoint is None:
            from zhuangzhou.memory.execution_history import TaskCheckpoint

            self._checkpoint = TaskCheckpoint()
        return self._checkpoint

    @property
    def workspace(self):
        if self._workspace is None:
            from zhuangzhou.memory.execution_history import WorkspaceTracker

            self._workspace = WorkspaceTracker()
        return self._workspace

    @property
    def workspace_intel(self):
        if self._workspace_intel is None:
            from zhuangzhou.memory.workspace_intelligence import WorkspaceIntelligence

            self._workspace_intel = WorkspaceIntelligence()
        return self._workspace_intel

    @property
    def journal(self):
        if self._journal is None:
            from zhuangzhou.memory.journal import Journal

            self._journal = Journal()
        return self._journal

    @property
    def session_archive(self):
        if self._session_archive is None:
            from zhuangzhou.memory.archive import SessionArchive

            self._session_archive = SessionArchive()
        return self._session_archive

    # -- KG convenience --

    def kg_query(self, query: str, limit: int = 5):
        """Query local KG triples (from kg_engine.py)."""
        from zhuangzhou.memory.kg_engine import LocalKGEngine

        return LocalKGEngine.query(query, limit=limit)

    def kg_stats(self):
        """Get local KG statistics."""
        from zhuangzhou.memory.kg_engine import LocalKGEngine

        return LocalKGEngine.stats()

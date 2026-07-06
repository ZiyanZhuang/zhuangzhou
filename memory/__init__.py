"""
Zhuangzhou Memory System
=========================
Persistent memory that makes Zhuangzhou "get smarter with use."

Six layers (with Journal now explicitly demoted to lightweight logging):

  1. Journal           - lightweight event log (UI/runtime breadcrumbs)
  2. Context           - session state (working: current state)
  3. Knowledge         - learned patterns (semantic: what was learned)
  4. Archive           - canonical full interaction recordings (complete lab notebook)
  5. ProjectMemory     - project-context-style accumulating plain text (high SNR context)
  6. ExecutionHistory  - persistent skill execution log + task checkpoint (cross-session)
"""

from .journal import Journal
from .context import SessionContext
from .knowledge import KnowledgeStore
from .archive import SessionArchive
from .project_memory import ProjectMemory
from .execution_history import ExecutionHistory, TaskCheckpoint, WorkspaceTracker
from .workspace_intelligence import WorkspaceIntelligence
from .registry import MemoryRegistry

__all__ = [
    "Journal",
    "SessionContext",
    "KnowledgeStore",
    "SessionArchive",
    "ProjectMemory",
    "ExecutionHistory",
    "TaskCheckpoint",
    "WorkspaceTracker",
    "WorkspaceIntelligence",
    "MemoryRegistry",
]


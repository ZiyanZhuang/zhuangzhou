"""
Zhuangzhou Workflow Engine
===========================
JSON-driven pipeline orchestration.

A workflow is a sequence of steps, each step calls a skill with parameters.
Workflows can be templated, versioned, and extended as new tools arrive.
"""

from .engine import WorkflowEngine

__all__ = ['WorkflowEngine']

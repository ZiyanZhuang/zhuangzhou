"""
Zhuangzhou Multi-Agent System — v0.8.0
=======================================
Hub-spoke orchestration: one Orchestrator routes tasks to domain Specialists.
Each specialist carries a tailored system prompt + curated skill subset,
boosting tool-call accuracy for its domain.

Components:
  pool.py         — APIKeyPool: round-robin, per-key rate tracking
  base.py         — SpecialistAgent: thin wrapper with domain system prompt
  specialists.py  — MDAgent, DockingAgent, LiteratureAgent, ChemAgent, StructureAgent, MemoryAgent
  orchestrator.py — OrchestratorAgent: intent routing + result aggregation

Quick start:
  from zhuangzhou.agents import get_orchestrator
  orc = get_orchestrator(backend='deepseek')
  result = orc.route("分析CYP3A4的V21L突变对底物结合的影响")
"""

from .orchestrator import OrchestratorAgent, get_orchestrator
from .pool import APIKeyPool
from .base import SpecialistAgent

__all__ = ["OrchestratorAgent", "get_orchestrator", "APIKeyPool", "SpecialistAgent"]

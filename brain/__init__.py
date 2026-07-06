"""
Zhuangzhou Brain — BeTaThink Integration
==========================================
BeTaThink is Zhuangzhou's "hippocampus" — the knowledge backend.

Provides:
  1. KG auto-construction (PDF -> entities -> graph)
  2. KG querying (entity search, path finding)
  3. RAG retrieval (vector + graph fusion)
  4. Agent orchestration (multi-agent pipelines)
  5. Health monitoring of backend services

Works in two modes:
  - Online:  BeTaThink API running (PostgreSQL, Neo4j, Milvus, Redis)
  - Offline: Local-only mode (file-based, no external DB needed)
"""

from .api import BeTaThinkClient
from .connector import BrainConnector
from .graphrag import GraphRAGBridge

__all__ = ['BeTaThinkClient', 'BrainConnector', 'GraphRAGBridge']

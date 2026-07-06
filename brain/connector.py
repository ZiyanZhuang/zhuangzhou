"""
Brain Connector
================
High-level interface that unifies:
  - BeTaThink API (online mode)
  - Local file-based fallback (offline mode)
  - Memory system integration

Automatically degrades gracefully when services are unavailable.
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional

from .api import BeTaThinkClient

logger = logging.getLogger(__name__)

BETATHINK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class BrainConnector:
    """Unified brain interface — online or offline."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        base_url = self.config.get('betathink_api', 'http://localhost:8000')
        self.client = BeTaThinkClient(base_url)
        self._mode = None  # 'online' or 'offline'

    @property
    def mode(self) -> str:
        if self._mode is None:
            self._mode = 'online' if self.client.is_online else 'offline'
        return self._mode

    def search(self, query: str, limit: int = 10) -> List[dict]:
        """Search for knowledge — tries API first, then local."""
        if self.mode == 'online':
            results = self.client.search_entities(query, limit)
            if results:
                return results

        # Offline fallback: local literature RAG
        return self._local_search(query, limit)

    def explore(self, source: str, target: str) -> List[dict]:
        """Find paths between concepts."""
        if self.mode == 'online':
            return self.client.explore_paths(source, target)
        return []

    def enrich_mutations(self, mutations: List[str], protein: str) -> List[dict]:
        """Enrich mutations with KG context (literature, known effects)."""
        enriched = []
        for mut in mutations:
            entry = {'mutation': mut, 'kg_context': []}

            if self.mode == 'online':
                # Search KG for this specific mutation
                results = self.client.search_entities(f"{protein} {mut}", limit=3)
                entry['kg_context'] = results

            # Also check local knowledge store
            try:
                from zhuangzhou.memory import KnowledgeStore
                ks = KnowledgeStore()
                similar = ks.get_similar_mutations(mut)
                if similar:
                    entry['historical_similar'] = similar
            except Exception:
                pass

            enriched.append(entry)
        return enriched

    def upload_and_learn(self, pdf_path: str) -> dict:
        """Upload a document and trigger KG construction."""
        if self.mode == 'online':
            return self.client.upload_document(pdf_path)
        return {'error': 'BeTaThink offline — cannot auto-construct KG',
                'suggestion': 'Start BeTaThink API first: cd BeTaThink && python -m api.server'}

    def get_status(self) -> dict:
        """Get brain status."""
        status = {
            'mode': self.mode,
            'api_url': self.config.get('betathink_api', 'http://localhost:8000'),
        }
        if self.mode == 'online':
            health = self.client.check_health()
            status['services'] = health.get('services', {})
            overview = self.client.get_overview()
            status['kg_stats'] = overview
        return status

    def _local_search(self, query: str, limit: int) -> List[dict]:
        """Fallback: search local literature RAG."""
        try:
            if BETATHINK_ROOT not in sys.path:
                sys.path.insert(0, BETATHINK_ROOT)
            from zhuangzhou.literature_rag import LiteratureRAG
            rag = LiteratureRAG()
            return rag.search(query, top_k=limit)
        except Exception as e:
            logger.debug("Local search failed: %s", e)
            return []

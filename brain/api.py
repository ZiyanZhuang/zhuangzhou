"""
BeTaThink API Client
=====================
HTTP client for the BeTaThink backend services.
Handles connection, retry, and graceful degradation to offline mode.
"""

import json
import logging
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BeTaThinkClient:
    """Client for BeTaThink API."""

    def __init__(self, base_url: str = 'http://localhost:8000', timeout: int = 10):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self._online = None  # None = unknown

    @property
    def is_online(self) -> bool:
        if self._online is None:
            self.check_health()
        return self._online or False

    def check_health(self) -> dict:
        """Check BeTaThink API health."""
        try:
            data = self._get('/api/v1/system/health')
            self._online = True
            return data
        except Exception:
            # Try alternate health endpoint
            try:
                data = self._get('/api/health')
                self._online = True
                return data
            except Exception:
                self._online = False
                return {'status': 'offline'}

    # ── Entity & KG Queries ────────────────────────────────────────────

    def search_entities(self, query: str, limit: int = 10,
                        entity_type: str = None) -> List[dict]:
        """Search for entities in the knowledge graph."""
        params = {'q': query, 'limit': limit}
        if entity_type:
            params['type'] = entity_type
        try:
            data = self._post('/api/v1/query/search-entity', params)
            return data.get('entities', data.get('results', []))
        except Exception as e:
            logger.debug("Entity search failed: %s", e)
            return []

    def explore_paths(self, source: str, target: str,
                      max_depth: int = 3) -> List[dict]:
        """Find paths between entities."""
        try:
            data = self._post('/api/v1/query/explore', {
                'source': source, 'target': target, 'max_depth': max_depth,
            })
            return data.get('paths', [])
        except Exception as e:
            logger.debug("Path exploration failed: %s", e)
            return []

    def get_overview(self) -> dict:
        """Get KG overview statistics."""
        try:
            return self._get('/api/v1/query/overview')
        except Exception:
            return {}

    # ── Literature & RAG ───────────────────────────────────────────────

    def search_literature(self, query: str, limit: int = 5) -> List[dict]:
        """Search literature via RAG."""
        try:
            data = self._post('/api/v1/query/literature', {
                'query': query, 'limit': limit,
            })
            return data.get('results', [])
        except Exception as e:
            logger.debug("Literature search failed: %s", e)
            return []

    # ── Chat ───────────────────────────────────────────────────────────

    def chat(self, message: str, session_id: str = None) -> dict:
        """Send a chat message to BeTaThink."""
        try:
            return self._post('/api/v1/chat', {
                'message': message,
                'session_id': session_id,
            })
        except Exception as e:
            return {'error': str(e)}

    # ── Document Upload ────────────────────────────────────────────────

    def upload_document(self, pdf_path: str) -> dict:
        """Upload PDF for KG construction."""
        import os
        boundary = '----ZhuangzhouBoundary'
        filename = os.path.basename(pdf_path)
        with open(pdf_path, 'rb') as f:
            file_data = f.read()
        body = (
            f'--{boundary}\r\n'
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f'Content-Type: application/pdf\r\n\r\n'
        ).encode() + file_data + f'\r\n--{boundary}--\r\n'.encode()

        url = f"{self.base_url}/api/v1/admin/upload"
        req = urllib.request.Request(
            url, data=body,
            headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except Exception as e:
            return {'error': str(e)}

    # ── SynBio Pipeline ────────────────────────────────────────────────

    def launch_synbio(self, query: str) -> dict:
        """Launch synthetic biology pipeline."""
        try:
            return self._post('/api/v1/synbio/run', {'query': query})
        except Exception as e:
            return {'error': str(e)}

    def get_job_status(self, job_id: str) -> dict:
        """Get job status."""
        try:
            return self._get(f'/api/v1/synbio/status/{job_id}')
        except Exception as e:
            return {'error': str(e)}

    # ── HTTP Helpers ───────────────────────────────────────────────────

    def _get(self, path: str) -> dict:
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url, headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def _post(self, path: str, data: dict) -> dict:
        url = f"{self.base_url}{path}"
        payload = json.dumps(data).encode()
        req = urllib.request.Request(
            url, data=payload,
            headers={
                'Content-Type': 'application/json',
                'Accept': 'application/json',
            })
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

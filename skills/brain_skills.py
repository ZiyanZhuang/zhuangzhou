"""Brain skills — BeTaThink KG, RAG, agents as Zhuangzhou skills."""

import os
import sys

from .registry import Skill, SkillParam, SkillRegistry

BETATHINK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_betathink():
    if BETATHINK_ROOT not in sys.path:
        sys.path.insert(0, BETATHINK_ROOT)


def kg_query(query: str, entity_type: str = None,
             limit: int = 10, config: dict = None, **kw) -> dict:
    """Query the BeTaThink knowledge graph."""
    import urllib.request
    import json
    base = (config or {}).get('betathink_api', 'http://localhost:8000')
    url = f"{base}/api/v1/query/search-entity"
    payload = json.dumps({'query': query, 'limit': limit}).encode()
    req = urllib.request.Request(url, data=payload,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {'error': str(e), 'entities': []}


def kg_path_explore(source: str, target: str,
                    max_depth: int = 3, config: dict = None, **kw) -> dict:
    """Find paths between entities in the knowledge graph."""
    import urllib.request
    import json
    base = (config or {}).get('betathink_api', 'http://localhost:8000')
    url = f"{base}/api/v1/query/explore"
    payload = json.dumps({'source': source, 'target': target, 'max_depth': max_depth}).encode()
    req = urllib.request.Request(url, data=payload,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {'error': str(e), 'paths': []}


def literature_search(query: str, limit: int = 5, config: dict = None, **kw) -> dict:
    """Search literature via BeTaThink RAG."""
    import urllib.request
    import json
    base = (config or {}).get('betathink_api', 'http://localhost:8000')
    url = f"{base}/api/v1/query/literature"
    payload = json.dumps({'query': query, 'limit': limit}).encode()
    req = urllib.request.Request(url, data=payload,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        # Fallback to local built-in knowledge base
        try:
            from zhuangzhou.literature_rag import (
                search_mutation_literature, get_mechanism_rules)
            patterns = search_mutation_literature('', query)
            mechanisms = get_mechanism_rules(query)
            return {
                'results': patterns,
                'mechanisms': mechanisms,
                'source': 'local_kb',
            }
        except Exception:
            return {'error': str(e), 'results': []}


def upload_document(pdf_path: str, config: dict = None, **kw) -> dict:
    """Upload a PDF document to BeTaThink for KG construction."""
    import urllib.request
    import json
    base = (config or {}).get('betathink_api', 'http://localhost:8000')
    url = f"{base}/api/v1/admin/upload"

    boundary = '----ZhuangzhouBoundary'
    filename = os.path.basename(pdf_path)
    with open(pdf_path, 'rb') as f:
        file_data = f.read()

    body = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f'Content-Type: application/pdf\r\n\r\n'
    ).encode() + file_data + f'\r\n--{boundary}--\r\n'.encode()

    req = urllib.request.Request(
        url, data=body,
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {'error': str(e)}


def synbio_pipeline(query: str, config: dict = None, **kw) -> dict:
    """Launch BeTaThink synthetic biology multi-agent pipeline."""
    import urllib.request
    import json
    base = (config or {}).get('betathink_api', 'http://localhost:8000')
    url = f"{base}/api/v1/synbio/run"
    payload = json.dumps({'query': query}).encode()
    req = urllib.request.Request(url, data=payload,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {'error': str(e)}


def register_skills(registry: SkillRegistry):
    registry.register(Skill(
        name='kg_query',
        description='Search BeTaThink knowledge graph for entities',
        category='brain',
        fn=kg_query,
        requires_network=True,
        inputs=[
            SkillParam('query', 'string', 'Search query'),
            SkillParam('entity_type', 'string', 'Filter by type', required=False),
            SkillParam('limit', 'int', 'Max results', required=False, default=10),
        ],
        outputs=[SkillParam('entities', 'list', 'Matching entities')],
        source='BeTaThink API',
    ))

    registry.register(Skill(
        name='kg_explore',
        description='Find paths between entities in knowledge graph',
        category='brain',
        fn=kg_path_explore,
        requires_network=True,
        inputs=[
            SkillParam('source', 'string', 'Source entity'),
            SkillParam('target', 'string', 'Target entity'),
            SkillParam('max_depth', 'int', 'Max path depth', required=False, default=3),
        ],
        outputs=[SkillParam('paths', 'list', 'Discovered paths')],
        source='BeTaThink API',
    ))

    registry.register(Skill(
        name='literature_search',
        description='Search literature via RAG (API + local fallback)',
        category='brain',
        fn=literature_search,
        inputs=[
            SkillParam('query', 'string', 'Search query'),
            SkillParam('limit', 'int', 'Max results', required=False, default=5),
        ],
        outputs=[SkillParam('results', 'list', 'Literature results')],
        source='BeTaThink RAG',
    ))

    registry.register(Skill(
        name='upload_document',
        description='Upload PDF for automatic KG construction',
        category='brain',
        fn=upload_document,
        requires_network=True,
        inputs=[SkillParam('pdf_path', 'path', 'PDF file to upload')],
        outputs=[SkillParam('result', 'dict', 'Upload status + document ID')],
        source='BeTaThink API',
        risk_level='medium',
    ))

    registry.register(Skill(
        name='synbio_pipeline',
        description='Launch synthetic biology multi-agent pipeline',
        category='brain',
        fn=synbio_pipeline,
        requires_network=True,
        inputs=[SkillParam('query', 'string', 'Design query')],
        outputs=[SkillParam('result', 'dict', 'Job ID + status')],
        source='BeTaThink SynBio Agents',
        risk_level='high',
        estimated_time='~5min',
    ))

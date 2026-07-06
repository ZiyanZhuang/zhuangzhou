"""
LiteratureFetcher — auto-fetches PubMed literature for report enrichment.
"""
import re
import time
import json
import urllib.request
import urllib.parse
from typing import List, Optional
import xml.etree.ElementTree as ET

_HEADERS = {'User-Agent': 'Zhuangzhou-HuiZi/1.0 (research tool)'}
_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_RATE_LIMIT = 0.35  # 3 req/sec max


def _get(url: str, timeout: int = 15) -> Optional[str]:
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode('utf-8')
    except Exception:
        return None


class LiteratureFetcher:
    """Fetches relevant PubMed papers for a protein/topic."""

    _TOPIC_MAP = {
        'esm2_scan': ['mutation scanning', 'protein fitness landscape'],
        'esm2_mutation_scan': ['mutation scanning', 'protein fitness'],
        'enrich_mutations': ['mutagenesis', 'variant effect'],
        'docking': ['molecular docking', 'binding affinity'],
        'smart_docking': ['molecular docking', 'ligand binding'],
        'md_simulation': ['molecular dynamics', 'simulation'],
        'plip': ['protein-ligand interaction', 'binding site'],
        'structure': ['crystal structure', 'structure determination'],
        'mutation_enrichment': ['site-directed mutagenesis', 'enzyme engineering'],
    }

    def fetch(self, protein_name: str, topics: List[str],
              max_results: int = 5, year_from: int = 2015) -> List[dict]:
        if not protein_name:
            return []

        query = self._build_query(protein_name, topics)
        pmids = self._esearch(query, max_results, year_from)

        if not pmids:
            # Simpler fallback query
            simple = f'"{protein_name}"[Title/Abstract]'
            pmids = self._esearch(simple, max_results, year_from)

        if not pmids:
            return []

        return self._efetch(pmids)

    def _build_query(self, protein_name: str, topics: List[str]) -> str:
        parts = [f'"{protein_name}"[Title/Abstract]']
        topic_kws = []
        for t in topics:
            topic_kws.extend(self._TOPIC_MAP.get(t, []))
        if topic_kws:
            joined = ' OR '.join(f'"{k}"' for k in topic_kws[:4])
            parts.append(f'({joined})')
        return ' AND '.join(parts)

    def _esearch(self, query: str, max_results: int, year_from: int) -> List[str]:
        params = urllib.parse.urlencode({
            'db': 'pubmed', 'term': query, 'retmax': max_results,
            'sort': 'relevance', 'datetype': 'pdat',
            'mindate': str(year_from), 'retmode': 'json',
        })
        text = _get(f"{_BASE}/esearch.fcgi?{params}")
        if not text:
            return []
        try:
            return json.loads(text).get('esearchresult', {}).get('idlist', [])
        except Exception:
            return []

    def _efetch(self, pmids: List[str]) -> List[dict]:
        time.sleep(_RATE_LIMIT)
        params = urllib.parse.urlencode({
            'db': 'pubmed', 'id': ','.join(pmids),
            'rettype': 'abstract', 'retmode': 'xml',
        })
        xml_text = _get(f"{_BASE}/efetch.fcgi?{params}")
        if not xml_text:
            return []
        return self._parse_xml(xml_text)

    def _parse_xml(self, xml_text: str) -> List[dict]:
        papers = []
        try:
            root = ET.fromstring(xml_text)
            for article in root.findall('.//PubmedArticle'):
                p = {}
                pmid_el = article.find('.//PMID')
                p['pmid'] = pmid_el.text if pmid_el is not None else ''
                title_el = article.find('.//ArticleTitle')
                p['title'] = self._text(title_el)
                authors = []
                for auth in article.findall('.//Author')[:5]:
                    last = auth.find('LastName')
                    init = auth.find('Initials')
                    if last is not None:
                        authors.append(f"{last.text} {init.text}" if init is not None else last.text)
                if len(article.findall('.//Author')) > 5:
                    authors.append('et al.')
                p['authors'] = ', '.join(authors)
                j = article.find('.//Journal/Title')
                p['journal'] = j.text if j is not None else ''
                yr = article.find('.//PubDate/Year')
                if yr is None:
                    yr = article.find('.//PubDate/MedlineDate')
                p['year'] = (yr.text or '')[:4] if yr is not None else ''
                ab_parts = []
                for ab in article.findall('.//AbstractText'):
                    t = self._text(ab)
                    if t:
                        lbl = ab.get('Label', '')
                        ab_parts.append(f"{lbl}: {t}" if lbl else t)
                full_ab = ' '.join(ab_parts)
                p['abstract'] = full_ab[:400] + '...' if len(full_ab) > 400 else full_ab
                p['doi'] = ''
                for id_el in article.findall('.//ArticleId'):
                    if id_el.get('IdType') == 'doi':
                        p['doi'] = id_el.text or ''
                        break
                p['citation'] = self._cite(p)
                if p.get('title'):
                    papers.append(p)
        except Exception:
            pass
        return papers

    @staticmethod
    def _text(el) -> str:
        if el is None:
            return ''
        return ''.join(el.itertext()).strip()

    @staticmethod
    def _cite(p: dict) -> str:
        parts = []
        if p.get('authors'):
            parts.append(p['authors'])
        if p.get('year'):
            parts.append(f"({p['year']})")
        if p.get('title'):
            parts.append(p['title'] + '.')
        if p.get('journal'):
            parts.append(p['journal'] + '.')
        if p.get('doi'):
            parts.append(f"doi:{p['doi']}")
        elif p.get('pmid'):
            parts.append(f"PMID:{p['pmid']}")
        return ' '.join(parts)

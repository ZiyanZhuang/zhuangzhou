"""
Research Skills -- Online scientific knowledge retrieval.

Provides agent-driven access to:
  - PubMed (NCBI E-utilities, free, no API key)
  - UniProt (REST API, protein annotations)
  - GraphRAG (local 7-domain knowledge graphs)
  - Literature RAG (built-in enzyme mutation KB + UniProt)

These are the "eyes and ears" of the agent -- it MUST research
before making experiment design decisions.
"""

import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import List, Optional

from .registry import Skill, SkillParam, SkillRegistry

logger = logging.getLogger(__name__)

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
UNIPROT_BASE = "https://rest.uniprot.org"


# ---------------------------------------------------------------------------
# Cross-lingual Query Translation (Chinese → English for KG/PubMed)
# ---------------------------------------------------------------------------

# Common Chinese biotech terms → English. This is a static dictionary
# that covers the most frequent terms users would search. It's NOT a
# general-purpose translator — it's a domain-specific term mapper.
_ZH_EN_TERMS = {
    # Enzyme engineering
    '定向进化': 'directed evolution',
    '理性设计': 'rational design',
    '半理性设计': 'semi-rational design',
    '突变': 'mutation',
    '点突变': 'point mutation',
    '饱和突变': 'saturation mutagenesis',
    '易错PCR': 'error-prone PCR',
    '活性位点': 'active site',
    '底物': 'substrate',
    '催化': 'catalysis',
    '催化机制': 'catalytic mechanism',
    '催化效率': 'catalytic efficiency',
    '酶活性': 'enzyme activity',
    '酶活': 'enzyme activity',
    '热稳定性': 'thermostability',
    '温度稳定性': 'thermostability',
    '耐热': 'thermostable',
    '耐酸': 'acid tolerance',
    '耐碱': 'alkaline tolerance',
    '有机溶剂耐受': 'organic solvent tolerance',
    '底物特异性': 'substrate specificity',
    '底物选择性': 'substrate selectivity',
    '对映选择性': 'enantioselectivity',
    '区域选择性': 'regioselectivity',
    '立体选择性': 'stereoselectivity',
    '辅因子': 'cofactor',
    '辅酶': 'coenzyme',
    '血红素': 'heme',
    '活性口袋': 'active site pocket',
    '结合口袋': 'binding pocket',
    # Protein types
    '脂肪酶': 'lipase',
    '蛋白酶': 'protease',
    '淀粉酶': 'amylase',
    '纤维素酶': 'cellulase',
    '氧化还原酶': 'oxidoreductase',
    '转氨酶': 'transaminase',
    '酮还原酶': 'ketoreductase',
    '细胞色素P450': 'cytochrome P450',
    '漆酶': 'laccase',
    '过氧化物酶': 'peroxidase',
    '水解酶': 'hydrolase',
    '转移酶': 'transferase',
    '裂解酶': 'lyase',
    '异构酶': 'isomerase',
    '连接酶': 'ligase',
    # Structural biology
    '蛋白质结构': 'protein structure',
    '二级结构': 'secondary structure',
    '三级结构': 'tertiary structure',
    '四级结构': 'quaternary structure',
    'α螺旋': 'alpha helix',
    'β折叠': 'beta sheet',
    '分子动力学': 'molecular dynamics',
    '分子对接': 'molecular docking',
    '同源建模': 'homology modeling',
    '力场': 'force field',
    '构象': 'conformation',
    '柔性': 'flexibility',
    '刚性': 'rigidity',
    '结合能': 'binding energy',
    '结合亲和力': 'binding affinity',
    '自由能': 'free energy',
    '氢键': 'hydrogen bond',
    '疏水相互作用': 'hydrophobic interaction',
    '静电作用': 'electrostatic interaction',
    '范德华力': 'van der Waals',
    '二硫键': 'disulfide bond',
    '盐桥': 'salt bridge',
    # Metabolic engineering
    '代谢工程': 'metabolic engineering',
    '合成生物学': 'synthetic biology',
    '代谢通量': 'metabolic flux',
    '途径': 'pathway',
    '表达': 'expression',
    '启动子': 'promoter',
    '密码子优化': 'codon optimization',
    '基因敲除': 'gene knockout',
    '基因过表达': 'gene overexpression',
    # Misc
    '药物代谢': 'drug metabolism',
    '生物催化': 'biocatalysis',
    '工业酶': 'industrial enzyme',
    '蛋白工程': 'protein engineering',
    '序列比对': 'sequence alignment',
    '进化': 'evolution',
    '保守性': 'conservation',
    '多序列比对': 'multiple sequence alignment',
    # Core biotech terms missing from original dict (v0.7.5+)
    '酶': 'enzyme',
    '突变': 'mutation',
    '对接': 'docking',
    '活性位点': 'active site',
    '抑制剂': 'inhibitor',
    '激活剂': 'activator',
    '催化': 'catalysis',
    '蛋白质': 'protein',
    '配体': 'ligand',
    '氨基酸': 'amino acid',
    '折叠': 'folding',
    '结构': 'structure',
    '功能': 'function',
    '辅因子': 'cofactor',
    '细胞色素': 'cytochrome',
    '氧化还原': 'redox',
    '磷酸化': 'phosphorylation',
    '糖基化': 'glycosylation',
    '基因': 'gene',
    '转录': 'transcription',
    '翻译': 'translation',
    '分析': 'analysis',
    '研究': 'research',
    '实验': 'experiment',
    '计算': 'computational',
    '模拟': 'simulation',
    '预测': 'prediction',
    '优化': 'optimization',
    '筛选': 'screening',
    '结合位点': 'binding site',
    '氢键': 'hydrogen bond',
    # Extended biology/chemistry terms (v0.7.5+)
    '细胞': 'cell',
    '核酸': 'nucleic acid',
    'DNA': 'DNA',
    'RNA': 'RNA',
    'mRNA': 'mRNA',
    '核糖体': 'ribosome',
    '线粒体': 'mitochondria',
    '质膜': 'plasma membrane',
    '受体': 'receptor',
    '信号通路': 'signaling pathway',
    '转导': 'transduction',
    '激酶': 'kinase',
    '磷酸酶': 'phosphatase',
    '蛋白酶体': 'proteasome',
    '泛素': 'ubiquitin',
    '分子伴侣': 'molecular chaperone',
    '热休克蛋白': 'heat shock protein',
    '脱氧核糖核酸': 'DNA',
    '核糖核酸': 'RNA',
    '碱基': 'nucleobase',
    '碱基对': 'base pair',
    '密码子': 'codon',
    '启动子': 'promoter',
    '增强子': 'enhancer',
    '操纵子': 'operon',
    '质粒': 'plasmid',
    '转化': 'transformation',
    '转染': 'transfection',
    '克隆': 'cloning',
    '重组': 'recombination',
    '杂交': 'hybridization',
    '引物': 'primer',
    'PCR': 'PCR',
    '测序': 'sequencing',
    '全基因组测序': 'whole genome sequencing',
    '单细胞测序': 'single cell sequencing',
    '代谢物': 'metabolite',
    '底物结合': 'substrate binding',
    '酶促反应': 'enzymatic reaction',
    '反应速率': 'reaction rate',
    '动力学': 'kinetics',
    'Km': 'Km',
    'kcat': 'kcat',
    '米氏常数': 'Michaelis constant',
    '最大反应速率': 'maximum reaction rate',
    '非竞争性抑制': 'noncompetitive inhibition',
    '竞争性抑制': 'competitive inhibition',
    '别构调节': 'allosteric regulation',
    '变构效应': 'allosteric effect',
    '氨基末端': 'N-terminus',
    '羧基末端': 'C-terminus',
    '疏水核心': 'hydrophobic core',
    '活化能': 'activation energy',
    '过渡态': 'transition state',
    '中间体': 'intermediate',
    '反应机制': 'reaction mechanism',
    '量子化学': 'quantum chemistry',
    '密度泛函理论': 'density functional theory',
    '从头算': 'ab initio',
    '半经验': 'semi-empirical',
    '分子轨道': 'molecular orbital',
    '电子密度': 'electron density',
    '能垒': 'energy barrier',
    '势能面': 'potential energy surface',
    '熵': 'entropy',
    '焓': 'enthalpy',
    '吉布斯自由能': 'Gibbs free energy',
    '平衡常数': 'equilibrium constant',
    '溶剂化': 'solvation',
    '极化': 'polarization',
}


def _translate_query(query: str) -> str:
    """Translate Chinese biotech terms to English for KG/PubMed search.

    Uses longest-match dictionary substitution. Terms not in the dict
    are left as-is (the user may be mixing Chinese text with English
    protein names like CYP3A4).
    """
    if not any('\u4e00' <= c <= '\u9fff' for c in query):
        return query  # No Chinese characters, skip

    result = query
    # Sort by length descending for longest-match-first
    for zh, en in sorted(_ZH_EN_TERMS.items(), key=lambda x: -len(x[0])):
        # Pad with spaces to prevent English tokens from merging
        result = result.replace(zh, f' {en} ')

    # Remove remaining Chinese particles/connectors (的, 和, 与, 了, etc.)
    result = re.sub(r'[\u4e00-\u9fff]', ' ', result)

    # Clean up: collapse multiple spaces
    result = re.sub(r'\s+', ' ', result).strip()
    return result

# Respect NCBI rate limits: max 3 requests/second without API key
_last_ncbi_call = 0.0


def _ncbi_throttle():
    global _last_ncbi_call
    elapsed = time.time() - _last_ncbi_call
    if elapsed < 0.34:
        time.sleep(0.34 - elapsed)
    _last_ncbi_call = time.time()


# ---------------------------------------------------------------------------
# PubMed Search
# ---------------------------------------------------------------------------

def search_pubmed(query: str, max_results: int = 5, **kw) -> dict:
    """
    Search PubMed for scientific papers.

    Uses NCBI E-utilities (free, no API key required for <3 req/s).
    Returns titles, abstracts, authors, PMIDs, and publication dates.
    Automatically translates Chinese biotech terms to English.
    """
    try:
        # Auto-translate Chinese terms for English-language PubMed
        original_query = query
        query = _translate_query(query)
        if query != original_query:
            logger.info("Query translated: %s -> %s", original_query, query)

        # Step 1: esearch to get PMIDs
        _ncbi_throttle()
        params = urllib.parse.urlencode({
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json',
            'sort': 'relevance',
        })
        url = f"{NCBI_BASE}/esearch.fcgi?{params}"
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Zhuangzhou/0.6.0 (enzyme engineering research tool)'
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())

        id_list = data.get('esearchresult', {}).get('idlist', [])
        total_count = int(data.get('esearchresult', {}).get('count', 0))

        if not id_list:
            return {
                'query': query,
                'original_query': original_query if query != original_query else None,
                'total_count': total_count,
                'papers': [],
                'hint': 'No results found. Try broader search terms.',
            }

        # Step 2: efetch to get details
        _ncbi_throttle()
        ids = ','.join(id_list)
        params = urllib.parse.urlencode({
            'db': 'pubmed',
            'id': ids,
            'retmode': 'xml',
            'rettype': 'abstract',
        })
        url = f"{NCBI_BASE}/efetch.fcgi?{params}"
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Zhuangzhou/0.6.0 (enzyme engineering research tool)'
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            xml_text = resp.read().decode('utf-8')

        # Parse XML (lightweight, no lxml needed)
        papers = _parse_pubmed_xml(xml_text)

        return {
            'query': query,
            'total_count': total_count,
            'returned': len(papers),
            'papers': papers,
        }

    except Exception as e:
        return {'error': str(e), 'hint': 'PubMed search failed. Check network.'}


def _parse_pubmed_xml(xml: str) -> list:
    """Parse PubMed efetch XML into structured paper list."""
    papers = []

    # Split by article
    articles = re.split(r'<PubmedArticle>', xml)
    for article in articles[1:]:  # skip preamble
        paper = {}

        # PMID
        m = re.search(r'<PMID[^>]*>(\d+)</PMID>', article)
        paper['pmid'] = m.group(1) if m else ''

        # Title
        m = re.search(
            r'<ArticleTitle>(.*?)</ArticleTitle>', article, re.DOTALL)
        paper['title'] = _clean_xml(m.group(1)) if m else ''

        # Abstract
        abstract_parts = re.findall(
            r'<AbstractText[^>]*>(.*?)</AbstractText>', article, re.DOTALL)
        paper['abstract'] = ' '.join(
            _clean_xml(p) for p in abstract_parts)[:1500]

        # Authors (first 5)
        authors = re.findall(
            r'<LastName>(.*?)</LastName>\s*<ForeName>(.*?)</ForeName>',
            article)
        paper['authors'] = [f"{ln} {fn}" for ln, fn in authors[:5]]
        if len(authors) > 5:
            paper['authors'].append(f"... (+{len(authors)-5} more)")

        # Journal
        m = re.search(r'<Title>(.*?)</Title>', article)
        paper['journal'] = _clean_xml(m.group(1)) if m else ''

        # Year
        m = re.search(r'<PubDate>.*?<Year>(\d{4})</Year>', article, re.DOTALL)
        paper['year'] = m.group(1) if m else ''

        # DOI
        m = re.search(
            r'<ArticleId IdType="doi">(.*?)</ArticleId>', article)
        paper['doi'] = m.group(1) if m else ''

        if paper.get('title'):
            papers.append(paper)

    return papers


def _clean_xml(text: str) -> str:
    """Remove XML tags and clean whitespace."""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------------------------------
# UniProt Protein Info
# ---------------------------------------------------------------------------

def search_uniprot(query: str, organism: str = None,
                   max_results: int = 5, **kw) -> dict:
    """
    Search UniProt for protein information.

    Returns protein names, accessions, functions, and key annotations.
    """
    try:
        search_query = query
        if organism:
            search_query += f" AND (organism_name:\"{organism}\")"

        params = urllib.parse.urlencode({
            'query': search_query,
            'format': 'json',
            'size': max_results,
            'fields': ','.join([
                'accession', 'protein_name', 'organism_name', 'length',
                'ft_act_site', 'ft_binding', 'ft_mutagen',
                'cc_function', 'cc_catalytic_activity',
            ]),
        })
        url = f"{UNIPROT_BASE}/uniprotkb/search?{params}"
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Zhuangzhou/0.6.0',
            'Accept': 'application/json',
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())

        results = []
        for entry in data.get('results', []):
            protein = {
                'accession': entry.get('primaryAccession', ''),
                'name': _extract_protein_name(entry),
                'organism': entry.get('organism', {}).get(
                    'scientificName', ''),
                'length': entry.get('sequence', {}).get('length', 0),
                'function': _extract_function(entry),
                'active_sites': _extract_features(entry, 'Active site'),
                'metal_binding': _extract_features(entry, 'Metal binding'),
                'mutagenesis': _extract_features(entry, 'Mutagenesis'),
            }
            results.append(protein)

        return {
            'query': query,
            'results': results,
            'count': len(results),
        }

    except Exception as e:
        return {'error': str(e), 'hint': 'UniProt search failed.'}


def get_uniprot_entry(accession: str, **kw) -> dict:
    """
    Get detailed UniProt entry by accession ID.

    Returns full protein annotation including active sites,
    metal binding, mutagenesis records, catalytic activity.
    """
    try:
        url = f"{UNIPROT_BASE}/uniprotkb/{accession}.json"
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Zhuangzhou/0.6.0',
            'Accept': 'application/json',
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            entry = json.loads(resp.read())

        result = {
            'accession': entry.get('primaryAccession', ''),
            'name': _extract_protein_name(entry),
            'organism': entry.get('organism', {}).get('scientificName', ''),
            'length': entry.get('sequence', {}).get('length', 0),
            'function': _extract_function(entry),
            'catalytic_activity': _extract_catalytic(entry),
            'active_sites': _extract_features(entry, 'Active site'),
            'metal_binding': _extract_features(entry, 'Metal binding'),
            'binding_sites': _extract_features(entry, 'Binding site'),
            'mutagenesis': _extract_features(entry, 'Mutagenesis'),
            'pdb_xrefs': _extract_pdb_refs(entry),
        }
        return result

    except Exception as e:
        return {'error': str(e), 'hint': f'UniProt entry {accession} not found.'}


# ---------------------------------------------------------------------------
# GraphRAG Local Knowledge
# ---------------------------------------------------------------------------

def search_graphrag(query: str, domain: str = 'auto',
                    limit: int = 10, **kw) -> dict:
    """
    Search local GraphRAG knowledge graphs (7 domains).

    Domains: Bioinformatics, Computational_Chemistry, Cytochrome_P450,
    Evolution, Structural_Biology, Synthetic_Biology, Systems_Biology.

    Set domain='auto' for automatic domain routing based on query keywords.
    IMPORTANT: KG corpus is English (except partial Structural_Biology).
    Chinese queries are auto-translated before search.
    """
    try:
        from zhuangzhou.brain.graphrag import GraphRAGBridge
        bridge = GraphRAGBridge()

        # Auto-translate Chinese terms for English KG corpus
        original_query = query
        query = _translate_query(query)

        if domain == 'auto':
            domain = bridge.select_domain(query)

        # Use query words as keywords for entity search
        keywords = [w for w in query.split() if len(w) > 1]
        if not keywords:
            keywords = [query]

        entities = bridge.search_entities(
            keywords, kg_id=domain or 'all', limit=limit)
        community = bridge.get_community_context(
            query, kg_id=domain or 'all', limit=3)

        total_hits = sum(
            len(v) for v in entities.values()) if isinstance(
                entities, dict) else 0

        result = {
            'query': query,
            'domain': domain or 'all (no domain matched)',
            'entities': entities,
            'community_context': community,
            'total_hits': total_hits,
        }
        if query != original_query:
            result['original_query'] = original_query
            result['translated'] = True
        if total_hits == 0:
            result['hint'] = (
                'No KG hits. Try different English keywords, '
                'or use search_pubmed for online literature.')
        return result

    except Exception as e:
        return {'error': str(e), 'hint': 'GraphRAG search failed.'}


def get_mutation_evidence(mutation: str, enzyme_class: str = '',
                          protein_name: str = '', **kw) -> dict:
    """
    Search for evidence about a specific mutation across all knowledge sources.

    Combines: GraphRAG KG + built-in catalytic mechanism rules + UniProt.
    Chinese enzyme_class/protein_name are auto-translated.
    """
    try:
        # Translate Chinese terms
        enzyme_class = _translate_query(enzyme_class) if enzyme_class else ''
        protein_name = _translate_query(protein_name) if protein_name else ''

        results = {
            'mutation': mutation,
            'graphrag_evidence': {},
            'mechanism_rules': {},
            'literature_context': {},
            'evidence_grade': 'E',  # default: no evidence
        }

        # 1. GraphRAG evidence
        try:
            from zhuangzhou.brain.graphrag import GraphRAGBridge
            bridge = GraphRAGBridge()
            results['graphrag_evidence'] = bridge.get_mutation_evidence(
                mutation, enzyme_class, protein_name)
        except Exception as _graphrag_err:
            logger.debug("GraphRAG unavailable for mutation evidence: %s", _graphrag_err)

        # 2. Mechanism rules from built-in KB
        try:
            from zhuangzhou.literature_rag import build_literature_context
            m = re.match(r'^([A-Z])(\d+)([A-Z])$', mutation)
            if m:
                wt, pos, mut = m.group(1), int(m.group(2)), m.group(3)
                lit_ctx = build_literature_context(
                    pos, wt, mut, enzyme_class)
                results['literature_context'] = lit_ctx
                if lit_ctx.get('literature_reasoning'):
                    has_mechanism = True
        except Exception:
            pass

        # 3. Compute evidence grade
        has_kg = (isinstance(results['graphrag_evidence'], dict) and
                  results['graphrag_evidence'].get('n_hits', 0) > 0)
        has_mechanism = bool(
            results.get('literature_context', {}).get('literature_reasoning'))
        has_uniprot = bool(
            results.get('literature_context', {}).get('known_mutagenesis'))

        if has_uniprot:
            results['evidence_grade'] = 'A'
            results['evidence_summary'] = (
                'UniProt有该位点的实验突变数据')
        elif has_kg:
            results['evidence_grade'] = 'B'
            results['evidence_summary'] = (
                f'GraphRAG知识图谱有相关实体'
                f'（{results["graphrag_evidence"].get("n_hits", 0)}条命中）')
        elif has_mechanism:
            results['evidence_grade'] = 'C'
            results['evidence_summary'] = (
                '基于内置催化机制规则的化学推理')
        else:
            results['evidence_grade'] = 'E'
            results['evidence_summary'] = (
                '当前知识源均无直接证据，分析为推测性质')

        return results

    except Exception as e:
        return {'error': str(e)}


# ---------------------------------------------------------------------------
# UniProt helpers
# ---------------------------------------------------------------------------

def _extract_protein_name(entry: dict) -> str:
    pn = entry.get('proteinDescription', {})
    rec = pn.get('recommendedName', {})
    if rec:
        return rec.get('fullName', {}).get('value', '')
    subs = pn.get('submissionNames', [])
    if subs:
        return subs[0].get('fullName', {}).get('value', '')
    return ''


def _extract_function(entry: dict) -> str:
    comments = entry.get('comments', [])
    for c in comments:
        if c.get('commentType') == 'FUNCTION':
            texts = c.get('texts', [])
            if texts:
                return texts[0].get('value', '')[:500]
    return ''


def _extract_catalytic(entry: dict) -> list:
    comments = entry.get('comments', [])
    activities = []
    for c in comments:
        if c.get('commentType') == 'CATALYTIC ACTIVITY':
            reaction = c.get('reaction', {})
            activities.append({
                'name': reaction.get('name', ''),
                'ec': reaction.get('ecNumber', ''),
            })
    return activities


def _extract_features(entry: dict, feature_type: str) -> list:
    features = entry.get('features', [])
    results = []
    for f in features:
        if f.get('type') == feature_type:
            loc = f.get('location', {})
            start = loc.get('start', {}).get('value')
            end = loc.get('end', {}).get('value')
            results.append({
                'start': start,
                'end': end,
                'description': f.get('description', ''),
            })
    return results


def _extract_pdb_refs(entry: dict) -> list:
    refs = entry.get('uniProtKBCrossReferences', [])
    pdbs = []
    for r in refs:
        if r.get('database') == 'PDB':
            pdbs.append({
                'id': r.get('id', ''),
                'method': r.get('properties', [{}])[0].get('value', '')
                if r.get('properties') else '',
            })
    return pdbs[:10]  # limit


def synthesis_knowledge(insights: str = None, patterns: str = None,
                        pitfalls: str = None, **kw) -> dict:
    """
    Reflect on the current session and record new knowledge.

    Appends insights, successful patterns, or pitfalls to ZHUANGZHOU.md.
    This provides the agent with 'Agentic Memory' across sessions.
    """
    from zhuangzhou.memory.project_memory import ProjectMemory
    try:
        memory = ProjectMemory()
        recorded = []
        
        if insights:
            memory._append_to_section("Learned Patterns", f"[Insight] {insights}")
            recorded.append("insights")
        
        if patterns:
            memory.add_pattern(patterns)
            recorded.append("patterns")
            
        if pitfalls:
            memory.add_pitfall(pitfalls)
            recorded.append("pitfalls")
            
        if not recorded:
            return {"error": "No knowledge provided to synthesize."}
            
        return {
            "success": True,
            "recorded": recorded,
            "memory_file": memory.memory_file
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# CrossRef Literature Search
# ---------------------------------------------------------------------------

CROSSREF_BASE = "https://api.crossref.org/works"


def search_crossref(query: str, max_results: int = 5, **kw) -> dict:
    """
    Search CrossRef for scientific papers.

    Uses the free CrossRef REST API (no key needed).
    Returns full citations: title, journal, year, DOI, authors, abstract snippet.
    Chinese biotech terms are auto-translated to English before search.
    """
    try:
        original_query = query
        query = _translate_query(query)
        if query != original_query:
            logger.info("CrossRef query translated: %s -> %s",
                        original_query, query)

        params = urllib.parse.urlencode({
            'query': query,
            'rows': max_results,
            'select': ','.join([
                'DOI', 'title', 'author', 'published-print', 'published-online',
                'container-title', 'abstract', 'URL',
            ]),
            'sort': 'relevance',
        })
        url = f"{CROSSREF_BASE}?{params}"
        req = urllib.request.Request(url, headers={
            'User-Agent': (
                'Zhuangzhou/0.6.0 (enzyme engineering; '
                'mailto:zhuangzhou@betathink.ai)'
            ),
            'Accept': 'application/json',
        })
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())

        items = data.get('message', {}).get('items', [])
        papers = []
        for item in items:
            title_list = item.get('title', [])
            title = title_list[0] if title_list else ''

            # Authors
            authors = []
            for a in item.get('author', [])[:5]:
                name = f"{a.get('family', '')} {a.get('given', '')}".strip()
                if name:
                    authors.append(name)
            if len(item.get('author', [])) > 5:
                authors.append(f"... (+{len(item['author'])-5} more)")

            # Year: prefer print, then online, then deposited
            year = ''
            for date_key in ('published-print', 'published-online', 'deposited'):
                date_parts = item.get(date_key, {}).get('date-parts', [[]])
                if date_parts and date_parts[0]:
                    year = str(date_parts[0][0])
                    break

            # Journal
            container = item.get('container-title', [])
            journal = container[0] if container else ''

            # Abstract (CrossRef sometimes has it)
            raw_abstract = item.get('abstract', '')
            abstract = re.sub(r'<[^>]+>', '', raw_abstract)[:1500]

            doi = item.get('DOI', '')

            if title:
                papers.append({
                    'title': title,
                    'authors': authors,
                    'journal': journal,
                    'year': year,
                    'doi': doi,
                    'abstract': abstract,
                    'url': f"https://doi.org/{doi}" if doi else item.get('URL', ''),
                    'source': 'CrossRef',
                })

        result = {
            'query': query,
            'returned': len(papers),
            'papers': papers,
        }
        if query != original_query:
            result['original_query'] = original_query
            result['translated'] = True
        return result

    except Exception as e:
        return {'error': str(e), 'hint': 'CrossRef search failed. Check network.'}


# ---------------------------------------------------------------------------
# Enhanced Literature Search (multi-source, intent expansion, synthesis)
# ---------------------------------------------------------------------------

def _expand_query_intent(query: str, context: str = None,
                          backend=None) -> list:
    """Generate 3-5 related sub-queries to broaden literature coverage.

    Uses LLM backend if available, otherwise applies keyword stemming rules.
    Returns a list of query strings starting with the original query.
    """
    sub_queries = [query]

    if backend is not None:
        try:
            prompt = (
                f"You are helping a researcher expand a literature search query "
                f"to find more relevant papers. Given the main query, generate "
                f"4 related sub-queries that would surface different aspects of "
                f"the topic. Return ONLY the sub-queries, one per line, no numbering.\n\n"
                f"Main query: {query}\n"
                f"Context: {context or 'enzyme engineering research'}"
            )
            if hasattr(backend, 'chat'):
                response = backend.chat(prompt)
            elif callable(backend):
                response = backend(prompt)
            else:
                response = None

            if response and isinstance(response, str):
                extras = [
                    line.strip()
                    for line in response.strip().split('\n')
                    if line.strip() and line.strip() != query
                ]
                sub_queries.extend(extras[:4])
                return sub_queries[:5]
        except Exception as e:
            logger.debug("LLM intent expansion failed: %s", e)

    # Fallback: keyword-based heuristic expansion
    q_lower = query.lower()

    # P450 / cytochrome expansion
    if any(t in q_lower for t in ('cyp', 'p450', 'cytochrome')):
        sub_queries.append(f"{query} active site substrate binding")
        sub_queries.append(f"drug metabolism {query}")
        sub_queries.append(f"P450 catalytic mechanism hydroxylation")

    # Mutation-related expansion
    if any(t in q_lower for t in ('mutant', 'mutation', 'variant')):
        sub_queries.append(f"{query} enzyme activity effect")
        sub_queries.append(f"{query} thermostability directed evolution")
        sub_queries.append(f"protein engineering {query}")

    # Docking / MD expansion
    if any(t in q_lower for t in ('docking', 'molecular dynamics', 'binding')):
        sub_queries.append(f"{query} free energy calculation")
        sub_queries.append(f"{query} force field simulation")

    # Enzyme-only expansion
    if 'enzyme' in q_lower and len(sub_queries) < 4:
        sub_queries.append(f"{query} catalytic efficiency substrate specificity")
        sub_queries.append(f"{query} thermostability industrial application")

    return sub_queries[:5]


def _deduplicate_papers(papers: list) -> list:
    """Remove duplicate papers by DOI and title similarity."""
    seen_dois = set()
    seen_titles = set()
    unique = []
    for p in papers:
        doi = p.get('doi', '').strip().lower()
        title = re.sub(r'\s+', ' ', p.get('title', '').lower()).strip()[:80]
        if doi and doi in seen_dois:
            continue
        if title and title in seen_titles:
            continue
        if doi:
            seen_dois.add(doi)
        if title:
            seen_titles.add(title)
        unique.append(p)
    return unique


def _synthesize_innovations(papers: list, query: str,
                              backend=None) -> list:
    """Use LLM to synthesize innovation directions from gathered papers.

    Returns a list of dicts:
        {direction, rationale, relevant_refs (list of DOI/title)}
    Falls back to a heuristic summary if no backend available.
    """
    if not papers:
        return []

    if backend is not None:
        try:
            paper_snippets = []
            for i, p in enumerate(papers[:15], 1):
                snippet = (
                    f"{i}. {p.get('title', 'Unknown title')} "
                    f"({p.get('journal', '')} {p.get('year', '')})"
                )
                if p.get('abstract'):
                    snippet += f" -- {p['abstract'][:200]}"
                paper_snippets.append(snippet)

            prompt = (
                f"You are an enzyme engineering research advisor. Based on the "
                f"following papers retrieved for the query '{query}', identify "
                f"2-3 specific research gaps or innovation directions.\n\n"
                f"Papers:\n" + "\n".join(paper_snippets) + "\n\n"
                f"Respond in JSON format as a list:\n"
                f'[{{"direction": "...", "rationale": "...", '
                f'"relevant_refs": ["paper title or DOI"]}}]'
            )

            if hasattr(backend, 'chat'):
                response = backend.chat(prompt)
            elif callable(backend):
                response = backend(prompt)
            else:
                response = None

            if response and isinstance(response, str):
                # Extract JSON from response
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    directions = json.loads(json_match.group())
                    if isinstance(directions, list):
                        return directions[:3]
        except Exception as e:
            logger.debug("LLM synthesis failed: %s", e)

    # Fallback: heuristic innovation directions based on paper metadata
    directions = []
    years = [int(p['year']) for p in papers if p.get('year', '').isdigit()]
    max_year = max(years) if years else 2024

    # Find most recent papers as "frontier"
    frontier_refs = [
        p.get('doi') or p.get('title', '')[:60]
        for p in papers
        if p.get('year', '') == str(max_year)
    ][:3]

    if len(papers) >= 3:
        directions.append({
            'direction': (
                f"Computational-experimental integration for {query}"),
            'rationale': (
                f"The {len(papers)} retrieved papers suggest active research "
                f"activity. Combining high-throughput computational screening "
                f"with targeted experimental validation could accelerate "
                f"discovery in this area."),
            'relevant_refs': frontier_refs,
        })
        directions.append({
            'direction': (
                f"Machine learning prediction models for {query}"),
            'rationale': (
                "Structure-activity relationship data from existing literature "
                "could train predictive models to identify promising variants "
                "or candidates without exhaustive experimental screening."),
            'relevant_refs': [p.get('doi', '') for p in papers[:2]],
        })

    return directions


def enhanced_literature_search(
    query: str,
    context: str = None,
    backend=None,
    max_results: int = 10,
    expand_intent: bool = True,
    **kw,
) -> dict:
    """
    Enhanced multi-source literature search with full citations and
    AI-synthesized innovation directions.

    Workflow:
      1. Translate Chinese query terms to English (auto)
      2. If expand_intent=True, generate 3-5 related sub-queries
      3. Search PubMed for each sub-query (primary source)
      4. Fall back to CrossRef for sub-queries with few PubMed results
      5. Deduplicate all results
      6. Use LLM backend (if available) to synthesize innovation directions

    Parameters
    ----------
    query        : main search query (Chinese/English/mixed)
    context      : extra context for intent expansion (enzyme name, goal)
    backend      : LLM backend object or callable for synthesis/expansion
    max_results  : max results per sub-query (default 10)
    expand_intent: generate sub-queries for broader coverage

    Returns
    -------
    dict with keys:
      query, sub_queries, papers (full citations with DOI),
      innovation_directions, total_papers, sources_used
    """
    try:
        # 1. Translate Chinese terms
        original_query = query
        query_en = _translate_query(query)
        translated = query_en != original_query

        # 2. Expand intent
        if expand_intent:
            sub_queries = _expand_query_intent(query_en, context=context,
                                               backend=backend)
        else:
            sub_queries = [query_en]

        all_papers = []
        sources_used = []

        # 3. Search PubMed for each sub-query
        for sq in sub_queries:
            pubmed_result = search_pubmed(sq, max_results=max(3, max_results // 2))
            sq_papers = pubmed_result.get('papers', [])

            # 4. CrossRef fallback if PubMed returns < 2 results
            if len(sq_papers) < 2:
                cr_result = search_crossref(sq, max_results=5)
                cr_papers = cr_result.get('papers', [])
                if cr_papers:
                    sq_papers.extend(cr_papers)
                    if 'CrossRef' not in sources_used:
                        sources_used.append('CrossRef')

            if sq_papers and 'PubMed' not in sources_used:
                sources_used.append('PubMed')
            all_papers.extend(sq_papers)

        # 5. Deduplicate
        unique_papers = _deduplicate_papers(all_papers)

        # 6. Synthesize innovation directions
        innovation_directions = _synthesize_innovations(
            unique_papers, query_en, backend=backend)

        result = {
            'query': query_en,
            'sub_queries': sub_queries,
            'total_papers': len(unique_papers),
            'papers': unique_papers,
            'innovation_directions': innovation_directions,
            'sources_used': sources_used or ['PubMed'],
        }
        if translated:
            result['original_query'] = original_query
            result['translated'] = True

        if not unique_papers:
            result['hint'] = (
                'No papers found. Try broader terms or check network access.')

        return result

    except Exception as e:
        return {'error': str(e), 'hint': 'Enhanced literature search failed.'}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_skills(registry: SkillRegistry):
    registry.register(Skill(
        name='synthesis_knowledge',
        description='Reflect on the current session and record new insights, '
                    'successful patterns, or pitfalls to ZHUANGZHOU.md. '
                    'Use this to "remember" what worked and what failed.',
        category='research',
        fn=synthesis_knowledge,
        inputs=[
            SkillParam('insights', 'string', 'New scientific or structural insights', required=False),
            SkillParam('patterns', 'string', 'Successful workflow patterns or parameter combos', required=False),
            SkillParam('pitfalls', 'string', 'Errors encountered and how they were solved', required=False),
        ],
        outputs=[SkillParam('success', 'bool', 'Whether knowledge was recorded')],
        source='Zhuangzhou Agentic Memory',
    ))

    registry.register(Skill(
        name='search_pubmed',
        description='Search PubMed for scientific papers. Returns titles, '
                    'abstracts, authors, PMIDs. Use for literature evidence '
                    'before experiment design.',
        category='research',
        fn=search_pubmed,
        requires_network=True,
        inputs=[
            SkillParam('query', 'string',
                       'PubMed search query (supports MeSH, Boolean)'),
            SkillParam('max_results', 'int', 'Max papers to return',
                       required=False, default=5),
        ],
        outputs=[SkillParam('papers', 'list', 'List of papers with abstracts')],
        source='NCBI E-utilities',
    ))

    registry.register(Skill(
        name='search_uniprot',
        description='Search UniProt for protein information. Returns protein '
                    'name, function, active sites, metal binding, mutagenesis '
                    'records.',
        category='research',
        fn=search_uniprot,
        requires_network=True,
        inputs=[
            SkillParam('query', 'string', 'Protein name or keyword'),
            SkillParam('organism', 'string', 'Organism filter',
                       required=False),
            SkillParam('max_results', 'int', 'Max results',
                       required=False, default=5),
        ],
        outputs=[SkillParam('results', 'list', 'UniProt protein entries')],
        source='UniProt REST API',
    ))

    registry.register(Skill(
        name='get_uniprot_entry',
        description='Get detailed UniProt entry by accession ID. Returns '
                    'full annotation: active sites, metal binding, '
                    'mutagenesis, catalytic activity, PDB cross-refs.',
        category='research',
        fn=get_uniprot_entry,
        requires_network=True,
        inputs=[
            SkillParam('accession', 'string', 'UniProt accession (e.g. P00183)'),
        ],
        outputs=[SkillParam('entry', 'dict', 'Full UniProt entry')],
        source='UniProt REST API',
    ))

    registry.register(Skill(
        name='search_graphrag',
        description='Search local GraphRAG knowledge graphs (7 domains: '
                    'Bioinformatics, Computational_Chemistry, Cytochrome_P450, '
                    'Evolution, Structural_Biology, Synthetic_Biology, '
                    'Systems_Biology). Returns entities and community context.',
        category='research',
        fn=search_graphrag,
        inputs=[
            SkillParam('query', 'string', 'Search query'),
            SkillParam('domain', 'string',
                       'KG domain or "auto" for auto-routing',
                       required=False, default='auto',
                       choices=['auto', 'all', 'Bioinformatics', 'Computational_Chemistry',
                                'Cytochrome_P450', 'Evolution', 'Structural_Biology',
                                'Synthetic_Biology', 'Systems_Biology']),
            SkillParam('limit', 'int', 'Max results',
                       required=False, default=10),
        ],
        outputs=[
            SkillParam('entities', 'list', 'Matching entities'),
            SkillParam('community_context', 'list', 'Community reports'),
        ],
        source='AutoGraphRAG (local)',
    ))

    registry.register(Skill(
        name='get_mutation_evidence',
        description='Search ALL knowledge sources for evidence about a '
                    'specific mutation. Combines GraphRAG + catalytic '
                    'mechanism rules + UniProt. ALWAYS call this before '
                    'recommending mutations.',
        category='research',
        fn=get_mutation_evidence,
        inputs=[
            SkillParam('mutation', 'string',
                       'Mutation in format like V21L'),
            SkillParam('enzyme_class', 'string',
                       'Enzyme class (p450, protease, etc.)',
                       required=False),
            SkillParam('protein_name', 'string', 'Protein name',
                       required=False),
        ],
        outputs=[SkillParam('evidence', 'dict', 'Multi-source evidence')],
        source='GraphRAG + LiteratureRAG + UniProt',
    ))

    registry.register(Skill(
        name='search_crossref',
        description='Search CrossRef for scientific literature. Returns '
                    'full citations with title, journal, year, DOI, authors, '
                    'and abstract snippet. Good fallback when PubMed returns '
                    'few results. Chinese queries are auto-translated.',
        category='research',
        fn=search_crossref,
        requires_network=True,
        inputs=[
            SkillParam('query', 'string', 'Search query'),
            SkillParam('max_results', 'int', 'Max results (default 5)',
                       required=False, default=5),
        ],
        outputs=[SkillParam('papers', 'list',
                            'Citations with DOI, authors, journal, year')],
        source='CrossRef API',
    ))

    registry.register(Skill(
        name='enhanced_literature_search',
        description='Enhanced multi-source literature search with full '
                    'citations (title, journal, year, DOI, authors, abstract) '
                    'and AI-synthesized innovation directions. Searches PubMed '
                    'first, CrossRef as fallback. Expands intent into 3-5 '
                    'sub-queries for broader coverage. Returns structured '
                    'innovation_directions with research gaps.',
        category='research',
        fn=enhanced_literature_search,
        requires_network=True,
        inputs=[
            SkillParam('query', 'string', 'Main research query'),
            SkillParam('context', 'string',
                       'Extra context (e.g. enzyme name, research goal)',
                       required=False),
            SkillParam('backend', 'string',
                       'LLM backend for synthesis (claude/deepseek/auto)',
                       required=False),
            SkillParam('max_results', 'int',
                       'Max results per sub-query (default 10)',
                       required=False, default=10),
            SkillParam('expand_intent', 'bool',
                       'Generate 3-5 sub-queries for broader coverage',
                       required=False, default=True),
        ],
        outputs=[
            SkillParam('papers', 'list', 'All papers with full citations'),
            SkillParam('innovation_directions', 'list',
                       'Synthesized research gaps and innovation directions'),
        ],
        source='PubMed + CrossRef + LLM synthesis',
    ))

"""
GraphRAG Bridge for Zhuangzhou
================================
Queries AutoGraphRAG parquet knowledge graphs via the sandboxed graphrag conda env.
Falls back to direct pandas loading if graphrag env unavailable (native mode).

Available KGs:
  Bioinformatics, Computational_Chemistry, Cytochrome_P450,
  Evolution, Synthetic_Biology, Systems_Biology, Structural Biology
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bilingual Query Translation (Chinese -> English)
# ---------------------------------------------------------------------------
# Static domain-specific dictionary. Covers the most common biotech/chem terms
# a researcher would type in Chinese. Terms NOT in the dict are left as-is so
# that English protein names (CYP3A4, etc.) pass through unchanged.

_ZH_EN_KG: dict = {
    # Core biology
    '酶': 'enzyme',
    '突变': 'mutation',
    '点突变': 'point mutation',
    '饱和突变': 'saturation mutagenesis',
    '定向进化': 'directed evolution',
    '理性设计': 'rational design',
    '活性位点': 'active site',
    '结合位点': 'binding site',
    '底物': 'substrate',
    '抑制剂': 'inhibitor',
    '激活剂': 'activator',
    '催化': 'catalysis',
    '催化机制': 'catalytic mechanism',
    '催化效率': 'catalytic efficiency',
    '催化活性': 'catalytic activity',
    '蛋白质': 'protein',
    '配体': 'ligand',
    '氨基酸': 'amino acid',
    '折叠': 'folding',
    '结构': 'structure',
    '功能': 'function',
    '表达': 'expression',
    '辅因子': 'cofactor',
    '辅酶': 'coenzyme',
    '细胞色素': 'cytochrome',
    '氧化还原': 'redox',
    '磷酸化': 'phosphorylation',
    '糖基化': 'glycosylation',
    '基因': 'gene',
    '转录': 'transcription',
    '翻译': 'translation',
    '对接': 'docking',
    '分子对接': 'molecular docking',
    '分子动力学': 'molecular dynamics',
    '代谢通路': 'metabolic pathway',
    '代谢途径': 'metabolic pathway',
    '途径': 'pathway',
    # Protein types
    '细胞色素P450': 'cytochrome P450',
    '脂肪酶': 'lipase',
    '蛋白酶': 'protease',
    '淀粉酶': 'amylase',
    '纤维素酶': 'cellulase',
    '氧化还原酶': 'oxidoreductase',
    '转氨酶': 'transaminase',
    '酮还原酶': 'ketoreductase',
    '漆酶': 'laccase',
    '过氧化物酶': 'peroxidase',
    '水解酶': 'hydrolase',
    '转移酶': 'transferase',
    '裂解酶': 'lyase',
    '异构酶': 'isomerase',
    '连接酶': 'ligase',
    # Structural/physical chemistry
    '蛋白质结构': 'protein structure',
    '二级结构': 'secondary structure',
    '三级结构': 'tertiary structure',
    '四级结构': 'quaternary structure',
    'α螺旋': 'alpha helix',
    'β折叠': 'beta sheet',
    '力场': 'force field',
    '构象': 'conformation',
    '结合能': 'binding energy',
    '结合亲和力': 'binding affinity',
    '自由能': 'free energy',
    '氢键': 'hydrogen bond',
    '疏水相互作用': 'hydrophobic interaction',
    '静电作用': 'electrostatic interaction',
    '范德华力': 'van der Waals',
    '二硫键': 'disulfide bond',
    '盐桥': 'salt bridge',
    '血红素': 'heme',
    '活性口袋': 'active site pocket',
    '结合口袋': 'binding pocket',
    # Metabolic / synthetic biology
    '代谢工程': 'metabolic engineering',
    '合成生物学': 'synthetic biology',
    '代谢通量': 'metabolic flux',
    '启动子': 'promoter',
    '密码子优化': 'codon optimization',
    '基因敲除': 'gene knockout',
    '基因过表达': 'gene overexpression',
    '生物催化': 'biocatalysis',
    '工业酶': 'industrial enzyme',
    '蛋白工程': 'protein engineering',
    '序列比对': 'sequence alignment',
    '多序列比对': 'multiple sequence alignment',
    '进化': 'evolution',
    '保守性': 'conservation',
    # Properties
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
    '酶活性': 'enzyme activity',
    '酶活': 'enzyme activity',
    # Drug / metabolism
    '药物代谢': 'drug metabolism',
    '羟基化': 'hydroxylation',
    '氧化': 'oxidation',
    '还原': 'reduction',
    '抑制': 'inhibition',
    '激活': 'activation',
    '药物': 'drug',
    '代谢物': 'metabolite',
    # Omics / bioinformatics
    '基因组': 'genome',
    '蛋白质组': 'proteome',
    '转录组': 'transcriptome',
    '同源': 'homolog',
    '系统发育': 'phylogenetics',
    '注释': 'annotation',
    # Misc
    '计算': 'computational',
    '模拟': 'simulation',
    '预测': 'prediction',
    '优化': 'optimization',
    '筛选': 'screening',
    '文献': 'literature',
    '研究': 'research',
    '实验': 'experiment',
}

# Pre-sort by length descending once, so longer phrases are matched first
_ZH_EN_KG_SORTED = sorted(_ZH_EN_KG.items(), key=lambda x: -len(x[0]))

_HAS_CHINESE_RE = re.compile(r'[\u4e00-\u9fff]')


def _is_chinese(text: str) -> bool:
    """Check if text contains significant Chinese content (>20% Chinese characters)."""
    if not text:
        return False
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return chinese_chars > len(text) * 0.2


def _translate_zh_query(query: str) -> dict:
    """
    Translate Chinese scientific query to English for KG search.
    Returns: {'primary': str, 'keywords': [...], 'is_chinese': bool}

    Applies longest-match dict substitution, then extracts any ASCII protein
    names/identifiers already embedded in the query for multi-keyword search.
    """
    if not _is_chinese(query):
        return {'primary': query, 'keywords': [query], 'is_chinese': False}

    translated = query
    keywords = []

    # Apply term substitutions (sorted by length descending — already sorted)
    for zh, en in _ZH_EN_KG_SORTED:
        if zh in translated:
            translated = translated.replace(zh, f' {en} ')
            keywords.append(en)

    # Strip remaining CJK characters
    translated = re.sub(r'[\u4e00-\u9fff]+', ' ', translated)
    translated = re.sub(r'\s+', ' ', translated).strip()

    # Extract ASCII terms already embedded in the original Chinese text
    ascii_terms = re.findall(r'[A-Z][A-Za-z0-9_-]{1,20}', query)
    keywords.extend(ascii_terms)

    # Deduplicate while preserving order
    seen: set = set()
    keywords = [k for k in keywords if k and not (k in seen or seen.add(k))]

    return {
        'primary': translated or query,
        'keywords': keywords or [translated or query],
        'is_chinese': True,
        'original': query,
    }


def translate_query_terms(text: str, backend=None) -> str:
    """Translate Chinese biotech/enzyme terms to English for KG search.

    If *text* contains no Chinese characters it is returned unchanged.
    With no backend (or if LLM call fails), uses the built-in
    domain-specific dictionary for longest-match substitution.

    Returns a bilingual string:
        "<original text> [EN: <translated english terms>]"
    so downstream KG search sees both the original (for any Chinese-aware
    indexing) and the English equivalents.

    Parameters
    ----------
    text    : query string (may be mixed Chinese/English)
    backend : optional LLM backend object with a .chat() method;
              reserved for future LLM-based expansion (not used yet)
    """
    if not _HAS_CHINESE_RE.search(text):
        return text  # Already English-only, nothing to do.

    translated = text
    for zh, en in _ZH_EN_KG_SORTED:
        translated = translated.replace(zh, f' {en} ')

    # Strip remaining CJK characters (particles, connectors not in dict)
    translated = re.sub(r'[\u4e00-\u9fff]+', ' ', translated)
    translated = re.sub(r'\s+', ' ', translated).strip()

    if translated == text.strip():
        # Nothing was substituted; return original
        return text

    return f"{text} [EN: {translated}]"

BETATHINK_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GRAPHRAG_ROOT = os.path.join(BETATHINK_ROOT, 'storage', 'graphrag')
BRIDGE_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              'envs', 'bridges', 'graphrag_bridge.py')

# Domain keyword routing (same as BeTaThink domain_agents.py but for enzyme engineering)
DOMAIN_KEYWORDS = {
    'Cytochrome_P450': [
        'cyp450', 'cytochrome', 'p450', 'cyp3a4', 'cyp2d6', 'cyp1a2',
        'drug metabolism', 'microsom', 'hydroxylat', 'oxidase',
    ],
    'Computational_Chemistry': [
        'dft', 'molecular dynamics', 'force field', 'docking', 'qm/mm',
        'gromacs', 'amber', 'orca', 'gaussian', 'basis set', 'functional',
        'binding energy', 'free energy', 'simulation',
    ],
    'Bioinformatics': [
        'sequence', 'alignment', 'blast', 'genome', 'proteome', 'phylogen',
        'uniprot', 'homolog', 'fasta', 'annotation',
    ],
    'Evolution': [
        'evolution', 'natural selection', 'phylogenet', 'conservation',
        'genetic drift', 'mutation rate', 'molecular clock', 'ancestral',
    ],
    'Synthetic_Biology': [
        'synthetic biology', 'gibson assembly', 'crispr', 'metabolic engineer',
        'promoter', 'plasmid', 'expression', 'chassis', 'directed evolution',
    ],
    'Systems_Biology': [
        'systems biology', 'flux balance', 'metabolic model', 'pathway',
        'stoichiometr', 'network motif', 'omics',
    ],
}


class GraphRAGBridge:
    """
    Query AutoGraphRAG parquet knowledge graphs.

    Primary mode: subprocess call to graphrag conda env (isolation).
    Fallback: direct pandas loading in native env (if pandas available).
    """

    def __init__(self, graphrag_root: str = None):
        self.root = graphrag_root or GRAPHRAG_ROOT
        self._sandbox = None
        self._fallback_mode = False

    @property
    def sandbox(self):
        if self._sandbox is None:
            try:
                from zhuangzhou.envs import get_sandbox
                self._sandbox = get_sandbox('graphrag')
            except Exception:
                self._sandbox = None
                self._fallback_mode = True
        return self._sandbox

    @property
    def available(self) -> bool:
        return os.path.isdir(self.root)

    def list_kgs(self) -> List[str]:
        """List available knowledge graph IDs."""
        if not self.available:
            return {'error': f'GraphRAG root not found: {self.root}',
                    'hint': 'Check BETATHINK_ROOT or graphrag_root parameter'}
        return [d for d in os.listdir(self.root)
                if os.path.isdir(os.path.join(self.root, d))
                and os.path.exists(os.path.join(self.root, d, 'entities.parquet'))]

    def _call_bridge(self, command: str, args: dict) -> dict:
        """Call the graphrag bridge script via sandbox or fallback."""
        # Try sandbox first
        sb = self.sandbox
        if sb and sb.available and not self._fallback_mode:
            result = sb.run_json_bridge(BRIDGE_SCRIPT, {
                'command': command, 'args': args
            })
            if result.success and isinstance(result.data, dict):
                return result.data.get('result', result.data)
            logger.debug("Sandbox bridge failed: %s, trying fallback", result.error)

        # Fallback: direct pandas loading (may not have pandas in native env)
        return self._fallback_call(command, args)

    def _fallback_call(self, command: str, args: dict) -> dict:
        """Fallback: run bridge logic directly if pandas is available."""
        try:
            import pandas as pd
        except ImportError:
            logger.warning("Neither graphrag sandbox nor pandas available")
            return {}

        # Import bridge functions directly
        import importlib.util
        spec = importlib.util.spec_from_file_location('graphrag_bridge', BRIDGE_SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        # Ensure the bridge can find pandas
        spec.loader.exec_module(mod)

        if command == 'search_entities':
            return mod.search_entities(**args)
        elif command == 'search_relationships':
            return mod.search_relationships(**args)
        elif command == 'community_reports':
            return mod.community_reports(**args)
        elif command == 'kg_stats':
            return mod.kg_stats(**args)
        elif command == 'list_kgs':
            return mod._list_kg_ids()
        return {}

    # ── Public API ────────────────────────────────────────────────────────

    def _search_entities_internal(self, keywords: List[str], kg_id: str,
                                   limit: int) -> dict:
        """Internal bridge call for entity search. Returns raw bridge result."""
        return self._call_bridge('search_entities', {
            'keywords': keywords, 'kg_id': kg_id, 'limit': limit
        })

    @staticmethod
    def _merge_entity_results(*result_dicts) -> Dict[str, list]:
        """Merge multiple entity result dicts, deduplicating by entity name."""
        merged: Dict[str, list] = {}
        for rd in result_dicts:
            if not isinstance(rd, dict):
                continue
            for kg_id, entities in rd.items():
                if not isinstance(entities, list):
                    continue
                existing = {e.get('entity', '') or e.get('id', '') for e in merged.get(kg_id, [])}
                new_entities = [
                    e for e in entities
                    if (e.get('entity', '') or e.get('id', '')) not in existing
                ]
                merged.setdefault(kg_id, []).extend(new_entities)
        return merged

    def search_entities(self, keywords: List[str], kg_id: str = 'all',
                        limit: int = 20) -> Dict[str, list]:
        """Search entities across knowledge graphs.

        Each keyword is translated if it contains Chinese characters so that
        the English-indexed parquet corpus is queried correctly.

        For Chinese queries, performs multi-pass search:
        1. Search with the fully translated query string
        2. Search with up to 2 top extracted English keywords
        Results are merged and deduplicated.
        """
        # Check if any keyword is Chinese — use enhanced multi-pass strategy
        if any(_is_chinese(kw) for kw in keywords):
            all_results = []
            extra_kws_seen: set = set()
            final_kws: List[str] = []

            for kw in keywords:
                if _is_chinese(kw):
                    info = _translate_zh_query(kw)
                    final_kws.append(info['primary'])
                    # Collect up to 2 unique extracted keywords per Chinese term
                    for ek in info.get('keywords', [])[:2]:
                        if ek and ek != info['primary'] and ek not in extra_kws_seen:
                            extra_kws_seen.add(ek)
                            extra_kws_seen.add(ek)  # idempotent
                else:
                    final_kws.append(kw)

            # Deduplicate final_kws while preserving order
            seen: set = set()
            final_kws = [k for k in final_kws if k and not (k in seen or seen.add(k))]

            # Primary search with translated keywords
            primary = self._search_entities_internal(final_kws or keywords, kg_id, limit)
            all_results.append(primary)

            # Secondary searches with top extracted keywords
            for ek in list(extra_kws_seen)[:2]:
                extra = self._search_entities_internal([ek], kg_id, max(3, limit // 3))
                all_results.append(extra)

            return self._merge_entity_results(*all_results)

        # Pure English path — translate individual terms as before
        translated_kws = []
        for kw in keywords:
            t = translate_query_terms(kw)
            # translate_query_terms returns "original [EN: ...]" format;
            # extract just the English portion for keyword matching
            if '[EN:' in t:
                en_part = t.split('[EN:', 1)[1].rstrip(']').strip()
                translated_kws.extend(en_part.split())
            else:
                translated_kws.append(t)
        # Deduplicate while preserving order
        seen = set()
        final_kws = []
        for kw in translated_kws:
            if kw and kw not in seen:
                seen.add(kw)
                final_kws.append(kw)
        return self._search_entities_internal(final_kws or keywords, kg_id, limit)

    def search_relationships(self, entity_name: str, kg_id: str = 'all',
                             limit: int = 20) -> Dict[str, list]:
        """Find relationships involving an entity."""
        return self._call_bridge('search_relationships', {
            'entity_name': entity_name, 'kg_id': kg_id, 'limit': limit
        })

    def get_community_context(self, query: str, kg_id: str = 'all',
                              limit: int = 5) -> Dict[str, list]:
        """Search community report summaries.

        Chinese queries are automatically translated to English before
        being forwarded to the parquet-backed bridge.
        """
        query = translate_query_terms(query)
        return self._call_bridge('community_reports', {
            'query': query, 'kg_id': kg_id, 'limit': limit
        })

    def get_kg_stats(self, kg_id: str) -> dict:
        """Get statistics for a knowledge graph."""
        return self._call_bridge('kg_stats', {'kg_id': kg_id})

    def select_domain(self, query: str) -> Optional[str]:
        """Select the most relevant domain KG for a query.

        Automatically translates Chinese terms before keyword matching so that
        Chinese queries are routed to the correct domain KG.
        """
        # Translate Chinese terms to English for domain routing
        translated = translate_query_terms(query)
        query_lower = translated.lower()
        scores = {}
        for kg_id, keywords in DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                scores[kg_id] = score
        if not scores:
            return None
        return max(scores, key=scores.get)

    def get_mutation_evidence(self, mutation: str, enzyme_class: str,
                              protein_name: str = '') -> dict:
        """
        Specialized evidence retrieval for enzyme mutations.
        Combines entity search + relationship search + community context.
        Chinese enzyme_class / protein_name are auto-translated before search.
        """
        # Translate Chinese terms in enzyme_class and protein_name
        if enzyme_class and _is_chinese(enzyme_class):
            enzyme_class = _translate_zh_query(enzyme_class)['primary']
        elif enzyme_class:
            enzyme_class = translate_query_terms(enzyme_class)
        if protein_name and _is_chinese(protein_name):
            protein_name = _translate_zh_query(protein_name)['primary']
        elif protein_name:
            protein_name = translate_query_terms(protein_name)
        domain = self.select_domain(enzyme_class) or 'all'

        # Search keywords: mutation itself + enzyme class terms
        keywords = [mutation]
        if protein_name:
            keywords.append(protein_name)
        # Add amino acid names from mutation string (e.g., V21L -> val, leu)
        aa_names = {
            'A': 'alanine', 'C': 'cysteine', 'D': 'aspartate', 'E': 'glutamate',
            'F': 'phenylalanine', 'G': 'glycine', 'H': 'histidine', 'I': 'isoleucine',
            'K': 'lysine', 'L': 'leucine', 'M': 'methionine', 'N': 'asparagine',
            'P': 'proline', 'Q': 'glutamine', 'R': 'arginine', 'S': 'serine',
            'T': 'threonine', 'V': 'valine', 'W': 'tryptophan', 'Y': 'tyrosine',
        }
        if len(mutation) >= 3:
            wt_aa = mutation[0]
            mut_aa = mutation[-1]
            if wt_aa in aa_names:
                keywords.append(aa_names[wt_aa])
            if mut_aa in aa_names:
                keywords.append(aa_names[mut_aa])

        entities = self.search_entities(keywords, kg_id=domain, limit=10)
        relationships = self.search_relationships(mutation, kg_id=domain, limit=10)
        community = self.get_community_context(
            f"{enzyme_class} {mutation} mutation catalytic effect",
            kg_id=domain, limit=3)

        total = sum(len(v) for v in entities.values()) if isinstance(entities, dict) else 0

        return {
            'domain': domain,
            'entities': entities,
            'relationships': relationships,
            'community_context': community,
            'n_hits': total,
        }

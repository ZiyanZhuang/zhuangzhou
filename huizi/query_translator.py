"""
QueryTranslator — convert Chinese scientific queries to English academic keywords.
Uses LLM when available, falls back to expanded dictionary.
"""
import re
import json
from typing import Optional, List, Dict

# Expanded Chinese→English scientific dictionary
_ZH_EN_DICT: Dict[str, List[str]] = {
    # Enzymes & proteins
    "尼古丁酰胺酶": ["nicotinamidase", "pyrazinamidase", "pncA"],
    "烟酰胺酶": ["nicotinamidase", "nicotinamide amidohydrolase"],
    "细胞色素P450": ["cytochrome P450", "CYP450", "CYP"],
    "羟化酶": ["hydroxylase", "monooxygenase"],
    "脱氢酶": ["dehydrogenase", "oxidoreductase"],
    "蛋白酶": ["protease", "peptidase"],
    "激酶": ["kinase", "phosphotransferase"],
    "转移酶": ["transferase"],
    "水解酶": ["hydrolase"],
    "氧化还原酶": ["oxidoreductase", "oxidase"],
    "合酶": ["synthase", "synthetase"],
    "异构酶": ["isomerase"],
    "裂解酶": ["lyase"],
    "氨基酸": ["amino acid", "residue"],
    "蛋白质": ["protein", "enzyme", "polypeptide"],
    # Structure
    "活性位点": ["active site", "catalytic site", "binding site"],
    "结合口袋": ["binding pocket", "active site pocket"],
    "螺旋": ["alpha-helix", "helix"],
    "折叠": ["beta-sheet", "fold"],
    "结构域": ["domain", "protein domain"],
    "氢键": ["hydrogen bond", "H-bond"],
    "疏水相互作用": ["hydrophobic interaction"],
    "盐桥": ["salt bridge", "ionic interaction"],
    "二硫键": ["disulfide bond"],
    # Mutations
    "突变": ["mutation", "amino acid substitution", "variant"],
    "点突变": ["point mutation"],
    "突变扫描": ["mutation scanning", "saturation mutagenesis"],
    "有益突变": ["beneficial mutation", "activating mutation"],
    "有害突变": ["deleterious mutation", "loss-of-function"],
    "热稳定性": ["thermostability", "thermal stability"],
    "催化活性": ["catalytic activity", "enzyme activity"],
    "底物特异性": ["substrate specificity"],
    "催化机制": ["catalytic mechanism", "reaction mechanism"],
    # Methods
    "分子对接": ["molecular docking", "protein-ligand docking"],
    "分子动力学": ["molecular dynamics", "MD simulation"],
    "自由能计算": ["free energy calculation", "binding free energy"],
    "同源建模": ["homology modeling"],
    "蛋白质语言模型": ["protein language model", "PLM", "ESM"],
    "序列比对": ["sequence alignment", "MSA"],
    "进化分析": ["evolutionary analysis", "phylogenetic analysis"],
    # Ligands/drugs
    "抑制剂": ["inhibitor", "IC50"],
    "底物": ["substrate", "natural substrate"],
    "激活剂": ["activator", "agonist"],
    "药物": ["drug", "small molecule", "compound"],
    "先导化合物": ["lead compound", "lead optimization"],
    # Protein features
    "辅因子": ["cofactor", "coenzyme"],
    "辅基": ["prosthetic group", "heme", "FAD"],
    "金属离子": ["metal ion", "zinc", "iron"],
    "变构": ["allosteric", "allostery"],
    "构象变化": ["conformational change", "conformational dynamics"],
    # Analysis
    "二级结构": ["secondary structure", "DSSP"],
    "溶剂可及性": ["solvent accessibility", "RSA", "SASA"],
    "保守性": ["evolutionary conservation", "sequence conservation"],
    "共进化": ["coevolution"],
    "折叠稳定性": ["folding stability", "thermodynamic stability"],
}

_LLM_PROMPT = (
    "You are a bilingual expert in protein engineering and structural biology.\n"
    "Extract English academic keywords from this Chinese scientific query for knowledge graph search.\n\n"
    "Chinese query: {query}\n\n"
    "Return ONLY valid JSON:\n"
    '{"keywords": ["term1", "term2", "term3"], "translated": "full English translation", '
    '"domain": "protein_engineering|structural_biology|pharmacology|computational_chemistry|biochemistry"}'
)


class QueryTranslator:
    """Translates Chinese scientific queries to English keywords."""

    def __init__(self, llm_backend=None):
        self._llm = llm_backend

    def translate(self, query: str) -> dict:
        """
        Returns:
            {'keywords': [...], 'translated': str, 'domain': str, 'source': str}
        """
        if not self._is_chinese(query):
            return {
                'keywords': [query], 'concepts': [],
                'translated': query, 'domain': 'biochemistry', 'source': 'passthrough'
            }

        # Try LLM first
        if self._llm:
            result = self._llm_translate(query)
            if result:
                result['source'] = 'llm'
                return result

        # Dictionary fallback
        result = self._dict_translate(query)
        result['source'] = 'dict'
        return result

    def _is_chinese(self, text: str) -> bool:
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        return chinese_chars > len(text) * 0.2

    def _llm_translate(self, query: str) -> Optional[dict]:
        try:
            prompt = _LLM_PROMPT.format(query=query)
            response = None
            if hasattr(self._llm, 'complete'):
                response = self._llm.complete(prompt, max_tokens=300)
            elif hasattr(self._llm, 'chat_raw'):
                response = self._llm.chat_raw(prompt)
            if response:
                m = re.search(r'\{[^{}]*\}', response, re.DOTALL)
                if m:
                    return json.loads(m.group())
        except Exception:
            pass
        return None

    def _dict_translate(self, query: str) -> dict:
        keywords: List[str] = []
        translated = query

        for zh, en_list in sorted(_ZH_EN_DICT.items(), key=lambda x: -len(x[0])):
            if zh in query:
                translated = translated.replace(zh, en_list[0])
                keywords.extend(en_list[:2])

        # ASCII terms already in query
        keywords.extend(re.findall(r'[A-Z][A-Za-z0-9_-]{1,20}', query))

        # Deduplicate
        seen: set = set()
        keywords = [k for k in keywords if not (k in seen or seen.add(k))]

        return {
            'keywords': keywords or [query],
            'concepts': [],
            'translated': translated,
            'domain': self._infer_domain(query),
        }

    def _infer_domain(self, query: str) -> str:
        if any(t in query for t in ['P450', 'CYP', '细胞色素']):
            return 'structural_biology'
        if any(t in query for t in ['突变', '扫描', '进化']):
            return 'protein_engineering'
        if any(t in query for t in ['药物', '抑制剂', 'IC50']):
            return 'pharmacology'
        if any(t in query for t in ['对接', '动力学', '自由能']):
            return 'computational_chemistry'
        return 'biochemistry'

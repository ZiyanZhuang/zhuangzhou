"""
Bio-design Mode — Zhuangzhou v0.8.0
=====================================
合成生物学 & 代谢工程 自动化设计引擎

像乐高一样组装生物元件 (启动子 → RBS → 酶 → 终止子)，
针对特定底盘细胞 (E. coli / 酵母) 优化代谢通路生产特定分子。

设计流程 (6个阶段):
  ┌──────────────────────────────────────────────────────────────────┐
  │  1. 目标定义    → 产物、底盘、产量目标                           │
  │  2. 通路挖掘    → metaKG BFS + GraphRAG 文献                   │
  │  3. 流量分析    → FBA 理论产量 + 瓶颈识别                       │
  │  4. 干扰消除    → 竞争通路 + 毒性筛查 → 敲除候选               │
  │  5. 元件设计    → 启动子/RBS/密码子优化 + 代谢负担评估          │
  │  6. 设计封装    → JSON设计包 + Markdown报告                     │
  └──────────────────────────────────────────────────────────────────┘

与 Plan Mode 的区别:
  - Plan Mode: 通用科研问题意图对齐
  - Bio-design Mode: 专门的代谢工程自动化设计流水线
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Stage configuration ────────────────────────────────────────────────────────

DESIGN_STAGES = [
    "target_definition",
    "pathway_mining",
    "flux_analysis",
    "interference_removal",
    "parts_design",
    "design_package",
]

STAGE_LABELS = {
    "target_definition":    "阶段1: 目标定义",
    "pathway_mining":       "阶段2: 通路挖掘",
    "flux_analysis":        "阶段3: 流量分析",
    "interference_removal": "阶段4: 干扰消除",
    "parts_design":         "阶段5: 元件设计",
    "design_package":       "阶段6: 设计封装",
}

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class DesignTarget:
    product: str                     # target molecule name
    bigg_metabolite: str = ""        # BiGG ID (e.g. 'tyr__L_c')
    chassis: str = "ecoli"           # 'ecoli' or 'yeast'
    titer_goal_g_l: float = 0.0      # production goal g/L (0 = not specified)
    carbon_source: str = "glucose"   # primary carbon source
    bigg_carbon: str = "glc__D_e"    # BiGG ID for carbon source


@dataclass
class PathwayDesign:
    source_metabolite: str = ""
    target_metabolite: str = ""
    pathway_reactions: List[str] = field(default_factory=list)
    pathway_enzymes: List[str] = field(default_factory=list)
    n_steps: int = 0
    native_to_chassis: bool = True   # exists in chassis naturally?
    heterologous_genes: List[str] = field(default_factory=list)
    literature_evidence: List[str] = field(default_factory=list)


@dataclass
class FluxAnalysis:
    theoretical_yield_mmol: float = 0.0
    growth_rate: float = 0.0
    bottleneck_reactions: List[str] = field(default_factory=list)
    cofactor_issues: List[str] = field(default_factory=list)
    shadow_prices: Dict[str, float] = field(default_factory=dict)
    # ── raw result dicts for passthrough to report ──
    fba_result_raw: Dict = field(default_factory=dict)
    bottleneck_result_raw: Dict = field(default_factory=dict)


@dataclass
class InterferenceMap:
    competing_reactions: List[Dict] = field(default_factory=list)
    knockout_candidates: List[str] = field(default_factory=list)
    toxic_intermediates: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)


@dataclass
class PartsDesign:
    expression_plan: List[Dict] = field(default_factory=list)   # from expression_balance
    crispr_designs: List[Dict] = field(default_factory=list)
    codon_optimized: bool = False
    burden_score: float = 0.0
    burden_risk: str = "UNKNOWN"
    # ── raw result dicts for passthrough to report ──
    burden_result_raw: Dict = field(default_factory=dict)
    protein_candidates: List[Dict] = field(default_factory=list)
    codon_sequences: Dict = field(default_factory=dict)


@dataclass
class BioDesignResult:
    target: DesignTarget
    pathway: PathwayDesign
    flux: FluxAnalysis
    interference: InterferenceMap
    parts: PartsDesign
    design_package_path: str = ""
    completed_stages: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    # ── raw pathway search result for report ──
    metakg_pathway_raw: Dict = field(default_factory=dict)

    def to_agent_message(self) -> str:
        """Convert to structured AgentExecutor message."""
        lines = [
            f"【Bio-design 已完成】目标产物: {self.target.product} | 底盘: {self.target.chassis}",
            "",
            f"完成阶段: {', '.join(self.completed_stages)}",
            "",
            "=== 通路设计 ===",
            f"  来源代谢物: {self.pathway.source_metabolite} → 目标: {self.pathway.target_metabolite}",
            f"  步骤数: {self.pathway.n_steps}",
            f"  关键酶: {', '.join(self.pathway.pathway_enzymes[:5]) or '待确定'}",
        ]

        if self.pathway.heterologous_genes:
            lines.append(f"  外源基因 (需引入): {', '.join(self.pathway.heterologous_genes)}")

        if self.flux.theoretical_yield_mmol > 0:
            lines += [
                "", "=== 流量分析 ===",
                f"  理论产量: {self.flux.theoretical_yield_mmol:.3f} mmol/gDW/h",
                f"  生长速率: {self.flux.growth_rate:.3f} h⁻¹",
            ]
            if self.flux.bottleneck_reactions:
                lines.append(f"  瓶颈反应: {', '.join(self.flux.bottleneck_reactions[:3])}")

        if self.interference.knockout_candidates:
            lines += ["", "=== 干扰消除 ===",
                      f"  敲除候选: {', '.join(self.interference.knockout_candidates[:5])}"]
        if self.interference.toxic_intermediates:
            lines.append(f"  ⚠ 毒性中间体: {', '.join(self.interference.toxic_intermediates)}")

        if self.parts.expression_plan:
            lines += ["", "=== 元件设计 ==="]
            for rec in self.parts.expression_plan[:5]:
                lines.append(
                    f"  {rec.get('enzyme','?')}: 启动子 {rec.get('promoter','?')} "
                    f"(强度{rec.get('promoter_strength',0):.2f}) + RBS {rec.get('rbs','?')}"
                )

        if self.design_package_path:
            lines += ["", f"  设计包: {self.design_package_path}"]

        if self.warnings:
            lines += ["", "⚠ 注意:"] + [f"  - {w}" for w in self.warnings]

        lines += [
            "", "请基于以上设计继续完成:",
            "1. 搜索每个酶的实验kcat/Km数据 (search_pubmed / get_protein_profile)",
            "2. 获取外源基因序列并密码子优化 (synbio_codon_optimize)",
            "3. 设计验证实验方案",
        ]
        return "\n".join(lines)


# ── Interactive clarification UI ──────────────────────────────────────────────

BIODESIGN_QUESTIONS = {
    "chassis": {
        "question": "底盘生物 (Chassis Organism)",
        "hint": "选择目标生产菌株",
        "options": [
            ("ecoli",  "E. coli K-12 MG1655 / BL21(DE3)", "模式菌株，遗传工具最成熟"),
            ("yeast",  "S. cerevisiae (酵母 BY4741/CEN.PK)", "耐受有机酸，分泌型表达好"),
            ("both",   "比较两者再决定 (推荐)",            "系统自动对比后推荐"),
        ],
    },
    "product_class": {
        "question": "目标产物类别",
        "hint": "影响通路挖掘策略",
        "options": [
            ("aromatic",   "芳香族化合物 (苯丙氨酸/酪氨酸/莽草酸/香草醛)", "莽草酸通路"),
            ("organic_acid","有机酸 (琥珀酸/乳酸/苹果酸/柠檬酸)",          "TCA/糖酵解衍生"),
            ("terpenoid",  "萜类化合物 (青蒿素/紫杉醇前体/角鲨烯)",         "MVA/MEP通路"),
            ("fatty_acid", "脂肪酸/生物柴油 (游离脂肪酸/脂肪醇)",           "FAS通路"),
            ("amino_acid", "氨基酸 (L-谷氨酸/L-赖氨酸/L-色氨酸)",         "中心代谢衍生"),
            ("other",      "其他/自定义",                                  ""),
        ],
    },
    "optimization_target": {
        "question": "主要优化目标",
        "hint": "决定设计侧重点",
        "options": [
            ("yield",    "产率最大化 (g产物/g碳源)",        "减少竞争通路，优化辅因子"),
            ("titer",    "产物浓度最大化 (g/L)",            "增强表达，减少毒性"),
            ("rate",     "生产速率最大化 (g/L/h)",          "优化酶活，减少代谢负担"),
            ("balanced", "均衡优化 (推荐: 产率+产量+稳定性)", "综合考虑"),
        ],
    },
    "engineering_depth": {
        "question": "工程深度",
        "hint": "决定分析粒度和工作量",
        "options": [
            ("quick",  "快速评估 (~10min): 通路挖掘+理论产量",          "概念验证阶段"),
            ("standard","标准设计 (~30min): 完整六阶段分析",            "实验室设计阶段"),
            ("deep",   "深度设计 (+酶工程+ALE+动态调控)",               "生产优化阶段"),
        ],
    },
}

# ── Known pathway seeds (common metabolic engineering targets) ────────────────

KNOWN_PATHWAYS = {
    # aromatic
    "tyrosine":   {"source": "pep_c",    "target": "tyr__L_c",  "exchange": "EX_tyr__L_e"},
    "phenylalanine": {"source": "pep_c", "target": "phe__L_c",  "exchange": "EX_phe__L_e"},
    "tryptophan": {"source": "pep_c",    "target": "trp__L_c",  "exchange": "EX_trp__L_e"},
    "shikimate":  {"source": "pep_c",    "target": "skm_c",     "exchange": "EX_skm_e"},
    "4hba":       {"source": "pep_c",    "target": "4hbz_c",    "exchange": "EX_4hbz_e"},
    # organic acids
    "succinate":  {"source": "pep_c",    "target": "succ_c",    "exchange": "EX_succ_e"},
    "malate":     {"source": "pyr_c",    "target": "mal__L_c",  "exchange": "EX_mal__L_e"},
    "citrate":    {"source": "pyr_c",    "target": "cit_c",     "exchange": "EX_cit_e"},
    "lactate":    {"source": "pyr_c",    "target": "lac__D_c",  "exchange": "EX_lac__D_e"},
    # amino acids
    "glutamate":  {"source": "akg_c",    "target": "glu__L_c",  "exchange": "EX_glu__L_e"},
    "lysine":     {"source": "asp__L_c", "target": "lys__L_c",  "exchange": "EX_lys__L_e"},
    "threonine":  {"source": "asp__L_c", "target": "thr__L_c",  "exchange": "EX_thr__L_e"},
    # fatty acids
    "palmitate":  {"source": "accoa_c",  "target": "hdcea_c",   "exchange": "EX_hdcea_e"},
}


# ── Main BioDesignMode class ───────────────────────────────────────────────────

class BioDesignMode:
    """Six-stage metabolic engineering auto-design pipeline.

    Usage:
        bdm = BioDesignMode(backend, registry)
        result = bdm.run(user_query, console=console)
        # result.to_agent_message() → structured AgentExecutor input
    """

    def __init__(self, backend, registry, verbose: bool = False):
        self.backend = backend
        self.registry = registry
        self.verbose = verbose

    @dataclass
    class Result:
        design: Optional[BioDesignResult] = None
        skipped: bool = False
        original_query: str = ""
        error: str = ""

        @property
        def agent_message(self) -> str:
            if self.skipped or self.design is None:
                return self.original_query
            return self.design.to_agent_message()

    def run(self, query: str, console=None) -> "BioDesignMode.Result":
        """Full pipeline entry point."""
        if console is None:
            from zhuangzhou.ui.core import console as _c
            console = _c

        from zhuangzhou.ui.core import ZZTheme
        from rich.panel import Panel
        from rich.text import Text

        console.print("")
        console.print(Panel(
            "[bold #00E5FF]Bio-design Mode[/]  🧬\n"
            "[dim]合成生物学 & 代谢工程自动化设计  │  6阶段流水线  │  Esc/n 退出[/]",
            border_style="#4CAF50", expand=False, padding=(0, 2),
        ))

        # Step 1: Clarify design target interactively
        target = self._clarify_target(query, console, ZZTheme)
        if target is None:
            return self.Result(skipped=True, original_query=query)

        design = BioDesignResult(
            target=target,
            pathway=PathwayDesign(),
            flux=FluxAnalysis(),
            interference=InterferenceMap(),
            parts=PartsDesign(),
        )

        depth = getattr(self, "_depth", "standard")

        # Step 2: Pathway mining
        console.print(f"\n  [bold #FFC107]▶ {STAGE_LABELS['pathway_mining']}[/]")
        self._run_pathway_mining(design, console)
        design.completed_stages.append("pathway_mining")

        # Step 3: Flux analysis
        if depth != "quick":
            console.print(f"\n  [bold #FFC107]▶ {STAGE_LABELS['flux_analysis']}[/]")
            self._run_flux_analysis(design, console)
            design.completed_stages.append("flux_analysis")

        # Step 4: Interference removal
        console.print(f"\n  [bold #FFC107]▶ {STAGE_LABELS['interference_removal']}[/]")
        self._run_interference_removal(design, console)
        design.completed_stages.append("interference_removal")

        # Step 5: Parts design
        console.print(f"\n  [bold #FFC107]▶ {STAGE_LABELS['parts_design']}[/]")
        self._run_parts_design(design, console)
        design.completed_stages.append("parts_design")

        # Step 6: Design package
        console.print(f"\n  [bold #FFC107]▶ {STAGE_LABELS['design_package']}[/]")
        self._run_design_package(design, console)
        design.completed_stages.append("design_package")

        self._print_summary(design, console)
        return self.Result(design=design, original_query=query)

    # ── Stage 1: Target clarification ─────────────────────────────────────────

    def _clarify_target(self, query: str, console, ZZTheme) -> Optional[DesignTarget]:
        """Interactive clarification of design target."""
        from rich.text import Text

        # Extract product name from query
        product = self._extract_product(query)
        if not product:
            product = self._prompt_input("请输入目标产物名称 (e.g. 酪氨酸/tyrosine): ", console)
            if not product:
                return None

        console.print(f"\n  [dim]目标产物:[/] [bold white]{product}[/]")

        # Chassis selection
        chassis_result = self._select_option(
            BIODESIGN_QUESTIONS["chassis"], console, ZZTheme
        )
        if chassis_result is None:
            return None
        chassis = chassis_result[0]
        if chassis == "both":
            chassis = "ecoli"  # default; comparison done in pathway mining

        # Optimization target
        opt_result = self._select_option(
            BIODESIGN_QUESTIONS["optimization_target"], console, ZZTheme
        )
        opt_target = opt_result[0] if opt_result else "balanced"

        # Engineering depth (save for later)
        depth_result = self._select_option(
            BIODESIGN_QUESTIONS["engineering_depth"], console, ZZTheme
        )
        self._depth = depth_result[0] if depth_result else "standard"

        # Look up BiGG IDs
        product_lower = product.lower()
        bigg_info = KNOWN_PATHWAYS.get(product_lower, {})
        bigg_met = bigg_info.get("target", "")

        if not bigg_met:
            console.print(f"  [dim]'{product}' 不在已知通路表中。将通过 metaKG 搜索...[/]")

        # Titer goal
        titer_str = self._prompt_input(
            f"目标产量 g/L (直接 Enter 跳过): ", console
        )
        try:
            titer_goal = float(titer_str) if titer_str else 0.0
        except ValueError:
            titer_goal = 0.0

        target = DesignTarget(
            product=product,
            bigg_metabolite=bigg_met,
            chassis=chassis,
            titer_goal_g_l=titer_goal,
            carbon_source="glucose",
            bigg_carbon="glc__D_e",
        )
        self._opt_target = opt_target
        return target

    # ── Stage 2: Pathway mining ────────────────────────────────────────────────

    def _run_pathway_mining(self, design: BioDesignResult, console) -> None:
        """Query metaKG + GraphRAG to find metabolic pathway."""
        product = design.target.product
        chassis = design.target.chassis
        product_lower = product.lower()

        # 1. Check known pathways
        known = KNOWN_PATHWAYS.get(product_lower)
        if known:
            design.pathway.source_metabolite = known["source"]
            design.pathway.target_metabolite = known["target"]
            design.target.bigg_metabolite = known["target"]
            console.print(f"  [dim]✓ 已知通路: {known['source']} → {known['target']}[/]")
        else:
            # 2. Search metaKG
            from zhuangzhou.skills.metakg import metakg_query
            search_result = metakg_query(product, organism=chassis)
            mets = [n for n in search_result.get("nodes", [])
                    if n.get("type") == "Metabolite"]
            if mets:
                best = mets[0]
                design.pathway.target_metabolite = best["id"]
                design.target.bigg_metabolite = best["id"]
                console.print(f"  [dim]✓ metaKG 找到代谢物: {best['id']} ({best.get('name','')})[/]")
            else:
                design.warnings.append(f"'{product}' 未在 metaKG 中找到。需要手动指定BiGG ID。")
                console.print(f"  [dim]⚠ metaKG 未找到 '{product}'[/]")

        # 3. BFS pathway finding
        if design.pathway.source_metabolite and design.pathway.target_metabolite:
            from zhuangzhou.skills.metakg import metakg_pathway
            path_result = metakg_pathway(
                source_metabolite=design.pathway.source_metabolite,
                target_metabolite=design.pathway.target_metabolite,
                organism=chassis,
                max_steps=8,
            )
            if path_result.get("paths_found", 0) > 0:
                best_path = path_result["paths"][0]
                design.pathway.n_steps = best_path["n_steps"]
                design.pathway.pathway_enzymes = list({
                    g for step in best_path["steps"] for g in step.get("genes", [])
                })
                design.pathway.pathway_reactions = [
                    s["reaction"] for s in best_path["steps"]
                ]
                design.metakg_pathway_raw = path_result  # ← save for report
                console.print(
                    f"  [dim]✓ 找到通路: {best_path['n_steps']} 步, "
                    f"酶: {', '.join(design.pathway.pathway_enzymes[:4])}[/]"
                )
            else:
                design.warnings.append("BFS 未找到直接通路 — 可能需要外源酶。")
                console.print(f"  [dim]⚠ BFS 未找到直接通路[/]")

        # 4. GraphRAG literature evidence
        try:
            from zhuangzhou.skills.synbio import synbio_graphrag_query
            lit = synbio_graphrag_query(
                query=f"{product} metabolic engineering {chassis} biosynthesis",
                kg_domain="Synthetic_Biology",
                limit=5,
            )
            ctx = lit.get("community_context", {})
            if ctx:
                design.pathway.literature_evidence = list(ctx.keys())[:3]
                console.print(f"  [dim]✓ GraphRAG: 找到 {len(ctx)} 个相关社区摘要[/]")
        except Exception as e:
            logger.debug("GraphRAG query failed: %s", e)

        # 5. LLM fallback — for compounds not in MetaKG/KNOWN_PATHWAYS
        if not design.pathway.pathway_enzymes:
            self._run_pathway_mining_llm_fallback(design, console)

    def _run_pathway_mining_llm_fallback(self, design: BioDesignResult, console) -> bool:
        """Use LLM backend to infer biosynthetic pathway for unknown compounds.

        Called when both KNOWN_PATHWAYS and MetaKG lookups fail to return enzyme
        information. Populates design.pathway fields from LLM knowledge and saves
        a synthetic metakg_pathway_raw dict so downstream stages work normally.

        Returns True if enzymes were successfully inferred.
        """
        product = design.target.product
        chassis = design.target.chassis
        console.print(f"  [dim]  → MetaKG 未找到通路，使用 LLM 推断 '{product}' 生物合成路线...[/]")

        prompt = (
            f'You are a metabolic engineering expert.\n'
            f'The target compound is: "{product}" (may be a Chinese name or transliteration).\n'
            f'Chassis organism: {chassis} (E. coli or S. cerevisiae).\n\n'
            f'Task: Identify the known biosynthetic pathway for this compound and return '
            f'structured information. If the name is Chinese, first identify the English name.\n\n'
            f'Respond ONLY with valid JSON (no markdown, no explanation):\n'
            f'{{\n'
            f'  "english_name": "English compound name",\n'
            f'  "compound_class": "e.g. terpenoid/polyketide/alkaloid/peptide/other",\n'
            f'  "source_metabolite": "precursor from central metabolism (e.g. acetyl-CoA, PEP, shikimate)",\n'
            f'  "pathway_enzymes": ["gene1", "gene2"],\n'
            f'  "pathway_reactions": ["rxn_id_or_EC1", "rxn_id_or_EC2"],\n'
            f'  "heterologous_genes": ["genes not native to {chassis}"],\n'
            f'  "n_steps": 3,\n'
            f'  "literature_notes": "Key references or notes on this pathway",\n'
            f'  "unknown": false\n'
            f'}}\n'
            f'If you cannot identify this compound, set "unknown": true and fill other fields as empty.'
        )

        try:
            raw = self.backend.chat(
                'You are a metabolic engineering expert. Respond ONLY with valid JSON.',
                [{'role': 'user', 'content': prompt}]
            )
            response = raw.text if hasattr(raw, 'text') else str(raw)
            # Extract JSON from response (strip markdown fences if present)
            import re as _re
            match = _re.search(r'\{[\s\S]*\}', response)
            if not match:
                logger.debug("LLM fallback: no JSON found in response")
                return False

            data = json.loads(match.group())

            if data.get('unknown'):
                console.print(f"  [dim]  ⚠ LLM 无法识别化合物 '{product}'[/]")
                design.warnings.append(f"LLM 也无法识别 '{product}' — 请手动指定BiGG ID和酶列表")
                return False

            # Populate pathway fields
            enzymes = [e for e in data.get('pathway_enzymes', []) if e]
            reactions = [r for r in data.get('pathway_reactions', []) if r]
            source = data.get('source_metabolite', '')
            english_name = data.get('english_name', product)

            if not enzymes:
                console.print(f"  [dim]  ⚠ LLM 未返回酶列表[/]")
                return False

            design.pathway.pathway_enzymes = enzymes
            design.pathway.pathway_reactions = reactions
            design.pathway.heterologous_genes = data.get('heterologous_genes', [])
            design.pathway.n_steps = data.get('n_steps', len(enzymes))
            if source:
                design.pathway.source_metabolite = source
            if data.get('literature_notes'):
                design.pathway.literature_evidence.append(data['literature_notes'])

            # Build synthetic metakg_pathway_raw so report template gets data
            design.metakg_pathway_raw = {
                'source': 'llm_inference',
                'product': product,
                'english_name': english_name,
                'compound_class': data.get('compound_class', ''),
                'paths_found': 1,
                'paths': [{
                    'n_steps': design.pathway.n_steps,
                    'steps': [
                        {'reaction': r, 'genes': [enzymes[i]] if i < len(enzymes) else []}
                        for i, r in enumerate(reactions)
                    ] or [{'reaction': 'unknown', 'genes': enzymes}],
                }],
                'notes': data.get('literature_notes', ''),
                'source_metabolite': source,
            }

            console.print(
                f"  [dim]✓ LLM 推断通路: {design.pathway.n_steps} 步 | "
                f"英文名: {english_name} | "
                f"酶: {', '.join(enzymes[:4])}[/]"
            )
            if design.pathway.heterologous_genes:
                console.print(
                    f"  [dim]  外源基因: {', '.join(design.pathway.heterologous_genes[:4])}[/]"
                )
            return True

        except json.JSONDecodeError as e:
            logger.debug("LLM fallback: JSON parse error: %s", e)
        except Exception as e:
            logger.debug("LLM pathway fallback failed: %s", e)
        return False

    # ── Stage 3: Flux analysis ─────────────────────────────────────────────────

    def _run_flux_analysis(self, design: BioDesignResult, console) -> None:
        """FBA + cofactor + bottleneck analysis."""
        chassis = design.target.chassis
        known = KNOWN_PATHWAYS.get(design.target.product.lower(), {})
        exchange_rxn = known.get("exchange", f"EX_{design.target.bigg_metabolite}")

        # FBA
        from zhuangzhou.skills.metakg import metakg_flux
        fba = metakg_flux(
            objective_reaction=exchange_rxn,
            organism=chassis,
            carbon_source="glc__D_e",
        )
        if "error" not in fba:
            design.flux.theoretical_yield_mmol = fba.get("objective_flux", 0.0)
            design.flux.growth_rate = fba.get("growth_rate", 0.0)
            design.flux.shadow_prices = fba.get("top_shadow_prices", {})
            design.flux.fba_result_raw = fba  # ← save raw dict for report
            console.print(
                f"  [dim]✓ FBA: 理论产量 {design.flux.theoretical_yield_mmol:.3f} mmol/gDW/h, "
                f"生长率 {design.flux.growth_rate:.3f} h⁻¹[/]"
            )
        else:
            design.warnings.append(f"FBA 不可用: {fba.get('error','')} — {fba.get('hint','')}")
            console.print(f"  [dim]⚠ FBA: {fba.get('hint', 'COBRApy 未安装')}[/]")

        # Bottleneck
        if design.pathway.pathway_reactions:
            from zhuangzhou.skills.metakg import metakg_bottleneck
            bn = metakg_bottleneck(design.pathway.pathway_reactions, organism=chassis)
            if "error" not in bn:
                design.flux.bottleneck_reactions = [
                    r["reaction"] for r in bn.get("bottleneck_reactions", [])
                ]
                design.flux.bottleneck_result_raw = bn  # ← save raw dict for report
                if design.flux.bottleneck_reactions:
                    console.print(
                        f"  [dim]✓ 瓶颈: {', '.join(design.flux.bottleneck_reactions[:3])}[/]"
                    )

        # Cofactor balance
        if design.pathway.pathway_reactions:
            from zhuangzhou.skills.metakg import metakg_cofactor
            cof = metakg_cofactor(design.pathway.pathway_reactions, organism=chassis)
            issues = [
                f"{v['label']}: {v['status']}"
                for k, v in cof.get("cofactor_summary", {}).items()
                if v["status"] != "balanced"
            ]
            design.flux.cofactor_issues = issues
            if issues:
                console.print(f"  [dim]⚠ 辅因子失衡: {'; '.join(issues[:2])}[/]")
            else:
                console.print(f"  [dim]✓ 辅因子平衡良好[/]")

    # ── Stage 4: Interference removal ─────────────────────────────────────────

    def _run_interference_removal(self, design: BioDesignResult, console) -> None:
        """Identify competing pathways, toxic intermediates, knockout targets."""
        chassis = design.target.chassis
        met_id = design.target.bigg_metabolite

        # Competing pathways
        if met_id:
            from zhuangzhou.skills.metakg import metakg_competitor
            comp = metakg_competitor(met_id, organism=chassis)
            if "error" not in comp:
                design.interference.competing_reactions = comp.get("competing_reactions", [])
                design.interference.knockout_candidates = comp.get("knockout_candidates", [])[:8]
                n_comp = len(design.interference.competing_reactions)
                console.print(
                    f"  [dim]✓ 竞争通路: {n_comp} 条 | "
                    f"敲除候选: {', '.join(design.interference.knockout_candidates[:4])}[/]"
                )

        # Toxicity screen
        all_mets = []
        if design.pathway.pathway_reactions:
            # Use intermediate metabolites from pathway
            for rxn_id in design.pathway.pathway_reactions[:6]:
                all_mets.append(rxn_id.lower() + "_c")  # crude approximation

        if all_mets:
            from zhuangzhou.skills.metakg import metakg_toxicity
            tox = metakg_toxicity(all_mets, organism=chassis)
            high_risk = tox.get("high_risk_metabolites", [])
            design.interference.toxic_intermediates = high_risk
            if high_risk:
                console.print(f"  [dim]⚠ 高毒性中间体: {', '.join(high_risk)}[/]")
                mits = [r.get("mitigations", []) for r in tox.get("results", [])
                        if r.get("risk_level") == "HIGH"]
                design.interference.mitigation_strategies = [
                    m for sublist in mits[:3] for m in sublist[:2]
                ]
            else:
                console.print(f"  [dim]✓ 毒性筛查: 未检测到高风险中间体[/]")

    # ── Stage 5: Parts design ──────────────────────────────────────────────────

    def _run_parts_design(self, design: BioDesignResult, console) -> None:
        """Promoter/RBS selection, codon optimization, burden assessment."""
        chassis = design.target.chassis
        enzymes = design.pathway.pathway_enzymes

        if enzymes:
            # Find bottleneck indices
            bn_rxns = set(design.flux.bottleneck_reactions)
            bn_indices = [
                i for i, rxn in enumerate(design.pathway.pathway_reactions)
                if rxn in bn_rxns
            ]

            from zhuangzhou.skills.synbio import synbio_expression_balance
            expr = synbio_expression_balance(
                enzyme_names=enzymes,
                organism=chassis,
                bottleneck_indices=bn_indices if bn_indices else None,
            )
            design.parts.expression_plan = expr.get("recommendations", [])

            # CRISPR designs for top knockouts
            if design.interference.knockout_candidates:
                for gene in design.interference.knockout_candidates[:3]:
                    from zhuangzhou.skills.synbio import synbio_crispr_design
                    cr = synbio_crispr_design(
                        target_gene=gene,
                        organism=chassis,
                        edit_type="knockout",
                    )
                    design.parts.crispr_designs.append(cr)
                console.print(
                    f"  [dim]✓ CRISPR 设计: {len(design.parts.crispr_designs)} 个敲除方案[/]"
                )

            # Burden assessment
            from zhuangzhou.skills.synbio import synbio_burden_assess
            levels = [r.get("target_flux", 0.5) for r in design.parts.expression_plan]
            level_labels = ["high" if l > 0.7 else "medium" if l > 0.3 else "low"
                            for l in levels]
            burden = synbio_burden_assess(
                heterologous_genes=enzymes,
                expression_levels=level_labels,
                organism=chassis,
            )
            design.parts.burden_score = burden.get("estimated_growth_reduction_pct", 0.0)
            design.parts.burden_risk  = burden.get("risk_level", "UNKNOWN")
            design.parts.burden_result_raw = burden  # ← save raw dict for report
            console.print(
                f"  [dim]✓ 代谢负担: {design.parts.burden_score:.1f}% 生长抑制 "
                f"[{design.parts.burden_risk}][/]"
            )

            # ── Protein candidate retrieval (for report enrichment) ────────
            # Strategy:
            #   1. Try get_protein_profile(identifier) — works when enzyme_name IS a UniProt accession
            #   2. Fallback: search_ncbi_protein(gene_name + product) → get_ncbi_protein_fasta(accession)
            #   3. Last resort: LLM inference of key protein info
            console.print("  [dim]  → 检索候选酶蛋白信息 (UniProt/NCBI)...[/]")
            product_name = design.target.product
            try:
                from zhuangzhou.skills.biodbapi import (
                    get_protein_profile, search_ncbi_protein, get_ncbi_protein_fasta,
                )
                for enzyme_name in enzymes[:6]:  # cap at 6 to avoid rate limiting
                    cand = self._retrieve_protein_candidate(
                        enzyme_name, product_name, chassis,
                        get_protein_profile, search_ncbi_protein, get_ncbi_protein_fasta,
                    )
                    design.parts.protein_candidates.append(cand)

                n_found = sum(1 for c in design.parts.protein_candidates if c.get('sequence'))
                console.print(
                    f"  [dim]  → 蛋白检索: {n_found}/{len(enzymes[:6])} 个找到序列[/]"
                )
            except Exception as _pe:
                logger.debug("Protein candidate retrieval failed: %s", _pe)
                # Still add placeholders so report sections exist
                for enzyme_name in enzymes[:6]:
                    if not any(c.get('gene_name') == enzyme_name
                               for c in design.parts.protein_candidates):
                        design.parts.protein_candidates.append({
                            'gene_name': enzyme_name, 'uniprot_id': '',
                            'name': enzyme_name, 'organism': chassis,
                            'sequence': '', 'sequence_length': 0, 'ec_numbers': [],
                        })

            # ── Codon optimization for enzymes with sequences ──────────────
            if design.parts.protein_candidates:
                console.print("  [dim]  → 密码子优化...[/]")
                try:
                    from zhuangzhou.skills.synbio import synbio_codon_optimize
                    for cand in design.parts.protein_candidates:
                        seq = cand.get('sequence', '')
                        if not seq:
                            continue
                        cod = synbio_codon_optimize(
                            protein_sequence=seq,
                            organism=chassis,
                        )
                        if not cod.get('error'):
                            design.parts.codon_sequences[cand['gene_name']] = cod
                    design.parts.codon_optimized = bool(design.parts.codon_sequences)
                    console.print(
                        f"  [dim]  → 密码子优化: {len(design.parts.codon_sequences)} 个基因完成[/]"
                    )
                except Exception as _ce:
                    logger.debug("Codon optimization failed: %s", _ce)
        else:
            design.warnings.append("未找到通路酶，跳过元件设计")
            console.print("  [dim]⚠ 未找到通路酶，跳过元件设计[/]")

    def _retrieve_protein_candidate(
        self, enzyme_name: str, product_name: str, chassis: str,
        get_protein_profile, search_ncbi_protein, get_ncbi_protein_fasta,
    ) -> dict:
        """Multi-strategy protein retrieval for a given enzyme gene name.

        Strategy order:
          1. UniProt accession lookup (works when enzyme_name IS a valid accession)
          2. NCBI Protein search by "gene_name + product_name" → fetch FASTA
          3. NCBI Protein search by gene_name alone → fetch FASTA
          4. Minimal placeholder (enzyme still shows in report table)
        """
        # Strategy 1: try as UniProt accession (e.g. Q3IWM5, P00924)
        profile = get_protein_profile(identifier=enzyme_name)
        if not profile.get('error') and profile.get('sequence'):
            return {
                'gene_name': enzyme_name,
                'uniprot_id': profile.get('identifier', enzyme_name),
                'name': profile.get('name', enzyme_name),
                'organism': profile.get('organism', ''),
                'sequence': profile.get('sequence', ''),
                'sequence_length': profile.get('length', 0),
                'ec_numbers': profile.get('ec_numbers', []),
                'function': profile.get('function', ''),
            }

        # Strategy 2: NCBI search with gene_name + compound context
        for query in [f"{enzyme_name} {product_name}", enzyme_name]:
            try:
                ncbi_result = search_ncbi_protein(query=query, max_results=3)
                hits = ncbi_result.get('results', [])
                if hits:
                    acc = hits[0].get('accession', '')
                    title = hits[0].get('title', enzyme_name)
                    org = hits[0].get('organism', chassis)
                    length = hits[0].get('length', 0)
                    seq = ''
                    if acc:
                        fasta = get_ncbi_protein_fasta(accession=acc)
                        seq = fasta.get('sequence', '')
                        if not fasta.get('error') and seq:
                            logger.debug(
                                "Protein '%s' found via NCBI (query=%r acc=%s len=%d)",
                                enzyme_name, query, acc, len(seq),
                            )
                    return {
                        'gene_name': enzyme_name,
                        'uniprot_id': acc,
                        'name': title[:80] if title else enzyme_name,
                        'organism': org,
                        'sequence': seq,
                        'sequence_length': len(seq) if seq else length,
                        'ec_numbers': [],
                        'function': title if title else '',
                    }
            except Exception as e:
                logger.debug("NCBI search failed for '%s': %s", enzyme_name, e)

        # Strategy 4: placeholder — enzyme still appears in all report tables
        return {
            'gene_name': enzyme_name,
            'uniprot_id': '',
            'name': enzyme_name,
            'organism': chassis,
            'sequence': '',
            'sequence_length': 0,
            'ec_numbers': [],
            'function': f'Enzyme involved in {product_name} biosynthesis',
        }

    # ── Stage 6: Design package ────────────────────────────────────────────────

    def _run_design_package(self, design: BioDesignResult, console) -> None:
        """Generate JSON design package + bilingual Word report."""
        from zhuangzhou.skills.synbio import synbio_design_package

        output_dir = os.path.join(
            os.getcwd(),
            f"{design.target.product.replace(' ','_')}_{design.target.chassis}_design"
        )

        # ── Assemble all enrichment data accumulated across stages ──────────
        # FBA results dict (with display-friendly fields)
        fba_results = dict(design.flux.fba_result_raw) if design.flux.fba_result_raw else {}
        if fba_results and design.flux.theoretical_yield_mmol:
            fba_results.setdefault('objective_value', design.flux.theoretical_yield_mmol)
            fba_results.setdefault('flux', design.flux.theoretical_yield_mmol)

        # Burden results dict
        burden_results = dict(design.parts.burden_result_raw) if design.parts.burden_result_raw else {}
        if not burden_results and design.parts.burden_score:
            burden_results = {
                'total_burden_score': design.parts.burden_score,
                'risk_level': design.parts.burden_risk,
                'estimated_growth_reduction_pct': design.parts.burden_score,
                'recommendations': [],
            }

        pkg = synbio_design_package(
            target_product=design.target.product,
            chassis=design.target.chassis,
            pathway_enzymes=design.pathway.pathway_enzymes or [],
            pathway_reactions=design.pathway.pathway_reactions or [],
            knockout_targets=design.interference.knockout_candidates or [],
            expression_recommendations=design.parts.expression_plan or [],
            output_dir=output_dir,
            # ── Enrichment data (new — previously all dropped) ──
            protein_candidates=design.parts.protein_candidates or [],
            codon_sequences=design.parts.codon_sequences or {},
            fba_results=fba_results,
            bottleneck_results=design.flux.bottleneck_result_raw or {},
            burden_results=burden_results,
            metakg_pathway=design.metakg_pathway_raw or {},
        )
        if "error" not in pkg:
            design.design_package_path = pkg.get("output_dir", "")
            word_en = pkg.get("word_report", "")
            word_zh = pkg.get("word_report_zh", "")
            console.print(
                f"  [dim]✓ 设计包已生成: {design.design_package_path}[/]"
            )
            if word_en:
                console.print(f"  [dim]✓ Word报告 (EN): {word_en}[/]")
            if word_zh:
                console.print(f"  [dim]✓ Word报告 (ZH): {word_zh}[/]")
        else:
            design.errors.append(pkg.get("error", ""))

    # ── Summary display ────────────────────────────────────────────────────────

    def _print_summary(self, design: BioDesignResult, console) -> None:
        from rich.table import Table
        from rich.panel import Panel
        from rich.text import Text
        from zhuangzhou.ui.core import ZZTheme

        console.print("")
        table = Table(title=f"Bio-design 完成: {design.target.product} @ {design.target.chassis}",
                      border_style=ZZTheme.MID)
        table.add_column("阶段", style=f"bold {ZZTheme.CYAN}", width=20)
        table.add_column("结果", style="white", width=50)

        table.add_row("目标产物", design.target.product)
        table.add_row("底盘菌株", design.target.chassis.upper())
        table.add_row(
            "通路长度",
            f"{design.pathway.n_steps} 步" if design.pathway.n_steps else "待确认"
        )
        table.add_row(
            "关键酶",
            ", ".join(design.pathway.pathway_enzymes[:4]) or "待确认"
        )
        if design.flux.theoretical_yield_mmol > 0:
            table.add_row(
                "理论产量",
                f"{design.flux.theoretical_yield_mmol:.3f} mmol/gDW/h"
            )
        if design.flux.bottleneck_reactions:
            table.add_row("瓶颈", ", ".join(design.flux.bottleneck_reactions[:3]))
        if design.interference.knockout_candidates:
            table.add_row(
                "敲除候选",
                ", ".join(design.interference.knockout_candidates[:4])
            )
        table.add_row(
            "代谢负担",
            f"{design.parts.burden_score:.1f}% [{design.parts.burden_risk}]"
            if design.parts.burden_score else "未评估"
        )
        if design.design_package_path:
            table.add_row("设计包路径", design.design_package_path)

        console.print(table)

        if design.warnings:
            console.print(f"  [bold #FFC107]⚠ 注意事项:[/]")
            for w in design.warnings:
                console.print(f"  [dim]  · {w}[/]")

    # ── Helper: interactive option selection ───────────────────────────────────

    def _select_option(self, question_cfg: Dict, console, ZZTheme) -> Optional[Tuple]:
        """Reuse AlignmentTree._select_option logic."""
        from rich.text import Text
        options = [(o[0], o[1], o[2]) for o in question_cfg["options"]]
        options.append(("skip", "跳过 (默认选第一项)", ""))

        t = Text()
        t.append(f"\n  ▶ {question_cfg['question']}", style=f"bold {ZZTheme.CYAN}")
        if question_cfg.get("hint"):
            t.append(f"\n  {question_cfg['hint']}", style="dim")
        console.print(t)

        # Quick fallback text selection
        for i, (val, label, hint) in enumerate(options):
            letter = chr(ord("a") + i)
            hint_str = f"  [{hint}]" if hint else ""
            console.print(f"  [dim]{letter})[/dim] {label}{hint_str}")
        console.print("[dim]  输入字母选择 (直接 Enter = 选a): [/dim]", end="")
        try:
            choice = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None

        if not choice:
            return options[0]
        for i, opt in enumerate(options):
            if choice == chr(ord("a") + i) or choice == opt[0][:3].lower():
                if opt[0] == "skip":
                    return options[0]
                return opt
        return options[0]

    def _extract_product(self, query: str) -> str:
        """Heuristic: extract product molecule name from query."""
        import re
        q = query.lower()

        # Exact match in known pathways
        for name in KNOWN_PATHWAYS:
            if name in q:
                return name

        # Chinese -> English mapping (common metabolic engineering targets)
        cn_map = {
            # amino acids
            "酪氨酸": "tyrosine", "色氨酸": "tryptophan", "苯丙氨酸": "phenylalanine",
            "谷氨酸": "glutamate", "赖氨酸": "lysine", "苏氨酸": "threonine",
            "天冬氨酸": "aspartate", "丙氨酸": "alanine",
            # organic acids
            "琥珀酸": "succinate", "苹果酸": "malate", "柠檬酸": "citrate",
            "乳酸": "lactate", "丙酮酸": "pyruvate", "衣康酸": "itaconate",
            "富马酸": "fumarate", "葡萄糖酸": "gluconate",
            # terpenoids / polyketides
            "莽草酸": "shikimate", "香草醛": "vanillin", "青蒿素": "artemisinin",
            "紫杉醇": "taxol", "角鲨烯": "squalene", "法尼醇": "farnesol",
            "番茄红素": "lycopene", "β-胡萝卜素": "beta-carotene",
            # natural products / others
            "对香豆酸": "p-coumaric acid", "白藜芦醇": "resveratrol",
            "柚皮素": "naringenin", "根皮素": "phloretin",
            "奥巴林": "aureobasidin A",  # cyclic depsipeptide antifungal
        }
        for cn, en in cn_map.items():
            if cn in query:
                return en

        # English patterns: "produce X", "X production", "X biosynthesis"
        m = re.search(r"(?:produce|synthesis of|biosynthesis of|生产|合成)\s+([\w\-]+)", q)
        if m:
            return m.group(1)

        return ""

    def _prompt_input(self, prompt: str, console) -> str:
        console.print(f"  [dim]{prompt}[/dim]", end="")
        try:
            return input().strip()
        except (EOFError, KeyboardInterrupt):
            return ""


# ── Slash command integration ──────────────────────────────────────────────────

def get_biodesign_mode(backend, registry) -> BioDesignMode:
    return BioDesignMode(backend=backend, registry=registry)

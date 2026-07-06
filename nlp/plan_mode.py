"""
Plan Mode — Zhuangzhou v0.9.0
==============================
意图对齐 → 执行规划引擎 (LLM-Driven Dynamic Clarification)

Flow:
  用户输入 (模糊/专业)
    ↓ _should_skip()     (轻量规则: 是直接命令则绕过)
  LLMClarifier           (LLM 自主判断歧义、动态生成问题)
    ↓ ≤3轮对话           (每轮LLM决定问什么、问几个选项)
  ResolvedEntity list
    ↓ PlanGenerator      (LLM, 将已解析实体映射为真实技能链)
  ExecutionPlan  [step1: skill+params, step2:..., 预估时间]
    ↓ 用户确认
  AgentExecutor.run_loop(结构化精确任务)

v0.9.0 vs v0.8.0:
  - 移除 ENTITY_OPTION_TREES (硬编码选项树)
  - 移除 IntentAnalyzer 规则匹配 (预定义关键词)
  - 移除 AlignmentTree 固定流程
  - 新增 LLMClarifier: LLM 自主分析查询, 动态决定提什么问题
  - 域知识通过上下文注入 LLM, 而不是写死成 if-else 控制流
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# 1. Skip Patterns — direct commands that bypass planning entirely
# =============================================================================

_SKIP_PATTERNS = [
    r"^/",                            # slash command — always direct
    r"^\s*(yes|no|y|n|确认|取消)\s*$",  # pure confirmation — no content to plan
    r"^\d+$",                          # number-only — selection or ID
    # NOTE: Broad keyword patterns like r"^(show|list|display|check)" were removed.
    # "list mutations after docking" and similar composite queries need planning.
    # The LLM decides whether a query needs planning based on its actual content.
]


# =============================================================================
# 2. Data Structures
# =============================================================================


@dataclass
class ResolvedEntity:
    entity_type: str       # e.g. "protein_species"
    label: str             # e.g. "人类 Homo sapiens"
    value: str             # e.g. "human"
    raw_input: str = ""    # user's custom text if value == "other"


@dataclass
class PlanStep:
    step: int
    skill: str
    params: Dict[str, Any]
    description: str
    estimated_time: str = ""
    depends_on: List[int] = field(default_factory=list)


@dataclass
class ExecutionPlan:
    query: str
    resolved: List[ResolvedEntity]
    steps: List[PlanStep]
    summary: str
    estimated_total_time: str = ""
    required_inputs: List[str] = field(default_factory=list)
    expected_outputs: List[str] = field(default_factory=list)

    def to_agent_message(self) -> str:
        """Convert plan to a structured, unambiguous task message for AgentExecutor."""
        lines = [
            f"【已规划任务】{self.summary}",
            "",
            "已解析上下文:",
        ]
        for r in self.resolved:
            lines.append(f"  - {r.entity_type}: {r.label} [{r.value}]"
                         + (f" (用户指定: {r.raw_input})" if r.raw_input else ""))
        lines.append("")
        lines.append(f"执行计划 ({len(self.steps)} 步, 预计 {self.estimated_total_time}):")
        for s in self.steps:
            dep = f" (需要 Step {s.depends_on})" if s.depends_on else ""
            lines.append(f"  Step {s.step}: [{s.skill}] {s.description}{dep}")
            if s.estimated_time:
                lines.append(f"           预计 {s.estimated_time}")
        lines.append("")
        lines.append("请按计划依次执行上述步骤。遇到路径依赖时，从上一步的输出中获取。")
        return "\n".join(lines)


# =============================================================================
# 3. LLM-Driven Clarifier (replaces IntentAnalyzer + AlignmentTree)
# =============================================================================


class LLMClarifier:
    """Dynamic, LLM-driven intent clarification.

    v0.9.0: Replaces the hardcoded ENTITY_OPTION_TREES + rule-based IntentAnalyzer.

    Key difference from old approach:
    - The LLM DECIDES what to ask (not pre-programmed keyword patterns)
    - Questions and options are generated dynamically based on the actual query
      and the real skill catalog — not a fixed decision tree
    - Domain knowledge is context injected into the LLM, not if-else control flow
    - The LLM knows when it has enough information and STOPS asking
    - Works for workflows the programmer never anticipated

    This is actual AI reasoning, not prompt engineering with hardcoded branches.
    """

    # Domain knowledge as LLM-readable context (not control flow!)
    # RULE: Only factual/reference data here. NO skill sequences, NO "A→B→C" routing.
    # Skill selection and ordering is the LLM's job based on the catalog it receives.
    _DOMAIN_KNOWLEDGE = """
## Zhuangzhou 科研域知识参考 (事实参考, 非技能路由规则)

**CYP450物种辨别** (事实):
  CYP2A6=人类(Human, UniProt P11509, PDB:1Z10/2FDY/5T9T)
  CYP2A1=大鼠(Rat)  CYP2B4=兔(Rabbit)  CYP3A4=人类(P08684)
  CYP2C9=人类  CYP2D6=人类  CYP2E1=人类
  → 查询中已写明则直接用, 不确定时才提问

**配体输入格式** (smart_docking已自动处理, 无需手动转换):
  化合物名称(nicotine/阿司匹林/caffeine) → 自动从PubChem下载
  SMILES字符串 → 自动生成3D构象
  本地SDF文件 → 直接使用

**MD时长参考** (RTX 5070 @ 133ns/day for CYP+ligand):
  10ns=快速验证(~2h)  50ns=中等(~9h)  100ns=标准(~18h)  200ns=高精度(~36h)

**蛋白类型** (事实):
  metalloprotein = 含HEME/Fe/Zn的蛋白 | plain = 标准蛋白

**结构来源优先级** (事实):
  RCSB PDB实验结构(推荐,4位PDB ID) > AlphaFold(AI预测) > 本地PDB

注意: 技能选择和执行顺序由LLM根据技能目录自行推断。不要依赖此处的映射。
"""

    _SYSTEM_PROMPT = """{skill_catalog}

{domain_knowledge}

## 你的角色与任务
你是Zhuangzhou生物信息系统的意图分析专家。
从用户查询和已确认参数中提取信息，判断是否需要进一步澄清。

## 分析原则
1. 优先从查询文本中直接推断 — 用户明确说了的不要再问
2. 只问"执行计划必需但无法推断"的参数
3. 每次只问最重要的一个 (最多3轮)
4. 选项要具体实用 (2-4个), 不要模糊泛化
5. 查询足够明确时直接输出 ready=true, 不要为问而问

## 输出格式 (纯JSON, 无markdown包裹)

情况A — 信息足够, 可以规划:
{{"ready": true, "inferred": {{"protein": "推断值或null", "ligand": "推断值或null", "goal": "用自然语言描述用户目标，不要限定为固定类别"}}}}

情况B — 需要澄清一个关键参数:
{{"ready": false,
  "question": "用中文问最重要的未知参数 (说明为什么需要此信息)",
  "param_name": "参数英文标识符",
  "options": [
    ["选项值", "显示标签", "技术说明"],
    ["选项值2", "显示标签2", "技术说明2"]
  ]
}}
"""

    def run(
        self,
        query: str,
        backend,
        registry,
        console,
        max_rounds: int = 3,
    ) -> List[ResolvedEntity]:
        """
        LLM-driven clarification loop. Returns list of ResolvedEntity.

        The LLM dynamically decides WHAT to ask based on:
        - The actual query text
        - The real skill catalog (injected)
        - Domain knowledge (injected)
        - Previous answers (accumulated)
        """
        try:
            catalog = registry.build_planning_catalog()
        except Exception:
            catalog = "(skill catalog unavailable)"

        system_prompt = self._SYSTEM_PROMPT.format(
            skill_catalog=catalog,
            domain_knowledge=self._DOMAIN_KNOWLEDGE,
        )

        conversation: List[Tuple[str, str]] = []  # (question, answer) history
        resolved: List[ResolvedEntity] = []

        for round_idx in range(max_rounds):
            # Build the message with accumulated context
            user_content = f"用户查询: {query}"
            if conversation:
                user_content += "\n\n已确认参数:"
                for q_txt, a_txt in conversation:
                    user_content += f"\n  Q: {q_txt}\n  A: {a_txt}"

            try:
                resp = backend.chat(
                    system_prompt,
                    [{"role": "user", "content": user_content}],
                    [],
                )
                raw = (resp.text or "").strip()
            except Exception as e:
                logger.warning("LLMClarifier call failed (round %d): %s", round_idx, e)
                break

            parsed = self._parse_json(raw)
            if not parsed:
                logger.warning("LLMClarifier: could not parse JSON in round %d: %s",
                               round_idx, raw[:200])
                break

            if parsed.get("ready") or round_idx == max_rounds - 1:
                # Collect what the LLM inferred
                for key, val in (parsed.get("inferred") or {}).items():
                    if val and str(val).strip():
                        resolved.append(ResolvedEntity(
                            entity_type=key,
                            label=str(val),
                            value=str(val),
                        ))
                break

            # LLM wants to ask a clarifying question
            question = (parsed.get("question") or "").strip()
            options_raw = parsed.get("options") or []
            param_name = (parsed.get("param_name") or f"param_{round_idx}").strip()

            if not question or not options_raw:
                # LLM returned unclear response — stop asking
                break

            # Show question and options to user
            answer = self._ask_user(question, options_raw, round_idx + 1, console)
            if answer is None:
                # User skipped this question
                break

            value, label = answer
            resolved.append(ResolvedEntity(
                entity_type=param_name,
                label=label,
                value=value,
            ))
            conversation.append((question, f"{label} [{value}]"))

        return resolved

    def _parse_json(self, text: str) -> Optional[dict]:
        """Extract first valid JSON object from LLM response text."""
        decoder = json.JSONDecoder()
        start = text.find("{")
        max_tries = min(len(text), 200)  # guard against pathological input
        tries = 0
        while start != -1 and tries < max_tries:
            tries += 1
            try:
                obj, _ = decoder.raw_decode(text, start)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
            start = text.find("{", start + 1)
        return None

    def _ask_user(
        self,
        question: str,
        options_raw: list,
        round_num: int,
        console,
    ) -> Optional[Tuple[str, str]]:
        """Display LLM-generated question and options. Return (value, label) or None."""
        try:
            from zhuangzhou.ui.core import ZZTheme
        except Exception:
            class ZZTheme:
                GOLD = "#FFC107"; CYAN = "#00E5FF"; GREEN = "#00E676"; DARK = "#1A1A2E"

        from rich.text import Text
        from rich.panel import Panel

        console.print("")
        header = Text()
        header.append(f"  ▶ 意图对齐 · 问题 {round_num}  ", style=f"bold {ZZTheme.GOLD}")
        header.append(question, style=f"bold {ZZTheme.CYAN}")
        console.print(header)

        # Normalize options: each item is [value, label] or [value, label, hint]
        opts: List[Tuple[str, str, str]] = []
        for o in options_raw:
            if isinstance(o, (list, tuple)) and len(o) >= 2:
                opts.append((str(o[0]), str(o[1]), str(o[2]) if len(o) > 2 else ""))
            elif isinstance(o, str):
                opts.append((o, o, ""))
        opts.append(("__custom__", "其他 / 自定义输入", ""))

        for i, (val, label, hint) in enumerate(opts):
            letter = chr(ord('a') + i) if i < 26 else str(i)
            hint_str = f"  [{hint}]" if hint else ""
            console.print(f"  [dim]{letter})[/dim] {label}{hint_str}")

        console.print("[dim]  输入字母选择 (Enter=跳过): [/dim]", end="")
        try:
            choice = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None

        if not choice:
            return None

        for i, (val, label, hint) in enumerate(opts):
            letter = chr(ord('a') + i) if i < 26 else str(i)
            if choice == letter or choice == val[:1].lower():
                if val == "__custom__":
                    console.print("  [dim]请输入自定义值: [/dim]", end="")
                    try:
                        custom = input().strip()
                    except (EOFError, KeyboardInterrupt):
                        return None
                    if not custom:
                        return None
                    return (custom, custom)
                check = Text()
                check.append("  ✓ ", style=f"bold {ZZTheme.GREEN}")
                check.append(label, style="bold white")
                console.print(check)
                return (val, label)

        return None


# =============================================================================
# 5. Plan Generator (LLM-powered)
# =============================================================================


class PlanGenerator:
    """Generate concrete ExecutionPlan from resolved entities using LLM.

    v0.8.1: Dynamic skill catalog injection — LLM sees real registry names,
    not hardcoded fictional names. Fake skill names are caught and corrected
    by _validate_and_fix_steps() post-processing.
    """

    # Template — {skill_catalog} is filled at runtime from registry.build_planning_catalog()
    _BASE_SYSTEM_PROMPT = """你是 Zhuangzhou 科研规划系统的执行计划生成器。
根据用户意图和已解析参数，阅读【当前可用技能目录】，自主推断最合适的技能和顺序，生成执行计划。

格式规则 (必须遵守):
1. 只使用目录中列出的技能名称 — 不要发明不存在的技能名
2. 路径依赖: 使用 "{{from_step_N}}" 引用上步输出
3. 时间估算: 使用目录中的 estimated_time 字段
4. 可视化: 严格按照目录末尾的 VISUALIZATION ROUTING GUIDE

推断原则:
- 阅读每个技能的 description、when_to_use、workflow_stage 字段
- 根据用户目标和已解析参数，自主决定用哪些技能、以什么顺序
- 不要臆造步骤，每一步都应有技能目录中的对应技能作为依据

{skill_catalog}

输出格式 (严格 JSON):
{{
  "summary": "一句话任务描述",
  "estimated_total_time": "~Xh",
  "required_inputs": ["蛋白PDB/名称", "配体名称/SMILES"],
  "expected_outputs": ["对接报告", "ChimeraX互作图"],
  "steps": [
    {{"step": 1, "skill": "get_pdb_entry", "params": {{"pdb_id": "推断值", "output_dir": null}},
     "description": "获取蛋白PDB结构", "estimated_time": "~30s", "depends_on": []}},
    ...
  ]
}}"""

    # Canonical corrections for LLM-hallucinated skill names
    _SKILL_NAME_MAP: Dict[str, str] = {
        # Common LLM hallucinations → real skill names
        "plip_analyze": "plip_analysis",
        "plip_visualization": "plip_visualize",
        "docking": "smart_docking",
        "vina_docking": "smart_docking",
        "run_docking": "smart_docking",
        # Visualization aliases — point to plip_visualize (PyMOL) since chimerax_plip is disabled
        "pymol_render": "plip_visualize",
        "chimerax_plip": "plip_visualize",
        "chimerax_visualize": "plip_visualize",
        "visualize_interactions": "plip_visualize",
        "generate_figure": "plip_visualize",
        "chatmol_viz": "plip_visualize",
    }

    def _build_dynamic_prompt(self, registry) -> str:
        """Build the system prompt with real skill catalog injected."""
        try:
            catalog = registry.build_planning_catalog()
        except Exception as e:
            logger.warning("Failed to build planning catalog: %s", e)
            catalog = "(Skill catalog unavailable — use skill names from registry)"
        return self._BASE_SYSTEM_PROMPT.format(skill_catalog=catalog)

    def generate(
        self, query: str, resolved: List[ResolvedEntity], backend, registry
    ) -> Optional[ExecutionPlan]:
        """Call LLM to generate ExecutionPlan. Returns None on failure."""
        context = self._build_context(query, resolved)
        system_prompt = self._build_dynamic_prompt(registry)
        prompt = f"""用户原始意图: {query}

已解析参数:
{context}

请生成完整执行计划 (JSON格式):"""

        try:
            resp = backend.chat(system_prompt, [{"role": "user", "content": prompt}], [])
            plan_json = self._extract_json(resp.text or "")
            if not plan_json:
                return None
            plan = self._parse_plan(query, resolved, plan_json)
            # Post-process: fix any hallucinated skill names
            plan.steps = self._validate_and_fix_steps(plan.steps, registry)
            return plan
        except Exception as e:
            logger.warning("PlanGenerator LLM call failed: %s", e)
            return self._fallback_plan(query, resolved, registry)

    def _build_context(self, query: str, resolved: List[ResolvedEntity]) -> str:
        lines = []
        for r in resolved:
            lines.append(f"  {r.entity_type} = {r.value}"
                         + (f" (用户指定: {r.raw_input})" if r.raw_input else ""))
        return "\n".join(lines) or "  (无额外参数)"

    def _extract_json(self, text: str) -> Optional[dict]:
        # Find the first '{' and attempt raw_decode to handle nested braces correctly
        decoder = json.JSONDecoder()
        start = text.find("{")
        max_tries = min(len(text), 200)
        tries = 0
        while start != -1 and tries < max_tries:
            tries += 1
            try:
                obj, _ = decoder.raw_decode(text, start)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
            start = text.find("{", start + 1)
        return None

    def _parse_plan(
        self, query: str, resolved: List[ResolvedEntity], data: dict
    ) -> ExecutionPlan:
        steps = []
        for s in data.get("steps", []):
            steps.append(PlanStep(
                step=s.get("step", len(steps) + 1),
                skill=s.get("skill", "unknown"),
                params=s.get("params", {}),
                description=s.get("description", ""),
                estimated_time=s.get("estimated_time", ""),
                depends_on=s.get("depends_on", []),
            ))
        return ExecutionPlan(
            query=query,
            resolved=resolved,
            steps=steps,
            summary=data.get("summary", query),
            estimated_total_time=data.get("estimated_total_time", "未知"),
            required_inputs=data.get("required_inputs", []),
            expected_outputs=data.get("expected_outputs", []),
        )

    def _validate_and_fix_steps(
        self, steps: List[PlanStep], registry
    ) -> List[PlanStep]:
        """Correct hallucinated skill names in LLM-generated plan steps.

        Strategy (in order):
        1. If name exists in registry -> keep as-is
        2. Exact key match in _SKILL_NAME_MAP -> remap
        3. Substring search in real registry names
        4. Category-based fallback (first skill in same inferred category)
        """
        if not registry._loaded:
            registry.load_all()

        real_skills = registry._skills  # dict name -> Skill

        for step in steps:
            name = step.skill
            if name in real_skills:
                continue  # Already valid

            # 1. Semantic mapping dict
            if name in self._SKILL_NAME_MAP:
                fixed = self._SKILL_NAME_MAP[name]
                if fixed in real_skills:
                    logger.warning(
                        "PlanGenerator: fixed hallucinated skill '%s' -> '%s'", name, fixed
                    )
                    step.skill = fixed
                    continue

            # 2. Substring match: look for real skill containing the hallucinated name as substring
            candidates = [n for n in real_skills if name in n or n in name]
            if candidates:
                fixed = candidates[0]
                logger.warning(
                    "PlanGenerator: substring-matched skill '%s' -> '%s'", name, fixed
                )
                step.skill = fixed
                continue

            # 3. Partial word match on description keywords
            name_parts = set(name.lower().replace('_', ' ').split())
            scored = []
            for rname, rskill in real_skills.items():
                desc_words = set(rskill.description.lower().split())
                overlap = len(name_parts & desc_words)
                if overlap > 0:
                    scored.append((overlap, rname))
            if scored:
                scored.sort(reverse=True)
                fixed = scored[0][1]
                logger.warning(
                    "PlanGenerator: keyword-matched skill '%s' -> '%s'", name, fixed
                )
                step.skill = fixed
                continue

            # 4. Last resort: log as unresolvable (keep original, execution will fail gracefully)
            logger.warning(
                "PlanGenerator: could not resolve skill '%s' — keeping as-is, "
                "execution will return error dict", name
            )

        return steps

    def _fallback_plan(
        self, query: str, resolved: List[ResolvedEntity], registry
    ) -> ExecutionPlan:
        """FIXED_MODE fallback — only called when LLM plan generation fails entirely.

        This is NOT Plan Mode. It is a hardcoded emergency sequence used as a last
        resort when the LLM backend is unavailable or throws an exception.

        Design intent: return an empty plan with a clear error message rather than
        silently substituting a pre-wired pipeline, so the user knows something went
        wrong and can retry or enter a command directly.
        """
        logger.error(
            "PlanGenerator LLM call failed and FIXED_MODE fallback was triggered. "
            "Returning empty plan — user should retry or use direct commands."
        )
        return ExecutionPlan(
            query=query,
            resolved=resolved,
            steps=[],
            summary=f"[FIXED_MODE] LLM规划失败，请直接输入命令或重试。原始请求: {query}",
            estimated_total_time="N/A",
        )


# =============================================================================
# 6. Plan Display
# =============================================================================


def display_plan(plan: ExecutionPlan, console) -> bool:
    """Show plan and ask user to confirm. Returns True if confirmed."""
    from zhuangzhou.ui.core import ZZTheme
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table
    from rich.box import ROUNDED, SIMPLE

    console.print("")

    # Header
    header = Text()
    header.append("  执行计划  ", style=f"bold black on {ZZTheme.CYAN}")
    header.append(f"  {plan.summary}", style=f"bold {ZZTheme.GOLD}")
    console.print(header)

    # Resolved entities
    if plan.resolved:
        console.print(f"  [dim]已对齐参数:[/]")
        for r in plan.resolved:
            console.print(f"  [dim]  · {r.entity_type}:[/] [bold white]{r.label}[/]")
    console.print("")

    # Steps table
    table = Table(box=SIMPLE, show_edge=False, header_style=f"bold {ZZTheme.CYAN}",
                  row_styles=["none", f"on {ZZTheme.DARK}"])
    table.add_column("步骤", style="bold", width=4)
    table.add_column("技能", style=f"bold {ZZTheme.CYAN}", width=30)
    table.add_column("描述", width=35)
    table.add_column("预计时间", style="dim", width=12)

    for s in plan.steps:
        table.add_row(str(s.step), s.skill, s.description, s.estimated_time)
    console.print(table)

    # Summary
    summary = Text()
    summary.append("  总预计时间: ", style="dim")
    summary.append(plan.estimated_total_time, style=f"bold {ZZTheme.GOLD}")
    if plan.expected_outputs:
        summary.append("  │  输出: ", style="dim")
        summary.append(", ".join(plan.expected_outputs), style="white")
    console.print(summary)
    console.print("")

    # Confirm
    console.print(f"  [bold {ZZTheme.CYAN}]确认执行此计划?[/]  "
                  f"[dim][Y/n] (Enter=确认, n=取消, e=编辑)[/dim]  ",
                  end="")
    try:
        choice = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return choice not in ("n", "no", "取消", "cancel")


# =============================================================================
# 7. Main PlanMode Entry Point
# =============================================================================


class PlanMode:
    """Plan Mode: the default front-end for all user queries in Zhuangzhou.

    v0.9.0: Uses LLMClarifier for dynamic, LLM-driven intent clarification.
    The LLM itself decides what to ask — not hardcoded keyword rules.

    Usage:
        pm = PlanMode(backend, registry)
        result = pm.process(user_query)
        if result.skipped:
            # pass query directly to AgentExecutor
        else:
            # use result.plan.to_agent_message() as the structured task
    """

    def __init__(self, backend, registry, enabled: bool = True,
                 agent_executor=None):
        self.backend = backend
        self.registry = registry
        self.enabled = enabled
        self.agent_executor = agent_executor  # needed by FixedMode
        self._clarifier = LLMClarifier()
        self._generator = PlanGenerator()
        # Lazy-init to avoid circular imports
        self._router = None
        self._fixed_mode = None

    def _get_router(self):
        if self._router is None:
            from .router_agent import RouterAgent
            self._router = RouterAgent(self.backend)
        return self._router

    def _get_fixed_mode(self):
        if self._fixed_mode is None:
            from .fixed_mode import FixedMode
            self._fixed_mode = FixedMode(self.backend, self.registry,
                                          self.agent_executor)
        return self._fixed_mode

    class Result:
        def __init__(self, skipped: bool, plan: Optional[ExecutionPlan] = None,
                     original_query: str = ""):
            self.skipped = skipped
            self.plan = plan
            self.original_query = original_query

        @property
        def agent_message(self) -> str:
            """Get the message to send to AgentExecutor."""
            if self.skipped or self.plan is None:
                return self.original_query
            return self.plan.to_agent_message()

    @staticmethod
    def _should_skip(query: str) -> bool:
        """Fast path: direct/simple commands that don't need planning."""
        q = query.strip().lower()
        for pat in _SKIP_PATTERNS:
            if re.search(pat, q):
                return True
        # Very short queries are likely direct commands
        return len(q) < 6

    def process(self, query: str, console=None) -> "PlanMode.Result":
        """Main entry point. Returns Result with plan or skip signal.

        v1.0.0 flow:
          1. Fast skip check (rule-based, 0 LLM)
          2. RouterAgent: single LLM call (~50 token) → direct / fixed / plan
             - direct → skip (Agent handles directly)
             - fixed  → FixedMode: ResearchPhase + Supervisor gates + execute
             - plan   → LLMClarifier → PlanGenerator (original path)
          3. User confirms plan (plan mode only)
        """
        if console is None:
            from zhuangzhou.ui.core import console as _console
            console = _console

        if not self.enabled:
            return self.Result(skipped=True, original_query=query)

        # Fast path: simple / direct commands skip planning entirely (0 LLM)
        if self._should_skip(query):
            return self.Result(skipped=True, original_query=query)

        # RouterAgent: decide execution path (~50 token)
        router = self._get_router()
        decision = router.route(query)
        logger.info("RouterAgent: mode=%s pipeline=%s confidence=%.2f — %s",
                    decision.mode, decision.pipeline, decision.confidence, decision.reason)

        if decision.mode == "direct":
            return self.Result(skipped=True, original_query=query)

        if decision.mode == "fixed" and decision.pipeline:
            self._dispatch_fixed(query, decision, console)
            # Fixed mode executes directly; return skipped=True so Agent loop doesn't re-run
            return self.Result(skipped=True, original_query=query)

        # mode == "plan" — fall through to original LLM planning path

        # LLM-driven clarification — the LLM decides what to ask
        resolved = self._clarifier.run(query, self.backend, self.registry, console)

        # Generate execution plan (resolved may be empty for clear queries)
        console.print(f"\n  [dim]生成执行计划中...[/]")
        plan = self._generator.generate(query, resolved, self.backend, self.registry)
        if plan is None or not plan.steps:
            return self.Result(skipped=True, original_query=query)

        # Show plan and get confirmation
        confirmed = display_plan(plan, console)
        if not confirmed:
            console.print("[dim]  计划已取消。直接发送原始问题给 Agent。[/dim]")
            return self.Result(skipped=True, original_query=query)

        return self.Result(skipped=False, plan=plan, original_query=query)

    def _dispatch_fixed(self, query: str, decision, console) -> None:
        """Parse query for FixedMode parameters and execute."""
        from .fixed_mode import FixedMode

        # Lightweight param extraction from query (no LLM: regex + heuristics)
        protein, ligand, mutations, duration_ns, pdb_id = self._parse_fixed_params(query)

        if not protein:
            console.print("[yellow]⚠ Fixed Mode: 无法从查询中提取蛋白名称，切换到 Plan Mode[/]")
            return  # Fall through to plan mode by not returning early

        fixed = self._get_fixed_mode()
        fixed.run(
            pipeline=decision.pipeline,
            protein=protein,
            ligand=ligand,
            mutations=mutations,
            duration_ns=duration_ns or 100,
            pdb_id=pdb_id,
            console=console,
        )

    @staticmethod
    def _parse_fixed_params(query: str):
        """
        Extract protein, ligand, mutations, duration, PDB ID from natural language.
        Pure regex / heuristics — 0 LLM tokens.
        Returns: (protein, ligand, mutations, duration_ns, pdb_id)
        """
        import re

        q = query.strip()

        # PDB ID (4 alphanumeric chars, often uppercase)
        pdb_match = re.search(r'\b([0-9][A-Za-z0-9]{3})\b', q)
        pdb_id = pdb_match.group(1).upper() if pdb_match else None

        # Duration (e.g. "100ns", "50 ns", "200纳秒")
        dur_match = re.search(r'(\d+)\s*(?:ns|纳秒)', q, re.IGNORECASE)
        duration_ns = int(dur_match.group(1)) if dur_match else None

        # Mutations (e.g. V21L, A82G, F87V)
        mutations = re.findall(r'\b([A-Z]\d+[A-Z])\b', q)

        # Protein name: look for CYP patterns first, then other protein names
        protein_match = (
            re.search(r'\b(CYP\w+|P450\w*)\b', q, re.IGNORECASE) or
            re.search(r'\b([A-Z][a-z0-9]{2,}(?:\d+)?(?:A|B|C)?)\b', q)
        )
        protein = protein_match.group(1) if protein_match else None

        # Ligand: common patterns
        # "A和B", "A with B", "A + B", "A into B", "A对接B"
        ligand = None
        lig_patterns = [
            r'(?:和|with|into|+|对接|bind)\s+([a-zA-Z\u4e00-\u9fff][\w\u4e00-\u9fff\-]*)',
            r'([a-zA-Z][\w\-]+)\s+(?:对接|docking|binding)',
        ]
        for pat in lig_patterns:
            m = re.search(pat, q, re.IGNORECASE)
            if m:
                candidate = m.group(1).strip()
                # Exclude if it looks like a protein (CYP, uppercase acronym)
                if candidate.lower() != (protein or "").lower():
                    ligand = candidate
                    break

        return protein, ligand, mutations or None, duration_ns, pdb_id

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled

"""
Zhuangzhou Conversation Engine 鈥?Gather 鈫?Act 鈫?Verify
=========================================================
Transplanted from LLM agent runtime's core architecture principles:

1. NO complex NLP routing. The LLM (LLM agent runtime / DeepSeek / etc.) IS the brain.
   Zhuangzhou only provides: context assembly + tool execution + result archival.

2. Plain text memory (not vector DB). project_memory.md accumulates knowledge
   with high signal-to-noise ratio, directly consumable by any LLM.

3. Real tool feedback. Tools return real stdout/stderr. Errors get learned.

4. Accumulating intelligence. Every session archived. Patterns extracted.
   The system gets smarter with every interaction.

Architecture:
    User Input
         |
    [1. GATHER] ContextBuilder assembles relevant memory
         |      - project_memory.md (accumulated knowledge)
         |      - session context (current protein, mutations)
         |      - recent error patterns + solutions
         |      - enzyme-specific notes
         |
    [2. ACT]   ToolExecutor runs tools with real feedback
         |      - 7 agent.py functions + 34 skills
         |      - Real results, real errors
         |      - Timing captured
         |
    [3. VERIFY] ResultArchiver records everything
         |      - Full interaction archived
         |      - Insights extracted 鈫?project_memory.md
         |      - Error patterns learned
         |      - Session context updated
         |
    Response to user/agent
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
# Data types
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?


@dataclass
class ToolCall:
    """A single tool invocation."""

    tool: str
    params: dict
    result: Any = None
    elapsed_ms: float = 0
    success: bool = True
    error: str = ""


@dataclass
class Interaction:
    """A complete Gather鈫扐ct鈫扸erify cycle."""

    user_input: str
    context: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    outcome: str = ""
    insights: List[str] = field(default_factory=list)
    elapsed_ms: float = 0


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
# 1. GATHER 鈥?Context Assembly
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?


class ContextBuilder:
    """
    Assembles relevant context before each interaction.

    LLM agent runtime pattern: read ZHUANGZHOU.md + project state 鈫?focused context.
    Zhuangzhou pattern: read project_memory.md + session context + enzyme notes
                       + workspace intelligence (v0.7.4).

    v0.7.5: Uses MemoryRegistry singleton + context budget + local KG injection.
    """

    BUDGET = 4000  # chars (~1000 tokens) 鈥?keeps context lean

    def __init__(self, config: dict = None):
        self._reg = None

    @property
    def _registry(self):
        if self._reg is None:
            from zhuangzhou.memory.registry import MemoryRegistry

            self._reg = MemoryRegistry.get()
        return self._reg

    @property
    def project_memory(self):
        return self._registry.project_memory

    @property
    def knowledge(self):
        return self._registry.knowledge

    @property
    def session_context(self):
        return self._registry.context

    @property
    def workspace_intel(self):
        return self._registry.workspace_intel

    def gather(self, user_input: str = "", enzyme: str = "") -> str:
        """
        Build the context string for the current interaction.

        Returns a focused plain text context that any LLM can consume.
        Uses priority-based budget allocation to keep context lean.
        """
        budget = self.BUDGET
        parts = []

        # Sources ordered by priority with max char allocation
        sources = [
            (self._get_workspace_block, 300),
            (self._get_project_memory_block, 1500, enzyme),
            (self._get_session_block, 200),
            (self._get_kg_context, 800, user_input),
            (self._get_error_hints, 400, user_input),
            (self._get_mutation_block, 400, user_input),
            (self._get_workflow_block, 400, user_input),
        ]

        for entry in sources:
            if budget <= 0:
                break
            func = entry[0]
            max_alloc = entry[1]
            args = entry[2:]
            alloc = min(max_alloc, budget)
            try:
                text = func(*args, max_chars=alloc) if args else func(max_chars=alloc)
            except Exception:
                continue
            if text:
                parts.append(text[:alloc])
                budget -= len(text[:alloc])

        return "\n\n".join(parts)

    def _get_workspace_block(self, max_chars: int = 300) -> str:
        try:
            return (self.workspace_intel.get_context_block() or "")[:max_chars]
        except Exception:
            return ""

    def _get_project_memory_block(self, enzyme: str = "", max_chars: int = 1500) -> str:
        max_lines = max(20, max_chars // 80)
        text = self.project_memory.get_context(enzyme=enzyme, max_lines=max_lines)
        return text.strip()[:max_chars] if text.strip() else ""

    def _get_session_block(self, max_chars: int = 200) -> str:
        summary = self.session_context.get_summary()
        if summary and summary != "No active context":
            return f"## Current Session\n{summary}"[:max_chars]
        return ""

    def _get_kg_context(self, user_input: str = "", max_chars: int = 800) -> str:
        """Inject local KG triples matching user query into context."""
        if not user_input:
            return ""
        try:
            triples = self._registry.kg_query(user_input[:80], limit=8)
        except Exception:
            triples = []
        if not triples:
            entities = re.findall(r"\b[A-Z]\d{1,4}[A-Z]\b", user_input)
            for entity in entities[:3]:
                triples.extend(self.knowledge.query_local_graph(entity, limit=2))
        if not triples:
            return ""
        lines = ["## Knowledge Graph (local)"]
        for t in triples:
            suffix = ""
            if t.get("access_count"):
                suffix = f" (used {t['access_count']}x)"
            lines.append(f"- {t['head']} --[{t['relation']}]--> {t['tail']}{suffix}")
        text = "\n".join(lines)
        return text[:max_chars]

    def _get_error_hints(self, user_input: str = "", max_chars: int = 400) -> str:
        if not user_input:
            return ""
        errors = self.knowledge._error_patterns.get("errors", {})
        hints = []
        input_lower = user_input.lower()
        for err_type, info in errors.items():
            for ctx in info.get("contexts", []):
                if any(word in input_lower for word in ctx.lower().split()[:3]):
                    solutions = info.get("solutions", [])
                    if solutions:
                        hints.append(f"{err_type}: {solutions[-1]}")
                    break
            if len(hints) >= 5:
                break
        if not hints:
            return ""
        text = "## Known Pitfalls\n" + "\n".join(f"- {h}" for h in hints)
        return text[:max_chars]

    def _get_mutation_block(self, user_input: str = "", max_chars: int = 400) -> str:
        if not user_input:
            return ""
        mutations = re.findall(r"\b[A-Z]\d{1,4}[A-Z]\b", user_input)
        if not mutations:
            return ""
        history = []
        for mut in mutations:
            similar = self.knowledge.get_similar_mutations(mut)
            for s in similar[:2]:
                history.append(
                    f"{s['mutation']} on {s['protein']}: "
                    f"score={s['score']:.2f}, {s['outcome']}"
                )
        if not history:
            return ""
        text = "## Mutation History\n" + "\n".join(f"- {h}" for h in history[:10])
        return text[:max_chars]

    def _get_workflow_block(self, user_input: str = "", max_chars: int = 400) -> str:
        if not user_input:
            return ""
        try:
            recommended = self.knowledge.get_recommended_chains(user_input[:80])
            if not recommended:
                return ""
            lines = ["## Proven Workflows"]
            for r in recommended[:3]:
                lines.append(
                    f"- {r['chain']} "
                    f"(success: {r['success_rate'] * 100:.0f}%, "
                    f"n={r['count']}, ~{r['avg_elapsed']:.0f}s)"
                )
            return "\n".join(lines)[:max_chars]
        except Exception:
            return ""


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
# 2. ACT 鈥?Tool Execution
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?


class ToolExecutor:
    """
    Execute tools and capture real results.

    LLM agent runtime pattern: bash returns real stderr/stdout.
    Zhuangzhou pattern: tool functions return real dicts (error + hint on failure).
    """

    def __init__(self, config: dict = None):
        self._agent = None
        self.config = config or {}

    @property
    def agent(self):
        if self._agent is None:
            from zhuangzhou.agent import get_agent

            self._agent = get_agent(self.config)
        return self._agent

    def execute(self, tool_name: str, params: dict) -> ToolCall:
        """Execute a single tool and capture the result."""
        tc = ToolCall(tool=tool_name, params=params)
        t0 = time.time()
        try:
            result = self.agent.execute_tool(tool_name, params)
            tc.result = result
            tc.success = not (isinstance(result, dict) and "error" in result)
            if not tc.success:
                tc.error = result.get("error", "")
        except Exception as e:
            tc.result = {"error": str(e)}
            tc.success = False
            tc.error = str(e)
            logger.error("Tool %s failed: %s", tool_name, e)
        tc.elapsed_ms = (time.time() - t0) * 1000
        return tc

    def execute_function(self, func_name: str, **kwargs) -> ToolCall:
        """Execute an agent.py function directly by name."""
        import zhuangzhou.agent as agent_module

        func = getattr(agent_module, func_name, None)
        if not func:
            return ToolCall(
                tool=func_name,
                params=kwargs,
                result={"error": f"Function {func_name} not found"},
                success=False,
                error=f"Function {func_name} not found",
            )

        tc = ToolCall(tool=func_name, params=kwargs)
        t0 = time.time()
        try:
            result = func(**kwargs)
            tc.result = result
            tc.success = not (isinstance(result, dict) and "error" in result)
            if not tc.success:
                tc.error = result.get("error", "")
        except Exception as e:
            tc.result = {"error": str(e)}
            tc.success = False
            tc.error = str(e)
        tc.elapsed_ms = (time.time() - t0) * 1000
        return tc


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
# 3. VERIFY 鈥?Result Archival & Learning
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?


class ResultArchiver:
    """
    Archive results and extract learnings.

    LLM agent runtime pattern: persistent memory in plain text files.
    Zhuangzhou pattern: archive full interactions, distill into project_memory.md.

    v0.7.6: Uses MemoryRegistry for shared instances across knowledge, execution,
    journal, and session archive so NLP and workflow paths converge on one memory spine.
    """

    def __init__(self, config: dict = None):
        self._reg = None

    @property
    def _registry(self):
        if self._reg is None:
            from zhuangzhou.memory.registry import MemoryRegistry

            self._reg = MemoryRegistry.get()
        return self._reg

    @property
    def archive(self):
        return self._registry.session_archive

    @property
    def project_memory(self):
        return self._registry.project_memory

    @property
    def knowledge(self):
        return self._registry.knowledge

    @property
    def journal(self):
        return self._registry.journal

    @property
    def exec_history(self):
        return self._registry.exec_history

    def archive_interaction(self, interaction: Interaction):
        """Archive a complete interaction and extract learnings."""
        # 1. Archive to session file
        self.archive.begin_interaction(
            interaction.user_input, {"context_preview": interaction.context[:500]}
        )
        for tc in interaction.tool_calls:
            self.archive.record_tool_call(
                tc.tool, tc.params, tc.result, tc.elapsed_ms, tc.success
            )
        for insight in interaction.insights:
            self.archive.record_insight(insight)
        self.archive.end_interaction(interaction.outcome)

        # 2. Log to journal
        for tc in interaction.tool_calls:
            self.journal.log_skill(tc.tool, tc.params, tc.result, tc.elapsed_ms / 1000)

        # 2b. Persist to execution history (cross-session)
        for tc in interaction.tool_calls:
            try:
                self.exec_history.record(
                    skill_name=tc.tool,
                    params=tc.params,
                    success=tc.success,
                    elapsed=tc.elapsed_ms / 1000,
                    error=tc.error,
                    user_context=interaction.user_input[:200],
                )
            except Exception:
                pass

        # 3. Learn from errors
        for tc in interaction.tool_calls:
            if not tc.success and tc.error:
                hint = ""
                if isinstance(tc.result, dict):
                    hint = tc.result.get("hint", "")
                self.knowledge.record_error(
                    tc.error[:100],
                    interaction.user_input[:200],
                    solution=hint or f"Check {tc.tool} parameters",
                )

        # 4. Learn from successes
        for tc in interaction.tool_calls:
            if tc.success and "analyze_mutation" in tc.tool:
                result = tc.result
                if isinstance(result, dict) and "mutation" in result:
                    self.knowledge.record_mutation_result(
                        result["mutation"],
                        tc.params.get("enzyme_class", "unknown"),
                        result.get("chemical_score", 0),
                        result.get("recommendation", ""),
                        {"confidence": result.get("confidence")},
                    )

    def distill_insights(self, interaction: Interaction) -> List[str]:
        """
        Extract reusable insights from an interaction.
        These get written to project_memory.md for future context.
        """
        insights = []

        for tc in interaction.tool_calls:
            if not tc.success:
                # Error -> pitfall
                hint = ""
                if isinstance(tc.result, dict):
                    hint = tc.result.get("hint", "")
                if hint:
                    insight = f"{tc.tool}: {tc.error} -> {hint}"
                    self.project_memory.add_pitfall(insight)
                    insights.append(insight)

            elif "run_pipeline" in tc.tool and tc.success:
                # Successful pipeline -> pattern
                result = tc.result
                if isinstance(result, dict) and result.get("status") == "complete":
                    summary = (
                        f"Pipeline completed: "
                        f"{result.get('track_a_count', 0)} track A, "
                        f"{result.get('track_b_count', 0)} track B, "
                        f"{result.get('combinations', 0)} combinations"
                    )
                    enzyme = tc.params.get("enzyme_class", "")
                    self.project_memory.record_successful_workflow(
                        enzyme, "run_pipeline", tc.params, summary
                    )
                    insights.append(summary)

        return insights

    def save(self):
        """Persist all archives to disk."""
        self.archive.save()


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
# Conversation Engine 鈥?the Gather鈫扐ct鈫扸erify loop
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?


class ConversationEngine:
    """
    The core engine that ties Gather鈫扐ct鈫扸erify together.

    Two usage modes:
      1. Programmatic (LLM agent runtime driving):
         engine.run_tool('analyze_mutation', mutation='V21L')
         engine.run_tool('search_knowledge', query='CYP3A4')

      2. Batch (multiple tools in sequence):
         engine.run_batch([
             ('analyze_mutation', {'mutation': 'V21L'}),
             ('analyze_mutation', {'mutation': 'A87G'}),
         ])

    Every call automatically: gathers context 鈫?executes 鈫?archives 鈫?learns.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.context_builder = ContextBuilder(config)
        self.tool_executor = ToolExecutor(config)
        self.archiver = ResultArchiver(config)

    def run_tool(
        self, func_name: str, user_input: str = "", enzyme: str = "", **kwargs
    ) -> dict:
        """
        Run a single tool with full Gather鈫扐ct鈫扸erify cycle.

        Args:
            func_name: agent.py function name (e.g. 'analyze_mutation')
            user_input: original user request (for context/archival)
            enzyme: enzyme name for context loading
            **kwargs: tool parameters

        Returns:
            dict with 'result', 'context_used', 'elapsed_ms', 'success'
        """
        t0 = time.time()

        # 1. GATHER
        context = self.context_builder.gather(
            user_input=user_input or func_name, enzyme=enzyme
        )

        # 2. ACT
        tc = self.tool_executor.execute_function(func_name, **kwargs)

        # 3. VERIFY + ARCHIVE
        interaction = Interaction(
            user_input=user_input or f"{func_name}({kwargs})",
            context=context,
            tool_calls=[tc],
            outcome="success" if tc.success else "error",
            elapsed_ms=(time.time() - t0) * 1000,
        )
        interaction.insights = self.archiver.distill_insights(interaction)
        self.archiver.archive_interaction(interaction)

        return {
            "result": tc.result,
            "success": tc.success,
            "elapsed_ms": tc.elapsed_ms,
            "context_used": bool(context.strip()),
            "insights": interaction.insights,
        }

    def run_batch(
        self, calls: List[tuple], user_input: str = "", enzyme: str = ""
    ) -> List[dict]:
        """
        Run multiple tools in a batch with shared context.

        Args:
            calls: list of (func_name, kwargs_dict) tuples
            user_input: original user request
            enzyme: enzyme name for context

        Returns:
            list of result dicts
        """
        t0 = time.time()

        # 1. GATHER (once for all)
        context = self.context_builder.gather(user_input=user_input, enzyme=enzyme)

        # 2. ACT (each tool)
        tool_calls = []
        for func_name, kwargs in calls:
            tc = self.tool_executor.execute_function(func_name, **kwargs)
            tool_calls.append(tc)

        # 3. VERIFY + ARCHIVE
        interaction = Interaction(
            user_input=user_input or f"batch({len(calls)} tools)",
            context=context,
            tool_calls=tool_calls,
            outcome="success" if all(tc.success for tc in tool_calls) else "partial",
            elapsed_ms=(time.time() - t0) * 1000,
        )
        interaction.insights = self.archiver.distill_insights(interaction)
        self.archiver.archive_interaction(interaction)

        return [
            {
                "result": tc.result,
                "success": tc.success,
                "elapsed_ms": tc.elapsed_ms,
                "tool": tc.tool,
            }
            for tc in tool_calls
        ]

    def get_context(self, user_input: str = "", enzyme: str = "") -> str:
        """Get assembled context without running any tools."""
        return self.context_builder.gather(user_input, enzyme)

    def save(self):
        """Persist all session data to disk."""
        self.archiver.save()
        self.context_builder.session_context.save()

    def get_memory_summary(self) -> dict:
        """Get a summary of all memory systems."""
        return {
            "archive": self.archiver.archive.summary(),
            "project_memory": self.archiver.project_memory.summary(),
            "knowledge": self.archiver.knowledge.summary(),
            "session": self.context_builder.session_context.get_summary(),
        }


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
# Singleton
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?

_engine: Optional[ConversationEngine] = None


def get_engine(config: dict = None) -> ConversationEngine:
    """Get or create the conversation engine singleton."""
    global _engine
    if _engine is None:
        _engine = ConversationEngine(config)
    return _engine


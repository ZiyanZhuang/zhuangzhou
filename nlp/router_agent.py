"""
RouterAgent — 50-token Intent Router
=====================================
Single LLM call with structured JSON output.
Decides between: direct / fixed / plan

Design principle (industry: Droptica 95% cost reduction pattern):
  - Minimal system prompt (no skill catalog loaded)
  - Enum-constrained output prevents hallucination
  - Confidence threshold: < 0.65 → fallback to plan mode
  - Fast path rules (0 token) checked first
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Fast-path rules evaluated before any LLM call (0 token)
_DIRECT_PATTERNS = [
    r"^/",                             # slash command
    r"^\s*(yes|no|y|n|确认|取消)\s*$",  # confirmation only
    r"^\d+$",                           # number selection
    r"^(help|帮助|\?|？)$",            # help request
]

# Routing decision threshold
_CONFIDENCE_THRESHOLD = 0.65

_ROUTER_SYSTEM = """你是科研工作流路由器。根据用户输入，判断应该走哪条执行路径。

路径定义:
  direct — 单步查询/操作，无需计划（搜索、下载、查看某个文件、问一个问题）
  fixed  — 用户目标明确且属于标准流水线（对接/MD/突变扫描），参数基本清楚
  plan   — 目标模糊、涉及多种分析、有条件逻辑、或超出标准流水线范围

fixed 触发条件（必须同时满足）:
  · 涉及 对接/docking/MD/模拟/突变扫描 之一
  · 有明确的蛋白（名称/PDB ID）或配体信息
  · 无复杂条件逻辑

pipeline 值（仅 fixed 时填写）:
  docking        — 分子对接 + PLIP 分析
  md             — 分子动力学模拟
  mutation_scan  — 突变扫描

仅输出JSON，无其他内容:
{"mode": "direct|fixed|plan", "pipeline": "docking|md|mutation_scan|null",
 "confidence": 0.0-1.0, "reason": "一句话说明"}"""


@dataclass
class RouteDecision:
    mode: str           # "direct" | "fixed" | "plan"
    pipeline: str       # "docking" | "md" | "mutation_scan" | None
    confidence: float
    reason: str
    from_fast_path: bool = False  # True if decided by rule, not LLM


class RouterAgent:
    """
    Routes user query to the appropriate execution path.

    Usage:
        router = RouterAgent(backend)
        decision = router.route("对接 CYP3A4 和 nicotine")
        # RouteDecision(mode='fixed', pipeline='docking', confidence=0.92)
    """

    def __init__(self, backend):
        self.backend = backend

    def route(self, query: str) -> RouteDecision:
        # 0. Fast path — rule-based, 0 LLM tokens
        fast = self._fast_path(query)
        if fast:
            return fast

        # 1. Single LLM call with constrained JSON output
        try:
            resp = self.backend.chat(
                _ROUTER_SYSTEM,
                [{"role": "user", "content": query}],
                [],
            )
            raw = resp.text or ""
            data = self._parse_json(raw)
            if data:
                mode = data.get("mode", "plan")
                pipeline = data.get("pipeline") or None
                confidence = float(data.get("confidence", 0.5))
                reason = data.get("reason", "")

                # Low confidence → force plan mode (safer)
                if confidence < _CONFIDENCE_THRESHOLD and mode == "fixed":
                    logger.info("RouterAgent: low confidence %.2f for fixed → plan", confidence)
                    mode = "plan"
                    reason += f" (confidence {confidence:.2f} < threshold, downgraded to plan)"

                return RouteDecision(
                    mode=mode,
                    pipeline=pipeline if mode == "fixed" else None,
                    confidence=confidence,
                    reason=reason,
                )
        except Exception as e:
            logger.warning("RouterAgent LLM call failed: %s — defaulting to plan", e)

        # Fallback: safe default
        return RouteDecision(mode="plan", pipeline=None, confidence=0.0,
                             reason="Router failed — defaulting to plan mode")

    def _fast_path(self, query: str) -> Optional[RouteDecision]:
        """Rule-based fast path. 0 LLM tokens."""
        q = query.strip()
        q_lower = q.lower()

        for pat in _DIRECT_PATTERNS:
            if re.search(pat, q_lower):
                return RouteDecision(
                    mode="direct", pipeline=None, confidence=1.0,
                    reason=f"Fast path: matched pattern '{pat}'",
                    from_fast_path=True,
                )

        # Very short queries (< 8 chars) that aren't numbers
        if len(q) < 8 and not q.isdigit():
            return RouteDecision(
                mode="direct", pipeline=None, confidence=0.85,
                reason="Fast path: very short query",
                from_fast_path=True,
            )

        return None

    def _parse_json(self, text: str) -> Optional[dict]:
        decoder = json.JSONDecoder()
        start = text.find("{")
        tries = 0
        while start != -1 and tries < 20:
            tries += 1
            try:
                obj, _ = decoder.raw_decode(text, start)
                if isinstance(obj, dict) and "mode" in obj:
                    return obj
            except json.JSONDecodeError:
                pass
            start = text.find("{", start + 1)
        return None

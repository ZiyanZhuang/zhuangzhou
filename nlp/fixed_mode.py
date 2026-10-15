"""
Fixed Mode — Template Pipelines with Intelligent Parameter Resolution
======================================================================
Fixed topology (deterministic step order) + smart ResearchPhase (fills params from literature).

Pipelines:
  docking       → research → pre_docking_gate → smart_docking → plip_analysis → plip_visualize → enriched_docking_report (.docx)
  md            → research → pre_md_gate → auto_md → gromacs_analysis → generate_md_report (.docx)
  mutation_scan → research → esm2_mutation_scan → generate_mutation_report (.docx)

Design:
  - Step order is fixed (no LLM decides topology)
  - Parameters ARE NOT fixed: ResearchPhase uses search_pubmed + LLM extraction
  - Supervisor gates catch bad parameters BEFORE expensive execution
  - Total LLM token cost: ~800 (research) + ~150 (gate) = ~950 per run
    vs Plan Mode: ~3000+ tokens

Scientific rigor contract:
  - Every parameter must have a literature reference or structural justification
  - SupervisorGate rejects unsupported parameters (blocking=True)
  - Gate failures are shown to user with specific issues — never silent
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .supervisor import PipelineSupervisor, GateResult
from zhuangzhou.skills.results import result_error

logger = logging.getLogger(__name__)


# =============================================================================
# Research Phase — PubMed search + LLM parameter extraction
# =============================================================================

_RESEARCH_SYSTEM = """你是生物计算实验的文献调研专家。
从以下PubMed摘要中提取计算参数，为后续实验提供文献支持。

只提取有文献明确记载的信息，推断内容标注 inferred=true。
缺乏文献支持的参数填 null，不要猜测。

输出JSON:
{
  "active_site_residues": ["Phe120", "Glu210"] 或 null,
  "active_site_center": {"x": 0.0, "y": 0.0, "z": 0.0} 或 null,
  "cofactors": ["HEME", "FAD"] 或 [],
  "recommended_chain": "A" 或 null,
  "md_duration_ns": 100 或 null,
  "forcefield_notes": "文献推荐的力场说明" 或 null,
  "key_residues_for_binding": ["Arg105", "Ser209"] 或 null,
  "literature_support_count": 3,
  "inferred": false,
  "notes": "其他重要说明"
}"""


@dataclass
class ResearchResult:
    active_site_residues: List[str] = field(default_factory=list)
    active_site_center: Optional[Dict[str, float]] = None
    cofactors: List[str] = field(default_factory=list)
    recommended_chain: Optional[str] = None
    md_duration_ns: Optional[int] = None
    forcefield_notes: Optional[str] = None
    key_residues_for_binding: List[str] = field(default_factory=list)
    literature_support_count: int = 0
    inferred: bool = False
    notes: str = ""
    literature_snippets: List[str] = field(default_factory=list)  # for gate validation
    raw: dict = field(default_factory=dict)


class ResearchPhase:
    """
    Search PubMed for protein/ligand information, then extract parameters via LLM.
    Total cost: 0 LLM tokens for search + ~800 tokens for extraction.
    """

    def __init__(self, backend):
        self.backend = backend

    def run(self, protein: str, ligand: Optional[str] = None,
            pipeline: str = "docking") -> ResearchResult:
        """
        Run literature research for the given protein/ligand pair.
        Returns ResearchResult with extracted parameters.
        """
        snippets = self._search_literature(protein, ligand, pipeline)
        if not snippets:
            logger.warning("ResearchPhase: no literature found for %s", protein)
            return ResearchResult(notes="No literature found — parameters will need manual review")

        return self._extract_params(protein, ligand, snippets, pipeline)

    def _search_literature(self, protein: str, ligand: Optional[str],
                            pipeline: str) -> List[str]:
        """Call search_pubmed skill for relevant papers."""
        try:
            from zhuangzhou.skills.research import search_pubmed

            queries = self._build_queries(protein, ligand, pipeline)
            snippets = []
            for q in queries:
                result = search_pubmed(query=q, max_results=3, return_abstracts=True)
                if isinstance(result, dict) and "results" in result:
                    for paper in result["results"]:
                        abstract = paper.get("abstract", "")
                        title = paper.get("title", "")
                        if abstract:
                            snippets.append(f"[{title}]\n{abstract[:600]}")
                if len(snippets) >= 6:
                    break

            logger.info("ResearchPhase: collected %d literature snippets", len(snippets))
            return snippets
        except Exception as e:
            logger.warning("ResearchPhase search_pubmed failed: %s", e)
            return []

    def _build_queries(self, protein: str, ligand: Optional[str],
                        pipeline: str) -> List[str]:
        """Build targeted PubMed queries."""
        queries = []
        # Primary: protein active site
        queries.append(f"{protein} active site binding pocket")
        # Secondary: cofactor
        queries.append(f"{protein} cofactor HEME crystal structure")
        # Ligand-specific
        if ligand and pipeline == "docking":
            queries.append(f"{protein} {ligand} binding interaction")
        # MD-specific
        if pipeline == "md":
            queries.append(f"{protein} molecular dynamics simulation")
        return queries

    def _extract_params(self, protein: str, ligand: Optional[str],
                         snippets: List[str], pipeline: str) -> ResearchResult:
        """LLM call to extract structured parameters from literature snippets."""
        combined = "\n\n---\n\n".join(snippets[:5])  # cap to ~3000 chars
        user_msg = (
            f"蛋白质: {protein}\n"
            + (f"配体: {ligand}\n" if ligand else "")
            + f"任务类型: {pipeline}\n\n"
            f"以下为PubMed检索到的相关文献摘要:\n\n{combined}\n\n"
            "请提取计算参数（JSON格式）:"
        )

        try:
            resp = self.backend.chat(
                _RESEARCH_SYSTEM,
                [{"role": "user", "content": user_msg}],
                [],
            )
            raw = resp.text or ""
            decoder = json.JSONDecoder()
            start = raw.find("{")
            tries = 0
            while start != -1 and tries < 20:
                tries += 1
                try:
                    data, _ = decoder.raw_decode(raw, start)
                    if isinstance(data, dict):
                        result = ResearchResult(
                            active_site_residues=data.get("active_site_residues") or [],
                            active_site_center=data.get("active_site_center"),
                            cofactors=data.get("cofactors") or [],
                            recommended_chain=data.get("recommended_chain"),
                            md_duration_ns=data.get("md_duration_ns"),
                            forcefield_notes=data.get("forcefield_notes"),
                            key_residues_for_binding=data.get("key_residues_for_binding") or [],
                            literature_support_count=data.get("literature_support_count", len(snippets)),
                            inferred=data.get("inferred", False),
                            notes=data.get("notes", ""),
                            literature_snippets=snippets,
                            raw=data,
                        )
                        logger.info("ResearchPhase extracted: cofactors=%s, active_site=%s",
                                    result.cofactors, result.active_site_residues)
                        return result
                except json.JSONDecodeError:
                    pass
                start = raw.find("{", start + 1)
        except Exception as e:
            logger.warning("ResearchPhase LLM extraction failed: %s", e)

        return ResearchResult(
            literature_snippets=snippets,
            notes="Parameter extraction failed — manual review required",
        )


# =============================================================================
# Fixed Pipeline Templates
# =============================================================================

@dataclass
class FixedStep:
    skill: str
    params: Dict[str, Any]
    description: str
    gate_before: Optional[str] = None   # gate to run before this step
    gate_after: Optional[str] = None    # gate to run after this step


def _build_docking_pipeline(research: ResearchResult,
                             protein: str, ligand: str,
                             pdb_id: Optional[str] = None) -> List[FixedStep]:
    """Docking pipeline: research-informed parameters, supervisor-gated."""
    return [
        FixedStep(
            skill="get_pdb_entry",
            params={"pdb_id": pdb_id or protein,
                    "output_dir": None},
            description=f"获取 {protein} 结构",
            gate_after="pre_docking",   # TypeA checks run after download
        ),
        FixedStep(
            skill="smart_docking",
            params={
                "pdb_path": "{from_step_1}",
                "ligand": ligand,
                "output_dir": None,
                # Research-informed box hint (may be None → auto-detect from structure)
                "active_site_residues": research.active_site_residues or None,
                "chain": research.recommended_chain,
            },
            description=f"分子对接 {ligand} → {protein}",
        ),
        FixedStep(
            skill="plip_analysis",
            params={"complex_pdb": "{from_step_2}", "output_dir": None},
            description="PLIP 相互作用分析",
            gate_after="post_docking",
        ),
        FixedStep(
            skill="plip_visualize",
            params={"complex_pdb": "{from_step_2}",
                    "plip_result": "{dict_step_3}",
                    "output_dir": None},
            description="生成互作可视化图",
        ),
        FixedStep(
            skill="enriched_docking_report",
            params={
                "results": "{dict_step_2}",  # full smart_docking result dict
                "title": f"{protein} \u2014 {ligand} Molecular Docking Report",
            },
            description="生成英文 Word 对接报告（含文献、UniProt、KEGG）",
        ),
    ]


def _build_md_pipeline(research: ResearchResult,
                        protein: str,
                        duration_ns: int = 100,
                        pdb_id: Optional[str] = None) -> List[FixedStep]:
    """MD pipeline: auto-selects metalloprotein_md if HEME detected by gate."""
    # Skill will be corrected by pre_md gate if HEME detected
    md_skill = "metalloprotein_md" if research.cofactors else "auto_md"

    return [
        FixedStep(
            skill="get_pdb_entry",
            params={"pdb_id": pdb_id or protein, "output_dir": None},
            description=f"获取 {protein} 结构",
            gate_after="pre_md",   # TypeA+TypeB gate: metal detection + LLM validation
        ),
        FixedStep(
            skill=md_skill,
            params={
                "pdb_path": "{from_step_1}",
                "duration_ns": research.md_duration_ns or duration_ns,
                "output_dir": None,
            },
            description=f"MD 模拟 ({research.md_duration_ns or duration_ns} ns)",
        ),
        FixedStep(
            skill="gromacs_analysis",
            params={"work_dir": "{from_step_2}"},
            description="MD 结果分析 (RMSD/RMSF/Rg)",
        ),
        FixedStep(
            skill="generate_md_report",
            params={
                "results": "{dict_step_3}",  # full gromacs_analysis result dict
                "title": f"{protein} Molecular Dynamics Simulation Report",
            },
            description="生成英文 Word MD 报告",
        ),
    ]


def _build_mutation_pipeline(research: ResearchResult,
                               protein: str,
                               mutations: List[str],
                               pdb_id: Optional[str] = None) -> List[FixedStep]:
    """Mutation scan pipeline."""
    return [
        FixedStep(
            skill="get_pdb_entry",
            params={"pdb_id": pdb_id or protein, "output_dir": None},
            description=f"获取 {protein} 野生型结构",
            gate_after="pre_docking",
        ),
        FixedStep(
            skill="esm2_mutation_scan",
            params={
                "pdb_path": "{from_step_1}",
                "mutations": mutations,
                "key_residues": research.key_residues_for_binding or None,
            },
            description=f"ESM-2 突变扫描 ({len(mutations)} 个位点)",
        ),
        FixedStep(
            skill="generate_mutation_report",
            params={
                "results": "{dict_step_2}",  # full esm2_mutation_scan result dict
                "title": f"{protein} Mutation Scan Report",
            },
            description="生成英文 Word 突变分析报告",
        ),
    ]


# =============================================================================
# FixedMode — main entry point
# =============================================================================

class FixedMode:
    """
    Execute a fixed-topology pipeline with intelligent parameter resolution.

    Flow:
      1. ResearchPhase: PubMed search → LLM param extraction (~800 token)
      2. post_research gate: TypeB LLM validates literature support (~150 token)
      3. Build pipeline steps with research-informed parameters
      4. Execute each step; TypeA gates run at structural decision points (0 token)
      5. TypeB pre_md gate if MD pipeline (~100 token)

    Total: ~950-1100 tokens vs Plan Mode ~3000+
    """

    def __init__(self, backend, registry, agent_executor):
        self.backend = backend
        self.registry = registry
        self.executor = agent_executor
        self.research_phase = ResearchPhase(backend)
        self.supervisor = PipelineSupervisor(backend)

    def run(self, pipeline: str, protein: str, ligand: Optional[str] = None,
            mutations: Optional[List[str]] = None,
            duration_ns: int = 100, pdb_id: Optional[str] = None,
            console=None) -> dict:
        """
        Execute a fixed pipeline end-to-end.

        Returns: {"success": bool, "results": [...], "gate_issues": [...]}
        """
        if console is None:
            from zhuangzhou.ui.core import console as _console
            console = _console

        console.print(f"\n  [bold cyan]Fixed Mode[/] [{pipeline}]  "
                      f"[dim]{protein}" + (f" + {ligand}" if ligand else "") + "[/]\n")

        # ── Step 1: Research Phase ──────────────────────────────────────────
        console.print("  [dim]📚 文献调研中...[/]")
        research = self.research_phase.run(protein, ligand, pipeline)
        self._print_research_summary(research, console)

        # ── Step 2: post_research Gate (TypeB) ──────────────────────────────
        gate_result = self.supervisor.gate_post_research(
            research_params=research.raw,
            literature_snippets=research.literature_snippets,
        )
        if not self._handle_gate(gate_result, console):
            return {"success": False, "reason": "post_research gate failed",
                    "gate_result": gate_result}

        # Apply gate corrections to research params
        if gate_result.corrections:
            research = self._apply_corrections(research, gate_result.corrections)

        # ── Step 3: Build pipeline ──────────────────────────────────────────
        steps = self._build_steps(pipeline, research, protein, ligand,
                                   mutations, duration_ns, pdb_id)
        if not steps:
            return {"success": False, "reason": f"Unknown pipeline: {pipeline}"}

        # ── Step 4: Execute steps with TypeA gates ──────────────────────────
        results = []
        step_outputs: Dict[int, dict] = {}
        gate_issues_all = []
        failed_step = None

        for i, step in enumerate(steps, 1):
            console.print(f"  [dim]Step {i}/{len(steps)}:[/] [cyan]{step.skill}[/]  "
                          f"[dim]{step.description}[/]")

            # Resolve {from_step_N} references
            try:
                resolved_params = self._resolve_refs(step.params, step_outputs)
            except (ValueError, KeyError, TypeError) as e:
                results.append({'step': i, 'skill': step.skill, 'error': str(e)})
                failed_step = i
                break

            if step.gate_before:
                gate = self._run_gate(step.gate_before, resolved_params, step_outputs, i)
                gate_issues_all.extend(gate.issues)
                if gate.blocking or not gate.passed:
                    results.append({'step': i, 'skill': step.skill, 'error': f'Gate {step.gate_before} blocked execution'})
                    failed_step = i
                    break

            # Run the skill
            skill = self.registry.get(step.skill)
            if not skill:
                console.print(f"  [red]✗ Skill '{step.skill}' not found[/]")
                results.append({"step": i, "skill": step.skill,
                                 "error": f"Skill not found: {step.skill}"})
                failed_step = i
                break

            try:
                result = (self.executor._execute_skill_with_ui(step.skill, resolved_params)
                          if self.executor is not None else skill.execute(**resolved_params))
            except Exception as e:
                result = {'error': str(e)}
            step_outputs[i] = result
            results.append({"step": i, "skill": step.skill, "result": result})
            if result_error(result) or isinstance(result, dict) and result.get('needs_confirmation'):
                failed_step = i
                break

            # Post-step TypeA gate
            if step.gate_after:
                gate = self._run_gate(step.gate_after, result, step_outputs, i)
                if gate.issues:
                    gate_issues_all.extend(gate.issues)
                    self._print_gate_issues(gate, console)
                if gate.blocking or not gate.passed:
                    console.print(f"  [red bold]✗ Gate '{step.gate_after}' blocked execution[/]")
                    failed_step = i
                    break
                # Apply gate corrections to next step
                if gate.corrections and i + 1 <= len(steps):
                    steps[i] = self._patch_step(steps[i], gate.corrections)

            if "error" in result:
                console.print(f"  [red]✗ {result['error']}[/]")
                break
            else:
                console.print(f"  [green]✓[/] 完成")

        return {
            "success": failed_step is None and len(results) == len(steps),
            "failed_step": failed_step,
            "results": results,
            "gate_issues": gate_issues_all,
            "research": research.raw,
        }

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _build_steps(self, pipeline, research, protein, ligand,
                      mutations, duration_ns, pdb_id) -> List[FixedStep]:
        if pipeline == "docking" and ligand:
            return _build_docking_pipeline(research, protein, ligand, pdb_id)
        elif pipeline == "md":
            return _build_md_pipeline(research, protein, duration_ns, pdb_id)
        elif pipeline == "mutation_scan" and mutations:
            return _build_mutation_pipeline(research, protein, mutations, pdb_id)
        return []

    def _resolve_refs(self, params: dict, outputs: Dict[int, dict]) -> dict:
        """Replace {from_step_N} / {dict_step_N} with actual output values.

        {from_step_N}  — extracts a path-like value (pdb_path, output_dir, …)
        {dict_step_N}  — returns the entire result dict (for report skills)
        """
        resolved = {}
        for k, v in params.items():
            if isinstance(v, str) and v.startswith("{dict_step_"):
                try:
                    step_n = int(v.split("_")[-1].rstrip("}"))
                    if step_n not in outputs:
                        raise ValueError(f'Missing result for step {step_n}')
                    resolved[k] = outputs[step_n]
                except (ValueError, KeyError):
                    raise ValueError(f'Unresolved pipeline reference: {v}')
            elif isinstance(v, str) and v.startswith("{from_step_"):
                try:
                    step_n = int(v.split("_")[-1].rstrip("}"))
                    if step_n not in outputs:
                        raise ValueError(f'Missing result for step {step_n}')
                    prev = outputs[step_n]
                    # Try common output keys
                    resolved[k] = (prev.get(k) or prev.get("complex_pdb") or prev.get("pdb_path")
                                   or prev.get("work_dir") or prev.get("output_dir")
                                   or prev.get('output_file'))
                    if not resolved[k]:
                        raise ValueError(f'No output artifact for {v}')
                except (ValueError, KeyError):
                    raise ValueError(f'Unresolved pipeline reference: {v}')
            else:
                resolved[k] = v
        return resolved

    def _run_gate(self, gate_name: str, step_result: dict,
                   outputs: Dict[int, dict], step_num: int) -> GateResult:
        """Dispatch TypeA gate based on name."""
        pdb_path = step_result.get("pdb_path") or step_result.get("output_file", "")

        if gate_name == "pre_docking":
            return self.supervisor.gate_pre_docking(pdb_path)
        elif gate_name == "pre_md":
            md_params = {
                "duration_ns": outputs.get(step_num, {}).get("duration_ns", 100),
                "protein": "unknown",
            }
            return self.supervisor.gate_pre_md(pdb_path, md_params)
        elif gate_name == "post_docking":
            docking_dir = step_result.get("output_dir", "")
            return self.supervisor.gate_post_docking(docking_dir)
        else:
            from .supervisor import GateResult
            return GateResult(passed=True, gate_name=gate_name)

    def _handle_gate(self, gate: GateResult, console) -> bool:
        """Display gate result to user. Return False if blocking failure."""
        if not gate.issues and gate.passed:
            return True

        for issue in gate.issues:
            level = "[yellow]⚠[/]" if not gate.blocking else "[red]✗[/]"
            console.print(f"  {level} {issue}")

        if gate.blocking:
            console.print(f"\n  [red bold]文献验证未通过，无法继续执行。[/]")
            console.print("  建议: 切换到 Plan Mode 或手动指定参数。\n")
            return False

        if gate.corrections:
            console.print(f"  [cyan]→ 已自动修正参数:[/] {list(gate.corrections.keys())}")
        return True

    def _print_gate_issues(self, gate: GateResult, console) -> None:
        for issue in gate.issues:
            console.print(f"  [yellow]⚠ Gate[{gate.gate_name}]:[/] {issue}")

    def _print_research_summary(self, research: ResearchResult, console) -> None:
        console.print(f"  [dim]  文献数: {research.literature_support_count}篇[/]"
                      + (f"  辅因子: [cyan]{', '.join(research.cofactors)}[/]"
                         if research.cofactors else "")
                      + (f"  活性位点: [cyan]{', '.join(research.active_site_residues[:3])}[/]"
                         if research.active_site_residues else ""))
        if research.notes:
            console.print(f"  [dim]  备注: {research.notes}[/]")

    def _apply_corrections(self, research: ResearchResult,
                             corrections: dict) -> ResearchResult:
        """Apply gate corrections back to research result."""
        if "active_site_residues" in corrections:
            research.active_site_residues = corrections["active_site_residues"]
        if "recommended_chain" in corrections:
            research.recommended_chain = corrections["recommended_chain"]
        return research

    def _patch_step(self, step: FixedStep, corrections: dict) -> FixedStep:
        """Apply gate corrections to a future step's params."""
        if "skill" in corrections:
            step.skill = corrections["skill"]
        step.params.update({k: v for k, v in corrections.items() if k != "skill"})
        return step

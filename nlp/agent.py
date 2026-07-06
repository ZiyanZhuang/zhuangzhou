"""
Zhuangzhou Agent - ReAct Engine v5 (Definitive Edition)
=======================================================
Full capabilities restoration + Context Engineering Integration.

Merged Features:
1.  Deferred Tools & ToolGenerator
2.  Dual-Mode Reasoning (Native Tools + Manual ReAct Fallback)
3.  Expert Domain Guidance (MD/Docking/QM/Mutation)
4.  Dynamic Pattern Memory (Success rate & pattern learning)
5.  Workspace Awareness (via ContextManager)
6.  Self-Correction & Multi-turn Retry
7.  Pre-flight Validation (Timeout guards & thread safety)
8.  Dependency Graph Reasoning
9.  Context Summarization (>30 messages)
10. Integrated rich UI & Stream Rendering
"""

import json
import logging
import os
import re
import sys
import time
import threading
import traceback
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Set

from zhuangzhou.memory.knowledge import KnowledgeStore
from zhuangzhou.memory.context import SessionContext
from zhuangzhou.memory.execution_history import (
    ExecutionHistory,
    TaskCheckpoint,
    WorkspaceTracker,
)
from zhuangzhou.nlp.context import context_manager, FEW_SHOT_REACT
from zhuangzhou.nlp.router import ToolRouter
from zhuangzhou.nlp.parallel import ParallelSkillExecutor, BackgroundTaskManager

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.spinner import Spinner
    from rich.text import Text

    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 50
CORE_CATEGORIES = {"research", "structure", "chemistry"}
MAX_TOOL_SCHEMAS = 50

CATEGORY_INDEX = {
    "simulation": "11 tools — auto_md, GROMACS MD, analysis, trajectory, CGenFF",
    "qm": "9 tools — DFT optimization, TS search, frequency analysis, energy profiles",
    "visualization": "4 tools — pymol_view, plip_visualize, chimerax_visualize, generate_report",
    "docking": "9 tools — Vina docking, SmartDocking, batch mutant docking, LigBuilder",
    "mutagenesis": "1 tool — mutate_protein (PyMOL/BioPython, optional GROMACS EM)",
    "plm": "1 tool — esm2_mutation_scan + esm2_load",
    "sequence": "4 tools — codon optimization, codon usage analysis",
    "brain": "5 tools — KG query, literature search, document upload",
}

MANUAL_TOOL_INSTRUCTIONS = """
## 工具调用说明 (REASONING MODE)
当前后端不支持原生 Function Calling。你必须严格按照以下格式在回复中调用工具：

Thought: [你的分析逻辑]
Action: {{"name": "[工具名称]", "input": {{"[参数]": "[值]"}}}}

### 可用工具手册 (Skill Manual)
{skill_manual}

### 操作示例 (Few-shot)
{few_shot}
"""

# =============================================================================
# 1. Helper Utilities
# =============================================================================


def _compact_dict(d: dict, depth: int = 0) -> dict:
    compact = {}
    for k, v in d.items():
        if isinstance(v, str) and len(v) > 2000:
            compact[k] = v[:2000] + "...(truncated)"
        elif isinstance(v, list) and len(v) > 20:
            compact[k] = v[:20] + ["...(truncated)"]
        elif isinstance(v, dict) and depth < 2:
            compact[k] = _compact_dict(v, depth + 1)
        elif isinstance(v, dict):
            compact[k] = f"{{...{len(v)} keys}}"
        else:
            compact[k] = v
    return compact


class ToolGenerator:
    @staticmethod
    def _map_type(skill_type: str) -> str:
        mapping = {
            "string": "string",
            "path": "string",
            "int": "integer",
            "float": "number",
            "list": "array",
            "set": "array",
            "dict": "object",
            "bool": "boolean",
        }
        return mapping.get(skill_type, "string")

    @classmethod
    def generate_schema(cls, skill) -> dict:
        properties = {}
        required = []
        for param in skill.inputs:
            prop_schema = {
                "type": cls._map_type(param.type),
                "description": param.description or f"Parameter for {param.name}",
            }
            if param.type in ("list", "set"):
                prop_schema["items"] = {"type": "string"}
            if param.default is not None:
                prop_schema["default"] = param.default
            if param.choices:
                prop_schema["enum"] = param.choices
            properties[param.name] = prop_schema
            if param.required:
                required.append(param.name)
        return {
            "name": skill.name,
            "description": f"[{skill.category}] {skill.description}",
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    @classmethod
    def generate_schemas(cls, registry, categories: Set[str] = None) -> List[dict]:
        schemas = []
        for skill in registry.list_all():
            if categories and skill.category not in categories:
                continue
            schemas.append(cls.generate_schema(skill))
        return schemas


LOAD_TOOLS_SCHEMA = {
    "name": "load_tools",
    "description": "Load additional tool categories on demand. Once loaded, tools remain available.",
    "input_schema": {
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": list(CATEGORY_INDEX.keys())},
            "query": {"type": "string", "description": "Search across all tools"},
        },
    },
}

# Background task management schemas (for long-running skills like MD, docking)
BACKGROUND_TASK_SCHEMAS = [
    {
        "name": "submit_background_task",
        "description": "Submit a long-running skill to execute in the background. "
                       "Use for skills estimated to take >60 seconds (MD simulation, "
                       "large docking, etc.). Returns a task_id for tracking.",
        "input_schema": {
            "type": "object",
            "properties": {
                "skill_name": {"type": "string", "description": "Name of the skill to run"},
                "skill_params": {"type": "object", "description": "Parameters for the skill"},
                "description": {"type": "string", "description": "Human-readable description"},
                "estimated_seconds": {"type": "number", "description": "Estimated run time in seconds", "default": 120},
            },
            "required": ["skill_name", "skill_params"],
        },
    },
    {
        "name": "check_background_tasks",
        "description": "Check status of background tasks. Returns running/completed task list.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "Specific task ID (optional, omit for all)"},
            },
        },
    },
    {
        "name": "collect_background_result",
        "description": "Retrieve the result of a completed background task. The task is removed after collection.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "Task ID to collect"},
            },
            "required": ["task_id"],
        },
    },
]

# =============================================================================
# 2. Agent Executor (The Reasoning Core)
# =============================================================================

from zhuangzhou.ui.core import (
    ZZInterface,
    ZZStatus,
    ToolSpinner,
    format_tokens,
    format_elapsed,
)


class AgentExecutor:
    def __init__(self, backend, registry, verbose=False):
        self.backend = backend
        self.registry = registry
        self.verbose = verbose
        self._loaded_categories: Set[str] = set(CORE_CATEGORIES)
        self._rebuild_tools()
        self.messages: List[dict] = []
        self.execution_log: List[dict] = []
        self._execution_log_lock = threading.Lock()
        self._confirmed_actions: set = set()
        self._exec_history = ExecutionHistory()
        self._checkpoint = TaskCheckpoint()
        self._workspace = WorkspaceTracker()
        self._ui_session_context = SessionContext()
        self._compaction_count = 0
        self._last_compaction_reason = ""
        self._last_input_tokens = 0
        self._last_output_tokens = 0
        self._last_total_tokens = 0
        self._last_stream_estimated_output_tokens = 0
        self._session_id = (
            __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        # Parallel skill execution (inspired by Claude Code's tool orchestration)
        self._parallel_executor = ParallelSkillExecutor(
            registry=self.registry,
            executor_fn=self._execute_skill_with_ui,
            max_workers=5,
        )
        self._bg_tasks = BackgroundTaskManager(max_workers=3)
        self._huizi_lock = threading.Lock()  # Thread-safe HuiZi updates

        # HuiZi context awareness
        try:
            from zhuangzhou.huizi import HuiZi
            self._huizi = HuiZi(session_id=self._session_id or "default")
            self._huizi_enabled = True
        except Exception as e:
            logger.warning("HuiZi init failed (context features disabled): %s", e)
            self._huizi = None
            self._huizi_enabled = False
        # Specialist lesson router — learns from tool outcomes across sessions
        try:
            self.router = ToolRouter(backend=self.backend, registry=self.registry)
        except Exception:
            self.router = None
        # Register HuiZi context skills
        if getattr(self, '_huizi_enabled', False):
            try:
                from zhuangzhou.skills.huizi_skill import register_huizi_skills
                register_huizi_skills(self.registry)
            except Exception:
                pass
        self._current_user_task: str = ""  # updated per run_loop() call

    def _build_prompt_blocks(self) -> List[dict]:
        snapshot = context_manager.get_snapshot()
        blocks = []

        if snapshot["constitution"]:
            blocks.append(
                {
                    "name": "constitution",
                    "title": "Project Constitution",
                    "content": f"## PROJECT CONSTITUTION (ABSOLUTE PRECEDENCE)\n{snapshot['constitution']}\n",
                }
            )

        core_prompt = f"""你是庄周（Zhuangzhou），一个理性酶工程/计算生物学自主智能体。
当前工作目录: {snapshot["cwd"]}
文件快照: {snapshot["files"]}

## 运行准则
1. ReAct 循环: Thought -> Action -> Observation
2. 证据分级: [A 文献] [B KG] [C 计算] [D 推测]
3. 自动纠错 (Feature 6): 阅读报错原因并修正路径或参数重试。
4. 工作流链推理: 自动衔接多步工具 (e.g. 突变 -> 对接 -> 分析)。
5. 研究阶段约束: 文献/数据库检索最多使用 5 次工具，收集到关键信息即停止，避免重复查询同类数据源。
6. next_steps 强制执行: 若工具返回结果包含 "next_steps" 字段，必须将其中每一步作为后续 Action 依次执行，不得跳过。
7. 报告生成: 完成 ESM-2/突变扫描后，若存在 beneficial_mutations_json 路径，MUST 调用 generate_report 生成 Word 报告，即使用户未明确要求。

## HuiZi Context Protocol
1. When the user mentions a protein/PDB file, immediately call `set_analysis_target` to record the analysis target.
   If there is no PDB file but a UniProt ID or sequence is available, call `fetch_protein_structure` to obtain a structure automatically.
2. Before generating a report, call `get_analysis_context`, then pass `protein_context` to `generate_report`.
3. For Chinese knowledge-graph queries, call `translate_scientific_query` first to obtain English keywords.
4. When important findings are produced, call `record_key_finding`.
5. Reports must use the correct protein name; never write "Unknown Protein".
"""
        blocks.append({"name": "core", "title": "Core Prompt", "content": core_prompt})

        if not getattr(self.backend, "supports_tools", True):
            manual = context_manager.get_skill_manual(
                self.registry, self._loaded_categories
            )
            tool_instr = MANUAL_TOOL_INSTRUCTIONS.format(
                skill_manual=manual, few_shot=FEW_SHOT_REACT
            )
        else:
            tool_instr = (
                f"\n## TOOLS\n已加载类别: {', '.join(self._loaded_categories)}\n"
            )
        blocks.append(
            {"name": "tools", "title": "Tool Instructions", "content": tool_instr}
        )

        # NOTE: _detect_domain_guidance() removed — keyword-triggered expert injection
        # violates LLM emergent reasoning. Domain guidance now lives in skill catalog
        # (when_to_use / workflow_stage fields), which the LLM reads and reasons from.

        pattern_hints = self._get_pattern_hints()
        if pattern_hints:
            blocks.append(
                {"name": "patterns", "title": "Pattern Hints", "content": pattern_hints}
            )

        # HuiZi context injection
        if getattr(self, '_huizi_enabled', False) and self._huizi:
            try:
                huizi_block = self._huizi.get_system_prompt_block()
                if huizi_block:
                    blocks.append({
                        "name": "huizi",
                        "title": "HuiZi Context",
                        "content": huizi_block,
                    })
            except Exception:
                pass

        for block in blocks:
            block["estimated_tokens"] = self._estimate_text_tokens(block["content"])
        return blocks

    def _rebuild_tools(self):
        self.tools = ToolGenerator.generate_schemas(
            self.registry, self._loaded_categories
        )
        self.tools = [t for t in self.tools
                       if t["name"] not in ("load_tools", "submit_background_task",
                                            "check_background_tasks", "collect_background_result")]
        self.tools.insert(0, LOAD_TOOLS_SCHEMA)
        self.tools.extend(BACKGROUND_TASK_SCHEMAS)

    def _handle_load_tools(self, params: dict) -> dict:
        category = params.get("category")
        query = params.get("query")
        loaded_names = []
        if category:
            if category in self._loaded_categories:
                return {"status": "already_loaded", "category": category}
            self._loaded_categories.add(category)
            loaded_names = [
                s.name for s in self.registry.list_all() if s.category == category
            ]
        elif query:
            query_lower = query.lower()
            for skill in self.registry.list_all():
                if (
                    query_lower in skill.name.lower()
                    or query_lower in (skill.description or "").lower()
                ):
                    self._loaded_categories.add(skill.category)
                    loaded_names.append(skill.name)

        self._rebuild_tools()

        # Dependency Graph Resolution (Feature 12)
        dep_hints = {}
        for name in loaded_names:
            skill = self.registry.get(name)
            if skill and skill.depends_on:
                dep_hints[name] = f"Needs: {', '.join(skill.depends_on)}"

        return {
            "status": "loaded",
            "count": len(loaded_names),
            "loaded_categories": sorted(self._loaded_categories),
            "workflow_hints": dep_hints,
        }

    def build_system_prompt(self) -> str:
        return "\n".join(
            block["content"].strip("\n")
            for block in self._build_prompt_blocks()
            if block["content"]
        ).strip()

    def _get_pattern_hints(self) -> str:
        """Inject historical pattern warnings (Feature 10)."""
        try:
            ks = KnowledgeStore()
            hints = []
            for entry in self.execution_log[-3:]:
                f_hints = ks.get_skill_failure_hints(entry["skill"])
                for h in f_hints:
                    hints.append(f"- ⚠ {entry['skill']}: {h}")
            return "\n\n## 动态记忆提示\n" + "\n".join(hints) if hints else ""
        except Exception:
            return ""

    def _update_huizi_context(self, skill_name: str, params: dict, result):
        """Extract protein/ligand context from skill calls and update HuiZi. Non-fatal.

        Thread-safe: guarded by _huizi_lock for parallel skill execution.
        """
        if not getattr(self, '_huizi_enabled', False) or not self._huizi:
            return
        if not isinstance(result, dict) or result.get('error'):
            return

        self._huizi_lock.acquire()
        try:
            hz = self._huizi
            pdb_path = params.get('pdb_path') or params.get('pdb_file')
            protein_name = params.get('protein_name') or params.get('name')

            # Protein-focused skills
            if skill_name in ('esm2_mutation_scan', 'esm2_scan', 'pdb_to_fasta',
                              'structural_analysis', 'annotate_dssp', 'pocket_detection',
                              'enrich_mutations', 'mutate_protein_enhanced'):
                if pdb_path:
                    hz.set_protein(pdb_path=pdb_path, name=protein_name, auto_resolve=False)
                # esm2_scan uses 'fasta_path' param — extract protein name from FASTA header
                if skill_name in ('esm2_mutation_scan', 'esm2_scan'):
                    fasta_path = params.get('fasta_path') or params.get('fasta')
                    if fasta_path and os.path.exists(fasta_path) and not hz.session.target.protein_name:
                        try:
                            # Prefer filename stem (e.g. CYP6AB328.fasta → CYP6AB328)
                            _stem = os.path.splitext(os.path.basename(fasta_path))[0]
                            with open(fasta_path, encoding='utf-8', errors='replace') as _f:
                                hdr = _f.readline().strip().lstrip('>')
                            _hdr_name = hdr.split()[0] if hdr else ''
                            # Use filename stem unless the header looks more descriptive
                            # (Header like "6J_chain_A" → PDB ID, use stem instead)
                            import re as _re
                            _is_pdb_chain = bool(_re.match(r'^[0-9A-Z]{4}[_\s]', _hdr_name, _re.I))
                            _pname = _stem if (_is_pdb_chain or len(_hdr_name) <= 4) else _hdr_name
                            if _pname:
                                hz.set_protein(name=_pname, auto_resolve=False)
                        except Exception:
                            pass
                    # Load beneficial_mutations_json to extract sequence and top mutations
                    bmj = result.get('beneficial_mutations_json')
                    if bmj and os.path.exists(bmj):
                        try:
                            import json as _json
                            with open(bmj, encoding='utf-8', errors='replace') as _f:
                                _esm_data = _json.load(_f)
                            _seq = _esm_data.get('sequence', '')
                            if _seq and not hz.session.target.fasta_sequence:
                                hz.session.target.fasta_sequence = _seq
                            # Extract protein name from 'protein' field if still unset
                            if not hz.session.target.protein_name and _esm_data.get('protein'):
                                import re as _re
                                _raw = str(_esm_data['protein'])
                                _clean = _re.sub(r'[_\s]chain[_\s]?[A-Za-z]$', '', _raw,
                                                 flags=_re.IGNORECASE).strip()
                                hz.set_protein(name=_clean or _raw, auto_resolve=False)
                            # Record top mutations as findings
                            _muts = sorted(_esm_data.get('mutations', []),
                                           key=lambda x: -x.get('delta_ll', 0))
                            if _muts:
                                top_str = ', '.join(
                                    f"{m['mutation']}(ΔLL={m.get('delta_ll',0):.2f})"
                                    for m in _muts[:5]
                                )
                                hz.record_analysis('esm2_scan',
                                    key_finding=f"ESM-2 top mutations: {top_str}")
                        except Exception:
                            pass
                fasta = result.get('fasta_sequence') or result.get('sequence')
                if fasta and not hz.session.target.fasta_sequence:
                    hz.session.target.fasta_sequence = fasta
                hz.record_analysis(
                    analysis_type=skill_name,
                    result_path=str(result.get('output_dir', result.get('output_file', '')) or ''),
                )

            # Docking skills: capture ligand info
            elif skill_name in ('smart_docking', 'vina_docking', 'diffdock'):
                if pdb_path:
                    hz.set_protein(pdb_path=pdb_path, name=protein_name, auto_resolve=False)
                ligand_name = params.get('ligand_name') or params.get('ligand')
                ligand_smiles = params.get('smiles') or params.get('ligand_smiles')
                if ligand_name or ligand_smiles:
                    hz.set_ligand(name=ligand_name, smiles=ligand_smiles)
                # Also capture SMILES from result if available
                result_smiles = result.get('ligand_smiles') or result.get('smiles')
                if result_smiles and not hz.session.target.ligand_smiles:
                    hz.session.target.ligand_smiles = result_smiles
                hz.record_analysis('docking', result_path=str(result.get('output_dir', '') or ''))

            # UniProt queries: capture protein identity
            elif skill_name == 'uniprot_query':
                if result.get('protein_name'):
                    hz.set_protein(
                        name=result.get('protein_name'),
                        uniprot_id=result.get('accession') or result.get('uniprot_id'),
                        organism=result.get('organism'),
                        auto_resolve=False,
                    )
                if result.get('sequence') and not hz.session.target.fasta_sequence:
                    hz.session.target.fasta_sequence = result['sequence']

            # Structure fetching: update pdb_path and quality info
            elif skill_name == 'fetch_protein_structure':
                pdb_path = result.get('pdb_path')
                if pdb_path and os.path.exists(pdb_path):
                    hz.session.target.pdb_path = pdb_path
                    hz.session.target.structure_source = result.get('source')
                    hz.session.target.plddt_mean = result.get('plddt_mean')
                    hz.session.target.structure_quality_tier = result.get('quality_tier')
                    if result.get('quality_tier') in ('low', 'very_low'):
                        _plddt_val = result.get('plddt_mean')
                        _plddt_str = f"{_plddt_val:.0f}" if _plddt_val is not None else "N/A"
                        hz.record_analysis('structure_fetch',
                            key_finding=f"结构pLDDT={_plddt_str}，置信度低，分析结果需谨慎解读")
                    else:
                        hz.record_analysis('structure_fetch',
                            result_path=pdb_path)

            # PLIP analysis
            elif skill_name in ('plip_visualize', 'plip_analysis', 'run_plip'):
                hz.record_analysis('plip', result_path=str(result.get('output_dir', '') or ''))

            hz.session.save()
        except Exception as e:
            logger.debug("HuiZi context update failed: %s", e)
        finally:
            self._huizi_lock.release()

    def _execute_skill_with_ui(self, name: str, params: dict, monitor=None) -> dict:
        if name == "load_tools":
            res = self._handle_load_tools(params)
            ZZInterface.print_action(f"load_tools", "DONE", 0)
            return res

        # Background task management pseudo-skills
        if name == "check_background_tasks":
            status = self._bg_tasks.check_status(params.get('task_id'))
            completed = self._bg_tasks.get_completed_ids()
            return {'tasks': status, 'completed_ids': completed,
                    'active_count': self._bg_tasks.active_count}
        if name == "collect_background_result":
            tid = params.get('task_id', '')
            if not tid:
                return {'error': 'task_id required'}
            return self._bg_tasks.collect_result(tid)
        if name == "submit_background_task":
            skill_name = params.get('skill_name', '')
            skill_params = params.get('skill_params', {})
            desc = params.get('description', '')
            if not skill_name:
                return {'error': 'skill_name required'}
            tid = self._bg_tasks.submit(
                skill_name, skill_params,
                executor_fn=self._execute_skill_with_ui,
                description=desc,
                estimated_seconds=float(params.get('estimated_seconds', 120)),
            )
            return {'task_id': tid, 'status': 'submitted', 'skill': skill_name}

        skill = self.registry.get(name)
        if not skill:
            return {"error": f"Skill '{name}' not found."}

        # Inject HuiZi context into generate_report
        if name == 'generate_report' and getattr(self, '_huizi_enabled', False) and self._huizi:
            if not params.get('protein_context'):
                try:
                    ctx = self._huizi.get_report_context(fetch_literature=True)
                    if ctx:
                        params = dict(params)
                        params['protein_context'] = ctx
                except Exception:
                    pass

        # Pre-flight Validation (Feature 11 & Timeout Guards)
        t0 = time.time()
        _TIMEOUT = 600  # 10 min default

        def _run():
            try:
                return skill.execute(**params)
            except Exception as e:
                return {"error": str(e), "traceback": traceback.format_exc()}

        if monitor:
            monitor.update(action=name, phase="act")

        # Execute with timeout guard
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run)
            try:
                result = future.result(timeout=_TIMEOUT)
            except concurrent.futures.TimeoutError:
                future.cancel()
                logger.warning("Tool execution timed out after %ss: %s", _TIMEOUT, name)
                executor.shutdown(wait=False)
                return {'error': f'Tool timeout after {_TIMEOUT}s', 'tool': name}

        elapsed = time.time() - t0
        if not isinstance(result, dict):
            result = {"error": f"Skill '{name}' returned {type(result).__name__} instead of dict",
                      "raw": str(result)[:200]}
        success = "error" not in result
        ZZInterface.print_action(name, "DONE" if success else "FAILED", elapsed)

        # Record to persistent history — always include error details for debugging
        _err_msg = ''
        _err_tb = ''
        if isinstance(result, dict) and not success:
            _err_msg = result.get('error', '') or result.get('hint', '')
            _err_tb = result.get('traceback', '')
        _user_ctx = getattr(self, '_current_user_task', '') or ''
        self._exec_history.record(
            name,
            params,
            success,
            elapsed,
            self._extract_paths(result),
            error=_err_msg,
            error_traceback=_err_tb,
            session_id=self._session_id,
            user_context=_user_ctx,
        )

        # Specialist lesson router — record outcome for domain learning
        if self.router is not None:
            self.router.record_tool_outcome(name, result, self._current_user_task)

        # Update HuiZi context
        self._update_huizi_context(name, params, result)

        return result

    @staticmethod
    def _extract_paths(result) -> list:
        paths = []
        if not isinstance(result, dict):
            return paths
        for v in result.values():
            if isinstance(v, str) and ("." in v) and ("/" in v or "\\" in v):
                paths.append(v)
        return paths

    def _parse_manual_tool_calls(self, text: str) -> List[Any]:
        from zhuangzhou.nlp.backends import ToolCallRequest
        import uuid

        clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        tool_calls = []
        decoder = json.JSONDecoder()
        # Scan each "Action:" marker and use raw_decode to correctly handle nested JSON
        search_from = 0
        for action_match in re.finditer(r"Action:\s*", clean_text):
            brace_pos = clean_text.find("{", action_match.end())
            if brace_pos == -1:
                continue
            try:
                data, _ = decoder.raw_decode(clean_text, brace_pos)
                if not isinstance(data, dict):
                    continue
                name = data.get("name")
                args = data.get("arguments") or data.get("input") or {}
                if name:
                    tool_calls.append(
                        ToolCallRequest(
                            id=f"m-{uuid.uuid4().hex[:4]}", name=name, arguments=args
                        )
                    )
            except Exception:
                continue
        return tool_calls

    def _summarize_history(self):
        """Summarize history to prevent context overflow (Feature 9)."""
        if len(self.messages) <= 30:
            return
        try:
            summary_prompt = (
                "Summarize the history, preserving all file paths and numeric results."
            )
            summary_resp = self.backend.chat(
                "Summarizer",
                [
                    {
                        "role": "user",
                        "content": summary_prompt + str(self.messages[1:-4]),
                    }
                ],
                [],
            )
            # Use role="user" for compatibility with OpenAI/DeepSeek which reject
            # role="system" in the messages array (only allowed as the first system param).
            self.messages = [
                self.messages[0],
                {"role": "user", "content": f"[Conversation summary]: {summary_resp.text}"},
                {"role": "assistant", "content": "Understood. I'll continue with this context."},
            ] + self.messages[-4:]
        except Exception as e:
            logger.warning("History summarization failed, using truncation: %s", e)

    def _fallback_compaction_summary(self, older_messages: List[dict]) -> str:
        points = []
        for msg in older_messages[-12:]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False, default=str)
            content = re.sub(r"\s+", " ", content).strip()
            if len(content) > 240:
                content = content[:237] + "..."
            points.append(f"- {role}: {content}")
        return "[Compressed conversation memory]\n" + "\n".join(points)

    def _summarize_for_budget(self, older_messages: List[dict]) -> str:
        serialized = []
        for msg in older_messages:
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False, default=str)
            serialized.append(f"{msg.get('role', 'unknown').upper()}: {content}")
        payload = "\n\n".join(serialized)
        prompt = (
            "Compress the following conversation into a short working-memory summary. "
            "Preserve concrete file paths, commands, mutations, numeric results, errors, and decisions. "
            "Use concise bullet points."
        )
        try:
            response = self.backend.chat(
                "You summarize conversation state for context-window compaction.",
                [{"role": "user", "content": prompt + "\n\n" + payload}],
                [],
            )
            summary = (response.text or "").strip()
            if summary:
                return "[Compressed conversation memory]\n" + summary
        except Exception:
            pass
        return self._fallback_compaction_summary(older_messages)

    def _estimate_context_usage(
        self, system_prompt: str = None
    ) -> tuple[int, int, float]:
        system_prompt = (
            system_prompt if system_prompt is not None else self.build_system_prompt()
        )
        system_tokens = self._estimate_text_tokens(system_prompt)
        message_tokens = sum(
            self._estimate_message_tokens(msg) for msg in self.messages
        )
        total = system_tokens + message_tokens
        limit = self._context_window_limit()
        ratio = (total / limit) if limit else 0.0
        return total, limit, ratio

    def _enforce_context_budget(self) -> bool:
        compacted = False
        for _ in range(3):
            total, limit, ratio = self._estimate_context_usage()
            if ratio < 0.82:
                break
            if len(self.messages) <= 8:
                break

            keep_recent = 8
            older = self.messages[:-keep_recent]
            recent = self.messages[-keep_recent:]
            if not older:
                break
            summary = self._summarize_for_budget(older)
            self.messages = [{"role": "system", "content": summary}] + recent
            self._compaction_count += 1
            self._last_compaction_reason = (
                f"auto-compacted at {int(ratio * 100)}% of context budget"
            )
            compacted = True
        return compacted

    @staticmethod
    def _estimate_text_tokens(text: Any) -> int:
        """Estimate token count for mixed Chinese/English text.

        Heuristic:
          - ASCII chars: ~4 chars per token (English prose)
          - Non-ASCII (CJK, etc.): ~1 char per token (each CJK char ≈ 1 token)

        This avoids the 4x underestimate of pure len()//4 on Chinese content.
        """
        if text is None:
            return 0
        if not isinstance(text, str):
            text = json.dumps(text, ensure_ascii=False, default=str)
        if not text:
            return 0
        ascii_count = sum(1 for c in text if ord(c) < 128)
        non_ascii_count = len(text) - ascii_count
        return max(1, ascii_count // 4 + non_ascii_count)

    def _estimate_message_tokens(self, message: dict) -> int:
        content = message.get("content", "")
        return self._estimate_text_tokens(content) + 8

    def _context_window_limit(self) -> int:
        backend_name = getattr(self.backend, "name", "") or "unknown"
        backend_name = backend_name.lower()
        if backend_name == "claude":
            return 200_000
        if backend_name == "openai":
            return 128_000
        if backend_name == "deepseek":
            return 64_000
        if backend_name == "ollama":
            return 32_000
        return 64_000

    def get_ui_status(self) -> dict:
        snapshot = context_manager.get_snapshot()
        session_summary = ""
        try:
            session_summary = self._ui_session_context.get_summary()
        except Exception:
            session_summary = ""

        prompt_blocks = self._build_prompt_blocks()
        system_prompt = "\n".join(
            block["content"].strip("\n") for block in prompt_blocks if block["content"]
        ).strip()
        total_tokens, limit, ratio = self._estimate_context_usage(system_prompt)
        system_tokens = self._estimate_text_tokens(system_prompt)
        message_tokens = sum(
            self._estimate_message_tokens(msg) for msg in self.messages
        )

        # Build pattern hints once — used both for layers_active and prompt_blocks
        _pattern_hints = self._get_pattern_hints()

        layers_active = []
        if snapshot.get("constitution"):
            layers_active.append("project")
        if snapshot.get("files"):
            layers_active.append("workspace")
        if session_summary and session_summary != "No active context":
            layers_active.append("session")
        if self.messages:
            layers_active.append("history")
        if _pattern_hints.strip():
            layers_active.append("patterns")

        reset_recommended = ratio >= 0.80 or len(self.messages) >= 24
        if ratio >= 0.90:
            reset_reason = "context window is nearly full"
        elif len(self.messages) >= 24:
            reset_reason = "conversation history is getting long"
        elif ratio >= 0.80:
            reset_reason = "context usage is above 80%"
        else:
            reset_reason = ""

        return {
            "estimated_tokens": total_tokens,
            "system_tokens": system_tokens,
            "message_tokens": message_tokens,
            "last_input_tokens": self._last_input_tokens,
            "last_output_tokens": self._last_output_tokens,
            "last_total_tokens": self._last_total_tokens,
            "context_limit": limit,
            "usage_ratio": ratio,
            "message_count": len(self.messages),
            "loaded_categories": sorted(self._loaded_categories),
            "loaded_tool_schemas": len(self.tools),
            "layers_active": layers_active,
            "reset_recommended": reset_recommended,
            "reset_reason": reset_reason,
            "session_summary": session_summary,
            "session_id": self._session_id,
            "backend": getattr(self.backend, "name", "unknown"),
            "model": getattr(self.backend, "model", ""),
            "compaction_count": self._compaction_count,
            "last_compaction_reason": self._last_compaction_reason,
            "prompt_blocks": [
                {
                    "name": block["name"],
                    "title": block["title"],
                    "content": block["content"],
                    "estimated_tokens": block["estimated_tokens"],
                }
                for block in prompt_blocks
            ],
            "system_prompt": system_prompt,
        }

    def list_resumable_sessions(self, limit: int = 10) -> List[dict]:
        return self._ui_session_context.list_recent_sessions(limit=limit)

    def resume_session_context(self, session_id: str = "") -> bool:
        return self._ui_session_context.resume_session(session_id=session_id)

    def resume_current_context(self) -> bool:
        return self._ui_session_context.resume_current()

    def _stream_llm_response(self, system: str, monitor=None):
        from zhuangzhou.nlp.backends import LLMResponse
        from zhuangzhou.ui.core import StreamRenderer

        text_parts, tool_calls, usage, stop_reason, renderer = [], [], {}, "", None
        self._last_stream_estimated_output_tokens = 0
        self._did_stream_thought = self._did_stream_final = False

        # Pre-seed estimated input tokens so the status bar shows a non-zero `in:`
        # value immediately at stream start, instead of staying 0 until stream ends.
        estimated_input, _limit, _ratio = self._estimate_context_usage(system)
        if monitor and estimated_input > 0:
            monitor.set_usage(
                input_tokens=estimated_input,
                output_tokens=monitor.output_tokens,  # keep existing output estimate
            )

        try:
            for event_type, data in self.backend.chat_stream(
                system, self.messages, self.tools
            ):
                if event_type == "text":
                    text_parts.append(data)
                    estimated_chunk_tokens = self._estimate_text_tokens(data)
                    if estimated_chunk_tokens > 0:
                        self._last_stream_estimated_output_tokens += (
                            estimated_chunk_tokens
                        )
                        if monitor:
                            monitor.update(
                                tokens=estimated_chunk_tokens,
                                output_tokens=estimated_chunk_tokens,
                            )
                    if renderer is None and not self.verbose:
                        if monitor and monitor._live:
                            monitor._live.stop()
                            monitor._live = None
                        renderer = StreamRenderer(
                            show_panel=True,
                            status_provider=self.get_ui_status,
                            monitor=monitor,
                        )
                        renderer.start()
                    if renderer:
                        renderer.feed(data)
                elif event_type == "tool_call":
                    tool_calls.append(data)
                elif event_type == "usage":
                    usage = data
                elif event_type == "done":
                    stop_reason = data
        except Exception as e:
            logger.error("Stream error: %s", e)
            if renderer:
                renderer.finish()
            return self.backend.chat(system, self.messages, self.tools)

        full_text = "".join(text_parts)
        if not tool_calls:
            tool_calls = self._parse_manual_tool_calls(full_text)
        if renderer:
            renderer.finish()
            if tool_calls:
                self._did_stream_thought = True
            else:
                self._did_stream_final = True
            if monitor and monitor._live is None:
                from rich.live import Live

                monitor._live = Live(
                    monitor, console=console, refresh_per_second=12, transient=True
                )
                monitor._live.start()

        return LLMResponse(
            text=full_text, tool_calls=tool_calls, stop_reason=stop_reason, usage=usage
        )

    def run_loop(self, user_message: str, monitor=None, **kwargs) -> str:
        self._current_user_task = user_message  # expose to router during this turn
        self.messages.append({"role": "user", "content": user_message})
        self._summarize_history()
        self._enforce_context_budget()
        consecutive_errors = 0
        turn_exact_input_tokens = 0
        turn_exact_output_tokens = 0
        max_iterations = kwargs.get('max_iterations', MAX_ITERATIONS)

        final_text = ""
        for iteration in range(max_iterations):
            self._enforce_context_budget()
            system = self.build_system_prompt()
            response = self._stream_llm_response(system, monitor)

            # --- Sync Token Usage to Status Bar ---
            # Input tokens: each API call returns the FULL context size (not a delta).
            # Replacing (not accumulating) prevents 2x/3x over-counting across iterations.
            # Output tokens: each call returns only its own output → accumulate correctly.
            if response.usage:
                it = int(response.usage.get("input", 0) or 0)
                ot = int(response.usage.get("output", 0) or 0)
                if it:
                    turn_exact_input_tokens = it   # replace: last call is the canonical total
                turn_exact_output_tokens += ot     # accumulate: each call's output is additive
                self._last_input_tokens = turn_exact_input_tokens
                self._last_output_tokens = turn_exact_output_tokens
                self._last_total_tokens = (
                    self._last_input_tokens + self._last_output_tokens
                )
                if monitor:
                    monitor.set_usage(
                        input_tokens=turn_exact_input_tokens,
                        output_tokens=turn_exact_output_tokens,
                    )
            elif self._last_stream_estimated_output_tokens > 0:
                # No exact usage — fallback to estimates
                self._last_output_tokens = (
                    turn_exact_output_tokens + self._last_stream_estimated_output_tokens
                )
                self._last_total_tokens = (
                    self._last_input_tokens + self._last_output_tokens
                )

            if not response.has_tool_calls():
                final_text = response.text or ""
                self.messages.append({"role": "assistant", "content": final_text})
                if not self._did_stream_final:
                    ZZInterface.display_message(
                        "assistant",
                        final_text,
                        status=self.get_ui_status(),
                        monitor=monitor,
                    )

                # --- NEW: Dual-Layer Memory Hooks ---
                try:
                    from zhuangzhou.memory.journal import archive
                    from zhuangzhou.memory.kg_engine import kg_engine

                    # 1. Full-Fidelity Archiving
                    turn_data = {
                        "input": user_message,
                        "reasoning_chain": final_text
                        if not self.execution_log
                        else str(self.messages[-2]),  # Simplified for brevity here
                        "final_response": final_text,
                        "skills": self.execution_log.copy(),
                        "timestamp": __import__("datetime").datetime.now().isoformat(),
                    }
                    dest = archive.record_turn(
                        user_input=user_message,
                        thought=turn_data["reasoning_chain"],
                        skills=turn_data["skills"],
                        output=final_text,
                    )
                    if self.verbose:
                        logger.info(f"Full-fidelity turn archived to: {dest}")

                    # 2. Trigger Asynchronous KG Distillation
                    kg_engine.add_to_queue(turn_data)
                except Exception as mem_exc:
                    logger.warning("Memory persistence failed (turn not archived): %s", mem_exc)
                # ------------------------------------

                return final_text

            assistant_content = []
            if response.text:
                assistant_content.append({"type": "text", "text": response.text})
                if not self._did_stream_thought:
                    ZZInterface.display_message("thought", response.text)
            for tc in response.tool_calls:
                assistant_content.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    }
                )
            self.messages.append({"role": "assistant", "content": assistant_content})

            # ── Parallel Skill Execution (v0.9.2) ────────────────────────────
            # Inspired by Claude Code's tool orchestration:
            #   - Concurrent-safe skills (research, brain, chemistry) run in parallel
            #   - File-writing skills (docking, simulation, report) run serially
            #   - Per-tool error isolation: one failure doesn't kill the batch
            self._parallel_executor.executor_fn = self._execute_skill_with_ui
            all_results, batch_has_error, pending_next_steps = (
                self._parallel_executor.execute_all(response.tool_calls, monitor)
            )

            tool_results = []
            for tc, obs_data in all_results:
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": json.dumps(obs_data, ensure_ascii=False)[:3000],
                    }
                )

            # Inject next_steps as an explicit follow-up reminder so the model
            # cannot miss them even when the tool result is truncated.
            if pending_next_steps:
                steps_text = "\n".join(f"  - {s}" for s in pending_next_steps)
                tool_results.append(
                    {
                        "type": "text",
                        "text": (
                            f"[System] 以下后续步骤是MANDATORY（必须执行，不得跳过）：\n{steps_text}"
                        ),
                    }
                )

            self.messages.append({"role": "user", "content": tool_results})
            consecutive_errors = consecutive_errors + 1 if batch_has_error else 0
            if consecutive_errors >= 3:
                return "[System Intervention] 连续多次工具调用失败，已强制中断。"

        return "[Warning] 达到最大推演轮数。"


# =============================================================================
# 3. Agent Public Interface
# =============================================================================


class ZhuangzhouAgent:
    def __init__(
        self,
        backend: str = None,
        api_key: str = None,
        model: str = None,
        verbose: bool = False,
    ):
        from zhuangzhou.nlp.backends import get_backend
        from zhuangzhou.skills.registry import get_registry

        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if model:
            kwargs["model"] = model
        self.backend = get_backend(backend, **kwargs)
        self.registry = get_registry()
        self.verbose = verbose
        self.executor = AgentExecutor(self.backend, self.registry, self.verbose)

    def chat(self, user_message: str, status_monitor=None, **kwargs) -> str:
        return self.executor.run_loop(user_message, monitor=status_monitor, **kwargs)

    def reset(self):
        self.executor.messages.clear()
        self.executor.execution_log.clear()
        self.executor._loaded_categories = set(CORE_CATEGORIES)
        self.executor._rebuild_tools()

    def get_ui_status(self) -> dict:
        return self.executor.get_ui_status()

    def get_context_debug(self) -> dict:
        return self.executor.get_ui_status()

    def list_resumable_sessions(self, limit: int = 10) -> List[dict]:
        return self.executor.list_resumable_sessions(limit=limit)

    def resume_session_context(self, session_id: str = "") -> bool:
        return self.executor.resume_session_context(session_id=session_id)

    def resume_current_context(self) -> bool:
        return self.executor.resume_current_context()

    @property
    def backend_name(self) -> str:
        return self.backend.name

    @backend_name.setter
    def backend_name(self, value: str):
        pass

    @property
    def skill_count(self) -> int:
        return len(self.executor.tools)

    @property
    def loaded_categories(self) -> Set[str]:
        return self.executor._loaded_categories


# =============================================================================
# 4. Convenience API (The "Full-blooded" interface)
# =============================================================================

_default_agent: Optional[ZhuangzhouAgent] = None


def create_agent(backend: str = None, **kwargs) -> ZhuangzhouAgent:
    """Factory to create a full-featured Zhuangzhou agent."""
    return ZhuangzhouAgent(backend=backend, **kwargs)


def chat(message: str, backend: str = None, **kwargs) -> str:
    """Global convenience function for one-shot or persistent chat."""
    global _default_agent
    if _default_agent is None:
        _default_agent = create_agent(backend=backend, **kwargs)
    return _default_agent.chat(message)

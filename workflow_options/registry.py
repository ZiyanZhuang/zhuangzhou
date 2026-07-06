"""
Workflow Registry — Central registry for user-selectable workflows.
===================================================================
Each workflow is a self-contained multi-step pipeline with:
  - Metadata (name, description, tags, references)
  - Stage definitions (ordered steps with skill mappings)
  - Execution logic (orchestrates skills through the agent)
  - Parameter requirements (what the user must provide)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_registry_instance = None


@dataclass
class WorkflowParam:
    """A parameter required by the workflow."""
    name: str
    type: str           # 'path', 'string', 'int', 'float', 'list', 'set', 'bool'
    description: str
    required: bool = True
    default: Any = None
    example: str = ''


@dataclass
class WorkflowStage:
    """A single stage in a workflow."""
    id: str             # e.g. 'define_objectives'
    name: str           # Human-readable: '明确目标函数与硬约束'
    name_en: str        # English: 'Define Objectives & Hard Constraints'
    description: str    # What this stage does
    skills: List[str]   # Zhuangzhou skills used in this stage
    inputs: List[str]   # Parameter names consumed
    outputs: List[str]  # Keys produced for downstream stages
    optional: bool = False
    estimated_time: str = ''


@dataclass
class Workflow:
    """A complete user-selectable workflow."""
    id: str
    name: str
    name_en: str
    version: str
    description: str
    description_en: str
    tags: List[str]                         # e.g. ['enzyme_engineering', 'active_learning']
    references: List[Dict[str, str]]        # [{title, authors, journal, year, url}]
    stages: List[WorkflowStage]
    params: List[WorkflowParam]
    applicability: str                      # When this workflow is appropriate
    not_recommended: str                    # When to avoid
    execute_fn: Optional[Callable] = None   # The execution function

    def describe(self, lang: str = 'zh') -> str:
        """Human-readable workflow description."""
        lines = []
        if lang == 'zh':
            lines.append(f"# {self.name}")
            lines.append(f"版本: {self.version}")
            lines.append(f"\n{self.description}")
            lines.append(f"\n## 适用范围\n{self.applicability}")
            lines.append(f"\n## 不推荐使用场景\n{self.not_recommended}")
            lines.append(f"\n## 执行流程 ({len(self.stages)} 步)")
            for i, stage in enumerate(self.stages, 1):
                opt = ' (可选)' if stage.optional else ''
                lines.append(f"  {i}. {stage.name}{opt}")
                lines.append(f"     技能: {', '.join(stage.skills)}")
                if stage.estimated_time:
                    lines.append(f"     预计: {stage.estimated_time}")
        else:
            lines.append(f"# {self.name_en}")
            lines.append(f"Version: {self.version}")
            lines.append(f"\n{self.description_en}")
            lines.append(f"\n## Applicability\n{self.applicability}")
            lines.append(f"\n## Stages ({len(self.stages)} steps)")
            for i, stage in enumerate(self.stages, 1):
                opt = ' (optional)' if stage.optional else ''
                lines.append(f"  {i}. {stage.name_en}{opt}")
                lines.append(f"     Skills: {', '.join(stage.skills)}")

        lines.append(f"\n## 参考文献 / References")
        for ref in self.references:
            lines.append(f"  - {ref.get('authors', '')} ({ref.get('year', '')}). "
                         f"{ref.get('title', '')}. {ref.get('journal', '')}. "
                         f"{ref.get('url', '')}")

        return '\n'.join(lines)

    def get_required_params(self) -> List[WorkflowParam]:
        return [p for p in self.params if p.required]

    def to_tool_schema(self) -> dict:
        """Generate LLM tool schema for this workflow."""
        properties = {}
        required = []
        for p in self.params:
            prop = {
                'type': {'path': 'string', 'string': 'string', 'int': 'integer',
                         'float': 'number', 'list': 'array', 'bool': 'boolean',
                         'set': 'array'}.get(p.type, 'string'),
                'description': p.description,
            }
            if p.example:
                prop['description'] += f' (e.g. {p.example})'
            if p.default is not None:
                prop['default'] = p.default
            properties[p.name] = prop
            if p.required:
                required.append(p.name)

        return {
            'name': f'workflow_{self.id}',
            'description': f'[Workflow] {self.name_en}: {self.description_en[:200]}',
            'input_schema': {
                'type': 'object',
                'properties': properties,
                'required': required,
            }
        }

    def to_catalog_entry(self) -> str:
        """Concise catalog entry for LLM workflow selection."""
        stage_names = ' → '.join(s.name_en for s in self.stages)
        tags_str = ', '.join(self.tags)
        return (f"  workflow: {self.id}\n"
                f"    name: {self.name_en}\n"
                f"    tags: {tags_str}\n"
                f"    stages: {stage_names}\n"
                f"    description: {self.description_en[:200]}\n")


class WorkflowRegistry:
    """Central registry of user-selectable workflows."""

    def __init__(self):
        self._workflows: Dict[str, Workflow] = {}
        self._loaded = False

    def register(self, workflow: Workflow):
        self._workflows[workflow.id] = workflow
        logger.info("Registered workflow: %s (%s)", workflow.id, workflow.name_en)

    def get(self, workflow_id: str) -> Optional[Workflow]:
        if not self._loaded:
            self.load_all()
        return self._workflows.get(workflow_id)

    def list_all(self) -> List[Workflow]:
        if not self._loaded:
            self.load_all()
        return list(self._workflows.values())

    def load_all(self):
        if self._loaded:
            return
        self._loaded = True

        # Load Pareto + Active Learning workflow
        try:
            from .pareto_active_learning import register_workflow
            register_workflow(self)
        except Exception as e:
            logger.warning("Failed to load pareto_active_learning workflow: %s", e)

        # Future workflows load here
        # from .directed_evolution import register_workflow
        # from .rational_design import register_workflow

        logger.info("Loaded %d workflows", len(self._workflows))

    def build_catalog(self) -> str:
        """Build catalog string for LLM workflow selection."""
        self.load_all()
        if not self._workflows:
            return "No workflows available."
        lines = ["=== AVAILABLE WORKFLOWS ===", ""]
        for wf in self._workflows.values():
            lines.append(wf.to_catalog_entry())
        return '\n'.join(lines)


def get_workflow_registry() -> WorkflowRegistry:
    """Get or create the singleton workflow registry."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = WorkflowRegistry()
    return _registry_instance


def list_workflows() -> List[dict]:
    """List all available workflows as dicts (for skill/API use)."""
    registry = get_workflow_registry()
    return [
        {
            'id': wf.id,
            'name': wf.name,
            'name_en': wf.name_en,
            'version': wf.version,
            'tags': wf.tags,
            'n_stages': len(wf.stages),
            'description': wf.description_en[:300],
        }
        for wf in registry.list_all()
    ]

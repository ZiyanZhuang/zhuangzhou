"""
Workflow Engine
================
Executes JSON-defined workflows by composing skills.

Workflow JSON format:
{
    "name": "enzyme_engineering_v1",
    "description": "Standard enzyme mutation screening",
    "version": "1.0",
    "steps": [
        {
            "id": "step_1",
            "skill": "esm2_load",
            "params": {"result_dirs": "$input.esm2_dirs"},
            "output_key": "mutations"
        },
        {
            "id": "step_2",
            "skill": "mutation_classification",
            "params": {
                "pdb_path": "$input.pdb_path",
                "mutations": "$step_1.mutations",
                "binding_site": "$input.binding_site"
            },
            "output_key": "classifications"
        },
        ...
    ]
}

Variable references:
  $input.xxx     -> workflow input parameter
  $step_N.xxx    -> output from step N
  $config.xxx    -> config value
  $env.xxx       -> environment variable
"""

import json
import inspect
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

from zhuangzhou.skills.results import result_error
from zhuangzhou.skills.registry import Skill, SkillParam

logger = logging.getLogger(__name__)

# Template directory
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), 'templates')


class WorkflowStep:
    """A single step in a workflow."""

    def __init__(self, step_def: dict):
        self.id = step_def['id']
        self.skill_name = step_def['skill']
        self.params = step_def.get('params', {})
        self.output_key = step_def.get('output_key', self.id)
        self.condition = step_def.get('condition', None)  # optional skip condition
        self.on_error = step_def.get('on_error', 'stop')  # 'stop', 'skip', 'continue'
        self.description = step_def.get('description', '')


class WorkflowResult:
    """Result of a workflow execution."""

    def __init__(self):
        self.steps_completed: List[str] = []
        self.steps_skipped: List[str] = []
        self.steps_failed: List[str] = []
        self.outputs: Dict[str, Any] = {}
        self.errors: Dict[str, str] = {}
        self.elapsed_seconds: float = 0
        self.success: bool = False


class WorkflowEngine:
    """Execute JSON-defined workflows by composing skills."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        from zhuangzhou.skills import get_registry
        self.registry = get_registry(config)
        self._progress_callback: Optional[Callable] = None

    def set_progress_callback(self, callback: Callable):
        """Set callback for progress updates: callback(step_id, status, detail)"""
        self._progress_callback = callback

    def load_workflow(self, path_or_name: str) -> dict:
        """Load a workflow from file path or template name."""
        # Try as file path first
        if os.path.exists(path_or_name):
            with open(path_or_name, encoding='utf-8', errors='replace') as f:
                return json.load(f)

        # Try as template name
        template_path = os.path.join(TEMPLATES_DIR, f'{path_or_name}.json')
        if os.path.exists(template_path):
            with open(template_path, encoding='utf-8', errors='replace') as f:
                return json.load(f)

        raise FileNotFoundError(f"Workflow not found: {path_or_name}")

    def list_templates(self) -> List[dict]:
        """List available workflow templates."""
        templates = []
        if os.path.isdir(TEMPLATES_DIR):
            for fname in os.listdir(TEMPLATES_DIR):
                if fname.endswith('.json'):
                    fpath = os.path.join(TEMPLATES_DIR, fname)
                    try:
                        with open(fpath, encoding='utf-8', errors='replace') as f:
                            wf = json.load(f)
                        templates.append({
                            'name': wf.get('name', fname),
                            'description': wf.get('description', ''),
                            'version': wf.get('version', '?'),
                            'steps': len(wf.get('steps', [])),
                            'file': fname,
                        })
                    except Exception:
                        pass
        return templates

    def execute(self, workflow: dict, inputs: dict) -> WorkflowResult:
        """Execute a workflow with given inputs."""
        result = WorkflowResult()
        t0 = time.monotonic()

        steps = [WorkflowStep(s) for s in workflow.get('steps', [])]
        ids = [s.id for s in steps]
        if len(ids) != len(set(ids)) or any(not s or s in ('input', 'config', 'env') for s in ids):
            raise ValueError('Workflow step IDs must be unique, non-empty and not reserved')
        if any(s.on_error not in ('stop', 'skip', 'continue') for s in steps):
            raise ValueError('on_error must be stop, skip or continue')
        output_keys = [s.output_key for s in steps]
        if len(output_keys) != len(set(output_keys)):
            raise ValueError('Workflow output keys must be unique')
        # Declared optional inputs may be omitted; undeclared/misspelled references may not.
        declared_inputs = dict(inputs)
        for name, definition in workflow.get('inputs', {}).items():
            if name not in declared_inputs:
                if definition.get('required', False) and 'default' not in definition:
                    raise ValueError(f'Missing workflow input: {name}')
                declared_inputs[name] = definition.get('default')
        if workflow.get('inputs'):
            params = [SkillParam(name=name, type=d.get('type', 'string'),
                description=d.get('description', ''), required=d.get('required', False),
                default=d.get('default'), choices=d.get('choices'), is_output_path=d.get('is_output_path', False))
                for name, d in workflow['inputs'].items()]
            validator = Skill('workflow_inputs', 'Workflow input contract', 'workflow',
                              lambda **values: {'validated_inputs': values}, inputs=params)
            checked = validator.execute(**declared_inputs)
            if result_error(checked):
                raise ValueError(result_error(checked))
            declared_inputs = checked['validated_inputs']
        context = {
            'input': declared_inputs,
            'config': self.config,
            'env': dict(os.environ),
        }

        logger.info("Starting workflow: %s (%d steps)",
                     workflow.get('name', 'unnamed'), len(steps))

        for step in steps:
            try:
                if step.condition and not self._eval_condition(step.condition, context):
                    result.steps_skipped.append(step.id)
                    self._notify(step.id, 'skipped', 'Condition not met')
                    continue
                resolved_params = self._resolve_params(step.params, context)
            except Exception as e:
                result.errors[step.id] = f"Condition/parameter resolution failed: {e}"
                result.steps_failed.append(step.id)
                context[step.id] = {'success': False, 'error': result.errors[step.id]}
                self._notify(step.id, 'failed', result.errors[step.id])
                if step.on_error == 'stop':
                    break
                continue

            # Get skill
            skill = self.registry.get(step.skill_name)
            if not skill:
                result.errors[step.id] = f"Skill not found: {step.skill_name}"
                result.steps_failed.append(step.id)
                context[step.id] = {'success': False, 'error': result.errors[step.id]}
                self._notify(step.id, 'failed', result.errors[step.id])
                if step.on_error == 'stop':
                    break
                continue

            # Execute
            self._notify(step.id, 'running', step.description or step.skill_name)
            try:
                signature = inspect.signature(skill.fn)
                if 'config' in signature.parameters or any(
                    p.kind == p.VAR_KEYWORD for p in signature.parameters.values()
                ):
                    resolved_params = {'config': self.config, **resolved_params}
                output = skill.execute(**resolved_params)
                context[step.id] = output
                result.outputs[step.output_key] = output
                error = result_error(output)
                if error:
                    raise RuntimeError(error)
                result.steps_completed.append(step.id)
                self._notify(step.id, 'done', '')
            except Exception as e:
                logger.error("Step %s failed: %s", step.id, e)
                result.errors[step.id] = str(e)
                result.steps_failed.append(step.id)
                context[step.id] = {'success': False, 'error': str(e)}
                self._notify(step.id, 'failed', str(e))
                if step.on_error == 'stop':
                    break

        result.elapsed_seconds = time.monotonic() - t0
        result.success = len(result.steps_failed) == 0
        return result

    def _resolve_params(self, params: dict, context: dict) -> dict:
        """Resolve $variable references in parameters."""
        resolved = {}
        for key, val in params.items():
            resolved[key] = self._resolve_value(val, context)
        return resolved

    def _resolve_value(self, val: Any, context: dict) -> Any:
        """Resolve a single value, handling $references."""
        if isinstance(val, str) and val.startswith('$'):
            value = context
            for part in val[1:].split('.'):
                if not isinstance(value, dict) or part not in value:
                    raise ValueError(f'Unresolved workflow reference: {val}')
                value = value[part]
            return value
        elif isinstance(val, dict):
            return {k: self._resolve_value(v, context) for k, v in val.items()}
        elif isinstance(val, list):
            return [self._resolve_value(v, context) for v in val]
        return val

    def _eval_condition(self, condition: str, context: dict) -> bool:
        """Evaluate a simple condition string."""
        parts = condition.split(maxsplit=2)
        if len(parts) != 3 or parts[1] not in ('==', '!=') or not parts[0].startswith('$'):
            raise ValueError(f'Invalid workflow condition: {condition}')
        left = self._resolve_value(parts[0], context)
        try:
            right = json.loads(parts[2])
        except json.JSONDecodeError:
            right = parts[2]
        return left == right if parts[1] == '==' else left != right

    def _notify(self, step_id: str, status: str, detail: str):
        if self._progress_callback:
            try:
                self._progress_callback(step_id, status, detail)
            except Exception:
                logger.exception('Workflow progress callback failed for %s (%s)', step_id, status)

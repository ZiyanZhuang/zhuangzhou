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
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

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
        t0 = time.time()

        steps = [WorkflowStep(s) for s in workflow.get('steps', [])]
        context = {
            'input': inputs,
            'config': self.config,
            'env': dict(os.environ),
        }

        logger.info("Starting workflow: %s (%d steps)",
                     workflow.get('name', 'unnamed'), len(steps))

        for step in steps:
            # Check condition
            if step.condition and not self._eval_condition(step.condition, context):
                result.steps_skipped.append(step.id)
                self._notify(step.id, 'skipped', 'Condition not met')
                continue

            # Resolve parameters
            try:
                resolved_params = self._resolve_params(step.params, context)
            except Exception as e:
                result.errors[step.id] = f"Parameter resolution failed: {e}"
                result.steps_failed.append(step.id)
                if step.on_error == 'stop':
                    break
                continue

            # Get skill
            skill = self.registry.get(step.skill_name)
            if not skill:
                result.errors[step.id] = f"Skill not found: {step.skill_name}"
                result.steps_failed.append(step.id)
                if step.on_error == 'stop':
                    break
                continue

            # Execute
            self._notify(step.id, 'running', step.description or step.skill_name)
            try:
                output = skill.execute(config=self.config, **resolved_params)
                context[step.id] = output
                result.outputs[step.output_key] = output
                result.steps_completed.append(step.id)
                self._notify(step.id, 'done', '')
            except Exception as e:
                logger.error("Step %s failed: %s", step.id, e)
                result.errors[step.id] = str(e)
                result.steps_failed.append(step.id)
                self._notify(step.id, 'failed', str(e))
                if step.on_error == 'stop':
                    break

        result.elapsed_seconds = time.time() - t0
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
            parts = val[1:].split('.', 1)
            if len(parts) == 2:
                scope, attr = parts
                scope_data = context.get(scope, {})
                if isinstance(scope_data, dict):
                    return scope_data.get(attr)
            return None
        elif isinstance(val, dict):
            return {k: self._resolve_value(v, context) for k, v in val.items()}
        elif isinstance(val, list):
            return [self._resolve_value(v, context) for v in val]
        return val

    def _eval_condition(self, condition: str, context: dict) -> bool:
        """Evaluate a simple condition string."""
        # Simple conditions like "$step_1.success == true"
        try:
            parts = condition.split()
            if len(parts) == 3:
                left = self._resolve_value(parts[0], context)
                op = parts[1]
                right = parts[2]
                if right == 'true':
                    right = True
                elif right == 'false':
                    right = False
                elif right.isdigit():
                    right = int(right)
                if op == '==':
                    return left == right
                if op == '!=':
                    return left != right
        except Exception:
            pass
        return True

    def _notify(self, step_id: str, status: str, detail: str):
        if self._progress_callback:
            self._progress_callback(step_id, status, detail)

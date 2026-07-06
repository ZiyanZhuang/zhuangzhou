"""
Workflow Harness — Execution Engine for Multi-Step Experimental Workflows
=========================================================================
Zhuangzhou v0.9.2

Provides the execution infrastructure that workflows need:
  1. Task decomposition — break workflows into atomic, checkpointed steps
  2. State management  — persist state across sessions, handle mid-workflow restart
  3. Checkpoint validation — verify preconditions before each step
  4. Failure recovery — retry, skip, or rollback on step failure

Architecture:
  WorkflowHarness wraps a Workflow and provides:
    - run(step_or_all, params) — single entry point
    - resume(state_path) — continue from last checkpoint
    - validate(step) — check preconditions without executing
    - status() — current progress summary

Design Principles (from 鲁棒性增强版工作流):
  - Never lose experimental data (wet-lab results are irreplaceable)
  - Checkpoint after every step (computational or data-entry)
  - Validate inputs before expensive computation
  - Clear error messages with actionable recovery hints
"""

import json
import logging
import os
import time
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Enums & Data Structures ──────────────────────────────────────────────────

class StepStatus(str, Enum):
    PENDING = 'pending'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'
    SKIPPED = 'skipped'
    BLOCKED = 'blocked'          # waiting on upstream step


class RecoveryAction(str, Enum):
    RETRY = 'retry'              # re-run same step with same params
    RETRY_MODIFIED = 'retry_modified'  # re-run with adjusted params
    SKIP = 'skip'                # skip step, continue to next
    ABORT = 'abort'              # stop workflow
    ROLLBACK = 'rollback'        # undo and restart from previous step
    WAIT_USER = 'wait_user'      # pause for user input/correction
    BLOCKED = 'blocked'          # waiting on upstream step completion


@dataclass
class StepRecord:
    """Record of a single step execution within the harness."""
    step_id: str
    step_name: str
    status: str = StepStatus.PENDING
    started_at: str = ''
    completed_at: str = ''
    elapsed_seconds: float = 0.0
    result: Optional[dict] = None
    error: str = ''
    error_traceback: str = ''
    retry_count: int = 0
    max_retries: int = 2
    recovery_action: str = ''
    # Preconditions that were checked
    preconditions_checked: List[str] = field(default_factory=list)
    preconditions_passed: bool = True


@dataclass
class HarnessState:
    """Complete harness state — serializable, persistent."""
    workflow_id: str
    workflow_version: str = ''
    session_id: str = ''

    # Config
    protein_name: str = ''
    output_dir: str = ''
    params: Dict[str, Any] = field(default_factory=dict)

    # Progress
    current_step_idx: int = 0
    steps: List[StepRecord] = field(default_factory=list)
    total_steps: int = 0

    # Timestamps
    created_at: str = ''
    updated_at: str = ''

    # Cumulative
    total_elapsed: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'HarnessState':
        steps_data = data.pop('steps', [])
        state = cls(**{k: v for k, v in data.items()
                       if k in cls.__dataclass_fields__})
        state.steps = []
        for sd in steps_data:
            try:
                state.steps.append(StepRecord(**{
                    k: v for k, v in sd.items()
                    if k in StepRecord.__dataclass_fields__
                }))
            except TypeError:
                state.steps.append(StepRecord(
                    step_id=sd.get('step_id', '?'),
                    step_name=sd.get('step_name', '?'),
                ))
        return state

    def get_step(self, step_id: str) -> Optional[StepRecord]:
        for s in self.steps:
            if s.step_id == step_id:
                return s
        return None


# ── Precondition Validators ──────────────────────────────────────────────────

class PreconditionError(Exception):
    """Raised when a step's preconditions are not met."""
    def __init__(self, message: str, hint: str = '', recovery: RecoveryAction = RecoveryAction.WAIT_USER):
        super().__init__(message)
        self.hint = hint
        self.recovery = recovery


def check_file_exists(path: str, label: str = 'file') -> str:
    """Validate that a file exists. Returns normalized path."""
    if not path:
        raise PreconditionError(
            f'{label} path is empty',
            hint=f'Provide a valid {label} file path',
            recovery=RecoveryAction.WAIT_USER
        )
    if not os.path.exists(path):
        raise PreconditionError(
            f'{label} not found: {path}',
            hint=f'Check the path. CWD={os.getcwd()}',
            recovery=RecoveryAction.WAIT_USER
        )
    return os.path.abspath(path)


def check_step_completed(state: HarnessState, step_id: str) -> None:
    """Validate that a prerequisite step has completed."""
    step = state.get_step(step_id)
    if not step or step.status != StepStatus.COMPLETED:
        raise PreconditionError(
            f'Prerequisite step "{step_id}" not completed',
            hint=f'Run step "{step_id}" first',
            recovery=RecoveryAction.BLOCKED
        )


def check_data_nonempty(data, label: str = 'data') -> None:
    """Validate that data is not empty."""
    if data is None or (isinstance(data, (list, dict, str)) and len(data) == 0):
        raise PreconditionError(
            f'{label} is empty',
            hint=f'Provide non-empty {label}',
            recovery=RecoveryAction.WAIT_USER
        )


# ── Step Definitions for Pareto+AL ───────────────────────────────────────────

# Precondition map: step_id → list of (check_fn, args) tuples
PARETO_PRECONDITIONS = {
    'step1': [],  # No preconditions for objectives
    'step2': [
        # Need either pdb_path or fasta_path
        ('check_protein_input', None),
    ],
    'step3': [
        ('check_step_completed', 'step2'),
    ],
    'step4': [
        ('check_step_completed', 'step3'),
        ('check_screening_data', None),
    ],
    'step5': [
        ('check_step_completed', 'step4'),
    ],
    'step6': [
        ('check_step_completed', 'step5'),
        ('check_new_screening_data', None),
    ],
    'step7': [
        ('check_step_completed', 'step4'),  # At least one round of screening
    ],
}


def _validate_pareto_preconditions(step_id: str, state: HarnessState, params: dict) -> List[str]:
    """Validate preconditions for a Pareto+AL step. Returns list of checked items."""
    checks = PARETO_PRECONDITIONS.get(step_id, [])
    checked = []

    for check_name, check_arg in checks:
        if check_name == 'check_step_completed':
            check_step_completed(state, check_arg)
            checked.append(f'{check_arg} completed')

        elif check_name == 'check_protein_input':
            pdb = params.get('pdb_path', '')
            fasta = params.get('fasta_path', '')
            if not pdb and not fasta:
                raise PreconditionError(
                    'No protein input: need pdb_path or fasta_path',
                    hint='Provide at least one of: pdb_path (PDB file), fasta_path (FASTA file)',
                    recovery=RecoveryAction.WAIT_USER
                )
            if pdb:
                check_file_exists(pdb, 'PDB')
            if fasta:
                check_file_exists(fasta, 'FASTA')
            checked.append('protein input valid')

        elif check_name == 'check_screening_data':
            tier1 = params.get('tier1_data')
            if not tier1:
                raise PreconditionError(
                    'No screening data for Step 4',
                    hint='Provide tier1_data (CSV/JSON path or list of results)',
                    recovery=RecoveryAction.WAIT_USER
                )
            if isinstance(tier1, str) and not os.path.exists(tier1):
                raise PreconditionError(
                    f'Screening data file not found: {tier1}',
                    hint='Check the file path',
                    recovery=RecoveryAction.WAIT_USER
                )
            checked.append('screening data present')

        elif check_name == 'check_new_screening_data':
            new_data = params.get('new_screening_data')
            if not new_data:
                raise PreconditionError(
                    'No new screening data for DBTL iteration',
                    hint='Provide new_screening_data (CSV/JSON path or list)',
                    recovery=RecoveryAction.WAIT_USER
                )
            checked.append('new screening data present')

    return checked


# ── The Harness ──────────────────────────────────────────────────────────────

class WorkflowHarness:
    """Execution harness for multi-step experimental workflows.

    Wraps a workflow with:
      - Precondition validation before each step
      - Automatic state persistence after each step
      - Failure handling with retry/skip/abort options
      - Progress tracking and status reporting
      - Session resumption from checkpoint

    Usage:
        harness = WorkflowHarness('pareto_active_learning')
        result = harness.run('design', params)      # Run steps 1-3
        # ... wet lab experiments ...
        harness = WorkflowHarness.resume(state_path)
        result = harness.run('step4', params)        # Continue
    """

    # Step ordering for the Pareto+AL workflow
    STEP_ORDER = ['step1', 'step2', 'step3', 'step4', 'step5', 'step6', 'step7']
    STEP_NAMES = {
        'step1': 'Define Objectives & Hard Constraints',
        'step2': 'Generate Candidates & Compress Design Space',
        'step3': 'Build First-Round Pareto Library',
        'step4': 'Record Two-Tier Screening Results',
        'step5': 'Surrogate Model Second-Round Proposals',
        'step6': 'Continue DBTL Iterations',
        'step7': 'Final Variant Validation',
    }
    # Steps that are purely computational (no wet-lab data needed)
    DESIGN_STEPS = ['step1', 'step2', 'step3']

    def __init__(self, workflow_id: str = 'pareto_active_learning',
                 output_dir: str = None, state: HarnessState = None):
        self.workflow_id = workflow_id
        self.output_dir = output_dir or os.getcwd()
        os.makedirs(self.output_dir, exist_ok=True)

        if state:
            self.state = state
        else:
            self.state = HarnessState(
                workflow_id=workflow_id,
                output_dir=self.output_dir,
                created_at=datetime.now().isoformat(),
                session_id=f"wf_{int(time.time())}",
                total_steps=len(self.STEP_ORDER),
            )
            # Initialize step records
            for step_id in self.STEP_ORDER:
                self.state.steps.append(StepRecord(
                    step_id=step_id,
                    step_name=self.STEP_NAMES[step_id],
                ))

        self._state_path = os.path.join(self.output_dir, 'harness_state.json')
        self._lock = threading.Lock()

    @classmethod
    def resume(cls, state_path: str) -> 'WorkflowHarness':
        """Resume a workflow from a saved harness state."""
        if not os.path.exists(state_path):
            raise FileNotFoundError(f'Harness state not found: {state_path}')

        with open(state_path, encoding='utf-8') as f:
            data = json.load(f)

        state = HarnessState.from_dict(data)
        output_dir = state.output_dir or os.path.dirname(state_path)

        harness = cls(
            workflow_id=state.workflow_id,
            output_dir=output_dir,
            state=state,
        )
        logger.info("Resumed workflow %s from step %d/%d",
                     state.workflow_id, state.current_step_idx, state.total_steps)
        return harness

    def _save_state(self):
        """Persist harness state to disk."""
        self.state.updated_at = datetime.now().isoformat()
        with self._lock:
            with open(self._state_path, 'w', encoding='utf-8') as f:
                json.dump(self.state.to_dict(), f, ensure_ascii=False, indent=2, default=str)

    def validate(self, step_id: str, params: dict) -> dict:
        """Check preconditions for a step without executing.

        Returns:
            {'valid': True, 'checked': [...]} or
            {'valid': False, 'error': '...', 'hint': '...', 'recovery': '...'}
        """
        try:
            checked = _validate_pareto_preconditions(step_id, self.state, params)
            return {'valid': True, 'checked': checked}
        except PreconditionError as e:
            return {
                'valid': False,
                'error': str(e),
                'hint': e.hint,
                'recovery': e.recovery.value,
            }

    def run(self, step: str, params: dict) -> dict:
        """Execute a workflow step (or sequential group) with full harness.

        Args:
            step: 'step1'-'step7', 'design' (1-3), 'all' (1-3), or step name
            params: Step parameters (merged with stored params)

        Returns:
            Dict with step results, harness status, and next_steps.
        """
        # Merge params into state
        self.state.params.update(params)
        self.state.protein_name = params.get('protein_name', self.state.protein_name)
        self.state.output_dir = params.get('output_dir', self.state.output_dir)
        params['output_dir'] = self.state.output_dir

        # Determine which steps to run
        if step in ('all', 'design'):
            steps_to_run = self.DESIGN_STEPS
        elif step in self.STEP_ORDER:
            steps_to_run = [step]
        else:
            # Try mapping step names
            name_map = {
                'define_objectives': 'step1',
                'generate_design_space': 'step2',
                'build_pareto_library': 'step3',
                'record_screening': 'step4',
                'surrogate_proposal': 'step5',
                'iterate_dbtl': 'step6',
                'final_validation': 'step7',
            }
            mapped = name_map.get(step)
            if mapped:
                steps_to_run = [mapped]
            else:
                return {'error': f'Unknown step: {step}',
                        'available': self.STEP_ORDER + ['design', 'all'],
                        'hint': 'Use step1-step7 or "design" for steps 1-3'}

        # Execute each step with harness protection
        results = []
        for step_id in steps_to_run:
            result = self._execute_step(step_id, params)
            results.append(result)
            if result.get('harness_status') == 'failed':
                break

        # Build summary
        completed = [s for s in self.state.steps if s.status == StepStatus.COMPLETED]
        failed = [s for s in self.state.steps if s.status == StepStatus.FAILED]

        summary = {
            'workflow_id': self.workflow_id,
            'steps_requested': steps_to_run,
            'steps_completed': [s.step_id for s in completed],
            'steps_failed': [s.step_id for s in failed],
            'progress': f"{len(completed)}/{self.state.total_steps}",
            'total_elapsed': round(self.state.total_elapsed, 1),
            'state_path': self._state_path,
            'results': results,
        }

        # Determine next steps
        next_step_idx = max(
            (self.STEP_ORDER.index(s.step_id) + 1
             for s in completed
             if s.step_id in self.STEP_ORDER),
            default=0
        )
        if next_step_idx < len(self.STEP_ORDER):
            next_id = self.STEP_ORDER[next_step_idx]
            next_name = self.STEP_NAMES[next_id]
            if next_id in ('step4', 'step6'):
                summary['next_steps'] = [
                    f'Wet lab: complete experiments, then run {next_id} ({next_name})',
                    f'Resume: WorkflowHarness.resume("{self._state_path}")',
                ]
            else:
                summary['next_steps'] = [f'{next_id}: {next_name}']
        else:
            summary['next_steps'] = ['Workflow complete!']

        if self.state.warnings:
            summary['warnings'] = self.state.warnings

        return summary

    def _execute_step(self, step_id: str, params: dict) -> dict:
        """Execute a single step with validation, timing, and error handling."""
        step_record = self.state.get_step(step_id)
        if not step_record:
            return {'error': f'Step {step_id} not found in harness'}

        # Skip if already completed
        if step_record.status == StepStatus.COMPLETED:
            logger.info("Step %s already completed, skipping", step_id)
            return {
                'step': step_id,
                'harness_status': 'skipped',
                'reason': 'already completed',
            }

        # ── 1. Validate preconditions ────────────────────────────────────
        try:
            checked = _validate_pareto_preconditions(step_id, self.state, params)
            step_record.preconditions_checked = checked
            step_record.preconditions_passed = True
        except PreconditionError as e:
            step_record.status = StepStatus.BLOCKED
            step_record.error = str(e)
            step_record.preconditions_passed = False
            step_record.recovery_action = e.recovery.value
            self._save_state()
            return {
                'step': step_id,
                'harness_status': 'blocked',
                'error': str(e),
                'hint': e.hint,
                'recovery': e.recovery.value,
            }

        # ── 2. Execute with retry loop ───────────────────────────────────
        step_record.status = StepStatus.RUNNING
        step_record.started_at = datetime.now().isoformat()
        self._save_state()

        last_error = None
        for attempt in range(step_record.max_retries + 1):
            step_record.retry_count = attempt
            t0 = time.time()

            try:
                from zhuangzhou.workflow_options.pareto_active_learning import execute_workflow
                step_params = {**params, 'step': step_id}
                result = execute_workflow(step_params)
                elapsed = time.time() - t0

                if 'error' in result:
                    last_error = result['error']
                    logger.warning("Step %s attempt %d failed: %s",
                                   step_id, attempt + 1, last_error)
                    if attempt < step_record.max_retries:
                        time.sleep(1)  # Brief pause before retry
                        continue
                    # All retries exhausted
                    step_record.status = StepStatus.FAILED
                    step_record.error = last_error
                    step_record.elapsed_seconds = elapsed
                    step_record.recovery_action = RecoveryAction.RETRY_MODIFIED.value
                    self._save_state()
                    return {
                        'step': step_id,
                        'harness_status': 'failed',
                        'error': last_error,
                        'hint': result.get('hint', ''),
                        'attempts': attempt + 1,
                        'recovery_options': [
                            RecoveryAction.RETRY.value,
                            RecoveryAction.RETRY_MODIFIED.value,
                            RecoveryAction.SKIP.value,
                            RecoveryAction.ABORT.value,
                        ],
                    }

                # Success
                step_record.status = StepStatus.COMPLETED
                step_record.completed_at = datetime.now().isoformat()
                step_record.elapsed_seconds = elapsed
                step_record.result = _truncate_result(result)
                self.state.total_elapsed += elapsed
                self.state.current_step_idx = self.STEP_ORDER.index(step_id) + 1
                self._save_state()

                return {
                    'step': step_id,
                    'harness_status': 'completed',
                    'elapsed': round(elapsed, 1),
                    'result': result,
                }

            except KeyboardInterrupt:
                elapsed = time.time() - t0
                step_record.status = StepStatus.FAILED
                step_record.error = 'Interrupted by user'
                step_record.elapsed_seconds = elapsed
                step_record.recovery_action = RecoveryAction.RETRY.value
                self._save_state()
                return {
                    'step': step_id,
                    'harness_status': 'interrupted',
                    'elapsed': round(elapsed, 1),
                    'hint': f'Resume with: WorkflowHarness.resume("{self._state_path}").run("{step_id}", params)',
                }

            except Exception as e:
                import traceback
                elapsed = time.time() - t0
                last_error = str(e)
                tb = traceback.format_exc()
                logger.warning("Step %s attempt %d exception: %s", step_id, attempt + 1, e)

                if attempt < step_record.max_retries:
                    time.sleep(1)
                    continue

                step_record.status = StepStatus.FAILED
                step_record.error = last_error
                step_record.error_traceback = tb
                step_record.elapsed_seconds = elapsed
                step_record.recovery_action = RecoveryAction.RETRY.value
                self._save_state()
                return {
                    'step': step_id,
                    'harness_status': 'failed',
                    'error': last_error,
                    'traceback': tb,
                    'attempts': attempt + 1,
                }

        # Should not reach here
        return {'step': step_id, 'harness_status': 'unknown'}

    def status(self) -> dict:
        """Get current workflow progress and status."""
        step_summary = []
        for s in self.state.steps:
            entry = {
                'step_id': s.step_id,
                'name': s.step_name,
                'status': s.status,
            }
            if s.elapsed_seconds > 0:
                entry['elapsed'] = round(s.elapsed_seconds, 1)
            if s.error:
                entry['error'] = s.error[:200]
            if s.retry_count > 0:
                entry['retries'] = s.retry_count
            step_summary.append(entry)

        completed = sum(1 for s in self.state.steps if s.status == StepStatus.COMPLETED)
        return {
            'workflow_id': self.workflow_id,
            'protein': self.state.protein_name,
            'progress': f"{completed}/{self.state.total_steps}",
            'steps': step_summary,
            'total_elapsed': round(self.state.total_elapsed, 1),
            'state_path': self._state_path,
            'session_id': self.state.session_id,
            'created': self.state.created_at,
            'updated': self.state.updated_at,
        }

    def reset_step(self, step_id: str) -> dict:
        """Reset a failed/completed step to pending for re-execution."""
        step = self.state.get_step(step_id)
        if not step:
            return {'error': f'Step {step_id} not found'}
        old_status = step.status
        step.status = StepStatus.PENDING
        step.error = ''
        step.error_traceback = ''
        step.retry_count = 0
        step.result = None
        step.recovery_action = ''
        self._save_state()
        return {'step': step_id, 'old_status': old_status, 'new_status': 'pending'}


def _truncate_result(result: dict, max_list: int = 20) -> dict:
    """Truncate large result dicts for state storage."""
    truncated = {}
    for k, v in result.items():
        if isinstance(v, list) and len(v) > max_list:
            truncated[k] = v[:max_list]
            truncated[f'{k}_truncated'] = True
            truncated[f'{k}_total'] = len(v)
        elif isinstance(v, str) and len(v) > 5000:
            truncated[k] = v[:5000]
        else:
            truncated[k] = v
    return truncated

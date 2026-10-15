"""
Parallel Skill Executor — Concurrent Tool Execution Engine
============================================================
Inspired by Claude Code's tool orchestration architecture:
  1. Each skill declares `concurrent_safe` — safe to run in parallel with others
  2. Tool calls are partitioned into batches: concurrent batches run in parallel,
     non-concurrent tools run serially
  3. Background tasks for long-running skills with progress tracking

Architecture:
  partition_tool_calls() → [(is_concurrent, [tool_calls])]
  ParallelSkillExecutor.execute_all() → ordered results with error isolation
  BackgroundTaskManager → fire-and-forget with status polling

Thread Safety:
  - Each concurrent skill runs in its own thread via ThreadPoolExecutor
  - Results are collected in order (matching tool_use_id to tool_result)
  - Per-tool error isolation: one failure doesn't kill the batch
"""

import concurrent.futures
import copy
import json
import logging
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from zhuangzhou.skills.results import result_error

logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

MAX_CONCURRENT_WORKERS = 5      # Max parallel skill threads
MAX_BACKGROUND_WORKERS = 3      # Max background task threads
TOOL_TIMEOUT = 600              # Per-tool timeout (seconds)

# Skills in these categories default to concurrent_safe=False
CONCURRENT_UNSAFE_CATEGORIES = frozenset({
    'docking',         # Writes files, runs Vina/LigBuilder, shared receptor files
    'simulation',      # GROMACS/AMBER — heavy I/O, file conflicts, long-running
    'mutagenesis',     # Writes mutated PDB files to disk
    'visualization',   # PyMOL/ChimeraX GUI subprocess, file output conflicts
    'report',          # Writes .docx files, needs HuiZi context injection
})

# Individual skills that override category defaults
FORCE_CONCURRENT_SAFE = frozenset({
    # Read-only query skills in otherwise-unsafe categories
    'pocket_detection',  # Just reads PDB, returns dict
})

FORCE_CONCURRENT_UNSAFE = frozenset({
    # State-modifying skills in otherwise-safe categories
    'load_tools',        # Modifies agent's loaded tool set
    'set_analysis_target',  # Modifies HuiZi session state
})


# ─── Batch Partitioning ──────────────────────────────────────────────────────

def is_skill_concurrent_safe(skill) -> bool:
    """Determine if a skill is safe to run concurrently.

    Priority:
      1. Explicit override lists (FORCE_CONCURRENT_SAFE / FORCE_CONCURRENT_UNSAFE)
      2. Skill's own concurrent_safe attribute (if explicitly set)
      3. Category-based default
    """
    if skill is None:
        return False  # Unknown tool → serial (safe default)

    name = skill.name

    # Override lists
    if name in FORCE_CONCURRENT_UNSAFE:
        return False
    if name in FORCE_CONCURRENT_SAFE:
        return True

    # Explicit skill-level override
    if getattr(skill, 'concurrent_safe', None) is not None:
        return skill.concurrent_safe

    if any(getattr(skill, flag, False) for flag in ('requires_gpu', 'requires_wsl', 'remote_preferred')):
        return False

    # Category-based default
    return skill.category not in CONCURRENT_UNSAFE_CATEGORIES


def partition_tool_calls(tool_calls, registry) -> List[Tuple[bool, list]]:
    """Partition tool calls into (is_concurrent, [calls]) batches.

    Consecutive concurrent-safe tools form one parallel batch.
    Each non-safe tool forms its own serial batch.

    Example:
      Input:  [search_uniprot, pubmed_search, esm2_scan, smart_docking, generate_report]
      Output: [(True,  [search_uniprot, pubmed_search, esm2_scan]),
               (False, [smart_docking]),
               (False, [generate_report])]
    """
    if not tool_calls:
        return []

    batches: List[Tuple[bool, list]] = []
    current_concurrent: list = []

    for tc in tool_calls:
        skill = registry.get(tc.name)
        is_safe = is_skill_concurrent_safe(skill)

        if is_safe:
            current_concurrent.append(tc)
        else:
            # Flush any pending concurrent batch
            if current_concurrent:
                batches.append((True, current_concurrent))
                current_concurrent = []
            batches.append((False, [tc]))

    # Flush trailing concurrent batch
    if current_concurrent:
        batches.append((True, current_concurrent))

    return batches


# ─── Batch Execution ─────────────────────────────────────────────────────────

def run_batch_concurrent(
    batch: list,
    executor_fn: Callable,
    max_workers: int = MAX_CONCURRENT_WORKERS,
) -> List[Tuple[Any, dict]]:
    """Run concurrent-safe tool calls in parallel, preserving order.

    Args:
        batch: List of ToolCallRequest objects.
        executor_fn: Callable(name, params) -> dict
        max_workers: Maximum parallel threads.

    Returns:
        List of (ToolCallRequest, result_dict) in original order.
        Per-tool error isolation: one failure returns error dict, others continue.
    """
    if not batch:
        return []
    if len(batch) == 1:
        # Single call — skip thread pool overhead
        tc = batch[0]
        return run_batch_serial(batch, executor_fn)

    n_workers = min(max_workers, len(batch))
    results: List[Optional[Tuple[Any, dict]]] = [None] * len(batch)

    logger.info("Parallel batch: %d skills × %d workers — %s",
                len(batch), n_workers,
                ', '.join(tc.name for tc in batch))

    with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix='zz-par') as pool:
        future_to_idx = {}
        for i, tc in enumerate(batch):
            future = pool.submit(executor_fn, tc.name, tc.arguments)
            future_to_idx[future] = i

        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            tc = batch[idx]
            try:
                result = future.result()
                results[idx] = (tc, result)
            except Exception as e:
                logger.warning("Parallel skill '%s' raised exception: %s", tc.name, e)
                results[idx] = (tc, {
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                })

    return results


def run_batch_serial(
    batch: list,
    executor_fn: Callable,
) -> List[Tuple[Any, dict]]:
    """Run tool calls sequentially (for non-concurrent-safe skills)."""
    results = []
    for tc in batch:
        try:
            result = executor_fn(tc.name, tc.arguments)
        except Exception as e:
            result = {'error': str(e), 'traceback': traceback.format_exc()}
        results.append((tc, result))
    return results


# ─── Parallel Skill Executor ─────────────────────────────────────────────────

class ParallelSkillExecutor:
    """Orchestrates parallel + serial execution of tool call batches.

    Usage in agent.py:
        executor = ParallelSkillExecutor(registry, agent._execute_skill_core)
        results, has_error, next_steps = executor.execute_all(tool_calls, monitor)

    The executor:
      1. Partitions tool_calls into concurrent/serial batches
      2. Runs concurrent batches in parallel via ThreadPoolExecutor
      3. Runs serial batches one-by-one
      4. Collects results in original order (preserving tool_use_id matching)
      5. Isolates errors per-tool (one failure doesn't kill others)
    """

    def __init__(self, registry, executor_fn: Callable,
                 max_workers: int = MAX_CONCURRENT_WORKERS):
        """
        Args:
            registry: SkillRegistry instance.
            executor_fn: Callable(name: str, params: dict) -> dict.
                         This should handle preflight, timeout, logging, etc.
            max_workers: Max parallel threads for concurrent batches.
        """
        self.registry = registry
        self.executor_fn = executor_fn
        self.max_workers = max_workers
        self._stats = {'parallel_batches': 0, 'serial_batches': 0,
                       'parallel_skills': 0, 'serial_skills': 0,
                       'total_time_saved_est': 0.0}

    def execute_all(
        self,
        tool_calls: list,
        monitor=None,
    ) -> Tuple[List[Tuple[Any, dict]], bool, list]:
        """Execute all tool calls with optimal parallelism.

        Args:
            tool_calls: List of ToolCallRequest from LLM response.
            monitor: Optional UI monitor for status updates.

        Returns:
            (all_results, batch_has_error, pending_next_steps)
            - all_results: [(ToolCallRequest, result_dict)] in original order
            - batch_has_error: True if any tool returned an error
            - pending_next_steps: Collected next_steps hints from all results
        """
        if not tool_calls:
            return [], False, []

        # Single tool call → fast path, no partitioning overhead
        if len(tool_calls) == 1:
            tc = tool_calls[0]
            result = run_batch_serial([tc], self.executor_fn)[0][1]
            has_error = result_error(result) is not None
            next_steps = (result.get('next_steps', [])
                          if isinstance(result, dict) else [])
            self._stats['serial_batches'] += 1
            self._stats['serial_skills'] += 1
            return [(tc, result)], has_error, next_steps

        batches = partition_tool_calls(tool_calls, self.registry)

        all_results: List[Tuple[Any, dict]] = []
        batch_has_error = False
        pending_next_steps: list = []

        for is_concurrent, batch in batches:
            if is_concurrent and len(batch) > 1:
                # ── Parallel batch ────────────────────────────────────────
                names = ', '.join(tc.name for tc in batch)
                if monitor:
                    monitor.update(action=f"⚡ parallel({len(batch)}): {names}",
                                   phase="act")
                logger.info("Running %d skills in parallel: %s", len(batch), names)

                t0 = time.time()
                results = run_batch_concurrent(
                    batch, self.executor_fn, self.max_workers
                )
                elapsed = time.time() - t0

                self._stats['parallel_batches'] += 1
                self._stats['parallel_skills'] += len(batch)

                logger.info("Parallel batch completed in %.1fs (%d skills)", elapsed, len(batch))
            else:
                # ── Serial batch ──────────────────────────────────────────
                results = run_batch_serial(batch, self.executor_fn)
                self._stats['serial_batches'] += 1
                self._stats['serial_skills'] += len(batch)

            # Collect results
            for tc, result in results:
                if isinstance(result, dict):
                    if result_error(result):
                        batch_has_error = True
                    if result.get('next_steps'):
                        pending_next_steps.extend(result['next_steps'])
                all_results.append((tc, result))

        # Deduplicate next_steps while preserving order
        pending_next_steps = list(dict.fromkeys(pending_next_steps))

        return all_results, batch_has_error, pending_next_steps

    @property
    def stats(self) -> dict:
        """Execution statistics for monitoring."""
        return dict(self._stats)


# ─── Background Task Manager ─────────────────────────────────────────────────

@dataclass
class BackgroundTask:
    """A long-running skill executing in the background."""
    task_id: str
    skill_name: str
    params: dict
    future: concurrent.futures.Future
    start_time: float
    description: str = ''
    estimated_seconds: float = 0.0

    @property
    def is_done(self) -> bool:
        return self.future.done()

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def status(self) -> str:
        if self.future.cancelled():
            return 'cancelled'
        if self.future.done():
            try:
                result = self.future.result()
                return 'failed' if result_error(result) else 'completed'
            except Exception:
                return 'failed'
        return 'running'

    def to_dict(self) -> dict:
        return {
            'task_id': self.task_id,
            'skill': self.skill_name,
            'description': self.description,
            'status': self.status,
            'elapsed': round(self.elapsed, 1),
            'estimated_seconds': self.estimated_seconds,
        }


def parameters_match(first: Any, second: Any) -> bool:
    """Match JSON inputs exactly; Python equality conflates bools and numbers."""
    if type(first) is not type(second):
        return False
    if type(first) is dict:
        return (all(type(key) is str for key in first)
                and first.keys() == second.keys()
                and all(parameters_match(value, second[key]) for key, value in first.items()))
    if type(first) is list:
        return len(first) == len(second) and all(parameters_match(a, b) for a, b in zip(first, second))
    if type(first) in (str, int, float, bool, type(None)):
        return first == second
    return False


class BackgroundTaskManager:
    """Manages background tasks with progress tracking.

    Usage:
        bg = BackgroundTaskManager()
        task_id = bg.submit('auto_md', params, executor_fn, estimated_seconds=300)
        ...later...
        status = bg.check_status(task_id)
        result = bg.collect_result(task_id)
    """

    def __init__(self, max_workers: int = MAX_BACKGROUND_WORKERS, max_pending: int = 32):
        if type(max_pending) is not int or max_pending < 1:
            raise ValueError('max_pending must be a positive integer')
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix='zz-bg'
        )
        self._tasks: Dict[str, BackgroundTask] = {}
        self._lock = threading.Lock()
        self.max_pending = max_pending

    def submit(
        self,
        skill_name: str,
        params: dict,
        executor_fn: Callable,
        description: str = '',
        estimated_seconds: float = 60.0,
        exclusive: bool = False,
    ) -> str:
        """Submit a skill for background execution.

        Args:
            skill_name: Skill to execute.
            params: Skill parameters.
            executor_fn: Callable(name, params) -> dict.
            description: Human-readable task description.
            estimated_seconds: Estimated run time (for progress display).
            exclusive: Reuse only an identical, unfinished invocation.

        Returns:
            task_id: Unique identifier for the background task.
        """
        task_id = f"bg-{skill_name}-{uuid.uuid4().hex}"
        params = copy.deepcopy(params)
        with self._lock:
            if exclusive:
                for task in self._tasks.values():
                    if (task.skill_name == skill_name and not task.is_done
                            and parameters_match(task.params, params)):
                        return task.task_id
            if len(self._tasks) >= self.max_pending:
                raise RuntimeError('Background task capacity reached; collect completed results before submitting')
            future = self._pool.submit(executor_fn, skill_name, params)
            self._tasks[task_id] = BackgroundTask(
                task_id=task_id, skill_name=skill_name, params=params, future=future,
                start_time=time.time(), description=description or f'Background: {skill_name}',
                estimated_seconds=estimated_seconds,
            )

        logger.info("Background task submitted: %s (%s)", task_id, skill_name)
        return task_id

    def wait_result(self, task_id: str, timeout: float) -> dict:
        """Bound the caller's wait; running Python threads cannot be forcibly killed."""
        with self._lock:
            task = self._tasks.get(task_id)
        if task is None:
            return {'error': f'No task with id: {task_id}'}
        try:
            task.future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return {'error': f'Tool wait timeout after {timeout}s; execution remains tracked',
                    'status': 'running', 'task_id': task_id, 'may_still_be_running': True,
                    'hint': 'Poll check_background_tasks and collect_background_result; do not restart this work'}
        except Exception:
            pass  # collect_result preserves the worker exception as a diagnostic.
        return self.collect_result(task_id)

    def check_status(self, task_id: str = None) -> Any:
        """Check status of one or all background tasks.

        Args:
            task_id: Specific task to check, or None for all.

        Returns:
            Single task dict or list of task dicts.
        """
        with self._lock:
            if task_id:
                task = self._tasks.get(task_id)
                if not task:
                    return {'error': f'No task with id: {task_id}'}
                return task.to_dict()
            return [t.to_dict() for t in self._tasks.values()]

    def collect_result(self, task_id: str) -> dict:
        """Retrieve the result of a completed background task.

        If the task is still running, returns a status dict.
        Once collected, the task is removed from tracking.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return {'error': f'No task with id: {task_id}'}

            if not task.is_done:
                return {
                    'status': 'running',
                    'elapsed': round(task.elapsed, 1),
                    'estimated_seconds': task.estimated_seconds,
                    'hint': f'Task still running (~{max(0, task.estimated_seconds - task.elapsed):.0f}s remaining)',
                }

            # Task is done — extract result and clean up
            try:
                result = task.future.result()
            except Exception as e:
                result = {'error': str(e)}

            del self._tasks[task_id]

        return result

    def get_completed_ids(self) -> List[str]:
        """Get IDs of tasks that have completed and are ready for collection."""
        with self._lock:
            return [tid for tid, t in self._tasks.items() if t.is_done]

    def cancel(self, task_id: str) -> dict:
        """Cancel a running background task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return {'error': f'No task with id: {task_id}'}
            if task.is_done:
                return {'status': 'already_completed'}
        # Future callbacks can run synchronously during cancel; never hold our lock.
        cancelled = task.future.cancel()
        if cancelled:
            with self._lock:
                self._tasks.pop(task_id, None)
        return {'cancelled': cancelled, 'task_id': task_id,
                'reason': 'cancelled_before_start' if cancelled else 'already_running'}

    def shutdown(self, wait: bool = False):
        """Shutdown the background thread pool."""
        self._pool.shutdown(wait=wait, cancel_futures=not wait)

    @property
    def active_count(self) -> int:
        """Number of currently running background tasks."""
        with self._lock:
            return sum(1 for t in self._tasks.values() if not t.is_done)

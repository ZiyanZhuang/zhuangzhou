"""
Tests for parallel skill execution engine (nlp/parallel.py)
============================================================
Validates:
  1. Batch partitioning logic (concurrent vs serial grouping)
  2. Parallel execution with order preservation
  3. Per-tool error isolation
  4. Background task manager
  5. Thread safety
"""

import sys
import time
import threading
import unittest
from pathlib import Path
from dataclasses import dataclass
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from zhuangzhou.nlp.parallel import (
    partition_tool_calls,
    run_batch_concurrent,
    run_batch_serial,
    ParallelSkillExecutor,
    BackgroundTaskManager,
    is_skill_concurrent_safe,
)


# ── Mock Objects ──────────────────────────────────────────────────────────────

@dataclass
class MockToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class MockSkill:
    name: str
    category: str
    concurrent_safe: bool = None


class MockRegistry:
    def __init__(self, skills: dict):
        self._skills = skills

    def get(self, name):
        return self._skills.get(name)


# ── Test Fixtures ─────────────────────────────────────────────────────────────

def make_registry():
    """Create a mock registry with mixed safe/unsafe skills."""
    return MockRegistry({
        'search_uniprot': MockSkill('search_uniprot', 'research'),
        'pubmed_search': MockSkill('pubmed_search', 'research'),
        'pdb_to_fasta': MockSkill('pdb_to_fasta', 'structure'),
        'esm2_scan': MockSkill('esm2_scan', 'plm'),
        'smart_docking': MockSkill('smart_docking', 'docking'),
        'generate_report': MockSkill('generate_report', 'report'),
        'auto_md': MockSkill('auto_md', 'simulation'),
        'mutate_protein': MockSkill('mutate_protein', 'mutagenesis'),
        'pocket_detection': MockSkill('pocket_detection', 'structure'),
        # Explicit override: docking but read-only
        'docking_score_only': MockSkill('docking_score_only', 'docking', concurrent_safe=True),
    })


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestConcurrencySafety(unittest.TestCase):
    """Test skill concurrent_safe classification."""

    def test_research_skills_safe(self):
        s = MockSkill('search_uniprot', 'research')
        self.assertTrue(is_skill_concurrent_safe(s))

    def test_docking_skills_unsafe(self):
        s = MockSkill('smart_docking', 'docking')
        self.assertFalse(is_skill_concurrent_safe(s))

    def test_simulation_skills_unsafe(self):
        s = MockSkill('auto_md', 'simulation')
        self.assertFalse(is_skill_concurrent_safe(s))

    def test_explicit_override_safe(self):
        s = MockSkill('docking_score_only', 'docking', concurrent_safe=True)
        self.assertTrue(is_skill_concurrent_safe(s))

    def test_explicit_override_unsafe(self):
        s = MockSkill('special_research', 'research', concurrent_safe=False)
        self.assertFalse(is_skill_concurrent_safe(s))

    def test_unknown_skill_unsafe(self):
        self.assertFalse(is_skill_concurrent_safe(None))

    def test_force_unsafe_override(self):
        s = MockSkill('load_tools', 'research')
        self.assertFalse(is_skill_concurrent_safe(s))


class TestBatchPartitioning(unittest.TestCase):
    """Test partition_tool_calls grouping logic."""

    def setUp(self):
        self.registry = make_registry()

    def test_all_concurrent(self):
        tcs = [
            MockToolCall('1', 'search_uniprot', {}),
            MockToolCall('2', 'pubmed_search', {}),
            MockToolCall('3', 'pdb_to_fasta', {}),
        ]
        batches = partition_tool_calls(tcs, self.registry)
        self.assertEqual(len(batches), 1)
        self.assertTrue(batches[0][0])  # is_concurrent
        self.assertEqual(len(batches[0][1]), 3)

    def test_all_serial(self):
        tcs = [
            MockToolCall('1', 'smart_docking', {}),
            MockToolCall('2', 'generate_report', {}),
        ]
        batches = partition_tool_calls(tcs, self.registry)
        self.assertEqual(len(batches), 2)
        self.assertFalse(batches[0][0])
        self.assertFalse(batches[1][0])

    def test_mixed_batches(self):
        """[safe, safe, unsafe, safe, unsafe] → 4 batches."""
        tcs = [
            MockToolCall('1', 'search_uniprot', {}),
            MockToolCall('2', 'pubmed_search', {}),
            MockToolCall('3', 'smart_docking', {}),
            MockToolCall('4', 'pdb_to_fasta', {}),
            MockToolCall('5', 'generate_report', {}),
        ]
        batches = partition_tool_calls(tcs, self.registry)
        self.assertEqual(len(batches), 4)
        # Batch 1: concurrent [search_uniprot, pubmed_search]
        self.assertTrue(batches[0][0])
        self.assertEqual(len(batches[0][1]), 2)
        # Batch 2: serial [smart_docking]
        self.assertFalse(batches[1][0])
        # Batch 3: concurrent [pdb_to_fasta] (single-element)
        self.assertTrue(batches[2][0])
        # Batch 4: serial [generate_report]
        self.assertFalse(batches[3][0])

    def test_empty_input(self):
        self.assertEqual(partition_tool_calls([], self.registry), [])

    def test_single_tool(self):
        tcs = [MockToolCall('1', 'search_uniprot', {})]
        batches = partition_tool_calls(tcs, self.registry)
        self.assertEqual(len(batches), 1)

    def test_unknown_tool_serial(self):
        tcs = [MockToolCall('1', 'nonexistent_tool', {})]
        batches = partition_tool_calls(tcs, self.registry)
        self.assertEqual(len(batches), 1)
        self.assertFalse(batches[0][0])  # Unknown → serial


class TestBatchExecution(unittest.TestCase):
    """Test parallel and serial batch execution."""

    def _slow_executor(self, name, params):
        """Simulates a 0.2s skill that records thread ID."""
        time.sleep(0.2)
        return {'skill': name, 'thread': threading.current_thread().name}

    def _failing_executor(self, name, params):
        if name == 'fail_skill':
            raise RuntimeError("Intentional failure")
        return {'skill': name, 'status': 'ok'}

    def test_concurrent_order_preserved(self):
        """Results must be in the same order as input, regardless of completion order."""
        batch = [
            MockToolCall(str(i), f'skill_{i}', {})
            for i in range(5)
        ]
        results = run_batch_concurrent(batch, self._slow_executor, max_workers=5)
        self.assertEqual(len(results), 5)
        for i, (tc, res) in enumerate(results):
            self.assertEqual(tc.name, f'skill_{i}')

    def test_concurrent_actually_parallel(self):
        """5 × 0.2s skills should complete in ~0.2s (not 1.0s) when run in parallel."""
        batch = [MockToolCall(str(i), f's{i}', {}) for i in range(5)]
        t0 = time.time()
        results = run_batch_concurrent(batch, self._slow_executor, max_workers=5)
        elapsed = time.time() - t0
        self.assertLess(elapsed, 0.8, f"Parallel batch took {elapsed:.2f}s, expected < 0.8s")
        self.assertEqual(len(results), 5)

    def test_concurrent_error_isolation(self):
        """One failing skill shouldn't prevent others from completing."""
        batch = [
            MockToolCall('1', 'good_skill', {}),
            MockToolCall('2', 'fail_skill', {}),
            MockToolCall('3', 'good_skill', {}),
        ]
        results = run_batch_concurrent(batch, self._failing_executor, max_workers=3)
        self.assertEqual(len(results), 3)
        # First and third succeed
        self.assertEqual(results[0][1]['status'], 'ok')
        self.assertIn('error', results[1][1])  # Second failed
        self.assertEqual(results[2][1]['status'], 'ok')

    def test_serial_order_preserved(self):
        batch = [MockToolCall(str(i), f's{i}', {}) for i in range(3)]
        results = run_batch_serial(batch, self._slow_executor)
        self.assertEqual(len(results), 3)
        for i, (tc, _) in enumerate(results):
            self.assertEqual(tc.name, f's{i}')

    def test_single_item_no_threadpool(self):
        """Single-item batch should skip ThreadPoolExecutor for efficiency."""
        batch = [MockToolCall('1', 'solo', {})]
        results = run_batch_concurrent(batch, self._slow_executor, max_workers=5)
        self.assertEqual(len(results), 1)
        # Should run on current thread (no 'zz-par' prefix)
        self.assertNotIn('zz-par', results[0][1]['thread'])


class TestParallelSkillExecutor(unittest.TestCase):
    """Test the full ParallelSkillExecutor pipeline."""

    def setUp(self):
        self.registry = make_registry()
        self.call_log = []
        self.log_lock = threading.Lock()

    def _tracking_executor(self, name, params):
        t0 = time.time()
        time.sleep(0.1)  # Simulate work
        elapsed = time.time() - t0
        entry = {'skill': name, 'thread': threading.current_thread().name, 'elapsed': elapsed}
        with self.log_lock:
            self.call_log.append(entry)
        return {'status': 'ok', 'skill': name}

    def test_mixed_execution(self):
        """Mixed tool calls: parallel batch + serial + parallel."""
        tcs = [
            MockToolCall('1', 'search_uniprot', {}),
            MockToolCall('2', 'pubmed_search', {}),
            MockToolCall('3', 'smart_docking', {}),
            MockToolCall('4', 'pdb_to_fasta', {}),
            MockToolCall('5', 'esm2_scan', {}),
        ]
        executor = ParallelSkillExecutor(self.registry, self._tracking_executor, max_workers=5)
        results, has_error, next_steps = executor.execute_all(tcs)

        self.assertFalse(has_error)
        self.assertEqual(len(results), 5)
        # Verify order preserved
        for i, (tc, res) in enumerate(results):
            self.assertEqual(tc.id, str(i + 1))

        # Verify stats
        stats = executor.stats
        self.assertGreater(stats['parallel_skills'], 0)

    def test_single_tool_fast_path(self):
        tcs = [MockToolCall('1', 'search_uniprot', {})]
        executor = ParallelSkillExecutor(self.registry, self._tracking_executor)
        results, has_error, _ = executor.execute_all(tcs)
        self.assertEqual(len(results), 1)
        self.assertFalse(has_error)

    def test_next_steps_collection(self):
        def executor_with_steps(name, params):
            return {'status': 'ok', 'next_steps': [f'step_from_{name}']}

        tcs = [
            MockToolCall('1', 'search_uniprot', {}),
            MockToolCall('2', 'pubmed_search', {}),
        ]
        executor = ParallelSkillExecutor(self.registry, executor_with_steps)
        _, _, next_steps = executor.execute_all(tcs)
        self.assertEqual(len(next_steps), 2)

    def test_error_detection(self):
        def error_executor(name, params):
            if name == 'search_uniprot':
                return {'error': 'API down'}
            return {'status': 'ok'}

        tcs = [
            MockToolCall('1', 'search_uniprot', {}),
            MockToolCall('2', 'pubmed_search', {}),
        ]
        executor = ParallelSkillExecutor(self.registry, error_executor)
        _, has_error, _ = executor.execute_all(tcs)
        self.assertTrue(has_error)


class TestBackgroundTaskManager(unittest.TestCase):
    """Test background task lifecycle."""

    def test_submit_and_collect(self):
        bg = BackgroundTaskManager(max_workers=2)
        try:
            def executor(name, params):
                time.sleep(0.2)
                return {'result': f'{name}_done'}

            tid = bg.submit('slow_skill', {}, executor, estimated_seconds=1.0)
            self.assertIn('bg-slow_skill', tid)

            # Check status while running
            status = bg.check_status(tid)
            self.assertEqual(status['status'], 'running')

            # Wait for completion
            time.sleep(0.5)
            result = bg.collect_result(tid)
            self.assertEqual(result['result'], 'slow_skill_done')

            # After collection, task is removed
            status2 = bg.check_status(tid)
            self.assertIn('error', status2)
        finally:
            bg.shutdown(wait=True)

    def test_completed_ids(self):
        bg = BackgroundTaskManager(max_workers=2)
        try:
            def fast(name, params):
                return {'done': True}

            tid = bg.submit('fast', {}, fast)
            time.sleep(0.2)
            completed = bg.get_completed_ids()
            self.assertIn(tid, completed)
        finally:
            bg.shutdown(wait=True)

    def test_error_handling(self):
        bg = BackgroundTaskManager(max_workers=1)
        try:
            def failing(name, params):
                raise ValueError("boom")

            tid = bg.submit('fail', {}, failing)
            time.sleep(0.3)
            result = bg.collect_result(tid)
            self.assertIn('error', result)
        finally:
            bg.shutdown(wait=True)


class TestThreadSafety(unittest.TestCase):
    """Verify thread safety under concurrent access."""

    def test_concurrent_executor_calls(self):
        """Multiple threads calling the executor simultaneously."""
        registry = make_registry()
        counter = {'value': 0}
        lock = threading.Lock()

        def counting_executor(name, params):
            with lock:
                counter['value'] += 1
            time.sleep(0.05)
            return {'count': counter['value']}

        executor = ParallelSkillExecutor(registry, counting_executor, max_workers=10)
        tcs = [MockToolCall(str(i), 'search_uniprot', {}) for i in range(10)]
        results, has_error, _ = executor.execute_all(tcs)

        self.assertEqual(len(results), 10)
        self.assertFalse(has_error)
        self.assertEqual(counter['value'], 10)


if __name__ == '__main__':
    unittest.main(verbosity=2)

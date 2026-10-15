"""Failure injection for real thread orchestration; all tasks are tiny fixtures."""

import threading
import subprocess
import sys
from unittest.mock import Mock, patch

import pytest

from zhuangzhou.nlp.backends import ToolCallRequest
from zhuangzhou.nlp.parallel import (
    BackgroundTaskManager, ParallelSkillExecutor, run_batch_concurrent, run_batch_serial,
)


def call(name="test"):
    return ToolCallRequest("1", name, {})


def fail(*args):
    raise ValueError("injected failure")


@pytest.mark.parametrize("runner,batch", [
    (run_batch_concurrent, []), (run_batch_concurrent, [call()]), (run_batch_serial, [call()]),
])
def test_empty_and_exception_batches_are_isolated(runner, batch):
    results = runner(batch, fail)
    assert len(results) == len(batch)
    if batch:
        assert "injected failure" in results[0][1]["error"]


def test_explicit_failure_propagates_to_agent_batch():
    executor = ParallelSkillExecutor(Mock(), lambda *a: {"success": False})
    _, has_error, _ = executor.execute_all([call()])
    assert has_error is True


def test_background_ids_do_not_collide_within_millisecond():
    manager = BackgroundTaskManager(max_workers=1)
    try:
        with patch("zhuangzhou.nlp.parallel.time.time", return_value=1000):
            ids = [manager.submit("same", {}, lambda *a: {}) for _ in range(8)]
        assert len(set(ids)) == 8
        assert len(manager.check_status()) == 8
    finally:
        manager.shutdown(wait=True)


def test_failed_result_is_failed_status():
    manager = BackgroundTaskManager(max_workers=1)
    done = threading.Event()
    try:
        task_id = manager.submit("fail", {}, lambda *a: {"success": False})
        manager._tasks[task_id].future.add_done_callback(lambda _: done.set())
        assert done.wait(2)
        assert manager.check_status(task_id)["status"] == "failed"
    finally:
        manager.shutdown(wait=True)


def test_background_capacity_is_bounded():
    manager = BackgroundTaskManager(max_workers=1, max_pending=2)
    release = threading.Event()
    try:
        manager.submit("one", {}, lambda *a: release.wait(2))
        manager.submit("two", {}, lambda *a: {})
        with pytest.raises(RuntimeError, match="capacity"):
            manager.submit("three", {}, lambda *a: {})
    finally:
        release.set()
        manager.shutdown(wait=True)


def test_queued_cancel_does_not_deadlock_or_leave_cache():
    # A real deadlock must be contained in a killable child process.
    code = '''
import threading
from zhuangzhou.nlp.parallel import BackgroundTaskManager
manager = BackgroundTaskManager(max_workers=1)
release, started = threading.Event(), threading.Event()
def wait(*args):
    started.set()
    release.wait(3)
    return {}
manager.submit("running", {}, wait)
assert started.wait(2)
queued = manager.submit("queued", {}, lambda *a: {})
assert manager.cancel(queued)["cancelled"] is True
assert queued not in getattr(manager, '_completed_results', {})
release.set()
manager.shutdown(wait=True)
'''
    subprocess.run([sys.executable, "-c", code], check=True, timeout=5, capture_output=True)


def test_submitted_parameters_are_a_snapshot():
    manager = BackgroundTaskManager(max_workers=1)
    release = threading.Event()
    try:
        manager.submit("first", {}, lambda *a: release.wait(2))
        params = {"nested": [1]}
        task_id = manager.submit("second", params, lambda name, p: p)
        params["nested"].append(2)
        release.set()
        manager.shutdown(wait=True)
        assert manager.collect_result(task_id) == {"nested": [1]}
    finally:
        release.set()
        manager.shutdown(wait=True)

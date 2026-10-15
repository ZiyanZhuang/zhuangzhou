"""Real agent orchestration with explicit model stubs; no scientific simulation."""

from unittest.mock import Mock, patch
import concurrent.futures
import threading
import json

import pytest

from zhuangzhou.nlp.agent import AgentExecutor
from zhuangzhou.nlp.backends import LLMBackend, LLMResponse, ToolCallRequest
from zhuangzhou.skills.registry import Skill, SkillRegistry


@pytest.fixture
def executor():
    registry = SkillRegistry()
    registry._loaded = True
    with patch("zhuangzhou.nlp.agent.ToolRouter", return_value=None):
        agent = AgentExecutor(LLMBackend(), registry, verbose=True)
    agent._huizi_enabled = False
    yield agent
    agent._bg_tasks.shutdown(wait=True)


def register(executor, output=None, fn=None, risk="low"):
    fn = fn or Mock(return_value=output or {"value": 3})
    executor.registry.register(Skill("calculate", "local deterministic fixture", "chemistry", fn, risk_level=risk))
    return fn


@pytest.mark.parametrize("output,success", [({"value": 1}, True), ({"success": False}, False), ({"error": "failed"}, False)])
def test_history_and_memory_agree(executor, output, success):
    register(executor, output)
    executor._execute_skill_with_ui("calculate", {})
    assert len(executor.execution_log) == 1
    assert executor.execution_log[0]["success"] is success
    assert executor._exec_history.get_recent(limit=1)[0]["success"] is success
    assert "calculate" in executor.build_system_prompt()


def test_high_risk_requires_explicit_one_shot_confirmation(executor):
    fn = register(executor, risk="high")
    first = executor._execute_skill_with_ui("calculate", {"value": 4})
    assert first.get("needs_confirmation") is True
    fn.assert_not_called()
    result = executor.confirm_action("calculate")
    assert result == {"value": 3}
    fn.assert_called_once_with(value=4)
    again = executor._execute_skill_with_ui("calculate", {"value": 5})
    assert again.get("needs_confirmation") is True
    assert fn.call_count == 1


def test_model_cannot_self_confirm_via_tool(executor):
    register(executor, risk="high")
    executor._execute_skill_with_ui("calculate", {})
    assert "error" in executor._execute_skill_with_ui("confirm_action", {"name": "calculate"})


def test_invalid_tool_arguments_do_not_dispatch(executor):
    fn = register(executor)
    result = executor._execute_skill_with_ui("calculate", ["bad"])
    assert "error" in result
    fn.assert_not_called()


def test_missing_tool_lists_real_alternatives(executor):
    register(executor)
    result = executor._execute_skill_with_ui("missing", {})
    assert "calculate" in result["available_skills"]


def test_iteration_exhaustion_saves_real_completed_steps(executor):
    register(executor)
    response = LLMResponse(tool_calls=[ToolCallRequest("id1", "calculate", {})])
    with patch.object(executor, "_stream_llm_response", return_value=response):
        assert "Warning" in executor.run_loop("test fixture computation", max_iterations=1)
    checkpoint = executor._checkpoint.load()
    assert checkpoint["status"] == "interrupted"
    assert checkpoint["completed_steps"][0]["skill"] == "calculate"


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_invalid_iteration_budget_rejected(executor, limit):
    with pytest.raises(ValueError):
        executor.run_loop("do nothing", max_iterations=limit)


def test_slash_confirmation_is_wired(executor):
    from zhuangzhou.nlp.slash import manager
    fn = register(executor, risk="high")
    executor._execute_skill_with_ui("calculate", {})
    context = Mock(executor=executor)
    assert manager.dispatch("/confirm calculate", context) is True
    fn.assert_called_once()


def test_timeout_returns_trackable_task_without_waiting_for_worker(executor, monkeypatch):
    release = threading.Event()
    timer = threading.Timer(0.5, release.set)
    original = concurrent.futures.Future.result
    def timed_result(future, timeout=None):
        if timeout is not None:
            raise concurrent.futures.TimeoutError()
        return original(future)
    monkeypatch.setattr(concurrent.futures.Future, 'result', timed_result)
    register(executor, fn=lambda **kw: (release.wait(2), {"value": 1})[1])
    timer.start()
    try:
        result = executor._execute_skill_with_ui("calculate", {})
        assert not release.is_set(), "timeout handler waited for the worker anyway"
        assert result["status"] == "running"
        assert result["task_id"]
        assert result["may_still_be_running"] is True
    finally:
        release.set()
        timer.cancel()


def test_logging_failure_does_not_erase_execution_or_trigger_retry(executor):
    fn = register(executor)
    with patch.object(executor._exec_history, 'record', side_effect=OSError('disk full')):
        result = executor._execute_skill_with_ui('calculate', {})
    fn.assert_called_once()
    assert result['value'] == 3
    assert 'persistence_warning' in result
    assert executor.execution_log[-1]['success'] is True


@pytest.mark.parametrize('response', [LLMResponse(text='API failed', stop_reason='error'),
    LLMResponse(text='truncated', stop_reason='length'), LLMResponse()])
def test_incomplete_model_turn_remains_resumable(executor, response):
    with patch.object(executor, '_stream_llm_response', return_value=response):
        executor.run_loop('incomplete model fixture', max_iterations=1)
    assert executor._checkpoint.has_pending()


def test_malformed_json_tool_call_cannot_dispatch(executor):
    fn = register(executor)
    result = executor._execute_skill_with_ui('calculate', {'_parse_error': '{bad'})
    assert 'error' in result
    fn.assert_not_called()


def test_large_observation_remains_valid_json(executor):
    register(executor, {'text': 'a' * 8000, 'values': [{'value': i} for i in range(100)]})
    response = LLMResponse(tool_calls=[ToolCallRequest('big', 'calculate', {})])
    with patch.object(executor, '_stream_llm_response', return_value=response):
        executor.run_loop('large output fixture', max_iterations=1)
    observation = executor.messages[-1]['content'][0]['content']
    assert isinstance(json.loads(observation), dict)
    assert len(observation) < 6000


def test_native_backend_text_example_cannot_trigger_tool_execution(executor):
    executor.backend.chat = Mock(return_value=LLMResponse(text='Example: Action: {"name":"calculate","input":{}}'))
    response = executor._stream_llm_response('fixture')
    assert response.tool_calls == []
    executor.backend.supports_tools = False
    assert executor._stream_llm_response('fixture').tool_calls[0].name == 'calculate'


@pytest.mark.parametrize('method', ['_summarize_history', '_enforce_context_budget'])
def test_compaction_preserves_tool_call_result_pairs(executor, method):
    executor.messages = [{'role': 'user', 'content': 'original request'}]
    for index in range(16):
        executor.messages.extend([
            {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': str(index), 'name': 'fixture', 'input': {}}]},
            {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': str(index), 'content': '{}'}]},
        ])
    executor.messages.append({'role': 'user', 'content': 'next request'})
    executor.backend.chat = Mock(return_value=LLMResponse(text='summary', stop_reason='stop'))
    with patch.object(executor, '_estimate_context_usage', side_effect=[(900, 1000, .9), (100, 1000, .1)]):
        getattr(executor, method)()
    pending = set()
    for message in executor.messages:
        content = message.get('content', [])
        if isinstance(content, list):
            for block in content:
                if block['type'] == 'tool_use':
                    pending.add(block['id'])
                elif block['type'] == 'tool_result':
                    assert block['tool_use_id'] in pending, 'compaction orphaned a tool result'
                    pending.remove(block['tool_use_id'])
    assert pending == set()


def test_failed_model_summary_uses_deterministic_fallback(executor):
    executor.backend.chat = Mock(return_value=LLMResponse(text='[API Error] unavailable', stop_reason='error'))
    summary = executor._summarize_for_budget([{'role': 'user', 'content': 'artifact fixture.pdb and error details'}])
    assert 'fixture.pdb' in summary
    assert '[API Error]' not in summary


def test_unrelated_chat_does_not_clear_interrupted_work(executor):
    executor._checkpoint.save('previous unfinished work', [], session_id='previous-session')
    with patch.object(executor, '_stream_llm_response', return_value=LLMResponse(text='Knowledge answer', stop_reason='stop')):
        executor.run_loop('unrelated question', max_iterations=1)
    checkpoint = executor._checkpoint.load()
    assert checkpoint['status'] == 'interrupted'
    assert checkpoint['user_request'] == 'previous unfinished work'


def test_final_prose_cannot_erase_unresolved_tool_failure(executor):
    register(executor, {'error': 'calculation fixture failed'})
    responses = [LLMResponse(tool_calls=[ToolCallRequest('fail', 'calculate', {})]),
                 LLMResponse(text='A final response', stop_reason='stop')]
    with patch.object(executor, '_stream_llm_response', side_effect=responses):
        final = executor.run_loop('failed calculation fixture', max_iterations=2)
    assert executor._checkpoint.has_pending()
    assert 'incomplete' in final.lower()

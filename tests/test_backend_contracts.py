from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from zhuangzhou.nlp.backends import OpenAICompatibleBackend


@pytest.fixture
def backend():
    result = OpenAICompatibleBackend(api_key="test-only", model="fixture")
    result._client = Mock()
    return result


def test_empty_choices_have_explicit_diagnostic(backend):
    backend.client.chat.completions.create.return_value = SimpleNamespace(choices=[])
    response = backend.chat("system", [], [])
    assert "Empty response" in response.text
    assert response.stop_reason == "error"


@pytest.mark.parametrize("streaming", [False, True])
def test_generic_bad_request_does_not_silently_disable_tools(backend, streaming):
    create = backend.client.chat.completions.create
    create.side_effect = RuntimeError("400: invalid messages: missing tool result")
    tools = [{"name": "fixture", "input_schema": {"type": "object"}}]
    if streaming:
        assert ("done", "error") in list(backend.chat_stream("system", [], tools))
    else:
        assert backend.chat("system", [], tools).stop_reason == "error"
    assert backend.supports_tools is True
    assert create.call_count == 1


def test_unfinished_stream_is_error_and_closed(backend):
    stream = Mock()
    stream.__iter__ = Mock(return_value=iter([]))
    backend.client.chat.completions.create.return_value = stream
    events = list(backend.chat_stream("system", [], []))
    assert ("done", "error") in events
    stream.close.assert_called_once()


def test_stream_has_output_budget(backend):
    backend.client.chat.completions.create.return_value = []
    list(backend.chat_stream("system", [], []))
    assert backend.client.chat.completions.create.call_args.kwargs["max_tokens"] == 8192


def test_usage_attached_to_final_choice_is_recorded(backend):
    delta = SimpleNamespace(content=None, tool_calls=[])
    choice = SimpleNamespace(delta=delta, finish_reason='stop')
    chunk = SimpleNamespace(choices=[choice], usage=SimpleNamespace(prompt_tokens=33, completion_tokens=7))
    backend.client.chat.completions.create.return_value = [chunk]
    assert ('usage', {'input': 33, 'output': 7}) in list(backend.chat_stream('system', []))


def test_fallback_conversion_does_not_keep_native_assistant_tool_calls(backend):
    backend.supports_tools = False
    messages = [{'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 'fixture', 'name': 'test', 'input': {}}]},
                {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'fixture', 'content': '{}'}]}]
    converted = backend._convert_messages(messages)
    assert 'tool_calls' not in converted[0]
    assert 'test' in converted[0]['content']
    assert converted[1]['role'] == 'user'

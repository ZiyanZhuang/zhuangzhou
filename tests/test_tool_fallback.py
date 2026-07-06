
import unittest
import json
import time
import re
from unittest.mock import MagicMock, patch
from zhuangzhou.nlp.backends import OpenAICompatibleBackend, LLMResponse, ToolCallRequest
from zhuangzhou.nlp.agent import AgentExecutor
from zhuangzhou.skills.registry import get_registry

class MockChoice:
    def __init__(self, content=None, tool_calls=None, finish_reason=None):
        self.delta = MagicMock()
        self.delta.content = content
        self.delta.tool_calls = tool_calls
        self.finish_reason = finish_reason

class MockChunk:
    def __init__(self, content=None, tool_calls=None, finish_reason=None, usage=None):
        choice = MagicMock()
        choice.delta.content = content
        choice.delta.tool_calls = tool_calls
        choice.finish_reason = finish_reason
        self.choices = [choice]
        self.usage = usage

def test_tool_fallback_mechanism():
    # 1. Setup Mock Backend
    backend = OpenAICompatibleBackend(api_key="test", model="test-model")
    mock_client = MagicMock()
    backend._client = mock_client
    
    # Mocking chat.completions.create
    def side_effect(model, messages, tools=None, stream=False, **kwargs):
        if tools is not None:
            # Simulate "does not support tools" error
            raise Exception("400: This model does not support tools.")
        
        if stream:
            # Return a list of chunks
            chunks = [
                MockChunk(content='Thought: I need to load skills.\nAction: {"name": "load_tools", "input": {"category": "simulation"}}'),
                MockChunk(finish_reason="stop"),
                MockChunk(usage=MagicMock(prompt_tokens=10, completion_tokens=20))
            ]
            # usage chunk usually has empty choices
            chunks[2].choices = []
            return chunks
        else:
            mock_resp = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = 'Thought: I need to load skills.\nAction: {"name": "load_tools", "input": {"category": "simulation"}}'
            mock_choice.message.tool_calls = None
            mock_choice.finish_reason = "stop"
            mock_resp.choices = [mock_choice]
            mock_resp.usage.prompt_tokens = 10
            mock_resp.usage.completion_tokens = 20
            return mock_resp

    mock_client.chat.completions.create.side_effect = side_effect

    # 2. Setup AgentExecutor
    registry = get_registry()
    executor = AgentExecutor(backend, registry)
    
    # 3. Run a loop
    with patch.object(executor, '_execute_skill_with_ui', return_value={"status": "loaded"}) as mock_exec:
        # We need to limit iterations
        with patch('zhuangzhou.nlp.agent.MAX_ITERATIONS', 1):
            executor.run_loop("Load simulation tools.")
            
        # 4. Verifications
        assert backend.supports_tools is False
        mock_exec.assert_called_once()
        args, kwargs = mock_exec.call_args
        assert args[0] == "load_tools"
        assert args[1] == {"category": "simulation"}

if __name__ == "__main__":
    test_tool_fallback_mechanism()
    print("Test passed!")

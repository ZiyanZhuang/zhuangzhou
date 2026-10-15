"""
LLM Backends -- Pluggable reasoning engines for Zhuangzhou.

Supported:
  - claude: Anthropic Claude API (tool_use native, best quality)
  - deepseek: DeepSeek API (OpenAI-compatible, cost-effective)
  - openai: OpenAI GPT-4o (OpenAI API)
  - ollama: Local models via Ollama (private, offline, robust OpenAI-compat)

Each backend implements: chat(system, messages, tools) -> Response
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _tools_unsupported(message: str) -> bool:
    text = message.lower()
    return any(phrase in text for phrase in (
        'does not support tools', 'tools are not supported', 'tool use is not supported',
        'tool calling is not supported', 'unsupported parameter: tools',
    ))


# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------

@dataclass
class ToolCallRequest:
    """A tool call requested by the LLM."""
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    """Unified response from any LLM backend."""
    text: str = ""
    tool_calls: List[ToolCallRequest] = field(default_factory=list)
    stop_reason: str = ""
    usage: dict = field(default_factory=dict)

    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------

class LLMBackend:
    """Base class for LLM backends."""

    name: str = "base"
    supports_tools: bool = True

    def chat(self, system: str, messages: List[dict],
             tools: List[dict] = None) -> LLMResponse:
        raise NotImplementedError

    def chat_stream(self, system: str, messages: List[dict],
                    tools: List[dict] = None):
        """Streaming chat — yields (event_type, data) tuples."""
        resp = self.chat(system, messages, tools)
        if resp.text:
            yield ('text', resp.text)
        for tc in resp.tool_calls:
            yield ('tool_call', tc)
        if resp.usage:
            yield ('usage', resp.usage)
        yield ('done', resp.stop_reason)

    def is_available(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Claude (Anthropic) backend
# ---------------------------------------------------------------------------

class ClaudeBackend(LLMBackend):
    """Anthropic Claude API with native tool_use."""

    name = "claude"

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY', '')
        self.model = model or os.environ.get(
            'ZHUANGZHOU_CLAUDE_MODEL', 'claude-3-5-sonnet-20240620')
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package required: pip install anthropic")
        return self._client

    def is_available(self) -> bool:
        if not self.api_key or not self.api_key.strip():
            return False
        return True

    def chat(self, system: str, messages: List[dict],
             tools: List[dict] = None) -> LLMResponse:
        kwargs = {
            'model': self.model,
            'max_tokens': 8192,
            'system': system,
            'messages': messages,
        }
        if tools:
            kwargs['tools'] = tools

        try:
            resp = self.client.messages.create(**kwargs)
        except Exception as e:
            logger.error("Claude API error: %s", e)
            return LLMResponse(
                text=f"[API Error] {type(e).__name__}: {e}",
                tool_calls=[], stop_reason='error',
                usage={'input': 0, 'output': 0})

        text_parts = []
        tool_calls = []
        for block in resp.content:
            if block.type == 'text':
                text_parts.append(block.text)
            elif block.type == 'tool_use':
                tool_calls.append(ToolCallRequest(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))

        _usage = getattr(resp, 'usage', None)
        return LLMResponse(
            text='\n'.join(text_parts),
            tool_calls=tool_calls,
            stop_reason=resp.stop_reason,
            usage={'input': _usage.input_tokens if _usage else 0,
                   'output': _usage.output_tokens if _usage else 0},
        )

    def chat_stream(self, system: str, messages: List[dict],
                    tools: List[dict] = None):
        kwargs = {
            'model': self.model,
            'max_tokens': 8192,
            'system': system,
            'messages': messages,
        }
        if tools:
            kwargs['tools'] = tools

        try:
            with self.client.messages.stream(**kwargs) as stream:
                current_tool = None
                input_json_parts = []

                for event in stream:
                    if event.type == 'content_block_start':
                        block = event.content_block
                        if block.type == 'tool_use':
                            current_tool = {'id': block.id, 'name': block.name}
                            input_json_parts = []
                        elif block.type == 'text':
                            current_tool = None

                    elif event.type == 'content_block_delta':
                        delta = event.delta
                        if delta.type == 'text_delta':
                            yield ('text', delta.text)
                        elif delta.type == 'input_json_delta':
                            input_json_parts.append(delta.partial_json)

                    elif event.type == 'content_block_stop':
                        if current_tool:
                            raw_json = ''.join(input_json_parts)
                            try:
                                args = json.loads(raw_json) if raw_json else {}
                            except json.JSONDecodeError:
                                args = {"_parse_error": raw_json[:500]}
                            yield ('tool_call', ToolCallRequest(
                                id=current_tool['id'],
                                name=current_tool['name'],
                                arguments=args,
                            ))
                            current_tool = None

                final = stream.get_final_message()
                _fusage = getattr(final, 'usage', None)
                yield ('usage', {
                    'input': _fusage.input_tokens if _fusage else 0,
                    'output': _fusage.output_tokens if _fusage else 0,
                })
                yield ('done', final.stop_reason)

        except Exception as e:
            logger.error("Claude stream error: %s", e)
            yield ('text', f"[API Error] {type(e).__name__}: {e}")
            yield ('done', 'error')


# ---------------------------------------------------------------------------
# OpenAI-compatible backend
# ---------------------------------------------------------------------------

class OpenAICompatibleBackend(LLMBackend):
    """OpenAI-compatible API (DeepSeek, GPT-4o, etc.)."""

    name = "openai"

    def __init__(self, api_key: str = None, model: str = None,
                 base_url: str = None, backend_name: str = "openai"):
        self.backend_name = backend_name
        self.name = backend_name

        if backend_name == 'deepseek':
            self.api_key = api_key or os.environ.get('DEEPSEEK_API_KEY', '')
            self.model = model or 'deepseek-chat'
            self.base_url = base_url or 'https://api.deepseek.com/v1'
        elif backend_name == 'ollama':
            self.api_key = "ollama"
            self.model = model or 'qwen2.5:latest'
            self.base_url = base_url or 'http://localhost:11434/v1'
        else:
            self.api_key = api_key or os.environ.get('OPENAI_API_KEY', '')
            self.model = model or 'gpt-4o'
            self.base_url = base_url

        self._client = None
        self.request_timeout = 300.0

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.request_timeout
                )
            except ImportError:
                raise ImportError(
                    "openai package required: pip install openai")
        return self._client

    def is_available(self) -> bool:
        if self.backend_name == 'ollama':
            try:
                import urllib.request
                # Check root API tags
                root_url = self.base_url.replace('/v1', '') + '/api/tags'
                with urllib.request.urlopen(root_url, timeout=2) as r:
                    return r.getcode() == 200
            except Exception as e:
                logger.debug("%s availability check failed: %s", self.__class__.__name__, e)
                return False
        if not self.api_key or not self.api_key.strip():
            return False
        return True

    def _convert_messages(self, messages: List[dict]) -> List[dict]:
        """Claude format -> OpenAI format."""
        converted = []
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            if isinstance(content, str):
                converted.append(msg)
                continue
            if not isinstance(content, list):
                converted.append(msg)
                continue

            if role == 'assistant':
                text_parts, tool_calls = [], []
                for block in content:
                    if block.get('type') == 'text':
                        text_parts.append(block.get('text', ''))
                    elif block.get('type') == 'tool_use':
                        if not self.supports_tools:
                            text_parts.append('Action: ' + json.dumps({
                                'name': block.get('name', ''), 'input': block.get('input', {}),
                            }))
                            continue
                        tool_calls.append({
                            'id': block.get('id', ''),
                            'type': 'function',
                            'function': {
                                'name': block.get('name', ''),
                                'arguments': json.dumps(block.get('input', {}))
                            }
                        })
                oai_msg = {'role': 'assistant', 'content': '\n'.join(text_parts) or ""}
                if tool_calls: oai_msg['tool_calls'] = tool_calls
                converted.append(oai_msg)
            elif role == 'user':
                has_results = any(b.get('type') == 'tool_result' for b in content if isinstance(b, dict))
                if has_results:
                    if self.supports_tools:
                        # Native tool mode: emit role='tool' messages as OpenAI expects
                        for block in content:
                            if block.get('type') == 'tool_result':
                                converted.append({
                                    'role': 'tool',
                                    'tool_call_id': block.get('tool_use_id', ''),
                                    'content': str(block.get('content', ''))
                                })
                    else:
                        # Manual ReAct mode: merge tool results as plain user text.
                        # Sending role='tool' without a preceding tool_calls causes 400
                        # on DeepSeek / Ollama when tools are not natively supported.
                        parts = []
                        for block in content:
                            if block.get('type') == 'tool_result':
                                parts.append(f"[工具结果]\n{block.get('content', '')}")
                            elif block.get('type') == 'text':
                                parts.append(block.get('text', ''))
                        converted.append({'role': 'user', 'content': '\n\n'.join(parts)})
                else:
                    text = "\n".join(b.get('text', '') for b in content if b.get('type') == 'text')
                    converted.append({'role': 'user', 'content': text})
        return converted

    def chat(self, system: str, messages: List[dict],
             tools: List[dict] = None) -> LLMResponse:
        oai_tools = None
        if tools and self.supports_tools:
            oai_tools = [{'type': 'function', 'function': {'name': t['name'], 
                         'description': t.get('description',''), 
                         'parameters': t.get('input_schema', {})}} for t in tools]

        full_msgs = [{'role': 'system', 'content': system}] + self._convert_messages(messages)
        
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=full_msgs,
                tools=oai_tools,
                max_tokens=8192
            )
            if not resp.choices:
                return LLMResponse(text="[API Error] Empty response (no choices)", stop_reason="error")
            choice = resp.choices[0]
            tool_calls = []
            if choice.message and hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.debug("Failed to parse tool call arguments for %s: %s", tc.function.name, e)
                        args = {"_parse_error": tc.function.arguments}
                    tool_calls.append(ToolCallRequest(id=tc.id, name=tc.function.name, arguments=args))

            usage = getattr(resp, 'usage', None)
            if usage:
                usage_dict = {'input': usage.prompt_tokens, 'output': usage.completion_tokens}
            else:
                usage_dict = {'input': 0, 'output': 0}
            return LLMResponse(
                text=(choice.message.content if choice.message else None) or "",
                tool_calls=tool_calls,
                stop_reason=choice.finish_reason or "",
                usage=usage_dict
            )
        except Exception as e:
            err_msg = str(e).lower()
            if oai_tools and _tools_unsupported(err_msg):
                if self.supports_tools:
                    logger.warning("%s does not support tools, falling back to manual mode.", self.model)
                    self.supports_tools = False
                    # Guard against infinite recursion
                    if getattr(self, '_fallback_in_progress', False):
                        raise
                    self._fallback_in_progress = True
                    try:
                        return self.chat(system, messages, tools)
                    finally:
                        self._fallback_in_progress = False

            logger.error("%s API error: %s", self.backend_name, e)
            return LLMResponse(text=f"[API Error] {e}", stop_reason="error")

    def chat_stream(self, system: str, messages: List[dict],
                    tools: List[dict] = None):
        oai_tools = None
        if tools and self.supports_tools:
            oai_tools = [{'type': 'function', 'function': {'name': t['name'], 
                         'description': t.get('description',''), 
                         'parameters': t.get('input_schema', {})}} for t in tools]

        full_msgs = [{'role': 'system', 'content': system}] + self._convert_messages(messages)
        
        stream = None
        finished = False
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=full_msgs,
                tools=oai_tools,
                stream=True,
                max_tokens=8192,
                stream_options={"include_usage": True}
            )
            
            tool_accum = {}
            for chunk in stream:
                if chunk.usage:
                    yield ('usage', {'input': chunk.usage.prompt_tokens, 'output': chunk.usage.completion_tokens})
                if not chunk.choices:
                    continue
                
                delta = chunk.choices[0].delta
                if delta.content: yield ('text', delta.content)
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        fn = getattr(tc, 'function', None)
                        if idx not in tool_accum:
                            tool_accum[idx] = {
                                'id': tc.id or '',
                                'name': (fn.name or '') if fn else '',
                                'parts': [],
                            }
                        if fn and fn.arguments:
                            tool_accum[idx]['parts'].append(fn.arguments)
                
                if chunk.choices[0].finish_reason:
                    finished = True
                    for idx in sorted(tool_accum.keys()):
                        data = tool_accum[idx]
                        try: args = json.loads("".join(data['parts']))
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.debug("Failed to parse streamed tool call arguments for %s: %s", data.get('name'), e)
                            args = {"_parse_error": "".join(data['parts'])}
                        yield ('tool_call', ToolCallRequest(id=data['id'], name=data['name'], arguments=args))
                    yield ('done', chunk.choices[0].finish_reason)
            if not finished:
                raise ValueError('Stream ended before a finish reason was received')
        except Exception as e:
            err_msg = str(e).lower()
            if oai_tools and _tools_unsupported(err_msg):
                if self.supports_tools:
                    logger.warning("%s does not support tools (stream), falling back to manual mode.", self.model)
                    self.supports_tools = False
                    # Pass tools=None to prevent any re-entry into this branch
                    yield from self.chat_stream(system, messages, tools=None)
                    return

            logger.error("%s Stream error: %s", self.backend_name, e)
            yield ('text', f"[Stream Error] {e}")
            yield ('done', 'error')
        finally:
            close = getattr(stream, 'close', None)
            if callable(close):
                close()


# ---------------------------------------------------------------------------
# Robust Ollama
# ---------------------------------------------------------------------------

class OllamaOpenAIBackend(OpenAICompatibleBackend):
    """Robust local Ollama with exponential backoff and context safety."""
    
    def __init__(self, **kwargs):
        super().__init__(backend_name='ollama', **kwargs)
        self.request_timeout = 600.0

    def chat(self, system: str, messages: List[dict], tools: List[dict] = None) -> LLMResponse:
        for attempt in range(3):
            try:
                return super().chat(system, messages, tools)
            except Exception as e:
                if attempt == 2: raise
                time.sleep(2 ** attempt)
                logger.warning("Ollama retry %d...", attempt + 1)

    def chat_stream(self, system: str, messages: List[dict], tools: List[dict] = None):
        # Stream has its own basic error handling
        return super().chat_stream(system, messages, tools)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_BACKENDS = {
    'claude': ClaudeBackend,
    'deepseek': lambda **kw: OpenAICompatibleBackend(backend_name='deepseek', **kw),
    'openai': lambda **kw: OpenAICompatibleBackend(backend_name='openai', **kw),
    'ollama': OllamaOpenAIBackend,
}

def _get_best_ollama_model() -> Optional[str]:
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1) as r:
            data = json.loads(r.read().decode('utf-8'))
            names = [m['name'] for m in data.get('models', [])]
            # Preference for bio/chem reasoning
            for pref in ['deepseek-r1:8b', 'qwen2.5:7b', 'llama3.1:8b', 'qwen2.5:latest']:
                if pref in names: return pref
            return names[0] if names else None
    except Exception: return None

def list_available_models() -> List[Dict[str, str]]:
    """List all available models, both local (Ollama) and cloud."""
    available = []
    
    # 1. Local Ollama models
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1) as r:
            data = json.loads(r.read().decode('utf-8'))
            for m in data.get('models', []):
                available.append({
                    'name': f"ollama:{m['name']}",
                    'type': 'local',
                    'size': f"{m.get('size', 0) / 1e9:.1f} GB"
                })
    except Exception:
        pass  # Ollama offline

    # 2. Known cloud backends (placeholders for configured keys)
    for backend in ['claude', 'deepseek', 'openai']:
        key_env = {
            'claude': 'ANTHROPIC_API_KEY',
            'deepseek': 'DEEPSEEK_API_KEY',
            'openai': 'OPENAI_API_KEY'
        }.get(backend)
        
        if os.environ.get(key_env):
            available.append({'name': backend, 'type': 'cloud', 'size': 'N/A'})

    return available


def get_backend(name: str = None, **kwargs) -> LLMBackend:
    # Handle global env override from cli.py
    if not name and os.environ.get('ZHUANGZHOU_DEFAULT_MODEL'):
        name = os.environ.get('ZHUANGZHOU_DEFAULT_MODEL')

    # Handle 'ollama:model' syntax
    if name and name.startswith('ollama:'):
        kwargs['model'] = name.split(':', 1)[1]
        name = 'ollama'
    
    if name is None:
        for cand in ['claude', 'deepseek', 'ollama', 'openai']:
            b = get_backend(cand, **kwargs)
            if b.is_available():
                if cand == 'ollama' and not kwargs.get('model'):
                    kwargs['model'] = _get_best_ollama_model()
                    b = OllamaOpenAIBackend(**kwargs)
                return b
        raise RuntimeError("No LLM backends available.")

    return _BACKENDS[name.lower()](**kwargs)

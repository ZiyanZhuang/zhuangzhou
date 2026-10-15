"""Actual Rich rendering and command dispatch; no model or scientific simulation."""
import io
from unittest.mock import Mock, patch

import pytest
from rich.console import Console

import zhuangzhou.ui.core as ui


@pytest.fixture
def display(monkeypatch):
    console = Console(file=io.StringIO(), width=80, height=60, legacy_windows=False,
                      record=True, theme=ui.ZZTheme.THEME)
    monkeypatch.setattr(ui, 'console', console)
    return console


def test_banner_uses_runtime_version_and_unambiguous_start_date(display, monkeypatch):
    import zhuangzhou
    monkeypatch.setattr(zhuangzhou, '__version__', '9.8.7')
    ui.ZZInterface.render_logo()
    text = display.export_text()
    assert 'v9.8.7' in text
    assert 'Started:' in text and 'Date:' not in text
    assert 'Fixed|Plan|Direct' not in text


@pytest.mark.parametrize('width', [60, 80, 100, 120])
def test_banner_metadata_never_splits_labels_from_values(display, width):
    display.width = width
    with patch.object(ui.os, 'getcwd', return_value='C:/work/' + 'long-project-' * 8):
        ui.ZZInterface.render_logo()
    lines = display.export_text().splitlines()
    for label in ('Path:',):
        matches = [line for line in lines if label in line]
        assert len(matches) == 1, f'{width}-column layout lost {label}'
        line = matches[0]
        assert line.split(label, 1)[1].strip(), f'{width}-column layout orphaned {label}'


@pytest.mark.parametrize('exception,expected', [(ValueError, '✗'), (KeyboardInterrupt, '◇')])
def test_monitor_exception_never_displays_success(display, exception, expected):
    monitor = ui.ZZStatus()
    monitor.__exit__(exception, exception(), None)
    text = display.export_text()
    assert expected in text and '✓' not in text


def test_waiting_monitor_is_not_reported_as_completed(display):
    monitor = ui.ZZStatus()
    monitor.update(phase='wait', action='Awaiting confirmation')
    monitor.__exit__(None, None, None)
    text = display.export_text()
    assert '◇' in text and 'Awaiting confirmation' in text and '✓' not in text


def test_generation_rate_excludes_input_tokens(display):
    with patch.object(ui.time, 'perf_counter', return_value=10):
        monitor = ui.ZZStatus(start_time=8)
        monitor.set_usage(input_tokens=1600, output_tokens=400)
        monitor._print_summary()
    text = display.export_text()
    assert '200/s' in text and '1.0k/s' not in text


def test_estimated_tokens_stay_labelled_until_backend_usage_arrives(display):
    monitor = ui.ZZStatus()
    monitor.update(tokens=20, output_tokens=20)
    monitor._print_summary()
    assert '~20 tokens' in display.export_text()
    monitor.set_usage(input_tokens=30, output_tokens=20)
    monitor._print_summary()
    assert '~50 tokens' not in display.export_text()


@pytest.mark.parametrize('streaming', [False, True])
def test_response_panels_label_estimated_usage(display, streaming):
    monitor = ui.ZZStatus()
    monitor.set_usage(input_tokens=1600, output_tokens=400, estimated=True)
    status = {'estimated_tokens': 2300, 'context_limit': 64000, 'usage_ratio': .04}
    if streaming:
        renderer = ui.StreamRenderer(status_provider=lambda: status, monitor=monitor)
        renderer.buffer = ['fixture response']
        display.print(renderer._render_content())
    else:
        ui.ZZInterface.display_message('assistant', 'fixture response', status, monitor)
    text = display.export_text()
    assert '~2.0k' in text and 'ctx~' in text


def test_help_lists_real_agent_commands_without_initializing_model(display, monkeypatch):
    import zhuangzhou.app as app
    monkeypatch.setattr(app, 'console', display)
    app._render_help_table()
    text = display.export_text()
    for command in ('/model', '/confirm', '/plan', '/resume'):
        assert command in text
    assert '/envs' not in text  # This app has no /envs dispatcher.


def test_version_flag_is_local_and_does_not_construct_app(display, monkeypatch):
    import zhuangzhou.app as app
    import zhuangzhou
    monkeypatch.setattr(app, 'console', display)
    monkeypatch.setattr('sys.argv', ['zhuangzhou', '--version'])
    monkeypatch.setattr(app, 'ZhuangzhouApp', Mock(side_effect=AssertionError('Version must be local')))
    app.main()
    assert zhuangzhou.__version__ in display.export_text()


def test_interactive_start_and_quit_do_not_require_model():
    from zhuangzhou.app import ZhuangzhouApp
    app = ZhuangzhouApp()
    session = Mock()
    session.prompt.return_value = '/quit'
    with patch.object(app, '_get_prompt_session', return_value=session), \
         patch.object(app, '_get_nlp_agent', side_effect=AssertionError('Startup must stay local')):
        app.run_interactive()
    assert app.running is False


@pytest.mark.parametrize('command', ['/status', '/unknown-command'])
def test_local_or_unknown_commands_do_not_initialize_model(command):
    from zhuangzhou.app import ZhuangzhouApp
    app = ZhuangzhouApp()
    app.commands['status'].execute = Mock()
    with patch.object(app, '_get_nlp_agent', side_effect=AssertionError('Local command needs no model')):
        app.dispatch(command)
    if command == '/status':
        app.commands['status'].execute.assert_called_once()


def test_registered_skill_slash_keeps_existing_agent_path():
    from zhuangzhou.app import ZhuangzhouApp
    app = ZhuangzhouApp()
    with patch.object(app, '_chat') as chat:
        app.dispatch('/esm2_load fixture')
    chat.assert_called_once_with('/esm2_load fixture')


@pytest.mark.parametrize('exception,expected', [(ValueError, '✗'), (KeyboardInterrupt, '◇')])
def test_chat_caught_failure_never_prints_success(display, monkeypatch, exception, expected):
    from zhuangzhou.app import ZhuangzhouApp
    app = ZhuangzhouApp()
    backend = Mock()
    backend.chat.side_effect = exception('injected fixture')
    with patch.object(app, '_get_nlp_agent', return_value=backend), \
         patch.object(app, '_get_plan_mode', return_value=None), \
         patch.object(ui.ZZStatus, '__enter__', lambda monitor: monitor):
        app._chat('UI fixture')
    text = display.export_text()
    assert expected in text and '✓' not in text


def test_unavailable_model_leaves_interactive_app_usable(display):
    from zhuangzhou.app import ZhuangzhouApp
    app = ZhuangzhouApp()
    with patch.object(app, '_get_nlp_agent', side_effect=RuntimeError('No model fixture')):
        app._chat('UI fixture')
    assert app.running
    assert 'Agent unavailable' in display.export_text()


def test_malformed_command_quotes_do_not_crash_app(display):
    from zhuangzhou.app import ZhuangzhouApp
    app = ZhuangzhouApp()
    app.dispatch('/config "unterminated')
    assert app.running
    assert 'Invalid quoting' in display.export_text()


def test_agent_command_initialization_error_keeps_prompt_usable(display):
    from zhuangzhou.app import ZhuangzhouApp
    app = ZhuangzhouApp()
    with patch.object(app, '_get_nlp_agent', side_effect=RuntimeError('No model fixture')):
        app.dispatch('/confirm fixture')
    assert app.running
    assert 'Command /confirm failed' in display.export_text()


def test_failed_chat_keeps_reported_token_accounting(display):
    from zhuangzhou.app import ZhuangzhouApp
    app = ZhuangzhouApp()
    def fail_chat(message, status_monitor):
        status_monitor.set_usage(input_tokens=12, output_tokens=3)
        raise ValueError('failure after usage fixture')
    backend = Mock()
    backend.chat.side_effect = fail_chat
    with patch.object(app, '_get_nlp_agent', return_value=backend), \
         patch.object(app, '_get_plan_mode', return_value=None), \
         patch.object(ui.ZZStatus, '__enter__', lambda monitor: monitor):
        app._chat('UI fixture')
    assert app.session_tokens == 15


def test_status_path_with_brackets_remains_literal(display, monkeypatch):
    from types import SimpleNamespace
    import zhuangzhou
    from zhuangzhou.commands.status import StatusCommand
    monkeypatch.setattr(ui.os, 'getcwd', lambda: 'C:/work/[fixture]')
    monkeypatch.setattr(zhuangzhou, '__file__', 'C:/packages/[fixture]/__init__.py')
    packages = {name: SimpleNamespace(__version__='fixture')
                for name in ('torch', 'numpy', 'scipy', 'Bio', 'rich', 'rdkit')}
    with patch.dict('sys.modules', packages), \
         patch('urllib.request.urlopen', side_effect=OSError('offline fixture')):
        StatusCommand(SimpleNamespace(console=display, config={})).execute([])
    assert '[fixture]' in display.export_text()


def test_prompt_completion_includes_the_actual_local_commands():
    from prompt_toolkit.document import Document
    from zhuangzhou.app import ZhuangzhouApp
    app = ZhuangzhouApp()
    with patch('zhuangzhou.app.PromptSession') as session, \
         patch('zhuangzhou.nlp.completer._get_skill_names', return_value=[]):
        app._get_prompt_session()
        completer = session.call_args.kwargs['completer']
        assert '/version' in [item.text for item in completer.get_completions(Document('/ver'), None)]
        assert '/status' in [item.text for item in completer.get_completions(Document('/st'), None)]


def test_chat_confirmation_displays_reviewable_parameters_without_credentials(display):
    from zhuangzhou.app import ZhuangzhouApp
    from zhuangzhou.nlp.agent import AgentExecutor, ZhuangzhouAgent
    from zhuangzhou.nlp.backends import LLMBackend, LLMResponse, ToolCallRequest
    from zhuangzhou.skills.registry import Skill, SkillRegistry

    registry = SkillRegistry()
    registry._loaded = True
    operation = Mock()
    registry.register(Skill('review_fixture', 'confirmation fixture', 'chemistry', operation, risk_level='high'))
    with patch('zhuangzhou.nlp.agent.ToolRouter', return_value=None):
        executor = AgentExecutor(LLMBackend(), registry)
    agent = object.__new__(ZhuangzhouAgent)
    agent.executor = executor
    app = ZhuangzhouApp()
    params = {'target': '[bold]fixture[/bold]', 'nested': {'api_key': 'fixture-sensitive-value'},
              'config': {'mode': 'fixture-mode'}, 'sequence': 'AC' * 180}
    try:
        with patch.object(app, '_get_nlp_agent', return_value=agent), \
             patch.object(app, '_get_plan_mode', return_value=None), \
             patch.object(executor, '_stream_llm_response', return_value=LLMResponse(
                 tool_calls=[ToolCallRequest('pending', 'review_fixture', params)])), \
             patch.object(ui.ZZStatus, '__enter__', lambda monitor: monitor):
            app._chat('confirmation fixture')
        text = display.export_text()
        assert text.count('/confirm review_fixture') == 1
        assert '[bold]fixture[/bold]' in text
        assert 'fixture-mode' in text and 'AC' * 180 in ''.join(ch for ch in text if ch.isalnum())
        assert 'REDACTED' in text and 'fixture-sensitive-value' not in text
        assert executor._pending_confirmations['review_fixture'] == params
        operation.assert_not_called()
    finally:
        executor._bg_tasks.shutdown(wait=True)

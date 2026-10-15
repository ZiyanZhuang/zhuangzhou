"""Model is a declared stub; selection, review, DOCX and history use real code/data."""

import json
from pathlib import Path
from unittest.mock import patch
import zipfile

import pytest

from zhuangzhou.nlp.agent import AgentExecutor
from zhuangzhou.nlp.backends import LLMBackend, LLMResponse, ToolCallRequest
from zhuangzhou.skills.registry import SkillRegistry


class DeterministicPlanner(LLMBackend):
    name = 'test-stub'

    def __init__(self, report_path):
        self.stage = 0
        self.report_path = report_path
        self.selected_name = ''

    def chat(self, system, messages, tools=None):
        if self.stage == 0:
            self.stage += 1
            return LLMResponse(tool_calls=[ToolCallRequest('select-1', 'synbio_yeast_parts_select', {'n_candidates': 1})])
        observations = [b for b in messages[-1]['content'] if b.get('type') == 'tool_result']
        observation = json.loads(observations[0]['content'])
        if self.stage == 1:
            # Deterministic review of the real selection, including source metadata.
            assert observation['validation']['valid'] is True
            assert observation['promoters'] and observation['terminators']
            self.selected_name = observation['promoters'][0]['name']
            assert observation['promoters'][0]['doi']
            self.stage += 1
            sections = [{'type': 'paragraph', 'text': 'Curated metadata selection; no experimental validation performed.'},
                        {'type': 'paragraph', 'text': self.selected_name}]
            return LLMResponse(tool_calls=[ToolCallRequest('report-1', 'generate_report', {
                'sections': sections, 'output_path': str(self.report_path), 'title': 'Offline integration fixture',
            })])
        assert 'docx_path' in observation, observation
        assert Path(observation['docx_path']) == self.report_path, observation
        assert self.report_path.is_file()
        return LLMResponse(text='Selection metadata and report recorded; biological performance remains unverified.', stop_reason='stop')


def test_real_selection_report_memory_and_next_context(tmp_path):
    pytest.importorskip('docx')
    backend = DeterministicPlanner(tmp_path / 'report.docx')
    registry = SkillRegistry()
    with patch('zhuangzhou.nlp.agent.ToolRouter', return_value=None):
        executor = AgentExecutor(backend, registry, verbose=True)
    executor._huizi_enabled = False
    try:
        final = executor.run_loop('Select one curated part and report metadata only.', max_iterations=3)
        assert 'unverified' in final
        assert [e['skill'] for e in executor.execution_log] == ['synbio_yeast_parts_select', 'generate_report']
        assert all(e['success'] for e in executor.execution_log)
        history = executor._exec_history.get_last_session_log(executor._session_id)
        assert len(history) == 2 and all(e['success'] for e in history)
        assert not executor._checkpoint.has_pending()
        assert 'synbio_yeast_parts_select' in executor.build_system_prompt()
        with zipfile.ZipFile(backend.report_path) as archive:
            document = archive.read('word/document.xml').decode('utf-8')
        assert backend.selected_name in document
        assert 'no experimental validation' in document
        from zhuangzhou.memory.registry import MemoryRegistry
        archive = MemoryRegistry.get().session_archive
        calls = archive.interactions[-1]['tool_calls']
        assert [item['tool'] for item in calls] == ['synbio_yeast_parts_select', 'generate_report']
        assert calls[-1]['result']['docx_path'] == str(backend.report_path)
    finally:
        executor._bg_tasks.shutdown(wait=True)

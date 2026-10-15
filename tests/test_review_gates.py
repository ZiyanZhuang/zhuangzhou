"""Review availability and execution outcome must never become fabricated success."""

from unittest.mock import Mock

import pytest

from zhuangzhou.nlp.backends import LLMResponse
from zhuangzhou.nlp.fixed_mode import FixedMode, FixedStep, ResearchResult
from zhuangzhou.nlp.supervisor import GateResult, PipelineSupervisor, SupervisorGate, TypeAValidator
from zhuangzhou.skills.registry import Skill, SkillRegistry


@pytest.mark.parametrize('text', ['', 'not JSON', '{}', '{"passed":"false"}',
    '{"passed":true,"confidence":NaN}', '{"passed":true,"issues":"bad"}'])
def test_invalid_model_review_is_blocking(text):
    backend = Mock(chat=Mock(return_value=LLMResponse(text=text)))
    result = SupervisorGate(backend).validate('post_research', {})
    assert result.passed is False
    assert result.blocking is True
    assert result.confidence == 0


def test_explicit_rejection_cannot_be_nonblocking():
    backend = Mock(chat=Mock(return_value=LLMResponse(text='{"passed":false,"blocking":false}')))
    result = SupervisorGate(backend).validate('post_research', {})
    assert result.passed is False and result.blocking is True


def test_review_without_backend_is_unavailable():
    result = PipelineSupervisor().gate_post_research({}, [])
    assert result.passed is False and result.blocking is True


def test_merge_does_not_erase_negative_review():
    result = PipelineSupervisor()._merge(GateResult(passed=False, blocking=False))
    assert result.passed is False


def test_missing_pdb_cannot_pass_scientific_precheck(tmp_path):
    result = TypeAValidator.check_pdb_has_metalloprotein(str(tmp_path / 'missing.pdb'))
    assert result.passed is False and result.blocking is True


def test_empty_pose_file_is_not_valid_docking_output(tmp_path):
    (tmp_path / 'docking_out.pdbqt').touch()
    result = TypeAValidator.check_docking_output(str(tmp_path))
    assert result.passed is False and result.blocking is True


@pytest.mark.parametrize('mode', ['missing', 'reported_failure', 'gate_blocked', 'exception'])
def test_fixed_pipeline_reports_failure_and_stops(mode):
    registry = SkillRegistry()
    registry._loaded = True
    fn = Mock(side_effect=RuntimeError('fixture error')) if mode == 'exception' else Mock(return_value={'success': mode != 'reported_failure'})
    if mode != 'missing':
        registry.register(Skill('first', 'fixture', 'test', fn))
    second = Mock(return_value={'success': True})
    registry.register(Skill('second', 'fixture', 'test', second))
    executor = Mock()
    # Delegate to real registry execution; the model/research/review inputs are stubs.
    executor._execute_skill_with_ui.side_effect = lambda name, params: registry.get(name).execute(**params)
    pipeline = FixedMode(None, registry, executor)
    pipeline.research_phase.run = Mock(return_value=ResearchResult())
    pipeline.supervisor.gate_post_research = Mock(return_value=GateResult(passed=True))
    pipeline._build_steps = Mock(return_value=[FixedStep('first', {}, 'fixture', 'post_docking'), FixedStep('second', {}, 'fixture')])
    pipeline._run_gate = Mock(return_value=GateResult(passed=mode != 'gate_blocked', blocking=mode == 'gate_blocked'))
    result = pipeline.run('md', 'test fixture', console=Mock())
    assert result['success'] is False
    assert result['failed_step'] == 1
    second.assert_not_called()

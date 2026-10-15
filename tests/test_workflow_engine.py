"""Offline regression tests: real workflow engine, deterministic local skills."""

from unittest.mock import Mock, patch

import pytest

from zhuangzhou.skills.registry import Skill, SkillRegistry
from zhuangzhou.workflow.engine import WorkflowEngine


@pytest.fixture
def engine():
    registry = SkillRegistry()
    registry._loaded = True
    with patch("zhuangzhou.skills.get_registry", return_value=registry):
        return WorkflowEngine()


def register(engine, name, fn):
    engine.registry.register(Skill(name, "test", "test", fn))


def step(name, **extra):
    return {"id": name, "skill": name, **extra}


@pytest.mark.parametrize("output", [{"error": "missing artifact"}, {"success": False}, {"status": "failed"}])
def test_returned_failure_stops_downstream(engine, output):
    downstream = Mock(return_value={"success": True})
    register(engine, "first", lambda **kw: output)
    register(engine, "second", downstream)
    result = engine.execute({"steps": [step("first"), step("second")]}, {})
    assert result.success is False
    assert result.steps_failed == ["first"]
    assert result.steps_completed == []
    assert result.outputs["first"] == output  # retain failure evidence
    downstream.assert_not_called()


def test_continue_retains_failed_state_and_runs_independent_step(engine):
    register(engine, "first", lambda **kw: {"error": "failed"})
    register(engine, "second", lambda **kw: {"value": 2})
    result = engine.execute({"steps": [step("first", on_error="continue"), step("second")]}, {})
    assert result.success is False
    assert result.steps_failed == ["first"]
    assert result.steps_completed == ["second"]


def test_missing_reference_fails_before_dispatch(engine):
    tool = Mock()
    register(engine, "first", tool)
    result = engine.execute({"steps": [step("first", params={"value": "$input.missing"})]}, {})
    assert result.steps_failed == ["first"]
    assert "missing" in result.errors["first"]
    tool.assert_not_called()


def test_nested_references_preserve_types_and_explicit_none(engine):
    register(engine, "first", lambda **kw: {"nested": {"value": [1, 2]}})
    tool = Mock(return_value={"success": True})
    register(engine, "second", tool)
    wf = {"steps": [step("first"), step("second", params={
        "values": ["$first.nested.value", {"optional": "$input.optional"}],
    })]}
    result = engine.execute(wf, {"optional": None})
    assert result.success is True
    assert tool.call_args.kwargs["values"] == [[1, 2], {"optional": None}]


@pytest.mark.parametrize("condition", ["$input.ok = true", "$input.ok > true", "$input.missing == true"])
def test_invalid_condition_fails_closed(engine, condition):
    tool = Mock()
    register(engine, "first", tool)
    result = engine.execute({"steps": [step("first", condition=condition)]}, {"ok": True})
    assert result.steps_failed == ["first"]
    tool.assert_not_called()


def test_false_condition_skips_and_valid_condition_runs(engine):
    tool = Mock(return_value={"value": 1})
    register(engine, "first", tool)
    result = engine.execute({"steps": [step("first", condition="$input.ok == true")]}, {"ok": False})
    assert result.success is True
    assert result.steps_skipped == ["first"]
    tool.assert_not_called()
    assert engine.execute({"steps": [step("first", condition="$input.ok == true")]}, {"ok": True}).success


@pytest.mark.parametrize("steps", [[step("first"), step("first")], [step("first", on_error="ignore")]])
def test_invalid_definition_rejected_before_side_effects(engine, steps):
    tool = Mock()
    register(engine, "first", tool)
    with pytest.raises(ValueError):
        engine.execute({"steps": steps}, {})
    tool.assert_not_called()


def test_failed_dependency_can_be_conditionally_skipped(engine):
    register(engine, "first", lambda **kw: {"error": "no artifact"})
    tool = Mock()
    register(engine, "second", tool)
    result = engine.execute({"steps": [step("first", on_error="continue"),
        step("second", condition="$first.success == true")]}, {})
    assert result.steps_failed == ["first"]
    assert result.steps_skipped == ["second"]
    tool.assert_not_called()


def test_progress_callback_failure_does_not_lose_successful_result(engine):
    register(engine, "first", lambda **kw: {"value": 1})
    engine.set_progress_callback(Mock(side_effect=RuntimeError("UI unavailable")))
    result = engine.execute({"steps": [step("first")]}, {})
    assert result.success
    assert result.outputs["first"] == {"value": 1}


def test_workflow_accepts_registered_function_without_config_parameter(engine):
    register(engine, 'first', lambda value: {'value': value})
    result = engine.execute({'steps': [step('first', params={'value': 3})]}, {})
    assert result.success
    assert result.outputs['first'] == {'value': 3}


def test_declared_workflow_input_contract_is_enforced_before_any_step(engine, tmp_path):
    tool = Mock()
    register(engine, 'first', tool)
    workflow = {'inputs': {'pdb': {'type': 'path', 'required': True}}, 'steps': [step('first')]}
    with pytest.raises(ValueError, match='File not found'):
        engine.execute(workflow, {'pdb': str(tmp_path / 'missing.pdb')})
    tool.assert_not_called()


def test_declared_workflow_inputs_are_normalized_once(engine, tmp_path):
    register(engine, 'first', lambda **kw: kw)
    workflow = {'inputs': {
        'mode': {'type': 'string', 'choices': ['vacuum', 'solvated'], 'default': 'vacuum'},
        'output_dir': {'type': 'path', 'is_output_path': True},
    }, 'steps': [step('first', params={'mode': '$input.mode', 'target': '$input.output_dir'})]}
    result = engine.execute(workflow, {'mode': 'VACUUM', 'output_dir': tmp_path / 'new'})
    assert result.outputs['first']['mode'] == 'vacuum'
    assert result.outputs['first']['target'] == str(tmp_path / 'new')

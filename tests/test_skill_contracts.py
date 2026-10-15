"""Scientific dispatch must reject malformed values before invoking tools."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from zhuangzhou.skills.registry import Skill, SkillParam


def skill(param, fn=None):
    return Skill("test", "test", "test", fn or (lambda **kw: kw), inputs=[param])


@pytest.mark.parametrize("kind,value", [
    ("bool", "maybe"), ("bool", 2), ("int", 1.5), ("int", True),
    ("int", float("inf")), ("float", "nan"), ("float", float("inf")),
    ("float", True), ("list", 42), ("dict", []), ("string", 5),
    ("set", [[1]]), ("path", 42),
])
def test_invalid_type_never_dispatches(kind, value):
    fn = Mock()
    result = skill(SkillParam("value", kind), fn).execute(value=value)
    assert "error" in result
    fn.assert_not_called()


@pytest.mark.parametrize("kind,value,expected", [
    ("bool", "false", False), ("bool", "yes", True), ("bool", 0, False),
    ("int", "3", 3), ("int", 3.0, 3), ("float", "1.25", 1.25),
    ("list", "A1G, V2L", ["A1G", "V2L"]), ("set", "A1G, A1G", {"A1G"}),
])
def test_supported_coercion(kind, value, expected):
    result = skill(SkillParam("value", kind)).execute(value=value)
    assert result["value"] == expected
    assert type(result["value"]) is type(expected)


@pytest.mark.parametrize("kind,value", [("path", " "), ("string", ""), ("list", []), ("dict", None)])
def test_required_empty_values_rejected(kind, value):
    fn = Mock()
    assert "error" in skill(SkillParam("value", kind), fn).execute(value=value)
    fn.assert_not_called()


def test_pathlib_supported(tmp_path):
    path = tmp_path / "蛋白.pdb"
    path.write_text("END\n", encoding="utf-8")
    result = skill(SkillParam("value", "path")).execute(value=path)
    assert result["value"] == str(path)


def test_choices_are_normalized_for_actual_call():
    result = skill(SkillParam("value", "string", choices=["vacuum", "solvated"])).execute(value="VACUUM")
    assert result["value"] == "vacuum"


def test_mutable_defaults_are_independent():
    def mutate(**kw):
        kw["value"].append("one")
        return kw
    s = skill(SkillParam("value", "list", required=False, default=[]), mutate)
    assert s.execute()["value"] == ["one"]
    assert s.execute()["value"] == ["one"]
    assert s.inputs[0].default == []


def test_output_path_may_not_exist(tmp_path):
    path = tmp_path / "new" / "result.json"
    result = skill(SkillParam("value", "path", is_output_path=True)).execute(value=str(path))
    assert result["value"] == str(path)

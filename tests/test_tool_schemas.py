from unittest.mock import Mock

import pytest

from zhuangzhou.nlp.agent import ToolGenerator
from zhuangzhou.nlp.tools import skill_to_tool
from zhuangzhou.skills.registry import Skill, SkillParam, get_registry


def test_tool_schema_generators_use_same_contract():
    skill = Skill('fixture', 'test', 'chemistry', Mock(), inputs=[SkillParam('values', 'list')])
    assert ToolGenerator.generate_schema(skill) == skill_to_tool(skill)
    assert skill_to_tool(skill)['input_schema']['properties']['values']['items'] == {}


@pytest.mark.parametrize('value', ['[{"type":"paragraph","text":"fixture"}]',
    [{'type': 'paragraph', 'text': 'fixture'}], {'sections': [{'text': 'fixture'}]}])
def test_report_contract_accepts_existing_supported_formats(value):
    report = get_registry().get('generate_report')
    with pytest.MonkeyPatch.context() as patcher:
        fn = Mock(return_value={'accepted': True})
        patcher.setattr(report, 'fn', fn)
        assert report.execute(sections=value) == {'accepted': True}
        assert fn.call_args.kwargs['sections'] == value

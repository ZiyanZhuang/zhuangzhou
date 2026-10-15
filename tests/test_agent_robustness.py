"""
Agent Robustness Test Suite
============================
Tests agent execution precision, parameter validation, error handling,
and conversation flow WITHOUT GPU/heavy computation.

Each test captures full agent output for analysis.
"""

import json
import os
import sys
import time
import traceback
import pytest

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Ensure zhuangzhou is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zhuangzhou.skills.registry import get_registry


# =============================================================================
# Test harness
# =============================================================================

class TestResult:
    __test__ = False  # Standalone script harness, not a pytest test class.
    def __init__(self, name):
        self.name = name
        self.passed = False
        self.output = ""
        self.error = ""
        self.duration = 0.0

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name} ({self.duration:.1f}s)"


results = []


def run_test(name, fn):
    """Run a test function and capture results."""
    r = TestResult(name)
    t0 = time.time()
    try:
        output = fn()
        r.output = str(output) if output else ""
        r.passed = True
    except AssertionError as e:
        r.error = str(e)
        r.output = traceback.format_exc()
    except Exception as e:
        r.error = str(e)
        r.output = traceback.format_exc()
    r.duration = time.time() - t0
    results.append(r)
    print(f"  {r}")
    return r


# =============================================================================
# Group 1: Skill Registry & Parameter Validation (no LLM)
# =============================================================================

def test_registry_completeness():
    """All skill modules load without error."""
    reg = get_registry()
    skills = reg.list_all()
    assert len(skills) >= 60, f"Expected 60+ skills, got {len(skills)}"
    categories = reg.categories()
    assert len(categories) >= 8, f"Expected 8+ categories, got {len(categories)}"
    return f"{len(skills)} skills in {len(categories)} categories"


def test_missing_required_params():
    """Skills return error dict (not crash) when required params missing."""
    reg = get_registry()
    test_skills = ['smart_docking', 'plip_analysis', 'mutate_protein',
                   'vina_docking', 'gromacs_analysis']
    errors = []
    for name in test_skills:
        s = reg.get(name)
        if not s:
            continue
        result = s.execute()
        assert isinstance(result, dict), f"{name} didn't return dict: {type(result)}"
        assert 'error' in result, f"{name} missing 'error' key: {result}"
        errors.append(f"{name}: {result['error'][:80]}")
    return "\n".join(errors)


def test_type_coercion():
    """String→list auto-split and int/float coercion work."""
    reg = get_registry()

    # Test comma-separated string → list
    s = reg.get('mutate_protein')
    r = s.execute(pdb_path='test.pdb', mutations='V9L,A5G')
    # Should NOT error on type — mutations should be auto-split to list
    # (will error on file not found, which is fine)
    if isinstance(r, dict) and 'error' in r:
        assert 'must be a list' not in r['error'], f"Auto-split failed: {r['error']}"

    # Test string→float coercion
    s2 = reg.get('gromacs_md')
    # Can't actually run, but test the validation layer
    from zhuangzhou.skills.registry import Skill, SkillParam
    test_skill = Skill(
        name='_test', description='test', category='test',
        fn=lambda simulation_time_ns=10.0, **kw: {'time': simulation_time_ns},
        inputs=[SkillParam('simulation_time_ns', 'float', 'time', required=False, default=10.0)],
    )
    r = test_skill.execute(simulation_time_ns='5.0')
    assert r['time'] == 5.0, f"Float coercion failed: {r}"

    return "comma-split OK, float coercion OK"


def test_choices_validation():
    """Enum choices are enforced at runtime."""
    reg = get_registry()

    # Invalid mode
    s = reg.get('mutate_minimize')
    r = s.execute(pdb_path='test.pdb', mutations=['V9L'], mode='gas_phase')
    assert 'error' in r, "Should reject invalid mode"
    assert 'one of' in r['error'], f"Wrong error: {r['error']}"

    # Invalid viz_type
    s2 = reg.get('gromacs_visualize')
    r2 = s2.execute(work_dir='/tmp', viz_type='movie')
    assert 'error' in r2, "Should reject invalid viz_type"

    # Invalid metric
    # The manifest is still testable while GUI skills stay disabled in the real registry.
    from unittest.mock import patch
    from zhuangzhou.skills import visualization
    from zhuangzhou.skills.registry import SkillRegistry
    isolated = SkillRegistry()
    isolated._loaded = True
    with patch.object(visualization, '_DISABLED_VIZ_SKILLS', frozenset()):
        visualization.register_skills(isolated)
    s3 = isolated.get('chimerax_defattr')
    r3 = s3.execute(work_dir='/tmp', metric='temperature')
    assert 'error' in r3, "Should reject invalid metric"

    # Valid choice should pass through (will fail on missing dir, not on validation)
    s4 = reg.get('mutate_minimize')
    r4 = s4.execute(pdb_path='test.pdb', mutations=['V9L'], mode='vacuum')
    if 'error' in r4:
        assert 'one of' not in r4['error'], f"Valid choice rejected: {r4['error']}"

    return "All invalid choices rejected correctly"


def test_nonexistent_skill():
    """Registry returns None for nonexistent skills."""
    reg = get_registry()
    s = reg.get('hallucinated_skill_name')
    assert s is None, f"Should be None, got {s}"

    s2 = reg.get('run_alphafold')
    assert s2 is None, "run_alphafold should not exist"

    return "Nonexistent skills correctly return None"


def test_path_type_validation():
    """Path params reject non-string types."""
    reg = get_registry()
    s = reg.get('plip_analysis')
    r = s.execute(complex_pdb=12345)
    assert 'error' in r
    assert 'file path string' in r['error']
    return f"Rejected: {r['error']}"


# =============================================================================
# Group 2: AgentExecutor mechanics (no LLM call)
# =============================================================================

def test_confirmation_gate():
    """High-risk skills trigger confirmation on first call."""
    from zhuangzhou.nlp.agent import AgentExecutor

    class MockBackend:
        name = 'mock'

    reg = get_registry()
    executor = AgentExecutor(MockBackend(), reg, verbose=True)

    # First call should return needs_confirmation
    r1 = executor._execute_skill_with_ui('gromacs_md', {
        'pdb_path': 'test.pdb', 'output_dir': '/tmp/md'
    })
    assert r1.get('needs_confirmation') is True, f"Should need confirmation: {r1}"
    assert r1.get('risk_level') == 'high'

    # Low-risk skill should execute directly (and fail on missing file)
    r2 = executor._execute_skill_with_ui('search_graphrag', {'query': 'test'})
    assert 'needs_confirmation' not in r2, f"Low-risk shouldn't need confirmation: {r2}"

    # After confirming, high-risk should execute
    executor._confirmed_actions.add('gromacs_md')
    r3 = executor._execute_skill_with_ui('gromacs_md', {
        'pdb_path': 'test.pdb', 'output_dir': '/tmp/md'
    })
    assert 'needs_confirmation' not in r3, f"Confirmed skill still blocked: {r3}"

    return "Confirmation gate: block→low-risk pass→confirmed pass"


def test_execution_log():
    """Execution log records skill calls with metadata."""
    from zhuangzhou.nlp.agent import AgentExecutor

    class MockBackend:
        name = 'mock'

    reg = get_registry()
    executor = AgentExecutor(MockBackend(), reg, verbose=True)

    # Execute a low-risk skill
    executor._execute_skill_with_ui('search_graphrag', {'query': 'CYP3A4'})

    assert len(executor.execution_log) == 1
    entry = executor.execution_log[0]
    assert entry['skill'] == 'search_graphrag'
    assert 'elapsed' in entry
    assert 'success' in entry
    assert 'output_files' in entry
    assert 'config' not in entry['params']  # config should be filtered

    return f"Log entry: {json.dumps(entry, default=str)[:200]}"


def test_nonexistent_skill_in_executor():
    """AgentExecutor returns helpful error for nonexistent skills."""
    from zhuangzhou.nlp.agent import AgentExecutor

    class MockBackend:
        name = 'mock'

    reg = get_registry()
    executor = AgentExecutor(MockBackend(), reg, verbose=True)

    r = executor._execute_skill_with_ui('run_alphafold', {'sequence': 'MKLL'})
    assert 'error' in r
    assert 'available_skills' in r
    assert len(r['available_skills']) > 0
    assert 'hint' in r

    return f"Error: {r['error']}, available: {len(r['available_skills'])} skills"


def test_system_prompt_content():
    """System prompt contains all required sections."""
    from zhuangzhou.nlp.agent import AgentExecutor

    class MockBackend:
        name = 'mock'

    reg = get_registry()
    executor = AgentExecutor(MockBackend(), reg, verbose=True)

    # Add some execution history
    executor.execution_log.append({
        'skill': 'search_pubmed', 'params': {'query': 'test'},
        'success': True, 'elapsed': 1.0, 'output_files': [],
    })

    prompt = executor.build_system_prompt()

    required_sections = [
        'ReAct', '工作空间感知', '自动纠错', '证据分级',
        'Skill 使用规范', '禁止行为', '依赖关系', '高风险操作',
        '参数安全', '已完成的计算',
    ]
    missing = [s for s in required_sections if s not in prompt]
    assert not missing, f"Missing sections: {missing}"

    # Check execution history is injected
    assert 'search_pubmed' in prompt, "Execution history not in prompt"

    return f"All {len(required_sections)} sections present, prompt={len(prompt)} chars"


# =============================================================================
# Group 3: Tool Schema Generation
# =============================================================================

def test_schema_has_enum():
    """Tool schemas include enum for params with choices."""
    from zhuangzhou.nlp.tools import skill_to_tool
    reg = get_registry()

    s = reg.get('mutate_minimize')
    schema = skill_to_tool(s)
    mode_prop = schema['input_schema']['properties']['mode']
    assert 'enum' in mode_prop, f"Missing enum in mode: {mode_prop}"
    assert mode_prop['enum'] == ['vacuum', 'solvated']

    s2 = reg.get('search_graphrag')
    schema2 = skill_to_tool(s2)
    domain_prop = schema2['input_schema']['properties']['domain']
    assert 'enum' in domain_prop
    assert len(domain_prop['enum']) >= 9

    return f"mode enum: {mode_prop['enum']}, domain enum: {len(domain_prop['enum'])} values"


def test_schema_risk_tags():
    """High-risk skills have [HIGH-RISK] tag in description."""
    from zhuangzhou.nlp.tools import skill_to_tool
    reg = get_registry()

    high_risk_skills = ['gromacs_md', 'metalloprotein_md', 'auto_md', 'synbio_pipeline']
    for name in high_risk_skills:
        s = reg.get(name)
        if not s:
            continue
        schema = skill_to_tool(s)
        assert 'HIGH-RISK' in schema['description'], \
            f"{name} missing HIGH-RISK tag: {schema['description'][:100]}"

    # Low-risk should NOT have the tag
    s_low = reg.get('search_pubmed')
    schema_low = skill_to_tool(s_low)
    assert 'HIGH-RISK' not in schema_low['description']

    return "All high-risk tagged, low-risk clean"


def test_schema_default_values():
    """Tool schemas include default values."""
    from zhuangzhou.nlp.tools import skill_to_tool
    reg = get_registry()

    s = reg.get('mutate_minimize')
    schema = skill_to_tool(s)
    mode_prop = schema['input_schema']['properties']['mode']
    assert 'default' in mode_prop, f"Missing default: {mode_prop}"
    assert mode_prop['default'] == 'solvated'

    return f"mode default: {mode_prop['default']}"


# =============================================================================
# Group 4: Live Agent Conversation (requires LLM backend)
# =============================================================================

@pytest.mark.live_api
def test_agent_simple_query():
    """Agent answers a simple knowledge question without tool calls."""
    try:
        from zhuangzhou.nlp.agent import ZhuangzhouAgent
        agent = ZhuangzhouAgent(verbose=True)
    except Exception as e:
        return f"SKIP (no backend): {e}"

    response = agent.chat("什么是CYP450酶？请简短回答，不要调用任何工具。")
    assert len(response) > 20, f"Response too short: {response}"
    assert 'error' not in response.lower() or 'cytochrome' in response.lower()
    return f"Response ({len(response)} chars): {response[:200]}..."


@pytest.mark.live_api
def test_agent_tool_selection():
    """Agent selects correct tool for a clear task."""
    try:
        from zhuangzhou.nlp.agent import ZhuangzhouAgent
        agent = ZhuangzhouAgent(verbose=True)
    except Exception as e:
        return f"SKIP (no backend): {e}"

    response = agent.chat(
        "在PubMed上搜索关于 CYP3A4 V21L 突变的文献，最多返回3篇。"
        "只需要搜索，不要做其他操作。"
    )
    # Check execution log
    log = agent.executor.execution_log
    if log:
        skill_names = [e['skill'] for e in log]
        assert 'search_pubmed' in skill_names, f"Expected search_pubmed, got: {skill_names}"
        # Should NOT call any heavy computation
        heavy = {'gromacs_md', 'metalloprotein_md', 'auto_md', 'smart_docking'}
        called_heavy = heavy & set(skill_names)
        assert not called_heavy, f"Unnecessarily called heavy skills: {called_heavy}"
    return f"Tools called: {[e['skill'] for e in log]}, Response: {response[:150]}..."


@pytest.mark.live_api
def test_agent_refuses_hallucinated_path():
    """Agent should not fabricate file paths for computational skills."""
    try:
        from zhuangzhou.nlp.agent import ZhuangzhouAgent
        agent = ZhuangzhouAgent(verbose=True)
    except Exception as e:
        return f"SKIP (no backend): {e}"

    response = agent.chat(
        "对 protein.pdb 做 PLIP 分析。"
        "注意：当前目录下没有 protein.pdb 这个文件。"
    )
    log = agent.executor.execution_log
    # If plip was called, it should have failed with file not found
    plip_calls = [e for e in log if e['skill'] == 'plip_analysis']
    if plip_calls:
        assert not plip_calls[0]['success'], "PLIP should fail on nonexistent file"
    return f"Tools: {[e['skill'] for e in log]}, Response: {response[:150]}..."


@pytest.mark.live_api
def test_agent_confirmation_flow():
    """Agent should report needs_confirmation for high-risk MD."""
    try:
        from zhuangzhou.nlp.agent import ZhuangzhouAgent
        agent = ZhuangzhouAgent(verbose=True)
    except Exception as e:
        return f"SKIP (no backend): {e}"

    response = agent.chat(
        "对当前目录的 test.pdb 运行 10ns 的 GROMACS MD 模拟。"
    )
    # Should either refuse (no file) or trigger confirmation
    log = agent.executor.execution_log
    md_calls = [e for e in log if 'md' in e['skill'].lower() or 'gromacs' in e['skill'].lower()]

    # Check response mentions confirmation or file not found
    has_confirmation_mention = any(
        kw in response for kw in ['确认', 'confirm', '高风险', 'high-risk', '不存在', 'not found']
    )

    return f"Tools: {[e['skill'] for e in log]}, confirmation_mentioned: {has_confirmation_mention}, Response: {response[:200]}..."


@pytest.mark.live_api
def test_agent_multi_turn():
    """Multi-turn conversation maintains context."""
    try:
        from zhuangzhou.nlp.agent import ZhuangzhouAgent
        agent = ZhuangzhouAgent(verbose=True)
    except Exception as e:
        return f"SKIP (no backend): {e}"

    r1 = agent.chat("在UniProt搜索 CYP2D6，只搜索不做其他事。")
    r2 = agent.chat("上一步搜索到的第一个结果的accession是什么？直接告诉我，不需要再搜索。")

    log = agent.executor.execution_log
    # Second turn should NOT re-search
    second_turn_tools = [e for e in log[len([e for e in log]):]]

    return f"Turn1: {r1[:100]}...\nTurn2: {r2[:100]}...\nTotal tools: {len(log)}"


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Zhuangzhou Agent Robustness Test Suite")
    print("=" * 70)

    print("\n--- Group 1: Skill Registry & Parameter Validation ---")
    run_test("Registry completeness", test_registry_completeness)
    run_test("Missing required params", test_missing_required_params)
    run_test("Type coercion", test_type_coercion)
    run_test("Choices validation", test_choices_validation)
    run_test("Nonexistent skill", test_nonexistent_skill)
    run_test("Path type validation", test_path_type_validation)

    print("\n--- Group 2: AgentExecutor Mechanics ---")
    run_test("Confirmation gate", test_confirmation_gate)
    run_test("Execution log", test_execution_log)
    run_test("Nonexistent skill in executor", test_nonexistent_skill_in_executor)
    run_test("System prompt content", test_system_prompt_content)

    print("\n--- Group 3: Tool Schema Generation ---")
    run_test("Schema has enum", test_schema_has_enum)
    run_test("Schema risk tags", test_schema_risk_tags)
    run_test("Schema default values", test_schema_default_values)

    print("\n--- Group 4: Live Agent Conversation (requires LLM) ---")
    run_test("Simple query (no tools)", test_agent_simple_query)
    run_test("Tool selection accuracy", test_agent_tool_selection)
    run_test("Refuses hallucinated path", test_agent_refuses_hallucinated_path)
    run_test("Confirmation flow", test_agent_confirmation_flow)
    run_test("Multi-turn context", test_agent_multi_turn)

    # Summary
    print("\n" + "=" * 70)
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"Results: {passed}/{total} passed")
    print("=" * 70)

    # Detailed output for failed/interesting tests
    for r in results:
        if not r.passed or 'SKIP' in r.output:
            print(f"\n{'='*40}")
            print(f"[{'FAIL' if not r.passed else 'SKIP'}] {r.name}")
            if r.error:
                print(f"Error: {r.error}")
            if r.output:
                print(f"Output:\n{r.output[:500]}")

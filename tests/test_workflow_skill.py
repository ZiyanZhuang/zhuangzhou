def test_pareto_harness_status_initializes_without_undefined_os(tmp_path, monkeypatch):
    from zhuangzhou.skills.workflow_skill import pareto_al_harness
    monkeypatch.chdir(tmp_path)
    status = pareto_al_harness(step='status')
    assert isinstance(status, dict)
    assert 'error' not in status

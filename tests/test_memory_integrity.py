import json
from unittest.mock import patch

import pytest

from zhuangzhou.memory.execution_history import ExecutionHistory, TaskCheckpoint, _safe_params
from zhuangzhou.utils.robustness import safe_json_parse


def test_json_cache_results_cannot_mutate_each_other():
    text = '{"nested":[1]}'
    safe_json_parse(text)['nested'].append(2)
    assert safe_json_parse(text) == {'nested': [1]}


def test_execution_params_redact_nested_credentials():
    params = {'api_key': 'fixture-secret', 'nested': {'Authorization': 'fixture-token'},
              'sequence': 'ACGT', 'config': {'password': 'fixture-password'}}
    result = _safe_params(params)
    assert 'fixture-secret' not in json.dumps(result)
    assert 'fixture-token' not in json.dumps(result)
    assert result['sequence'] == 'ACGT'
    assert params['api_key'] == 'fixture-secret'


def test_checkpoint_replace_failure_preserves_previous_file(tmp_path):
    checkpoint = TaskCheckpoint(str(tmp_path))
    checkpoint.save('previous', [])
    with patch('os.replace', side_effect=OSError('disk failure')):
        with pytest.raises(OSError):
            checkpoint.save('next', [])
    assert checkpoint.load()['user_request'] == 'previous'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['checkpoint.json']


def test_history_handles_invalid_records_and_bounded_limits(tmp_path):
    path = tmp_path / 'execution_log.jsonl'
    path.write_text('[]\ninvalid\n' + '\n'.join(json.dumps({'skill': 'test', 'i': i}) for i in range(25)), encoding='utf-8')
    history = ExecutionHistory(str(tmp_path))
    assert history.get_recent(limit=2) == [{'skill': 'test', 'i': 23}, {'skill': 'test', 'i': 24}]
    assert history.get_recent(limit=0) == []
    with pytest.raises(ValueError):
        history.get_recent(limit=-1)
    assert history.get_last_session_log('missing') == []
    assert history.get_session_ids() == []
    assert history.summary()['total_executions'] == 25


def test_session_archive_ids_cannot_escape_the_archive_directory(tmp_path):
    from zhuangzhou.memory.archive import SessionArchive
    outside = tmp_path / 'outside.json'
    outside.write_text('{"private":"fixture"}', encoding='utf-8')
    archive = SessionArchive(str(tmp_path / 'archive'))
    assert archive.load_session('../outside') is None


@pytest.mark.parametrize('data', [[], {'timestamp': 'not-a-date', 'status': 'interrupted'}])
def test_invalid_checkpoint_cannot_masquerade_as_resumable_task(tmp_path, data):
    (tmp_path / 'checkpoint.json').write_text(json.dumps(data), encoding='utf-8')
    assert TaskCheckpoint(str(tmp_path)).load() is None

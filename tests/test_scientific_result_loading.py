"""Validate saved score metadata only; no language model or PLM inference runs."""

import json
from unittest.mock import patch

import pytest

from zhuangzhou.tools import classify_mutations, load_esm2_results
from zhuangzhou.skills.registry import get_registry


def write_result(directory, model='fixture-A', **changes):
    directory.mkdir(exist_ok=True)
    record = {'position': 1, 'wt': 'A', 'mut': 'G', 'mutation': 'A1G', 'delta_ll': 2.0}
    record.update(changes)
    path = directory / 'esm2_beneficial_mutations.json'
    path.write_text(json.dumps({'model': model, 'mutations': [record]}), encoding='utf-8')
    return str(directory)


def test_aggregation_semantics_unchanged_and_skill_returns_structured_result(tmp_path):
    first = write_result(tmp_path / 'first')
    second = write_result(tmp_path / 'second', model='fixture-B', delta_ll=3.0)
    merged = load_esm2_results([first, second])
    assert merged[0]['delta_ll'] == 3.0
    assert merged[0]['avg_delta_ll'] == 2.5
    assert merged[0]['consensus_score'] == 3.25
    assert merged[0]['n_models'] == 2
    result = get_registry().get('esm2_load').execute(result_dirs=[first, second])
    assert result['mutations'] == merged
    assert result['sources'] == [first, second]


def test_missing_requested_model_cannot_silently_change_consensus(tmp_path):
    first = write_result(tmp_path / 'first')
    with pytest.raises(FileNotFoundError):
        load_esm2_results([first, str(tmp_path / 'missing')])


@pytest.mark.parametrize('changes', [{'delta_ll': float('nan')}, {'delta_ll': float('inf')},
    {'delta_ll': '2.0'}, {'position': 0}, {'position': True}, {'mutation': 'A2G'}, {'wt': 'AA'}])
def test_invalid_saved_scores_are_rejected(tmp_path, changes):
    source = write_result(tmp_path / 'first', **changes)
    with pytest.raises(ValueError):
        load_esm2_results([source])


@pytest.mark.parametrize('mut', ['G', 'L'])
def test_conflicting_reference_residue_is_not_merged(tmp_path, mut):
    first = write_result(tmp_path / 'first')
    second = write_result(tmp_path / 'second', model='fixture-B', wt='V', mut=mut, mutation=f'V1{mut}')
    with pytest.raises(ValueError, match='reference'):
        load_esm2_results([first, second])


def test_conflicting_duplicate_model_score_is_not_last_writer_wins(tmp_path):
    first = write_result(tmp_path / 'first')
    second = write_result(tmp_path / 'second', delta_ll=9.0)
    with pytest.raises(ValueError, match='Conflicting'):
        load_esm2_results([first, second])


def test_duplicate_identical_scores_do_not_inflate_model_count(tmp_path):
    source = write_result(tmp_path / 'first')
    assert load_esm2_results([source, source])[0]['n_models'] == 1


def test_unavailable_classifier_is_an_explicit_error():
    with patch.dict('sys.modules', {'mutation_classifier': None}):
        result = classify_mutations('fixture.pdb', ['A1G'])
    assert isinstance(result, dict)
    assert 'error' in result


def test_saved_result_reaches_template_classifier_with_correct_shape(tmp_path):
    from zhuangzhou.workflow.engine import WorkflowEngine
    source = write_result(tmp_path / 'scores')
    pdb = tmp_path / 'fixture.pdb'
    pdb.write_text('END\n', encoding='utf-8')
    engine = WorkflowEngine()
    template = engine.load_workflow('mutation_screening')
    template['steps'] = template['steps'][:2]
    classifier = engine.registry.get('mutation_classification')
    with patch.object(classifier, 'fn', side_effect=lambda mutations, **kw: {'seen': mutations}):
        result = engine.execute(template, {'pdb_path': str(pdb), 'binding_site': [1], 'esm2_dirs': [source]})
    assert result.success
    assert result.outputs['structural_roles']['seen'][0]['mutation'] == 'A1G'

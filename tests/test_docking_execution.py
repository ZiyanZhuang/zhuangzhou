"""External process is a fixture; real runner validates paths, scores and failures."""

from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import Mock, patch

import pytest

from zhuangzhou.smartdocking.vina import run_vina
from zhuangzhou.smartdocking.pipeline import SmartDocking


@pytest.fixture
def inputs(tmp_path):
    receptor = tmp_path / 'receptor.pdbqt'
    ligand = tmp_path / 'ligand.pdbqt'
    receptor.write_text('ATOM      1  C   ALA A   1       0.000   0.000   0.000\n', encoding='utf-8')
    ligand.write_text(receptor.read_text(encoding='utf-8'), encoding='utf-8')
    grid = dict(center_x=0., center_y=0., center_z=0., size_x=20., size_y=20., size_z=20.)
    return receptor, ligand, grid, tmp_path


def test_zero_exit_without_current_artifacts_is_failure(inputs):
    receptor, ligand, grid, directory = inputs
    # Old successful artifacts must not be mistaken for this invocation's outputs.
    (directory / 'vina_docked.pdbqt').write_text('REMARK VINA RESULT: -9 0 0\nATOM  stale\n', encoding='utf-8')
    (directory / 'vina_log.txt').write_text('-----+---\n1 -9 0 0\n', encoding='utf-8')
    with patch('zhuangzhou.smartdocking.vina._find_vina', return_value=('fixture-vina', False)), \
         patch('zhuangzhou.smartdocking.vina.subprocess.run', return_value=CompletedProcess([], 0, '', '')):
        result = run_vina(str(receptor), str(ligand), grid, str(directory))
    assert 'error' in result


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1, 0])
def test_invalid_box_is_rejected_before_dispatch(inputs, value):
    receptor, ligand, grid, directory = inputs
    grid['size_x'] = value
    runner = Mock(return_value=CompletedProcess([], 0, '', ''))
    with patch('zhuangzhou.smartdocking.vina._find_vina', return_value=('fixture-vina', False)), \
         patch('zhuangzhou.smartdocking.vina.subprocess.run', runner):
        assert 'error' in run_vina(str(receptor), str(ligand), grid, str(directory))
    runner.assert_not_called()


def write_success(command, **kwargs):
    output = Path(command[command.index('--out') + 1])
    log = Path(command[command.index('--log') + 1])
    output.write_text('REMARK VINA RESULT: -7.0 0.0 0.0\nATOM      1  C   LIG A   1       0.000   0.000   0.000\n', encoding='utf-8')
    log.write_text('-----+---\n1 -7.0 0.0 0.0\n', encoding='utf-8')
    return CompletedProcess(command, 0, '', '')


def test_successful_runs_have_distinct_artifacts_and_finite_scores(inputs):
    receptor, ligand, grid, directory = inputs
    with patch('zhuangzhou.smartdocking.vina._find_vina', return_value=('fixture-vina', False)), \
         patch('zhuangzhou.smartdocking.vina.subprocess.run', side_effect=write_success):
        first = run_vina(str(receptor), str(ligand), grid, str(directory))
        second = run_vina(str(receptor), str(ligand), grid, str(directory))
    assert 'error' not in first
    assert first['best_affinity'] == -7.0
    assert first['output_pdbqt'] != second['output_pdbqt']
    assert Path(first['output_pdbqt']).is_file()


def test_pipeline_propagates_vina_failure_at_top_level(inputs, monkeypatch):
    receptor, ligand, grid, directory = inputs
    import zhuangzhou.smartdocking.pipeline as module
    for name, result in {'preprocess_protein': {'pdb_path': str(receptor)}, 'detect_cofactors': [],
        'detect_ligands': [], 'define_active_site': grid, 'pdb_to_pdbqt': str(receptor),
        'sdf_to_pdbqt': str(ligand), 'run_vina': {'error': 'injected docking failure'}}.items():
        monkeypatch.setattr(module, name, Mock(return_value=result))
    docking = SmartDocking(str(receptor), str(directory / 'pipeline'))
    result = docking.run(ligand_sdf=str(ligand), chain='A')
    assert result['error'] == 'injected docking failure'
    assert result['success'] is False


def test_smart_docking_skip_flag_never_starts_compute(inputs):
    from zhuangzhou.skills.docking import smart_docking
    receptor, ligand, _, _ = inputs
    with patch('zhuangzhou.tools.run_smart_docking') as run:
        result = smart_docking(str(receptor), str(ligand), run_vina=False)
    run.assert_not_called()
    assert 'error' in result  # Unsupported preparation-only mode is explicit.


def test_smart_docking_plip_uses_the_docked_complex(inputs):
    from zhuangzhou.skills.docking import smart_docking
    receptor, ligand, _, directory = inputs
    complex_path = directory / 'pose_complex.pdb'
    complex_path.write_text('ATOM  fixture\n', encoding='utf-8')
    with patch('zhuangzhou.tools.run_smart_docking', return_value={'success': True, 'complex_pdb': str(complex_path)}), \
         patch('zhuangzhou.tools.run_plip_analysis', return_value={'interactions': []}) as plip:
        result = smart_docking(str(receptor), str(ligand), run_plip=True)
    assert result['plip'] == {'interactions': []}
    assert plip.call_args.args[0] == str(complex_path)


def test_docking_template_uses_docked_pose_for_interaction_analysis():
    import json
    root = Path(__file__).resolve().parents[1]
    template = json.loads((root / 'workflow/templates/docking_only.json').read_text(encoding='utf-8'))
    assert all(step['skill'] != 'generate_complex' for step in template['steps'])
    plip_step = next(step for step in template['steps'] if step['skill'] == 'plip_analysis')
    assert plip_step['params']['complex_pdb'] == '$step_dock.complex_pdb'


def test_real_template_routes_interaction_input_to_actual_pose(inputs):
    from zhuangzhou.workflow.engine import WorkflowEngine
    receptor, ligand, _, directory = inputs
    complex_path = directory / 'computed_pose.pdb'
    complex_path.write_text('ATOM  computed fixture\n', encoding='utf-8')
    engine = WorkflowEngine()
    with patch('zhuangzhou.tools.run_smart_docking', return_value={'success': True, 'complex_pdb': str(complex_path)}) as docking, \
         patch('zhuangzhou.tools.run_plip_analysis', return_value={'interactions': []}) as plip:
        result = engine.execute(engine.load_workflow('docking_only'), {
            'protein_pdb': str(receptor), 'ligand_sdf': str(ligand), 'center': [1, 2, 3],
        })
    assert result.success
    assert result.steps_completed == ['step_dock', 'step_plip']
    assert plip.call_args.args[0] == str(complex_path)
    assert plip.call_count == 1
    assert docking.call_args.kwargs['center'] == [1, 2, 3]


def test_native_pipeline_honors_manual_grid_and_exposes_actual_complex(inputs, monkeypatch):
    receptor, ligand, grid, directory = inputs
    import zhuangzhou.smartdocking.pipeline as module
    run_dir = directory / 'native-run'
    run_dir.mkdir()
    complex_path = run_dir / 'complex.pdb'
    complex_path.write_text('ATOM  fixture\n', encoding='utf-8')
    results = {'preprocess_protein': {'pdb_path': str(receptor)}, 'detect_cofactors': [],
        'detect_ligands': [], 'define_active_site': grid, 'pdb_to_pdbqt': str(receptor),
        'sdf_to_pdbqt': str(ligand), 'run_vina': {'success': True, 'output_pdbqt': str(ligand),
            'run_dir': str(run_dir), 'best_affinity': -7.0, 'affinities': [(1, -7.0, 0, 0)]},
        'generate_complexes': [{'path': str(complex_path), 'rank': 1, 'affinity': -7.0}]}
    mocks = {}
    for name, result in results.items():
        mocks[name] = Mock(return_value=result)
        monkeypatch.setattr(module, name, mocks[name])
    docking = SmartDocking(str(receptor), str(directory / 'pipeline'))
    result = docking.run(ligand_sdf=str(ligand), chain='A', center=[1, 2, 3], box_size=[15, 16, 17])
    assert result['success'] is True
    used_grid = mocks['run_vina'].call_args.args[2]
    assert [used_grid[f'center_{axis}'] for axis in 'xyz'] == [1, 2, 3]
    assert [used_grid[f'size_{axis}'] for axis in 'xyz'] == [15, 16, 17]
    assert result['complex_pdb'] == str(complex_path)
    assert Path(result['report_path']).is_file()

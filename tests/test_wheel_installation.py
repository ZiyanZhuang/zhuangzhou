"""Build and import the actual wheel outside the source tree and editable finder."""

import json
from pathlib import Path
import subprocess
import sys
import zipfile
from email.parser import Parser

import pytest


@pytest.fixture(scope='module')
def wheel(tmp_path_factory):
    directory = tmp_path_factory.mktemp('wheel')
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, '-m', 'pip', 'wheel', str(root), '--no-deps',
                             '--no-build-isolation', '--wheel-dir', str(directory)],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    return next(directory.glob('*.whl'))


def test_wheel_contains_runtime_templates_and_bridge(wheel):
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    assert 'zhuangzhou/CYPamber/cypforge_bridge.py' in names
    assert 'zhuangzhou/workflow/templates/docking_only.json' in names
    assert not any('/.venv/' in name or name.endswith(('.key', '.pem', '.env')) for name in names)


def test_running_version_matches_built_package_metadata(wheel):
    import zhuangzhou
    with zipfile.ZipFile(wheel) as archive:
        metadata_path = next(name for name in archive.namelist() if name.endswith('.dist-info/METADATA'))
        metadata = Parser().parsestr(archive.read(metadata_path).decode('utf-8'))
    assert metadata['Version'] == zhuangzhou.__version__


def test_installed_wheel_imports_and_lists_templates(wheel, tmp_path):
    target = tmp_path / 'site-packages'
    target.mkdir()
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(target)
    code = '''
import importlib, json, sys
from pathlib import Path
target = Path(sys.argv[1])
sys.meta_path = [f for f in sys.meta_path if '__editable__' not in getattr(f, '__module__', '')]
sys.path.insert(0, str(target))
for name in ('zhuangzhou.app', 'zhuangzhou.nlp.agent', 'zhuangzhou.CYPamber.cypforge_bridge',
             'zhuangzhou.synbio.assembor', 'cypforge', 'cypforge_core.orchestrator'):
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(target), name
from zhuangzhou.workflow.engine import WorkflowEngine
from zhuangzhou.synbio.assembor import validate_yeast_parts
assert len(WorkflowEngine().list_templates()) == 4
assert validate_yeast_parts()['valid'] is True
print(json.dumps({'imports': 'passed', 'templates': 4, 'parts': 'valid'}))
'''
    result = subprocess.run([sys.executable, '-I', '-c', code, str(target)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['imports'] == 'passed'

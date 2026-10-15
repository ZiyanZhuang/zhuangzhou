"""Offline tests never use the user's persistent agent state or live services."""

import os
import socket
import tempfile
from pathlib import Path

import pytest


# These are manual scientific runs, some with import-time subprocess/file effects.
# Keep their original direct `python tests/<file>.py` entry points intact.
collect_ignore = [
    "test_chimerax.py", "test_cyp_bonded.py", "test_cyp_cgenff.py",
    "test_cyp_md5.py", "test_metalloprotein_md.py",
]


def pytest_addoption(parser):
    parser.addoption("--run-live-api", action="store_true", help="Allow marked live model/API tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "live_api: opt-in tests that call external models/services")
    config._zz_test_home = tempfile.TemporaryDirectory(prefix="zz-test-home-")
    config._zz_home_patch = pytest.MonkeyPatch()
    original_expanduser = os.path.expanduser
    test_home = config._zz_test_home.name
    config._zz_home_patch.setattr(os.path, "expanduser", lambda path:
        test_home + path[1:] if isinstance(path, str) and (path == "~" or path.startswith(("~/", "~\\")))
        else original_expanduser(path))
    config._zz_home_patch.setattr(Path, "home", classmethod(lambda cls: Path(test_home)))


def pytest_unconfigure(config):
    if hasattr(config, "_zz_home_patch"):
        config._zz_home_patch.undo()
        config._zz_test_home.cleanup()


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-live-api"):
        skip = pytest.mark.skip(reason="Live API test; explicitly opt in with --run-live-api")
        for item in items:
            if "live_api" in item.keywords:
                item.add_marker(skip)


@pytest.fixture(autouse=True)
def isolated_execution_state(tmp_path, monkeypatch):
    import zhuangzhou.memory.execution_history as module
    monkeypatch.setattr(module, 'EXECUTION_DIR', str(tmp_path / 'execution'))


@pytest.fixture(autouse=True)
def offline_network(request, monkeypatch):
    if request.node.get_closest_marker("live_api") and request.config.getoption("--run-live-api"):
        return

    def blocked(*args, **kwargs):
        raise RuntimeError("Network access is disabled in offline tests")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    # A deferred global worker must not outlive this fixture and reach a live model.
    from zhuangzhou.memory.kg_engine import kg_engine
    monkeypatch.setattr(kg_engine, 'add_to_queue', lambda turn: False)

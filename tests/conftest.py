"""Puts this folder on sys.path so shared helper modules import from any subfolder run alone."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# git sets GIT_INDEX_FILE and other GIT_* variables for a commit hook. A test that runs git and
# inherits them would stage into the real repository, so drop them before any test module loads.
for _name in [name for name in os.environ if name.startswith("GIT_")]:
    del os.environ[_name]

HERE = str(Path(__file__).resolve().parent)
if HERE not in sys.path:
    sys.path.insert(0, HERE)


@pytest.fixture
def repo(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    return tmp_path


TIERS = ("fast", "medium", "slow")


def pytest_collection_modifyitems(items):
    for item in items:
        tiers = [tier for tier in TIERS if item.get_closest_marker(tier)]
        if len(tiers) > 1:
            raise pytest.UsageError(f"{item.nodeid} has more than one tier marker: {tiers}")
        if not tiers:
            item.add_marker(pytest.mark.fast)

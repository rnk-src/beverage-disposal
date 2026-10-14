"""The environment the tests run in is the one the project pins."""

import importlib
import importlib.metadata
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from support import lockversions
from support.paths import ROOT


def test_tests_run_on_python_3_12():
    assert sys.version_info[:2] == (3, 12)


@pytest.mark.parametrize("module", ["mujoco", "numpy", "gymnasium", "hypothesis", "PIL", "mink"])
def test_project_dependencies_import(module):
    importlib.import_module(module)


PINNED = ["mujoco", "numpy", "gymnasium", "pytest", "hypothesis", "pillow", "mink"]


def locked_versions():
    return lockversions.locked_versions((ROOT / "uv.lock").read_text())


@pytest.mark.parametrize("name", PINNED)
def test_installed_version_matches_the_lockfile(name):
    assert importlib.metadata.version(name) == locked_versions()[name]


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not on PATH")
def test_lockfile_is_up_to_date_with_pyproject():
    result = subprocess.run(["uv", "lock", "--check"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_project_package_is_installed_from_src():
    package = importlib.import_module("beverage_disposal")
    assert Path(package.__file__).resolve().is_relative_to(ROOT / "src")

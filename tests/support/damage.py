"""Runs a test module against a damaged copy of the vendored model, to prove the tests can fail."""

import os
import shutil
import subprocess
import sys

from support.paths import ROOT
from support.so101 import SO101


def edit(folder, name, old, new):
    path = folder / name
    text = path.read_text()
    assert old in text, f"{old!r} not found in {name}: the fault would not change anything"
    path.write_text(text.replace(old, new))


def run_on_damaged_copy(tmp_path, test_module, fault):
    """Copy the model, apply `fault(copy)`, run `test_module` against it, return pytest's result."""
    copy = tmp_path / "so101"
    shutil.copytree(SO101, copy)
    fault(copy)
    return _run_pytest(tmp_path, test_module, {"SO101_DIR": str(copy)})


def run_on_mutated_package(tmp_path, test_module, fault):
    """Copy our package, apply `fault(copy)`, run `test_module` on the copy, return the result."""
    src = tmp_path / "src"
    shutil.copytree(
        ROOT / "src" / "beverage_disposal",
        src / "beverage_disposal",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    fault(src / "beverage_disposal")
    # PYTHONPATH comes before site-packages, so the copy shadows the editable install.
    return _run_pytest(tmp_path, test_module, {"PYTHONPATH": str(src)})


def _run_pytest(tmp_path, test_module, env):
    # Runs from the scratch folder: MuJoCo appends its warnings to MUJOCO_LOG.TXT in the current
    # directory, and a damaged model can make it warn.
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(ROOT / test_module),
            "-q",
            "--tb=no",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
        ],
        cwd=tmp_path,
        env={**os.environ, **env, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
    )


def failed_ids(result):
    """Test IDs pytest reported as FAILED. Their path is relative to the directory pytest ran in."""
    return [
        line.removeprefix("FAILED ").split(" - ")[0]
        for line in result.stdout.splitlines()
        if line.startswith("FAILED ")
    ]


def assert_the_test_failed(result, test_module, failing_test):
    wanted = f"{test_module}::{failing_test}"
    assert any(id.endswith(wanted) for id in failed_ids(result)), (
        f"{wanted} did not fail on the damaged copy:\n{result.stdout}"
    )

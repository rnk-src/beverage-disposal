"""Tests must not inherit the git variables a commit hook is run with."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")

PLAIN_GIT_TEST = """
import subprocess


def test_a_plain_git_call_uses_its_own_index(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "f.txt").write_text("x")
    subprocess.run(["git", "add", "f.txt"], cwd=tmp_path, check=True)
    listed = subprocess.run(
        ["git", "ls-files"], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    assert listed.stdout.split() == ["f.txt"]
"""


def test_a_commit_hook_environment_does_not_reach_plain_git_calls_in_tests(tmp_path):
    # During `git commit <path>` git points a hook at the real commit's temporary index like this.
    hostile_index = tmp_path / "real-commit" / "next-index.lock"
    hostile_index.parent.mkdir()
    inner = tmp_path / "inner"
    inner.mkdir()
    shutil.copy(TESTS / "conftest.py", inner / "conftest.py")
    (inner / "test_plain_git.py").write_text(PLAIN_GIT_TEST)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "--color=no", "-q"],
        cwd=inner,
        env={**os.environ, "GIT_INDEX_FILE": str(hostile_index)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not hostile_index.exists(), "the inner test wrote into the hook's index"

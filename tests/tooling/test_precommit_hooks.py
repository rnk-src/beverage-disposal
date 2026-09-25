"""What the pre-commit hooks do, run for real in a throwaway git repository."""

import shutil
import subprocess
import sys

import pytest
from support.gitenv import ISOLATED_GIT
from support.paths import ROOT

# The hooks call git and uv, so without either there is nothing to run them against.
pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("uv") is None,
    reason="git and uv are needed to run the hooks",
)

IDENTITY = ["-c", "user.name=test", "-c", "user.email=hook-tests"]

# Fast, medium and slow tests in one file; only the first may run under the fast-tests hook.
TIERED_TESTS = """
import pytest

def test_fast_thing():
    pass

@pytest.mark.medium
def test_medium_thing():
    assert False, "the medium tier must not run in the hook"

@pytest.mark.slow
def test_slow_thing():
    assert False, "the slow tier must not run in the hook"
"""


def git(repo, *args, env=ISOLATED_GIT):
    return subprocess.run(
        ["git", *IDENTITY, *args],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture(scope="module")
def hook_env(tmp_path_factory):
    home = tmp_path_factory.mktemp("hook-home")
    env = {k: v for k, v in ISOLATED_GIT.items() if k != "VIRTUAL_ENV"}
    env["UV_PROJECT_ENVIRONMENT"] = str(home / "venv")
    env["PRE_COMMIT_HOME"] = str(home / "pre-commit-cache")
    return env


@pytest.fixture(scope="module")
def base_repo(tmp_path_factory, hook_env):
    repo = tmp_path_factory.mktemp("hook-repo")
    git(repo, "init", "-q")
    for name in (".pre-commit-config.yaml", "pyproject.toml", "uv.lock", ".python-version"):
        shutil.copy(ROOT / name, repo / name)
    (repo / "src" / "beverage_disposal").mkdir(parents=True)
    (repo / "src" / "beverage_disposal" / "__init__.py").write_text('"""Stub."""\n')
    (repo / "tests").mkdir()
    shutil.copy(ROOT / "tests" / "conftest.py", repo / "tests" / "conftest.py")
    git(repo, "add", "-A")
    assert git(repo, "commit", "-q", "-m", "base").returncode == 0
    return repo


def reset(repo):
    git(repo, "reset", "-q", "--hard")
    git(repo, "clean", "-fdq")


@pytest.fixture
def repo(base_repo):
    """The throwaway repo, put back to its base commit before and after each test."""
    reset(base_repo)
    yield base_repo
    reset(base_repo)


def pre_commit(repo, env, *args):
    return subprocess.run(
        [sys.executable, "-m", "pre_commit", *args],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def output(result):
    return result.stdout + result.stderr


def test_the_lint_hook_rejects_an_unused_import(repo, hook_env):
    (repo / "bad.py").write_text("import os\n")
    result = pre_commit(repo, hook_env, "run", "ruff-check", "--files", "bad.py")
    assert result.returncode == 1, output(result)
    assert "F401" in output(result)


def test_the_lint_hook_accepts_a_clean_file(repo, hook_env):
    (repo / "good.py").write_text('"""Fine."""\n\nVALUE = 1\n')
    result = pre_commit(repo, hook_env, "run", "ruff-check", "--files", "good.py")
    assert result.returncode == 0, output(result)


def test_the_format_hook_rewrites_a_messy_file_and_fails_that_run(repo, hook_env):
    (repo / "messy.py").write_text("x = { 'a':1 }\n")
    git(repo, "add", "messy.py")  # pre-commit sees "modified by the hook" through git diff
    first = pre_commit(repo, hook_env, "run", "ruff-format", "--files", "messy.py")
    assert first.returncode == 1, output(first)
    assert (repo / "messy.py").read_text() == 'x = {"a": 1}\n'
    second = pre_commit(repo, hook_env, "run", "ruff-format", "--files", "messy.py")
    assert second.returncode == 0, output(second)


def test_the_format_hook_leaves_a_formatted_file_alone(repo, hook_env):
    (repo / "tidy.py").write_text('x = {"a": 1}\n')
    git(repo, "add", "tidy.py")
    result = pre_commit(repo, hook_env, "run", "ruff-format", "--files", "tidy.py")
    assert result.returncode == 0, output(result)
    assert (repo / "tidy.py").read_text() == 'x = {"a": 1}\n'


def test_the_fast_tests_hook_runs_only_the_fast_tier(repo, hook_env):
    (repo / "tests" / "test_tiers.py").write_text(TIERED_TESTS)
    git(repo, "add", "tests/test_tiers.py")
    result = pre_commit(repo, hook_env, "run", "fast-tests", "--verbose")
    assert result.returncode == 0, output(result)
    assert "1 passed, 2 deselected" in output(result)


def test_the_fast_tests_hook_fails_when_a_fast_test_fails(repo, hook_env):
    (repo / "tests" / "test_broken.py").write_text("def test_broken():\n    assert False\n")
    git(repo, "add", "tests/test_broken.py")
    result = pre_commit(repo, hook_env, "run", "fast-tests")
    assert result.returncode == 1, output(result)
    assert "test_broken" in output(result)


def test_an_installed_hook_blocks_a_bad_commit_and_lets_a_clean_one_through(repo, hook_env):
    assert pre_commit(repo, hook_env, "install").returncode == 0
    try:
        # A passing test is staged too, so the tests hook cannot be what blocks this commit.
        (repo / "tests" / "test_ok.py").write_text("def test_ok():\n    pass\n")
        (repo / "bad.py").write_text("import os\n")
        git(repo, "add", "bad.py", "tests/test_ok.py")
        blocked = git(repo, "commit", "-q", "-m", "bad", env=hook_env)
        assert blocked.returncode != 0, output(blocked)
        assert git(repo, "rev-list", "--count", "HEAD").stdout.strip() == "1"

        git(repo, "reset", "-q", "--hard")
        git(repo, "clean", "-fdq")
        (repo / "good.py").write_text('"""Fine."""\n\nVALUE = 1\n')
        (repo / "tests" / "test_ok.py").write_text("def test_ok():\n    pass\n")
        git(repo, "add", "good.py", "tests/test_ok.py")
        allowed = git(repo, "commit", "-q", "-m", "good", env=hook_env)
        assert allowed.returncode == 0, output(allowed)
        assert git(repo, "rev-list", "--count", "HEAD").stdout.strip() == "2"
    finally:
        pre_commit(repo, hook_env, "uninstall")


def current_branch(repo):
    return git(repo, "branch", "--show-current").stdout.strip()


def test_the_branch_hook_refuses_to_commit_on_main(repo, hook_env):
    original = current_branch(repo)
    try:
        assert git(repo, "switch", "-q", "-C", "main").returncode == 0
        result = pre_commit(repo, hook_env, "run", "no-commit-to-branch")
        # A missing hook id also exits 1, so insist the hook ran and reported a failure.
        assert "No hook with id" not in output(result), output(result)
        assert "Failed" in output(result), output(result)
        assert result.returncode == 1, output(result)
    finally:
        git(repo, "switch", "-q", original)


def test_the_branch_hook_allows_other_branches_even_with_main_in_the_name(repo, hook_env):
    original = current_branch(repo)
    try:
        assert git(repo, "switch", "-q", "-c", "docs/main-notes").returncode == 0
        result = pre_commit(repo, hook_env, "run", "no-commit-to-branch")
        assert "Passed" in output(result), output(result)
        assert result.returncode == 0, output(result)
    finally:
        git(repo, "switch", "-q", original)
        git(repo, "branch", "-q", "-D", "docs/main-notes")


def test_an_installed_hook_still_lets_a_no_ff_merge_into_main_through(repo, hook_env):
    original = current_branch(repo)
    assert pre_commit(repo, hook_env, "install").returncode == 0
    try:
        assert git(repo, "switch", "-q", "-C", "main").returncode == 0
        assert git(repo, "switch", "-q", "-c", "iter/work").returncode == 0
        # A passing test is staged so the fast tests hook cannot be what blocks the commit.
        (repo / "tests" / "test_on_branch.py").write_text("def test_it():\n    pass\n")
        git(repo, "add", "tests/test_on_branch.py")
        committed = git(repo, "commit", "-q", "-m", "work", env=hook_env)
        assert committed.returncode == 0, output(committed)
        git(repo, "switch", "-q", "main")
        merged = git(repo, "merge", "--no-ff", "-q", "-m", "merge", "iter/work", env=hook_env)
        assert merged.returncode == 0, output(merged)
        assert len(git(repo, "log", "-1", "--format=%p").stdout.split()) == 2  # two parents
    finally:
        pre_commit(repo, hook_env, "uninstall")
        git(repo, "switch", "-q", original)
        git(repo, "branch", "-q", "-D", "iter/work")
        git(repo, "branch", "-q", "-D", "main")

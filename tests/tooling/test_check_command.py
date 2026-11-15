"""One command, `make check`, runs every gate, and the workflow runs that same command."""

import shutil
import subprocess

import pytest
from support.paths import ROOT

pytestmark = pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")

WORKFLOW = ROOT / ".github" / "workflows" / "check.yml"


def planned_steps():
    """What `make check` would run, in order, without running it."""
    result = subprocess.run(
        ["make", "-n", "check"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def test_the_check_command_runs_format_lint_types_and_tests_in_that_order():
    steps = planned_steps()
    wanted = ["ruff format --check", "ruff check", "mypy", "pytest"]
    positions = [next(i for i, step in enumerate(steps) if name in step) for name in wanted]
    assert positions == sorted(positions)


def test_the_check_command_leaves_out_the_docker_tests_that_only_fit_this_machine():
    # The Gazebo image record is for arm64 and the tests need a Docker daemon.
    assert any("--ignore=tests/gazebo" in step for step in planned_steps())


def test_the_check_command_stops_at_the_first_failing_step():
    # `make -n` hides the dash that tells make to carry on after a failure, so read the recipe.
    recipe = [
        line for line in (ROOT / "Makefile").read_text().splitlines() if line.startswith("\t")
    ]
    assert recipe
    assert not any(line.lstrip("\t@").startswith("-") for line in recipe)


def test_the_workflow_runs_make_check_on_pushes_to_main_and_on_pull_requests():
    text = WORKFLOW.read_text()
    assert "make check" in text
    assert "pull_request" in text
    assert "branches: [main]" in text


def test_the_workflow_installs_uv_so_the_locked_environment_is_used():
    assert "astral-sh/setup-uv" in WORKFLOW.read_text()


def test_the_workflow_gives_the_runner_software_rendering_for_the_offscreen_tests():
    # The first run on a hosted Linux runner aborted in MuJoCo's renderer: no display or GPU there.
    text = WORKFLOW.read_text()
    assert "MUJOCO_GL: osmesa" in text
    assert "libosmesa6" in text

"""How the fast/medium/slow markers behave, checked by running pytest on tiny generated tests."""

import shutil
import subprocess
import sys
import textwrap

import pytest
from support.paths import ROOT


def run_inner_pytest(tmp_path, test_source, *args):
    """Run pytest in a scratch folder that uses this project's real pyproject.toml and conftest."""
    shutil.copy(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    shutil.copy(ROOT / "tests" / "conftest.py", tmp_path / "conftest.py")
    (tmp_path / "test_inner.py").write_text(textwrap.dedent(test_source))
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "--color=no", *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )


def test_a_misspelled_tier_marker_is_an_error(tmp_path):
    result = run_inner_pytest(
        tmp_path,
        """
        import pytest

        @pytest.mark.slwo
        def test_typo():
            pass
        """,
    )
    assert result.returncode != 0
    assert "slwo" in result.stdout + result.stderr


def test_the_three_tier_markers_are_registered(tmp_path):
    result = run_inner_pytest(tmp_path, "", "--markers")
    for tier in ("fast", "medium", "slow"):
        assert f"@pytest.mark.{tier}" in result.stdout


ONE_TEST_PER_TIER = """
    import pytest

    @pytest.mark.fast
    def test_fast():
        pass

    @pytest.mark.medium
    def test_medium():
        pass

    @pytest.mark.slow
    def test_slow():
        assert False, "the slow tier must not run"

    def test_unmarked():
        pass
"""


def test_a_plain_run_skips_slow_and_runs_everything_else(tmp_path):
    result = run_inner_pytest(tmp_path, ONE_TEST_PER_TIER, "-q")
    assert result.returncode == 0, result.stdout
    assert "3 passed, 1 deselected" in result.stdout


ONLY_SLOW_MAY_RUN = ONE_TEST_PER_TIER.replace("pass", 'assert False, "only slow may run"').replace(
    'assert False, "the slow tier must not run"', "pass"
)


def test_asking_for_the_slow_tier_runs_only_the_slow_tier(tmp_path):
    result = run_inner_pytest(tmp_path, ONLY_SLOW_MAY_RUN, "-q", "-m", "slow")
    assert result.returncode == 0, result.stdout
    assert "1 passed, 3 deselected" in result.stdout


def test_an_unmarked_test_is_selected_by_the_fast_marker(tmp_path):
    result = run_inner_pytest(tmp_path, ONE_TEST_PER_TIER, "-q", "-m", "fast")
    assert result.returncode == 0, result.stdout
    assert "2 passed, 2 deselected" in result.stdout


def test_a_test_with_two_tier_markers_is_an_error(tmp_path):
    result = run_inner_pytest(
        tmp_path,
        """
        import pytest

        def test_healthy():
            pass

        @pytest.mark.fast
        @pytest.mark.slow
        def test_two_tiers():
            pass
        """,
    )
    assert result.returncode != 0
    assert "test_two_tiers" in result.stdout + result.stderr


def collected(tmp_path, source, *args):
    result = run_inner_pytest(tmp_path, source, "--collect-only", "-q", *args)
    assert result.returncode == 0, result.stdout + result.stderr
    return {line.split("::", 1)[1] for line in result.stdout.splitlines() if "::" in line}


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ([], {"test_fast", "test_medium", "test_unmarked"}),
        (["-m", "slow"], {"test_slow"}),
        (["-m", "medium"], {"test_medium"}),
        (["-m", "fast"], {"test_fast", "test_unmarked"}),
        (
            ["-m", "fast or medium or slow"],
            {"test_fast", "test_medium", "test_slow", "test_unmarked"},
        ),
        (["-m", "not fast"], {"test_medium", "test_slow"}),
    ],
)
def test_each_marker_expression_selects_exactly_its_tiers(tmp_path, args, expected):
    assert collected(tmp_path, ONE_TEST_PER_TIER, *args) == expected


CLASS_LEVEL_SLOW = """
    import pytest

    def test_plain():
        pass

    @pytest.mark.slow
    class TestTraining:
        def test_long_run(self):
            pass
"""


def test_a_slow_marker_on_a_class_makes_its_tests_slow(tmp_path):
    assert collected(tmp_path, CLASS_LEVEL_SLOW, "-m", "fast") == {"test_plain"}
    assert collected(tmp_path, CLASS_LEVEL_SLOW, "-m", "slow") == {"TestTraining::test_long_run"}

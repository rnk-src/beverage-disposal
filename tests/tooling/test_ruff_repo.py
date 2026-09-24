"""The repository as a whole satisfies its own lint and format rules."""

from support.ruffrun import ruff


def test_the_repository_has_no_lint_findings():
    result = ruff("check", ".")
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_repository_is_already_formatted():
    result = ruff("format", "--check", ".")
    assert result.returncode == 0, result.stdout + result.stderr

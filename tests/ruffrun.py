"""Runs ruff for the tests, with this project's own configuration."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def ruff(*args, cwd=ROOT, stdin=None):
    return subprocess.run(
        [sys.executable, "-m", "ruff", *args],
        cwd=cwd,
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )


def lint_codes(tmp_path, source):
    """Rule codes ruff reports for `source`, using a copy of the real pyproject.toml."""
    shutil.copy(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    (tmp_path / "snippet.py").write_text(source)
    result = ruff("check", "--no-cache", "--output-format", "json", "snippet.py", cwd=tmp_path)
    assert result.returncode in (0, 1), result.stderr
    return sorted(finding["code"] for finding in json.loads(result.stdout))


def formatted(tmp_path, source):
    """What `ruff format` makes of `source`, using a copy of the real pyproject.toml."""
    shutil.copy(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    result = ruff(
        "format", "--no-cache", "--stdin-filename", "snippet.py", "-", cwd=tmp_path, stdin=source
    )
    assert result.returncode == 0, result.stderr
    return result.stdout

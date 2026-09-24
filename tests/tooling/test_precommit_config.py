"""The pre-commit configuration, validated by pre-commit itself."""

import subprocess
import sys

from support.paths import ROOT


def test_the_pre_commit_config_is_valid():
    result = subprocess.run(
        [sys.executable, "-m", "pre_commit", "validate-config", ".pre-commit-config.yaml"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

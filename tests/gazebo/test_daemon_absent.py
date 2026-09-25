"""Without a Docker daemon the Gazebo tests skip instead of failing, so the suite runs anywhere."""

import pytest
from mutants import run_on_damaged_copy

pytestmark = pytest.mark.medium

NO_DAEMON = {"DOCKER_HOST": "unix:///nonexistent/docker.sock"}


def test_gazebo_tests_are_skipped_when_the_daemon_cannot_be_reached(tmp_path):
    result, found = run_on_damaged_copy(
        tmp_path, lambda copy: None, "tests/gazebo/test_image.py", env=NO_DAEMON
    )
    assert result.returncode == 0, result.stdout[-2000:]
    assert "1 skipped" in result.stdout, result.stdout[-2000:]
    assert not found["FAILED"]

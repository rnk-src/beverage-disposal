"""Talking to the Docker daemon from tests: is it there, build the Gazebo image, run in it."""

import json
import os
import shutil
import subprocess
from pathlib import Path

from support.paths import ROOT

# Both can be overridden so a test can build a deliberately damaged copy under another tag.
GAZEBO_DIR = Path(os.environ.get("GAZEBO_DOCKER_DIR", ROOT / "docker" / "gazebo"))
IMAGE = os.environ.get("GAZEBO_IMAGE_TAG", "beverage-disposal-gazebo:test")


def daemon_running():
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True, timeout=120).returncode == 0


def build():
    return subprocess.run(
        ["docker", "build", "-t", IMAGE, str(GAZEBO_DIR)], capture_output=True, text=True
    )


def run(*command, timeout=600):
    return subprocess.run(
        ["docker", "run", "--rm", IMAGE, *command], capture_output=True, text=True, timeout=timeout
    )


def step_world(world, steps):
    """Step a world baked into the image for `steps` iterations; return the printed JSON state."""
    result = run("step_world", f"/worlds/{world}", str(steps))
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout)

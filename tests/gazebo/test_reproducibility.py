"""A rebuild must not silently change what is inside the image."""

import difflib
import re

import pytest
from support import dockerenv

pytestmark = pytest.mark.medium

RECORD = dockerenv.GAZEBO_DIR / "apt-versions.txt"
LIST_PACKAGES = ["dpkg-query", "-W", "-f=${Package} ${Version} ${Architecture}\\n"]
REGENERATE = (
    "docker run --rm beverage-disposal-gazebo:test dpkg-query -W "
    "-f='${Package} ${Version} ${Architecture}\\n' | sort > docker/gazebo/apt-versions.txt"
)


def test_the_base_image_is_pinned_by_digest():
    # A tag such as jazzy-ros-base moves whenever the maintainers rebuild it; a digest does not.
    first = next(
        line
        for line in (dockerenv.GAZEBO_DIR / "Dockerfile").read_text().splitlines()
        if line.startswith("FROM ")
    )
    assert re.fullmatch(r"FROM \S+@sha256:[0-9a-f]{64}", first), first


def test_the_installed_packages_are_exactly_the_recorded_versions(built_image):
    assert built_image.returncode == 0, built_image.stderr[-2000:]
    assert RECORD.is_file(), f"no record of package versions; create it with: {REGENERATE}"
    result = dockerenv.run(*LIST_PACKAGES)
    assert result.returncode == 0, result.stderr[-2000:]
    installed = sorted(result.stdout.splitlines())
    recorded = sorted(RECORD.read_text().splitlines())
    drift = [
        line
        for line in difflib.unified_diff(
            recorded, installed, "recorded", "installed", lineterm="", n=0
        )
        if line[:1] in "+-" and line[:3] not in ("+++", "---")
    ]
    assert not drift, (
        f"the image drifted from {RECORD.name} ({len(drift)} lines):\n"
        + "\n".join(drift[:20])
        + f"\nIf the change is intended, regenerate it with: {REGENERATE}"
    )

"""The Gazebo validation image builds from the Dockerfile in the repository."""

import pytest

pytestmark = pytest.mark.medium


def test_the_image_builds_from_the_dockerfile(built_image):
    assert built_image.returncode == 0, built_image.stderr[-2000:]

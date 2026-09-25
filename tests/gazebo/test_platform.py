"""What the image contains: the decided platform (Ubuntu 24.04, Jazzy, Harmonic) and packages."""

import pytest
from support import dockerenv

pytestmark = pytest.mark.medium

# Binary packages, not "all": a package built for the container's own architecture is what
# proves the arm64 question (on an Apple Silicon Mac the container architecture is arm64).
PACKAGES = ["ros-jazzy-ros-gz-sim", "ros-jazzy-ros-gz-bridge", "ros-jazzy-gz-ros2-control"]


def in_image(built_image, *command):
    assert built_image.returncode == 0, built_image.stderr[-2000:]
    result = dockerenv.run(*command)
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout.strip()


def test_the_image_is_ubuntu_2404(built_image):
    assert 'VERSION_ID="24.04"' in in_image(built_image, "cat", "/etc/os-release")


def test_the_system_python_is_312(built_image):
    assert in_image(built_image, "python3", "--version").startswith("Python 3.12.")


def test_the_ros_distribution_is_jazzy(built_image):
    assert in_image(built_image, "printenv", "ROS_DISTRO") == "jazzy"


def test_gazebo_is_harmonic(built_image):
    # Harmonic is Gazebo Sim 8 (Garden 7, Ionic 9).
    assert in_image(built_image, "gz", "sim", "--versions").startswith("8.")


@pytest.mark.parametrize("package", PACKAGES)
def test_the_package_is_installed_for_the_container_architecture(built_image, package):
    installed = in_image(built_image, "dpkg-query", "-W", "-f=${Architecture}", package)
    assert installed == in_image(built_image, "dpkg", "--print-architecture")

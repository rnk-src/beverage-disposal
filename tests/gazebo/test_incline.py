"""A block on a 30 degree slope slides at the Coulomb rate in Gazebo and MuJoCo, or sticks."""

import math

import pytest
from mujoco_scenes import INCLINE, block_travel_on_incline
from support import dockerenv

pytestmark = pytest.mark.medium

G = 9.81
SECOND = 1000  # steps of 1 ms
SLIDING = ("block_on_incline.sdf", 0.3)
# tan(30 degrees) = 0.577, so a coefficient above that holds the block still (static friction).
STICKING = ("block_on_incline_sticky.sdf", 0.7)
# Slab 0.1 against block 0.3, so the engines' combination rules show.
MIXED = ("block_on_incline_mixed.sdf", 0.1, 0.3)


def coulomb_distance(friction, seconds):
    """Distance from rest under constant acceleration a = g (sin(theta) - mu cos(theta))."""
    return 0.5 * G * (math.sin(INCLINE) - friction * math.cos(INCLINE)) * seconds**2


def gazebo_travel(built_image, world, steps):
    assert built_image.returncode == 0, built_image.stderr[-2000:]
    x0, _, z0 = dockerenv.step_world(world, 0)["poses"]["block"]
    x1, _, z1 = dockerenv.step_world(world, steps)["poses"]["block"]
    return math.dist((x0, z0), (x1, z1))


def test_the_sliding_distance_matches_the_coulomb_formula(built_image):
    world, friction = SLIDING
    # 1 % covers the start-up transient; both engines measured within 0.1 %.
    assert gazebo_travel(built_image, world, SECOND) == pytest.approx(
        coulomb_distance(friction, 1.0), rel=0.01
    )


def test_gazebo_and_mujoco_slide_the_same_distance(built_image):
    world, friction = SLIDING
    assert gazebo_travel(built_image, world, SECOND) == pytest.approx(
        block_travel_on_incline(friction, SECOND), rel=0.005
    )


def test_gazebo_holds_the_block_still_when_friction_exceeds_the_slope(built_image):
    world, _ = STICKING
    assert gazebo_travel(built_image, world, SECOND) < 1e-3


def test_known_gap_mujoco_creeps_where_gazebo_holds(built_image):
    # Soft contacts have no true static friction, so a resting block slowly slides. Measured about
    # 4 mm in one second here; a tripwire so a MuJoCo upgrade that changes it gets noticed.
    _, friction = STICKING
    assert 0.5e-3 < block_travel_on_incline(friction, SECOND) < 20e-3


def test_known_gap_the_engines_combine_two_friction_coefficients_differently(built_image):
    # Measured: DART uses the smaller coefficient of the two surfaces, MuJoCo the larger one, so the
    # same pair slides 1.7 times further in Gazebo. A tripwire for the friction setup in 7.5.
    world, slab, block = MIXED
    gazebo = gazebo_travel(built_image, world, SECOND)
    mujoco = block_travel_on_incline(block, SECOND, slab_friction=slab)
    assert gazebo == pytest.approx(coulomb_distance(min(slab, block), 1.0), rel=0.01)
    assert mujoco == pytest.approx(coulomb_distance(max(slab, block), 1.0), rel=0.01)

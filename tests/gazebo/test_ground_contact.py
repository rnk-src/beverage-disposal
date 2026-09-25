"""A ball dropped on a static ground: Gazebo reports static models and the ball comes to rest."""

import pytest
from mujoco_scenes import ball_height_on_ground
from support import dockerenv

pytestmark = pytest.mark.medium

WORLD = "ball_on_ground.sdf"


def test_the_readout_lists_static_models_too(built_image):
    assert built_image.returncode == 0, built_image.stderr[-2000:]
    poses = dockerenv.step_world(WORLD, 10)["poses"]
    # Static models never move, so Gazebo's live pose topic omits them; the readout must not.
    assert poses["ground"] == pytest.approx([0.0, 0.0, -0.1])


RADIUS = 0.1
# Three seconds is long enough for both engines to settle (Gazebo takes about 2 s to push the ball
# out of its landing penetration). 1 mm allows contact softness: measured, MuJoCo rests 0.37 mm low.
SETTLED_STEPS = 3000
REST_TOLERANCE = 1e-3


def gazebo_rest_height(built_image):
    assert built_image.returncode == 0, built_image.stderr[-2000:]
    return dockerenv.step_world(WORLD, SETTLED_STEPS)["poses"]["ball"][2]


def test_the_ball_comes_to_rest_with_its_lowest_point_on_the_ground(built_image):
    assert gazebo_rest_height(built_image) == pytest.approx(RADIUS, abs=REST_TOLERANCE)


def test_gazebo_and_mujoco_rest_at_the_same_height(built_image):
    gazebo = gazebo_rest_height(built_image)
    assert gazebo == pytest.approx(ball_height_on_ground(SETTLED_STEPS), abs=REST_TOLERANCE)

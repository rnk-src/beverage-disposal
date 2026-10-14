"""Pitch: how far below horizontal the TCP's z axis points, positive down (D-15)."""

import mujoco
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from support.so101 import EXPECTED_RANGES, JOINTS, MJCF, SO101

from beverage_disposal.core.kinematics import HOME, pitch
from beverage_disposal.sim.arm import load_arm

POSES = 500
TOLERANCE = 1e-9


def angle_of(joint):
    return st.floats(*EXPECTED_RANGES[joint])


arm_poses = st.tuples(*(angle_of(joint) for joint in JOINTS[:5])).map(np.array)


def test_home_points_straight_down_so_its_pitch_is_plus_ninety_degrees():
    # D-15's convention, and the pose 1.2 worked out by hand: TCP z axis (0, 0, -1).
    assert pitch(HOME) == pytest.approx(np.pi / 2, abs=1e-6)


def test_pitch_is_zero_for_a_tool_pointing_straight_out_and_negative_for_one_pointing_up():
    straight_out = np.zeros(5)
    pointing_up = np.array([0, 0, 0, -np.pi / 2, 0])
    assert pitch(straight_out) == pytest.approx(0, abs=1e-6)
    assert pitch(pointing_up) == pytest.approx(-np.pi / 2, abs=1e-6)


def test_pitch_keeps_growing_when_the_tool_tilts_back_past_straight_down():
    # The reason pitch is the signed angle in the arm's plane and not the elevation angle: the
    # elevation would fold back down from 90 degrees, and have no derivative at home.
    past_vertical = HOME.copy()
    past_vertical[3] = 1.6
    assert pitch(past_vertical) == pytest.approx(1.6, abs=1e-6)


def mujoco_tcp_axes(model, count, seed):
    """Random in-limit poses with MuJoCo's TCP z axis in the base frame."""
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    for _ in range(count):
        data.qpos[:] = rng.uniform(*model.jnt_range.T)
        mujoco.mj_kinematics(model, data)
        base_rot = data.body("base").xmat.reshape(3, 3)
        yield data.qpos[:5].copy(), base_rot.T @ data.site("tcp").xmat.reshape(3, 3)[:, 2]


def test_pitch_is_the_angle_of_mujocos_tcp_z_axis_below_the_arms_plane_heading():
    arm = load_arm(SO101 / MJCF)
    for angles, z_axis in mujoco_tcp_axes(arm, POSES, seed=2):
        # The pan axis points down, so a positive pan turns the plane's heading clockwise seen
        # from above: the base x axis becomes (cos pan, -sin pan, 0).
        heading = np.array([np.cos(angles[0]), -np.sin(angles[0]), 0])
        expected = np.arctan2(-z_axis[2], z_axis @ heading)
        assert pitch(angles) == pytest.approx(expected, abs=TOLERANCE)


@settings(max_examples=200, derandomize=True)
@given(pose=arm_poses)
def test_pitch_is_the_sum_of_the_three_parallel_planar_joint_angles(pose):
    # Lift, elbow and wrist_flex turn about the same horizontal axis, so their angles add; pan only
    # turns the plane and roll turns about the tool axis, so neither changes the pitch.
    # The sum can pass 180 degrees inside the joint limits; pitch is an angle, so compare turns.
    difference = pitch(pose) - (pose[1] + pose[2] + pose[3])
    assert np.arctan2(np.sin(difference), np.cos(difference)) == pytest.approx(0, abs=TOLERANCE)

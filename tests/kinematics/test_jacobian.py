"""The Jacobian: how the TCP position and the pitch change when each arm joint turns."""

import mujoco
import numpy as np
import pytest
from support.so101 import EXPECTED_RANGES, JOINTS, MJCF, SO101

from beverage_disposal.core.kinematics import fk, jacobian, pitch
from beverage_disposal.sim.arm import load_arm

POSES = 300
STEP = 1e-6


def random_poses(count, seed):
    rng = np.random.default_rng(seed)
    low, high = np.array([EXPECTED_RANGES[joint] for joint in JOINTS[:5]]).T
    return rng.uniform(low, high, size=(count, 5))


def turn_between(after, before):
    # Pitch is an angle: readings either side of the +-180 degree seam differ by a small turn.
    difference = after - before
    return np.arctan2(np.sin(difference), np.cos(difference))


def finite_difference_columns(pose, quantity, subtract=np.subtract):
    """Central differences of `quantity` per joint: an oracle sharing no maths with jacobian."""
    columns = []
    for joint in range(5):
        plus, minus = pose.copy(), pose.copy()
        plus[joint] += STEP
        minus[joint] -= STEP
        columns.append(subtract(quantity(plus), quantity(minus)) / (2 * STEP))
    return np.array(columns).T


def test_the_jacobian_has_a_row_for_x_y_z_and_pitch_and_a_column_for_each_arm_joint():
    assert jacobian(np.zeros(5)).shape == (4, 5)


def test_the_position_rows_match_central_finite_differences_of_fk_over_random_poses():
    for pose in random_poses(POSES, seed=3):
        expected = finite_difference_columns(pose, lambda angles: fk(angles)[:3, 3])
        np.testing.assert_allclose(jacobian(pose)[:3], expected, rtol=0, atol=1e-8)


def test_the_pitch_row_matches_central_finite_differences_of_pitch_over_random_poses():
    for pose in random_poses(POSES, seed=4):
        expected = finite_difference_columns(pose, pitch, subtract=turn_between)
        np.testing.assert_allclose(jacobian(pose)[3], expected, rtol=0, atol=1e-8)


@pytest.fixture(scope="module")
def arm():
    return load_arm(SO101 / MJCF)


def mujoco_jacobians(model, count, seed):
    """MuJoCo's TCP linear and angular Jacobians over the five arm joints, in the base frame."""
    data = mujoco.MjData(model)
    for pose in random_poses(count, seed):
        data.qpos[:5] = pose
        mujoco.mj_forward(model, data)
        jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, model.site("tcp").id)
        # MuJoCo reports them in the world frame, our Jacobian is in the base frame.
        base_rot = data.body("base").xmat.reshape(3, 3)
        yield pose, base_rot.T @ jacp[:, :5], base_rot.T @ jacr[:, :5]


def test_the_position_rows_match_mujocos_site_jacobian(arm):
    # A second oracle that shares neither our chain file nor our code: MuJoCo's own mj_jacSite.
    for pose, jacp, _ in mujoco_jacobians(arm, POSES, seed=5):
        np.testing.assert_allclose(jacobian(pose)[:3], jacp, rtol=0, atol=1e-9)


def test_the_pitch_row_is_the_turn_rate_about_the_plane_normal_from_mujocos_angular_jacobian(arm):
    for pose, _, jacr in mujoco_jacobians(arm, POSES, seed=6):
        # Pitch grows when the tool turns about (sin pan, cos pan, 0): +y at pan 0, where a positive
        # turn about +y swings a forward-pointing tool downwards.
        normal = np.array([np.sin(pose[0]), np.cos(pose[0]), 0])
        np.testing.assert_allclose(jacobian(pose)[3], normal @ jacr, rtol=0, atol=1e-9)


def test_rolling_the_wrist_moves_neither_the_tcp_nor_the_pitch():
    # The TCP sits on the roll axis (D-77), which is what lets IK lock the roll joint.
    for pose in random_poses(POSES, seed=7):
        np.testing.assert_allclose(jacobian(pose)[:, 4], 0, rtol=0, atol=1e-12)


def test_the_three_parallel_planar_joints_each_add_one_for_one_to_pitch_and_pan_and_roll_add_none():
    for pose in random_poses(POSES, seed=8):
        np.testing.assert_allclose(jacobian(pose)[3], [0, 1, 1, 1, 0], rtol=0, atol=1e-12)

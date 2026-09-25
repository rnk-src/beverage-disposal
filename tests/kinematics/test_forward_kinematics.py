"""Forward kinematics in plain NumPy: five arm joint angles to the pose of the TCP."""

import mujoco
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from support.so101 import EXPECTED_RANGES, JOINTS, MJCF, SO101

from beverage_disposal.core.kinematics import HOME, fk
from beverage_disposal.sim.arm import load_arm

POSES = 1000
# Both sides do the same double-precision products, so they agree to rounding; a millimetre-scale
# slip or a value rounded on its way through the JSON file is far above this.
TOLERANCE = 1e-9


@pytest.fixture(scope="module")
def arm():
    return load_arm(SO101 / MJCF)


def mujoco_tcp_poses(model, count, seed):
    """Random in-limit poses and MuJoCo's TCP pose for each, in the arm's base frame."""
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    for _ in range(count):
        data.qpos[:] = rng.uniform(*model.jnt_range.T)
        mujoco.mj_kinematics(model, data)
        # FK works in the base frame; the base sits at the world origin today, but not once 2.1
        # places the arm in a scene.
        base_pos, base_rot = data.body("base").xpos, data.body("base").xmat.reshape(3, 3)
        position = base_rot.T @ (data.site("tcp").xpos - base_pos)
        rotation = base_rot.T @ data.site("tcp").xmat.reshape(3, 3)
        yield data.qpos[:5].copy(), position, rotation


def test_fk_puts_the_tcp_where_mujoco_does_over_random_in_limit_poses(arm):
    for angles, position, _ in mujoco_tcp_poses(arm, POSES, seed=0):
        np.testing.assert_allclose(fk(angles)[:3, 3], position, rtol=0, atol=TOLERANCE)


def test_fk_turns_the_tcp_frame_the_way_mujoco_does_over_random_in_limit_poses(arm):
    # D-77's frame: z towards the fingertips, x towards the moving jaw. 1.3 reads pitch off its z.
    for angles, _, rotation in mujoco_tcp_poses(arm, POSES, seed=1):
        np.testing.assert_allclose(fk(angles)[:3, :3], rotation, rtol=0, atol=TOLERANCE)


def test_the_home_pose_holds_the_tcp_under_the_wrist_pointing_straight_down():
    # Worked by hand from the MJCF's body offsets and quaternions, without MuJoCo: the
    # wrist-flex axis is at (0.2321345, -0.0182778, 0.23437), and turning it +90 degrees swings the
    # 0.1592274 m tool straight down. Rounding in the hand work is below 1e-7 m.
    pose = fk(HOME)
    np.testing.assert_allclose(pose[:3, 3], [0.2321345, -0.0001778, 0.0751426], rtol=0, atol=1e-6)
    np.testing.assert_allclose(pose[:3, 2], [0, 0, -1], rtol=0, atol=1e-6)


def angle_of(joint):
    return st.floats(*EXPECTED_RANGES[joint])


arm_poses = st.tuples(*(angle_of(joint) for joint in JOINTS[:5])).map(np.array)
# Read off the MJCF: the shoulder body sits at (0.0388353, -8.97657e-09, 0.0624), turned half a turn
# about y, so the pan joint's z axis points straight down through that point.
PAN_AXIS_XY = np.array([0.0388353, -8.97657e-09])


def turned_about_the_pan_axis(point, angle):
    # Turning by `angle` about a downward axis is turning by -angle about the upward one.
    c, s = np.cos(-angle), np.sin(-angle)
    x, y = point[:2] - PAN_AXIS_XY
    return np.array([*(PAN_AXIS_XY + [c * x - s * y, s * x + c * y]), point[2]])


@settings(max_examples=200, derandomize=True)
@given(pose=arm_poses, new_pan=angle_of("shoulder_pan"))
def test_turning_the_pan_joint_turns_the_tcp_by_the_same_angle_about_the_pan_axis(pose, new_pan):
    turned = pose.copy()
    turned[0] = new_pan
    expected = turned_about_the_pan_axis(fk(pose)[:3, 3], new_pan - pose[0])
    np.testing.assert_allclose(fk(turned)[:3, 3], expected, rtol=0, atol=TOLERANCE)


@settings(max_examples=200, derandomize=True)
@given(pose=arm_poses, new_roll=angle_of("wrist_roll"))
def test_rolling_the_wrist_leaves_the_tcp_where_it_is(pose, new_roll):
    # D-77 puts the TCP on the roll axis, which is what lets 1.3 and 1.4 lock the roll.
    rolled = pose.copy()
    rolled[4] = new_roll
    np.testing.assert_allclose(fk(rolled)[:3, 3], fk(pose)[:3, 3], rtol=0, atol=TOLERANCE)

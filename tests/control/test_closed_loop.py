"""The closed loop: joint targets through the PD actuators, one 20 Hz control step at a time."""

import mujoco
import numpy as np
import pytest
from support.so101 import MJCF, SO101

from beverage_disposal.core.controller import EEDeltaController
from beverage_disposal.core.kinematics import HOME, fk
from beverage_disposal.sim.arm import load_arm
from beverage_disposal.sim.loop import ArmLoop

GRIPPER_START = 0.5


@pytest.fixture(scope="module")
def arm():
    return load_arm(SO101 / MJCF)


def test_one_control_step_is_a_twentieth_of_a_second_of_physics(arm):
    # Locked decision: 20 Hz policy rate with frame skip over the physics steps.
    loop = ArmLoop(arm)
    loop.reset(HOME, GRIPPER_START)
    loop.step(HOME, GRIPPER_START)
    assert loop.time == pytest.approx(0.05, abs=1e-9)
    loop.step(HOME, GRIPPER_START)
    assert loop.time == pytest.approx(0.10, abs=1e-9)


def test_reset_puts_the_arm_at_rest_in_the_given_pose_and_the_tip_where_fk_says(arm):
    loop = ArmLoop(arm)
    pose = np.array([0.3, -0.2, 0.5, 0.7, -0.4])
    loop.reset(pose, GRIPPER_START)
    np.testing.assert_allclose(loop.joints, pose, rtol=0, atol=1e-12)
    assert loop.gripper == pytest.approx(GRIPPER_START)
    np.testing.assert_array_equal(loop.joint_velocities, np.zeros(5))
    np.testing.assert_allclose(loop.tip, fk(pose)[:3, 3], rtol=0, atol=1e-9)


def test_a_substep_callback_sees_every_physics_step_of_a_control_step(arm):
    # The step-response metrics need the joint angle at the physics rate, not the 20 Hz one.
    loop = ArmLoop(arm)
    loop.reset(HOME, GRIPPER_START)
    seen = []
    loop.step(HOME, GRIPPER_START, on_substep=lambda: seen.append(loop.time))
    assert len(seen) == 25
    np.testing.assert_allclose(np.diff(seen), arm.opt.timestep, rtol=0, atol=1e-12)


def settled(arm, pose, seconds=3.0):
    loop = ArmLoop(arm)
    loop.reset(pose, GRIPPER_START)
    for _ in range(round(seconds / 0.05)):
        loop.step(pose, GRIPPER_START)
    return loop


def test_a_held_pose_sags_by_the_gravity_torque_over_the_servo_stiffness(arm):
    # Expected values from MuJoCo's own inverse dynamics (qfrc_bias is the gravity torque at rest),
    # not from the closed loop: sag = torque / kp while the servo is not at its force limit.
    kp = arm.actuator_gainprm[0, 0]
    rng = np.random.default_rng(0)
    low, high = arm.jnt_range[:5].T
    for _ in range(8):
        pose = rng.uniform(0.6 * low, 0.6 * high)
        data = mujoco.MjData(arm)
        data.qpos[:] = [*pose, GRIPPER_START]
        mujoco.mj_forward(arm, data)
        predicted = -data.qfrc_bias[:5] / kp
        np.testing.assert_allclose(settled(arm, pose).joints - pose, predicted, rtol=0, atol=5e-5)


def test_a_held_pose_comes_to_rest_and_stays_there(arm):
    pose = np.array([0.4, 0.3, -0.5, 0.9, 0.2])
    loop = settled(arm, pose)
    assert np.max(np.abs(loop.joint_velocities)) < 1e-6
    before = loop.joints
    for _ in range(20):
        loop.step(pose, GRIPPER_START)
    np.testing.assert_allclose(loop.joints, before, rtol=0, atol=1e-7)


def random_target_run(arm, steps=200):
    rng = np.random.default_rng(1)
    low, high = arm.jnt_range.T
    loop = ArmLoop(arm)
    loop.reset(HOME, GRIPPER_START)
    history = []
    for _ in range(steps):
        targets = rng.uniform(low, high)
        loop.step(targets[:5], targets[5])
        history.append([*loop.joints, loop.gripper])
    return np.array(history)


def test_the_loop_never_produces_nan_or_leaves_the_joint_limits_and_repeats_exactly(arm):
    first = random_target_run(arm)
    assert np.all(np.isfinite(first))
    low, high = arm.jnt_range.T
    assert np.all(first >= low - 1e-6) and np.all(first <= high + 1e-6)
    np.testing.assert_array_equal(first, random_target_run(arm))


def five_centimetre_move(model, hold_steps=10):
    """The tip after every control step of a 5 cm move along x, and its distance from the goal."""
    loop = ArmLoop(model)
    loop.reset(HOME, GRIPPER_START)
    controller = EEDeltaController()
    controller.reset(HOME, GRIPPER_START)
    start = loop.tip
    tips, errors = [], []
    # Three full 1.5 cm deltas and a third of one: 5.0 cm along x.
    for action in [[1, 0, 0, 0]] * 3 + [[1 / 3, 0, 0, 0]] + [[0, 0, 0, 0]] * hold_steps:
        command = controller.step(np.array(action, dtype=float), loop.joints, loop.gripper)
        loop.step(command.joints, command.gripper)
        tips.append(loop.tip - start)
        errors.append(np.linalg.norm(loop.tip - start - [0.05, 0, 0]))
    return np.array(tips), errors


def test_a_commanded_five_centimetre_move_is_achieved_within_a_millimetre_in_eight_steps(arm):
    # Measured in 1.6: the tip is 2.2 mm off after 5 steps and 0.2 mm off from the 6th; the
    # tolerance and the step count are fixed after those numbers, with a little room.
    _, errors = five_centimetre_move(arm)
    assert all(error < 1e-3 for error in errors[7:])


def test_halving_the_physics_step_barely_changes_the_move_and_not_at_all_once_it_has_settled(arm):
    # Measured in 1.6: halving moves the tip by up to 0.91 mm in mid-motion (1.4 mm for a quarter
    # step: the Euler integrator is not fully converged at 2 ms while the arm is moving) and by
    # under a micron once settled. The bounds are fixed after those numbers.
    finer = load_arm(SO101 / MJCF)
    finer.opt.timestep = arm.opt.timestep / 2
    coarse, _ = five_centimetre_move(arm)
    fine, _ = five_centimetre_move(finer)
    difference = np.linalg.norm(coarse - fine, axis=1)
    assert difference.max() < 2e-3
    assert difference[-1] < 1e-5


def test_the_gripper_target_is_followed_like_a_joint(arm):
    loop = ArmLoop(arm)
    loop.reset(HOME, GRIPPER_START)
    for _ in range(20):
        loop.step(HOME, 1.2)
    assert loop.gripper == pytest.approx(1.2, abs=2e-3)


@pytest.mark.parametrize("joint", range(6))
@pytest.mark.parametrize("side", ["low", "high"])
def test_commanding_a_joint_hard_into_its_stop_overshoots_it_by_under_three_degrees_and_settles(
    arm, joint, side
):
    # The stops are soft constraints. Measured in 1.6: a full-range step to a limit pushes through
    # it by up to 0.038 rad (2.2 degrees) before the stop wins; the bound is fixed after that.
    limit = arm.jnt_range[joint][0 if side == "low" else 1]
    loop = ArmLoop(arm)
    loop.reset(HOME, GRIPPER_START)
    targets = np.array([*HOME, GRIPPER_START])
    targets[joint] = limit
    worst = []

    def watch():
        angle = loop.gripper if joint == 5 else loop.joints[joint]
        worst.append(max(arm.jnt_range[joint][0] - angle, angle - arm.jnt_range[joint][1]))

    for _ in range(20):
        loop.step(targets[:5], targets[5], on_substep=watch)
    assert max(worst) < 0.05
    settled_angle = loop.gripper if joint == 5 else loop.joints[joint]
    if (joint, side) == (2, "high"):
        # Known from 0.x: the gripper meets the shoulder at about 1.5 rad, so this stop is not hit.
        assert settled_angle < 1.55
    else:
        assert settled_angle == pytest.approx(limit, abs=2e-3)

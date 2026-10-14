"""The end-effector delta controller: a policy action [dx, dy, dz, gripper] to joint targets."""

import numpy as np
import pytest
from support.so101 import EXPECTED_RANGES

from beverage_disposal.core.controller import MAX_DELTA, MAX_GRIPPER_STEP, EEDeltaController
from beverage_disposal.core.kinematics import HOME, fk

GRIPPER_START = 0.5
POSITION_TOLERANCE = 1e-4


def tip(joints):
    return fk(joints)[:3, 3]


def fresh_controller(**settings):
    controller = EEDeltaController(**settings)
    controller.reset(HOME, GRIPPER_START)
    return controller


def test_a_zero_delta_leaves_the_arm_where_it_is():
    command = fresh_controller().step(np.zeros(4), HOME, GRIPPER_START)
    np.testing.assert_allclose(command.joints, HOME, rtol=0, atol=1e-12)
    np.testing.assert_allclose(command.target, tip(HOME), rtol=0, atol=1e-12)
    assert command.ik_ok


@pytest.mark.parametrize("scale", [1.0, 0.5, -1.0])
def test_a_full_action_moves_the_target_by_the_maximum_delta_and_a_partial_one_by_its_share(scale):
    command = fresh_controller().step(np.array([scale, 0, 0, 0]), HOME, GRIPPER_START)
    np.testing.assert_allclose(command.target - tip(HOME), [scale * 0.015, 0, 0], atol=1e-12)
    assert MAX_DELTA == 0.015


def test_the_joint_targets_put_the_tip_on_the_new_target_and_leave_the_roll_alone():
    start = HOME.copy()
    start[4] = 0.4
    controller = EEDeltaController()
    controller.reset(start, GRIPPER_START)
    command = controller.step(np.array([0.3, -0.7, 0.2, 0]), start, GRIPPER_START)
    np.testing.assert_allclose(tip(command.joints), command.target, rtol=0, atol=POSITION_TOLERANCE)
    assert command.joints[4] == 0.4


def test_actions_outside_minus_one_to_one_act_like_the_nearest_allowed_action():
    wild = fresh_controller().step(np.array([5.0, -3.0, 0.2, 9.0]), HOME, GRIPPER_START)
    clipped = fresh_controller().step(np.array([1.0, -1.0, 0.2, 1.0]), HOME, GRIPPER_START)
    np.testing.assert_array_equal(wild.joints, clipped.joints)
    np.testing.assert_array_equal(wild.target, clipped.target)
    assert wild.gripper == clipped.gripper


@pytest.mark.parametrize(
    "action", [[np.nan, 0, 0, 0], [0, np.inf, 0, 0], [0, 0, 0], [0, 0, 0, 0, 0]], ids=str
)
def test_an_action_that_is_not_four_finite_numbers_is_refused_not_guessed_at(action):
    # A NaN reaching the IK would end up as NaN joint targets sent to the servos.
    with pytest.raises(ValueError, match="action"):
        fresh_controller().step(np.array(action, dtype=float), HOME, GRIPPER_START)


SMALL_BOX = (np.array([0.15, -0.05, 0.05]), np.array([0.25, 0.05, 0.12]))


def follow(controller, actions, joints=HOME, gripper=GRIPPER_START):
    """Feed actions to the controller with an arm that tracks every command exactly."""
    commands = []
    for action in actions:
        command = controller.step(np.array(action, dtype=float), joints, gripper)
        commands.append(command)
        joints, gripper = command.joints, command.gripper
    return commands


def test_the_target_never_leaves_the_workspace_box_whatever_the_policy_does():
    rng = np.random.default_rng(0)
    controller = fresh_controller(workspace=SMALL_BOX)
    low, high = SMALL_BOX
    for command in follow(controller, rng.uniform(-1, 1, size=(300, 4))):
        assert np.all(command.target >= low - 1e-12) and np.all(command.target <= high + 1e-12)
        # The tip follows the target to the IK tolerance, or the controller held a good target.
        assert np.all(tip(command.joints) >= low - 2 * POSITION_TOLERANCE)
        assert np.all(tip(command.joints) <= high + 2 * POSITION_TOLERANCE)


def test_pushing_on_the_box_wall_stops_exactly_at_the_wall():
    commands = follow(fresh_controller(workspace=SMALL_BOX), [[1, 0, 0, 0]] * 20)
    assert commands[-1].target[0] == SMALL_BOX[1][0]
    np.testing.assert_array_equal(commands[-1].target, commands[-2].target)


def test_the_target_never_runs_further_from_the_measured_tip_than_the_lead_limit():
    # A blocked gripper: the measured arm stays at HOME while the policy keeps pushing. Without the
    # clamp the target runs away and the PD controller pushes harder and harder (windup, D-19).
    lead = 0.04
    controller = fresh_controller(max_tip_lead=lead)
    distances = []
    for _ in range(20):
        command = controller.step(np.array([1.0, 0, 0, 0]), HOME, GRIPPER_START)
        distances.append(np.linalg.norm(command.target - tip(HOME)))
    assert max(distances) <= lead + 1e-12
    assert distances[-1] == pytest.approx(lead, abs=1e-12)


FAR_BOX = (np.array([0.05, -0.5, 0.0]), np.array([2.0, 0.5, 1.0]))


def test_when_the_ik_cannot_reach_the_target_the_last_good_command_is_held():
    controller = fresh_controller(workspace=FAR_BOX, max_tip_lead=5.0)
    joints, gripper, previous = HOME, GRIPPER_START, None
    for _ in range(100):
        command = controller.step(np.array([1.0, 0, 0, 0]), joints, gripper)
        if not command.ik_ok:
            break
        previous, (joints, gripper) = command, (command.joints, command.gripper)
    else:
        pytest.fail("the target was never out of reach: the box is too small for this test")
    np.testing.assert_array_equal(command.joints, previous.joints)
    np.testing.assert_array_equal(command.target, previous.target)
    # The held target is the one the next action starts from, so backing off works at once.
    back = controller.step(np.array([-1.0, 0, 0, 0]), joints, gripper)
    assert back.ik_ok
    np.testing.assert_allclose(back.target - previous.target, [-0.015, 0, 0], atol=1e-12)


def gripper_range():
    return EXPECTED_RANGES["gripper"]


def test_the_gripper_is_an_absolute_target_that_moves_at_most_one_slew_step_per_command():
    commands = follow(fresh_controller(), [[0, 0, 0, 1.0]] * 30)
    angles = np.array([GRIPPER_START] + [c.gripper for c in commands])
    assert np.max(np.abs(np.diff(angles))) <= MAX_GRIPPER_STEP + 1e-12
    assert angles[1] - angles[0] == pytest.approx(MAX_GRIPPER_STEP)
    # Fully open is the top of the gripper's range, and it stays there.
    assert angles[-1] == pytest.approx(gripper_range()[1], abs=1e-5)
    assert angles[-2] == angles[-1]


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (-1.0, gripper_range()[0]),
        (0.0, sum(gripper_range()) / 2),
        (0.5, 0.75 * gripper_range()[1] + 0.25 * gripper_range()[0]),
    ],
)
def test_the_gripper_action_maps_linearly_onto_the_gripper_range(action, expected):
    commands = follow(fresh_controller(), [[0, 0, 0, action]] * 30)
    assert commands[-1].gripper == pytest.approx(expected, abs=1e-5)


def test_the_gripper_target_stays_inside_the_gripper_range_whatever_the_policy_does():
    rng = np.random.default_rng(1)
    low, high = gripper_range()
    for command in follow(fresh_controller(), rng.uniform(-1, 1, size=(200, 4))):
        assert low - 1e-5 <= command.gripper <= high + 1e-5


def test_repeated_equal_deltas_trace_a_straight_line_in_equal_steps():
    commands = follow(fresh_controller(), [[0, 1.0, 0, 0]] * 6)
    tips = np.array([tip(HOME)] + [tip(c.joints) for c in commands])
    steps = np.diff(tips, axis=0)
    np.testing.assert_allclose(
        steps, np.tile([0, 0.015, 0], (6, 1)), rtol=0, atol=2 * POSITION_TOLERANCE
    )


def test_a_delta_adds_to_the_previous_target_not_to_where_the_arm_measured_to_be():
    # The arm has not moved (HOME is measured both times), yet the second delta stacks on the first:
    # that is what lets the PD controller's sag be corrected (D-19). Measured + delta: 0.015.
    controller = fresh_controller()
    controller.step(np.array([1.0, 0, 0, 0]), HOME, GRIPPER_START)
    second = controller.step(np.array([1.0, 0, 0, 0]), HOME, GRIPPER_START)
    np.testing.assert_allclose(second.target - tip(HOME), [0.03, 0, 0], atol=1e-12)


def test_the_same_actions_give_the_same_commands_and_reset_starts_over_from_the_measured_arm():
    actions = np.random.default_rng(2).uniform(-1, 1, size=(20, 4))
    first = follow(fresh_controller(), actions)
    second = follow(fresh_controller(), actions)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a.joints, b.joints)
    controller = fresh_controller()
    follow(controller, actions)
    controller.reset(HOME, GRIPPER_START)
    again = follow(controller, actions)
    np.testing.assert_array_equal(again[-1].joints, first[-1].joints)


def test_the_joint_targets_stay_inside_the_joint_limits_over_a_long_random_run():
    low = np.array([EXPECTED_RANGES[j][0] for j in EXPECTED_RANGES][:5])
    high = np.array([EXPECTED_RANGES[j][1] for j in EXPECTED_RANGES][:5])
    rng = np.random.default_rng(3)
    for command in follow(fresh_controller(), rng.uniform(-1, 1, size=(300, 4))):
        assert np.all(command.joints >= low - 1e-6) and np.all(command.joints <= high + 1e-6)
        assert np.all(np.isfinite(command.joints))

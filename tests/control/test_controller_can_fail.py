"""The controller tests must notice a mutated controller, one fault at a time (medium tier)."""

import pytest
from support.damage import assert_the_test_failed, edit, run_on_mutated_package

pytestmark = pytest.mark.medium

CONTROLLER = "core/controller.py"
CONTROLLER_TESTS = "tests/control/test_ee_delta_controller.py"
CORE_TESTS = "tests/kinematics/test_core_needs_no_mujoco.py"
IK_TESTS = "tests/kinematics/test_ik.py"


def double_the_delta_scale(package):
    edit(package, CONTROLLER, "self._max_delta * action[:3]", "2 * self._max_delta * action[:3]")


def stop_clipping_the_action(package):
    edit(package, CONTROLLER, "action = np.clip(action, -1.0, 1.0)", "pass")


def accept_any_action(package):
    edit(
        package,
        CONTROLLER,
        "if action.shape != (4,) or not np.all(np.isfinite(action)):",
        "if False:",
    )


def ignore_the_workspace_box(package):
    edit(
        package,
        CONTROLLER,
        "np.clip(self._target + self._max_delta * action[:3], *self._workspace)",
        "self._target + self._max_delta * action[:3]",
    )


def ignore_the_tip_lead(package):
    edit(package, CONTROLLER, "if distance > self._max_tip_lead:", "if False:")


def add_the_delta_to_the_measured_tip(package):
    edit(
        package,
        CONTROLLER,
        "self._target + self._max_delta",
        "fk(np.asarray(joints, dtype=float))[:3, 3] + self._max_delta",
    )


def keep_going_after_the_ik_fails(package):
    edit(package, CONTROLLER, "if ik_ok:", "if True:")


def drop_the_gripper_slew_limit(package):
    edit(
        package,
        CONTROLLER,
        "step = np.clip(wanted - self._gripper, -self._max_gripper_step, self._max_gripper_step)",
        "step = wanted - self._gripper",
    )


def invert_the_gripper_mapping(package):
    edit(
        package,
        CONTROLLER,
        "wanted = low + (action + 1) / 2 * (high - low)",
        "wanted = high - (action + 1) / 2 * (high - low)",
    )


def ignore_the_given_pitch(package):
    edit(
        package,
        CONTROLLER,
        "result = solve_ik(target, self._pitch, start=self._joints)",
        "result = solve_ik(target, np.pi / 2, start=self._joints)",
    )


def nudge_the_roll(package):
    edit(
        package,
        CONTROLLER,
        "result = solve_ik(target, self._pitch, start=self._joints)",
        "result = solve_ik(target, self._pitch, start=self._joints + [0, 0, 0, 0, 0.1])",
    )


def drift_sideways_every_step(package):
    edit(
        package,
        CONTROLLER,
        "self._target + self._max_delta",
        "self._target + [0, 0.001, 0] + self._max_delta",
    )


def keep_the_old_target_on_reset(package):
    edit(
        package,
        CONTROLLER,
        "self._target = fk(self._joints)[:3, 3]",
        'self._target = getattr(self, "_target", fk(self._joints)[:3, 3])',
    )


def stop_parking_joints_on_their_limits(package):
    edit(
        package,
        "core/ik.py",
        "outside = active & ((proposed < _LOW) | (proposed > _HIGH))",
        "outside = np.zeros(IK_JOINTS, dtype=bool)",
    )


def keep_iterating_at_a_stationary_point(package):
    edit(package, "core/ik.py", "if np.max(np.abs(angles - before)) < STALL_STEP:", "if False:")


def import_mujoco_in_the_controller(package):
    edit(
        package,
        CONTROLLER,
        "import numpy as np\n",
        "import mujoco  # noqa: F401\nimport numpy as np\n",
    )


def row(fault, module, test):
    return fault, module, test


MUTANTS = [
    row(
        double_the_delta_scale,
        CONTROLLER_TESTS,
        "test_a_full_action_moves_the_target_by_the_maximum_delta_and_a_partial_one_by_its_share[1.0]",
    ),
    row(
        stop_clipping_the_action,
        CONTROLLER_TESTS,
        "test_actions_outside_minus_one_to_one_act_like_the_nearest_allowed_action",
    ),
    row(
        accept_any_action,
        CONTROLLER_TESTS,
        "test_an_action_that_is_not_four_finite_numbers_is_refused_not_guessed_at[[nan, 0, 0, 0]]",
    ),
    row(
        ignore_the_workspace_box,
        CONTROLLER_TESTS,
        "test_the_target_never_leaves_the_workspace_box_whatever_the_policy_does",
    ),
    row(
        ignore_the_workspace_box,
        CONTROLLER_TESTS,
        "test_pushing_on_the_box_wall_stops_exactly_at_the_wall",
    ),
    row(
        ignore_the_tip_lead,
        CONTROLLER_TESTS,
        "test_the_target_never_runs_further_from_the_measured_tip_than_the_lead_limit",
    ),
    row(
        add_the_delta_to_the_measured_tip,
        CONTROLLER_TESTS,
        "test_a_delta_adds_to_the_previous_target_not_to_where_the_arm_measured_to_be",
    ),
    row(
        keep_going_after_the_ik_fails,
        CONTROLLER_TESTS,
        "test_when_the_ik_cannot_reach_the_target_the_last_good_command_is_held",
    ),
    row(
        drop_the_gripper_slew_limit,
        CONTROLLER_TESTS,
        "test_the_gripper_is_an_absolute_target_that_moves_at_most_one_slew_step_per_command",
    ),
    row(
        invert_the_gripper_mapping,
        CONTROLLER_TESTS,
        "test_the_gripper_action_maps_linearly_onto_the_gripper_range[-1.0--0.17453292519943295]",
    ),
    row(
        ignore_the_given_pitch,
        CONTROLLER_TESTS,
        "test_the_controller_holds_the_tool_at_the_pitch_it_was_given",
    ),
    row(
        nudge_the_roll,
        CONTROLLER_TESTS,
        "test_the_joint_targets_put_the_tip_on_the_new_target_and_leave_the_roll_alone",
    ),
    row(
        drift_sideways_every_step,
        CONTROLLER_TESTS,
        "test_repeated_equal_deltas_trace_a_straight_line_in_equal_steps",
    ),
    row(
        keep_the_old_target_on_reset,
        CONTROLLER_TESTS,
        "test_the_same_actions_give_the_same_commands_and_reset_starts_over_from_the_measured_arm",
    ),
    row(
        stop_parking_joints_on_their_limits,
        CONTROLLER_TESTS,
        "test_the_joint_targets_stay_inside_the_joint_limits_over_a_long_random_run",
    ),
    row(
        keep_iterating_at_a_stationary_point,
        IK_TESTS,
        "test_a_pitch_the_arm_cannot_reach_ends_the_solve_early_with_the_position_met",
    ),
    row(
        import_mujoco_in_the_controller,
        CORE_TESTS,
        "test_the_controller_imports_and_runs_with_mujoco_unavailable",
    ),
]


@pytest.mark.parametrize(
    ("fault", "test_module", "failing_test"),
    MUTANTS,
    ids=[f"{f.__name__}-{t[5:40]}" for f, _, t in MUTANTS],
)
def test_the_controller_tests_fail_on_a_mutated_package(tmp_path, fault, test_module, failing_test):
    result = run_on_mutated_package(tmp_path, test_module, fault)
    assert_the_test_failed(result, test_module, failing_test)

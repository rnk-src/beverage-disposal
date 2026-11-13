"""The closed-loop and actuator tests must notice a mutated loop, one fault at a time (medium)."""

import pytest
from support.damage import assert_the_test_failed, edit, run_on_mutated_package

pytestmark = pytest.mark.medium

LOOP = "sim/loop.py"
RESPONSE = "sim/actuator_response.py"
METRICS = "core/step_response.py"
LOOP_TESTS = "tests/control/test_closed_loop.py"
RESPONSE_TESTS = "tests/control/test_actuator_response.py"
METRIC_TESTS = "tests/control/test_step_response_metrics.py"


def lengthen_the_control_period(package):
    edit(package, LOOP, "CONTROL_PERIOD = 0.05", "CONTROL_PERIOD = 0.1")


def forget_the_gripper_on_reset(package):
    edit(
        package,
        LOOP,
        "self._data.qpos[: ARM_JOINTS + 1] = [*joints, gripper]",
        "self._data.qpos[: ARM_JOINTS + 1] = [*joints, 0.0]",
    )


def ignore_the_gripper_target(package):
    edit(
        package,
        LOOP,
        "self._data.ctrl[:] = [*joint_targets, gripper_target]",
        "self._data.ctrl[:] = [*joint_targets, 0.0]",
    )


def never_call_the_substep_callback(package):
    edit(
        package, LOOP, "            if on_substep is not None:\n                on_substep()\n", ""
    )


def offset_the_first_target_by_a_milliradian(package):
    edit(
        package,
        LOOP,
        "self._data.ctrl[:] = [*joint_targets, gripper_target]",
        "self._data.ctrl[:] = [joint_targets[0] + 0.001, *joint_targets[1:], gripper_target]",
    )


def make_the_first_target_wobble(package):
    edit(
        package,
        LOOP,
        "self._data.ctrl[:] = [*joint_targets, gripper_target]",
        "self._data.ctrl[:] = [joint_targets[0] + 0.01 * __import__('math').sin(self._data.time),"
        " *joint_targets[1:], gripper_target]",
    )


def add_a_little_randomness_to_the_targets(package):
    edit(
        package,
        LOOP,
        "self._data.ctrl[:] = [*joint_targets, gripper_target]",
        "self._data.ctrl[:] = [joint_targets[0] + 1e-6 * __import__('random').random(),"
        " *joint_targets[1:], gripper_target]",
    )


def give_each_control_step_only_five_physics_steps(package):
    edit(package, LOOP, "round(CONTROL_PERIOD / model.opt.timestep)", "5")


def settle_for_no_time_before_the_step(package):
    edit(package, RESPONSE, "SETTLE_SECONDS = 2.0", "SETTLE_SECONDS = 0.0")


def step_by_a_tenth_of_a_radian(package):
    edit(package, RESPONSE, "STEP_SIZE = 0.2", "STEP_SIZE = 0.1")


def record_the_first_joint_whatever_joint_was_stepped(package):
    edit(
        package,
        RESPONSE,
        "angles.append(loop.gripper if joint == 5 else loop.joints[joint])",
        "angles.append(loop.joints[0])",
    )


def always_step_upwards(package):
    edit(package, RESPONSE, "1 if start < (low + high) / 2 else -1", "1")


def flip_the_sign_of_the_recorded_damping(package):
    edit(
        package,
        RESPONSE,
        "-model.actuator_biasprm[actuator, 2]",
        "model.actuator_biasprm[actuator, 2]",
    )


def rise_from_twenty_to_eighty_percent(package):
    edit(package, METRICS, "RISE_LOW, RISE_HIGH = 0.1, 0.9", "RISE_LOW, RISE_HIGH = 0.2, 0.8")


def never_report_overshoot(package):
    edit(package, METRICS, "max(0.0, progress.max() - 1)", "0.0")


def settle_within_fifty_percent(package):
    edit(package, METRICS, "SETTLE_BAND = 0.02", "SETTLE_BAND = 0.5")


def average_the_whole_record_for_the_final_value(package):
    edit(
        package,
        METRICS,
        "tail = values[-max(1, int(len(values) * TAIL_FRACTION)) :]",
        "tail = values",
    )


def keep_the_sign_of_the_miss(package):
    edit(package, METRICS, "abs(tail.mean() - target)", "(tail.mean() - target)")


MUTANTS = [
    (
        lengthen_the_control_period,
        LOOP_TESTS,
        "test_one_control_step_is_a_twentieth_of_a_second_of_physics",
    ),
    (
        forget_the_gripper_on_reset,
        LOOP_TESTS,
        "test_reset_puts_the_arm_at_rest_in_the_given_pose_and_the_tip_where_fk_says",
    ),
    (ignore_the_gripper_target, LOOP_TESTS, "test_the_gripper_target_is_followed_like_a_joint"),
    (
        never_call_the_substep_callback,
        LOOP_TESTS,
        "test_a_substep_callback_sees_every_physics_step_of_a_control_step",
    ),
    (
        offset_the_first_target_by_a_milliradian,
        LOOP_TESTS,
        "test_a_held_pose_sags_by_the_gravity_torque_over_the_servo_stiffness",
    ),
    (make_the_first_target_wobble, LOOP_TESTS, "test_a_held_pose_comes_to_rest_and_stays_there"),
    (
        add_a_little_randomness_to_the_targets,
        LOOP_TESTS,
        "test_the_loop_never_produces_nan_or_leaves_the_joint_limits_and_repeats_exactly",
    ),
    (
        give_each_control_step_only_five_physics_steps,
        LOOP_TESTS,
        "test_a_commanded_five_centimetre_move_is_achieved_within_a_millimetre_in_eight_steps",
    ),
    (
        give_each_control_step_only_five_physics_steps,
        LOOP_TESTS,
        "test_halving_the_physics_step_barely_changes_the_move_and_not_at_all_once_it_has_settled",
    ),
    (
        settle_for_no_time_before_the_step,
        RESPONSE_TESTS,
        "test_the_committed_reference_file_is_what_the_measurement_gives_today",
    ),
    (
        step_by_a_tenth_of_a_radian,
        RESPONSE_TESTS,
        "test_every_joint_has_a_response_and_the_protocol_that_produced_it",
    ),
    (
        record_the_first_joint_whatever_joint_was_stepped,
        RESPONSE_TESTS,
        "test_every_joint_follows_a_step_fast_with_moderate_overshoot_and_no_sag",
    ),
    (
        always_step_upwards,
        RESPONSE_TESTS,
        "test_every_joint_follows_a_step_fast_with_moderate_overshoot_and_no_sag",
    ),
    (
        flip_the_sign_of_the_recorded_damping,
        RESPONSE_TESTS,
        "test_the_servo_parameters_in_the_record_are_the_mjcfs",
    ),
    (
        settle_within_fifty_percent,
        RESPONSE_TESTS,
        "test_the_alternative_servo_numbers_from_joints_properties_sag_and_never_settle",
    ),
    (
        rise_from_twenty_to_eighty_percent,
        METRIC_TESTS,
        "test_a_first_order_response_rises_in_ln9_time_constants_and_never_overshoots",
    ),
    (
        never_report_overshoot,
        METRIC_TESTS,
        "test_an_underdamped_second_order_response_overshoots_by_the_textbook_amount",
    ),
    (
        settle_within_fifty_percent,
        METRIC_TESTS,
        "test_a_first_order_response_settles_within_two_percent_after_ln50_time_constants",
    ),
    (
        average_the_whole_record_for_the_final_value,
        METRIC_TESTS,
        "test_a_response_that_ends_short_reports_the_missing_part_as_the_steady_state_error",
    ),
    (
        keep_the_sign_of_the_miss,
        METRIC_TESTS,
        "test_the_steady_state_error_is_a_size_in_the_units_of_the_signal_whichever_side_it_misses_on",
    ),
]


@pytest.mark.parametrize(
    ("fault", "test_module", "failing_test"),
    MUTANTS,
    ids=[f"{f.__name__}-{t[5:40]}" for f, _, t in MUTANTS],
)
def test_the_loop_tests_fail_on_a_mutated_package(tmp_path, fault, test_module, failing_test):
    result = run_on_mutated_package(tmp_path, test_module, fault)
    assert_the_test_failed(result, test_module, failing_test)

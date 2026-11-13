"""The task-test checks are run against mutated copies of our package and must notice."""

import pytest
from support.damage import assert_the_test_failed, edit, run_on_mutated_package

TASK_TESTS = "tests/tasktests/test_task_tests.py"
RECORDER_TESTS = "tests/tasktests/test_recorder.py"
JUDGING_TESTS = "tests/scripted/test_judging.py"
RULES = "core/task_tests.py"
RECORDER = "sim/recorder.py"


def accept_a_flick_as_a_grasp(package):
    edit(package, RULES, "LIFT_STEPS = 5", "LIFT_STEPS = 1")


def accept_a_can_far_from_the_tool(package):
    edit(package, RULES, "& (distance <= HELD_DISTANCE)", "& (distance <= 100.0)")


def allow_any_speed(package):
    edit(package, RULES, "MAX_SPEED = 2.5", "MAX_SPEED = 1000.0")


def allow_any_tilt(package):
    edit(package, RULES, "MIN_UPRIGHT = math.cos(math.radians(60.0))", "MIN_UPRIGHT = -1.0")


def allow_any_arm_force(package):
    edit(package, RULES, "MAX_ARM_FORCE = 20.0", "MAX_ARM_FORCE = 1.0e9")


def ignore_the_finished_flag(package):
    edit(package, RULES, '"in_time": t.finished,', '"in_time": True,')


def accept_a_floating_can(package):
    edit(package, RULES, "REST_HEIGHT_TOLERANCE = 0.003", "REST_HEIGHT_TOLERANCE = 1.0")


def add_up_the_jumps_instead_of_averaging(package):
    edit(package, RULES, "jumps.mean()", "jumps.sum()")


def measure_distance_to_the_wrong_corner(package):
    edit(package, RULES, "[goal.center_x, goal.center_y]", "[goal.center_y, goal.center_x]")


def count_the_container_as_part_of_the_arm(package):
    edit(
        package,
        RECORDER,
        "            if body in (0, self._container):",
        "            if body == 0:",
    )


def stamp_every_row_with_time_zero(package):
    edit(package, RECORDER, 'rows["time"].append(loop.time)', 'rows["time"].append(0.0)')


FAULTS = [
    (TASK_TESTS, accept_a_flick_as_a_grasp, "test_a_can_lifted_for_only_a_moment_was_not_grasped"),
    (
        TASK_TESTS,
        accept_a_can_far_from_the_tool,
        "test_a_can_up_in_the_air_far_from_the_tool_was_not_grasped",
    ),
    (TASK_TESTS, allow_any_speed, "test_a_can_flung_fast_fails_not_thrown"),
    (TASK_TESTS, allow_any_tilt, "test_a_can_that_tips_over_on_the_way_fails_not_knocked_over"),
    (TASK_TESTS, allow_any_arm_force, "test_an_arm_pressing_hard_on_the_table_fails_arm_safe"),
    (TASK_TESTS, ignore_the_finished_flag, "test_an_episode_that_runs_out_of_time_fails_in_time"),
    (
        JUDGING_TESTS,
        accept_a_floating_can,
        "test_a_can_floating_above_the_surface_is_not_a_success",
    ),
    (
        TASK_TESTS,
        add_up_the_jumps_instead_of_averaging,
        "test_smoothness_is_the_mean_jump_between_consecutive_actions",
    ),
    (
        TASK_TESTS,
        measure_distance_to_the_wrong_corner,
        "test_the_indicators_are_numbers_that_say_how_close_the_run_came",
    ),
    (
        RECORDER_TESTS,
        count_the_container_as_part_of_the_arm,
        "test_the_arm_pressing_into_the_table_is_recorded_as_force_and_a_resting_can_is_not",
    ),
    (
        RECORDER_TESTS,
        stamp_every_row_with_time_zero,
        "test_a_recorded_episode_has_one_row_per_step_in_the_control_period",
    ),
]


@pytest.mark.parametrize(
    ("module", "fault", "failing_test"),
    FAULTS,
    ids=lambda value: getattr(value, "__name__", None) if callable(value) else None,
)
def test_the_task_test_checks_fail_on_a_mutated_package(tmp_path, module, fault, failing_test):
    result = run_on_mutated_package(tmp_path, module, fault)
    assert_the_test_failed(result, module, failing_test)

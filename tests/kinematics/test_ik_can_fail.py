"""The IK tests must notice a mutated solver, one fault at a time (medium: each run is a pytest)."""

import pytest
from support.damage import assert_the_test_failed, edit, run_on_mutated_package

pytestmark = pytest.mark.medium

IK = "core/ik.py"
IK_TESTS = "tests/kinematics/test_ik.py"
MINK_TESTS = "tests/kinematics/test_ik_against_mink.py"
CORE_TESTS = "tests/kinematics/test_core_needs_no_mujoco.py"


def step_the_wrong_way(package):
    edit(
        package, IK, "proposed = angles[:IK_JOINTS] + step", "proposed = angles[:IK_JOINTS] - step"
    )


def drop_the_pitch_step(package):
    edit(
        package,
        IK,
        "step = step + _fit_secondary(step, secondary, max_step)",
        "step = step",
    )


def stop_parking_joints_on_their_limits(package):
    edit(
        package,
        IK,
        "outside = active & ((proposed < _LOW) | (proposed > _HIGH))",
        "outside = np.zeros(IK_JOINTS, dtype=bool)",
    )


def remove_the_damping(package):
    edit(package, IK, "damping**2 * np.eye", "0.0 * np.eye")


def remove_the_step_cap(package):
    edit(
        package,
        IK,
        "step = _limit_norm(position_pinv @ position_error, max_step)",
        "step = position_pinv @ position_error",
    )


def ignore_the_start(package):
    edit(
        package, IK, "angles = np.array(start, dtype=float)", "angles = np.array(HOME, dtype=float)"
    )


def skip_the_position_correction_after_the_pitch_step(package):
    edit(package, IK, "advance(with_pitch=False)", "pass")


def aim_at_a_slightly_wrong_pitch(package):
    edit(
        package,
        IK,
        "turned = None if target_pitch is None else _turn(target_pitch - pitch(angles))",
        "turned = None if target_pitch is None else _turn(target_pitch + 0.1 - pitch(angles))",
    )


def stop_after_two_iterations(package):
    edit(package, IK, "MAX_ITERATIONS = 100", "MAX_ITERATIONS = 2")


def start_from_a_random_nudge(package):
    edit(
        package,
        IK,
        "angles = np.array(start, dtype=float)",
        "angles = np.array(start, dtype=float) + np.random.default_rng().normal(0, 1e-3, 5)",
    )


def count_a_phantom_iteration(package):
    edit(package, IK, "    iterations = 0\n", "    iterations = 1\n")


def count_every_iteration_twice(package):
    # Dropping the increment altogether would loop forever on an unreachable target.
    edit(package, IK, "        iterations += 1\n", "        iterations += 2\n")


def import_mujoco_in_the_ik(package):
    edit(package, IK, "import numpy as np\n", "import mujoco  # noqa: F401\nimport numpy as np\n")


MUTANTS = [
    (
        step_the_wrong_way,
        IK_TESTS,
        "test_position_only_ik_reaches_targets_that_forward_kinematics_says_exist",
    ),
    (
        drop_the_pitch_step,
        IK_TESTS,
        "test_ik_also_reaches_the_pitch_of_a_known_pose_when_the_arm_has_the_freedom",
    ),
    (
        drop_the_pitch_step,
        IK_TESTS,
        "test_ik_points_the_tool_straight_down_by_default_at_targets_where_that_is_possible",
    ),
    (
        stop_parking_joints_on_their_limits,
        IK_TESTS,
        "test_ik_never_leaves_the_joint_limits_and_never_touches_the_roll",
    ),
    (
        remove_the_damping,
        IK_TESTS,
        "test_damping_keeps_a_step_from_a_stretched_arm_small_where_plain_inversion_explodes",
    ),
    (
        remove_the_step_cap,
        IK_TESTS,
        "test_one_iteration_never_moves_the_joints_further_than_the_step_cap[0.2]",
    ),
    (
        remove_the_step_cap,
        IK_TESTS,
        "test_one_iteration_never_moves_the_joints_further_than_the_step_cap[0.05]",
    ),
    (
        remove_the_step_cap,
        IK_TESTS,
        "test_an_unreachable_target_gives_the_best_effort_and_converged_false_not_an_exception",
    ),
    (ignore_the_start, IK_TESTS, "test_the_solution_stays_near_the_start_for_a_nearby_target"),
    (
        skip_the_position_correction_after_the_pitch_step,
        IK_TESTS,
        "test_position_has_strict_priority_over_a_pitch_the_arm_cannot_reach",
    ),
    (
        aim_at_a_slightly_wrong_pitch,
        IK_TESTS,
        "test_ik_matches_the_closed_form_planar_solution_in_the_arms_plane",
    ),
    (start_from_a_random_nudge, IK_TESTS, "test_ik_is_deterministic"),
    (
        count_a_phantom_iteration,
        IK_TESTS,
        "test_a_target_the_arm_is_already_at_takes_no_iterations",
    ),
    (
        count_every_iteration_twice,
        IK_TESTS,
        "test_the_iteration_count_stops_at_the_cap_when_the_target_cannot_be_reached",
    ),
    (
        stop_after_two_iterations,
        MINK_TESTS,
        "test_ik_and_mink_find_the_same_joint_angles_for_a_reachable_pose_near_home",
    ),
    (
        stop_after_two_iterations,
        MINK_TESTS,
        "test_ik_ends_no_further_from_a_target_out_of_reach_than_mink_and_both_respect_the_limits",
    ),
    (
        import_mujoco_in_the_ik,
        CORE_TESTS,
        "test_inverse_kinematics_imports_and_runs_with_mujoco_unavailable",
    ),
]


@pytest.mark.parametrize(
    ("fault", "test_module", "failing_test"),
    MUTANTS,
    ids=[f"{f.__name__}-{t[5:45]}" for f, _, t in MUTANTS],
)
def test_the_ik_tests_fail_on_a_mutated_solver(tmp_path, fault, test_module, failing_test):
    result = run_on_mutated_package(tmp_path, test_module, fault)
    assert_the_test_failed(result, test_module, failing_test)

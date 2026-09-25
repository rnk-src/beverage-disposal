"""The damaged-copy harness must not pass when the damage is missing or the wrong test fails."""

import pytest
from support.damage import assert_the_test_failed, edit, run_on_damaged_copy, run_on_mutated_package

STEP_TESTS = "tests/model/test_so101_steps.py"
POSE_TEST = "test_the_arm_holds_its_rest_pose_under_gravity"


def weaken_the_servos(folder):
    edit(folder, "so101_new_calib.xml", 'kp="998.22"', 'kp="0.5"')


def test_a_fault_that_changes_nothing_is_reported(tmp_path):
    result = run_on_damaged_copy(tmp_path, STEP_TESTS, lambda folder: None)
    with pytest.raises(AssertionError):
        assert_the_test_failed(result, STEP_TESTS, POSE_TEST)


def test_a_test_that_did_not_fail_is_reported_even_when_others_did(tmp_path):
    result = run_on_damaged_copy(tmp_path, STEP_TESTS, weaken_the_servos)
    with pytest.raises(AssertionError):
        assert_the_test_failed(
            result,
            STEP_TESTS,
            "test_a_hard_push_never_carries_the_joint_past_its_stop[shoulder_pan-toward_upper_stop]",
        )


def test_a_test_that_did_fail_is_accepted(tmp_path):
    result = run_on_damaged_copy(tmp_path, STEP_TESTS, weaken_the_servos)
    assert_the_test_failed(result, STEP_TESTS, POSE_TEST)


ARM_TESTS = "tests/model/test_arm_model.py"
ROLL_TEST = "test_rolling_the_wrist_does_not_move_the_tcp"


def put_the_tcp_on_gripperframe(package):
    edit(package, "sim/arm.py", "(0.0, 0.0, -0.0981274)", "(-0.0079, -0.000218121, -0.0981274)")


def test_a_package_fault_that_changes_nothing_is_reported(tmp_path):
    result = run_on_mutated_package(tmp_path, ARM_TESTS, lambda package: None)
    with pytest.raises(AssertionError):
        assert_the_test_failed(result, ARM_TESTS, ROLL_TEST)


def test_the_tests_import_the_mutated_copy_of_the_package(tmp_path):
    # If the run imported the installed package instead, every mutant would pass unnoticed.
    result = run_on_mutated_package(tmp_path, ARM_TESTS, put_the_tcp_on_gripperframe)
    assert_the_test_failed(result, ARM_TESTS, ROLL_TEST)

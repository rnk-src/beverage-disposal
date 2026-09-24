"""The damaged-copy harness must not pass when the damage is missing or the wrong test fails."""

import pytest
from damage import assert_the_test_failed, edit, run_on_damaged_copy

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

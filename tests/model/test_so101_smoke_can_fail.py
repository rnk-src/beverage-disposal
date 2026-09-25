"""The step and render tests are run against damaged copies of the model and must notice."""

import pytest
from support.damage import assert_the_test_failed, edit, run_on_damaged_copy

STEP_TESTS = "tests/model/test_so101_steps.py"


def make_the_joint_damping_negative(folder):
    # Negative damping adds energy every step. MuJoCo then warns (mjWARN_BADQACC) and resets the
    # state to something finite, which is why the stability test reads the warning counters.
    edit(folder, "so101_new_calib.xml", 'damping="0.60"', 'damping="-50"')


def use_a_huge_timestep(folder):
    # MuJoCo does not call this unstable (no warning, finite state), but the arm drifts ~2 rad.
    edit(
        folder,
        "so101_new_calib.xml",
        '<mujoco model="so101_new_calib">',
        '<mujoco model="so101_new_calib">\n<option timestep="0.5"/>',
    )


def weaken_the_servos(folder):
    edit(folder, "so101_new_calib.xml", 'kp="998.22"', 'kp="0.5"')


def wire_the_gripper_actuator_to_the_wrist(folder):
    edit(
        folder,
        "so101_new_calib.xml",
        'name="gripper" joint="gripper"',
        'name="gripper" joint="wrist_roll"',
    )


def disable_joint_limits(folder):
    edit(
        folder,
        "so101_new_calib.xml",
        '<mujoco model="so101_new_calib">',
        '<mujoco model="so101_new_calib">\n<option><flag limit="disable"/></option>',
    )


def disable_contacts(folder):
    edit(
        folder,
        "so101_new_calib.xml",
        '<mujoco model="so101_new_calib">',
        '<mujoco model="so101_new_calib">\n<option><flag contact="disable"/></option>',
    )


FAULTS = [
    (
        make_the_joint_damping_negative,
        "test_a_thousand_steps_at_rest_stay_finite_and_raise_no_mujoco_warning",
    ),
    (use_a_huge_timestep, "test_the_arm_holds_its_rest_pose_under_gravity"),
    (weaken_the_servos, "test_the_arm_holds_its_rest_pose_under_gravity"),
    (weaken_the_servos, "test_a_position_command_moves_the_joint_to_its_target[shoulder_pan]"),
    (
        disable_joint_limits,
        "test_a_hard_push_never_carries_the_joint_past_its_stop[shoulder_pan-toward_upper_stop]",
    ),
    (
        disable_joint_limits,
        "test_a_full_range_step_command_reaches_both_ends_and_never_passes_them[shoulder_lift]",
    ),
    (
        disable_contacts,
        "test_the_gripper_touches_the_shoulder_before_the_elbow_reaches_its_upper_stop",
    ),
    (
        wire_the_gripper_actuator_to_the_wrist,
        "test_a_position_command_moves_the_joint_to_its_target[gripper]",
    ),
]


@pytest.mark.parametrize(
    ("fault", "failing_test"),
    FAULTS,
    ids=[f"{f.__name__}-{t.split('test_', 1)[1][:30]}" for f, t in FAULTS],
)
def test_the_step_tests_fail_on_a_damaged_copy(tmp_path, fault, failing_test):
    result = run_on_damaged_copy(tmp_path, STEP_TESTS, fault)
    assert_the_test_failed(result, STEP_TESTS, failing_test)


RENDER_TESTS = "tests/model/test_so101_render.py"


def hide_the_visual_geoms(folder):
    # The default view draws geom groups 0 to 2 only; group 5 is not drawn.
    edit(folder, "so101_new_calib.xml", 'group="2"', 'group="5"')


RENDER_FAULTS = [
    (hide_the_visual_geoms, "test_the_arm_is_visible_and_does_not_fill_the_frame"),
    (hide_the_visual_geoms, "test_the_render_follows_the_arms_pose"),
]


@pytest.mark.parametrize(
    ("fault", "failing_test"), RENDER_FAULTS, ids=[t for _, t in RENDER_FAULTS]
)
def test_the_render_tests_fail_on_a_damaged_copy(tmp_path, fault, failing_test):
    result = run_on_damaged_copy(tmp_path, RENDER_TESTS, fault)
    assert_the_test_failed(result, RENDER_TESTS, failing_test)

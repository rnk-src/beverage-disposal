"""The arm model tests are run against mutated copies of our package and must notice."""

import pytest
from support.damage import assert_the_test_failed, edit, run_on_mutated_package

ARM_TESTS = "tests/model/test_arm_model.py"
ARM = "sim/arm.py"


def rename_the_tcp_site(package):
    edit(package, ARM, 'name="tcp"', 'name="tip"')


def put_the_tcp_on_gripperframe(package):
    # Reusing upstream's frame is the likely mistake: it sits on the fixed pad, 7.9 mm off the axis.
    edit(package, ARM, "(0.0, 0.0, -0.0981274)", "(-0.0079, -0.000218121, -0.0981274)")


def leave_the_tcp_frame_unturned(package):
    edit(package, ARM, "TCP_QUAT = (0.0, 1.0, 0.0, 0.0)", "TCP_QUAT = (1.0, 0.0, 0.0, 0.0)")


def hang_the_tcp_on_the_moving_jaw(package):
    # The same point at gripper angle 0, written in the jaw's frame. Reusing TCP_POS there would put
    # the site on the jaw's hinge axis, where the jaw's own rotation cannot move it.
    edit(package, ARM, 'spec.body("gripper")', 'spec.body("moving_jaw_so101_v1")')
    edit(package, ARM, "(0.0, 0.0, -0.0981274)", "(-0.0202, -0.074727, 0.0188)")


def change_the_timestep_while_loading(package):
    edit(
        package,
        ARM,
        "    return spec\n",
        "    spec.option.timestep = 0.001\n    return spec\n",
    )


FAULTS = [
    (rename_the_tcp_site, "test_the_arm_model_has_a_tcp_site_on_the_gripper_body"),
    (
        put_the_tcp_on_gripperframe,
        "test_at_gripper_zero_the_tcp_is_halfway_between_the_jaws_at_the_depth_of_gripperframe",
    ),
    (
        leave_the_tcp_frame_unturned,
        "test_the_tcp_z_axis_points_to_the_fingertips_and_x_towards_the_moving_jaw",
    ),
    (put_the_tcp_on_gripperframe, "test_rolling_the_wrist_does_not_move_the_tcp"),
    (hang_the_tcp_on_the_moving_jaw, "test_opening_the_jaws_does_not_move_the_tcp"),
    (change_the_timestep_while_loading, "test_the_arm_model_is_the_vendored_model_plus_one_site"),
]


@pytest.mark.parametrize(
    ("fault", "failing_test"), FAULTS, ids=[f"{f.__name__}-{t[5:40]}" for f, t in FAULTS]
)
def test_the_arm_model_tests_fail_on_a_mutated_package(tmp_path, fault, failing_test):
    result = run_on_mutated_package(tmp_path, ARM_TESTS, fault)
    assert_the_test_failed(result, ARM_TESTS, failing_test)

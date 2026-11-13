"""The grasp tests are run against mutated copies of our package and must notice."""

import pytest
from support.damage import assert_the_test_failed, edit, run_on_mutated_package

PADS_TESTS = "tests/grasp/test_gripper_pads.py"
GRASP_TESTS = "tests/grasp/test_prescribed_grasp.py"
PADS = "sim/pads.py"
GRASP = "sim/grasp.py"
SCENE = "sim/scene.py"


def put_the_moving_pad_ten_millimetres_too_far(package):
    edit(
        package,
        PADS,
        "FIXED_FACE_X + config.container.diameter +",
        "FIXED_FACE_X + 0.01 + config.container.diameter +",
    )


def leave_the_moving_pad_unturned(package):
    edit(package, PADS, "mujoco.mju_mat2Quat(quat, rotation.T.flatten())", "quat[0] = 1.0")


def lift_the_moving_pad_ten_millimetres(package):
    edit(
        package,
        PADS,
        "0.0, pads.depth]",
        "0.0, pads.depth + 0.01]",
    )


def leave_the_jaw_collisions_on(package):
    edit(
        package,
        PADS,
        "            geom.contype = 0\n            geom.conaffinity = 0",
        "            pass",
    )


def ignore_the_force_limit(package):
    edit(package, SCENE, "[-config.grasp.force_limit, config.grasp.force_limit]", "[-3.35, 3.35]")


def ignore_the_timestep(package):
    edit(package, SCENE, "spec.option.timestep = config.physics.timestep", "pass")


def skip_the_contact_softness(package):
    edit(package, SCENE, "            geom.solref = solref", "            pass")


def aim_the_tcp_on_the_wrong_side(package):
    edit(package, GRASP, "- FACE_TO_TCP\n    tcp_xy", "+ 0.04\n    tcp_xy")


def never_close_the_gripper(package):
    edit(package, GRASP, "        if close:", "        if False:")


def report_the_height_not_the_gain(package):
    edit(package, GRASP, "lift_gain=float(after[2] - before[2])", "lift_gain=float(after[2])")


PADS_FAULTS = [
    (
        put_the_moving_pad_ten_millimetres_too_far,
        "test_the_pad_faces_are_one_container_diameter_apart_at_the_design_angle",
    ),
    (leave_the_moving_pad_unturned, "test_the_two_pad_faces_are_parallel_at_the_design_angle"),
    (lift_the_moving_pad_ten_millimetres, "test_the_pads_meet_the_can_at_the_same_height"),
    (leave_the_jaw_collisions_on, "test_only_the_pads_collide_on_the_gripper_and_the_jaw"),
    (ignore_the_force_limit, "test_the_gripper_servo_is_limited_to_the_configured_torque"),
    (ignore_the_timestep, "test_the_configured_timestep_and_contact_softness_reach_the_model"),
    (
        skip_the_contact_softness,
        "test_the_configured_timestep_and_contact_softness_reach_the_model",
    ),
]

GRASP_FAULTS = [
    (
        aim_the_tcp_on_the_wrong_side,
        "test_the_can_is_lifted_with_the_gripper_and_does_not_slip[empty]",
    ),
    (never_close_the_gripper, "test_the_can_is_lifted_with_the_gripper_and_does_not_slip[empty]"),
    (report_the_height_not_the_gain, "test_a_gripper_that_never_closes_does_not_lift_the_can"),
    (ignore_the_force_limit, "test_the_can_is_lifted_with_the_gripper_and_does_not_slip[empty]"),
]


@pytest.mark.parametrize(
    ("module", "fault", "failing_test"),
    [(PADS_TESTS, f, t) for f, t in PADS_FAULTS] + [(GRASP_TESTS, f, t) for f, t in GRASP_FAULTS],
    ids=lambda value: getattr(value, "__name__", None) if callable(value) else None,
)
def test_the_grasp_tests_fail_on_a_mutated_package(tmp_path, module, fault, failing_test):
    result = run_on_mutated_package(tmp_path, module, fault)
    assert_the_test_failed(result, module, failing_test)

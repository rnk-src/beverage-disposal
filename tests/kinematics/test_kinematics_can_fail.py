"""The kinematics tests must notice a mutated package or a damaged model, one fault at a time."""

import pytest
from support.damage import (
    assert_the_test_failed,
    edit,
    run_on_damaged_copy,
    run_on_mutated_package,
)
from support.so101 import MJCF

FK_TESTS = "tests/kinematics/test_forward_kinematics.py"
CHAIN_TESTS = "tests/kinematics/test_chain_file.py"
CORE_TESTS = "tests/kinematics/test_core_needs_no_mujoco.py"
JACOBIAN_TESTS = "tests/kinematics/test_jacobian.py"
PITCH_TESTS = "tests/kinematics/test_pitch.py"
CHAIN_FILE = "core/so101_chain.json"
KINEMATICS = "core/kinematics.py"

TCP_OFFSET = '"offset": [\n      0.0,\n      0.0,\n      -0.0981274\n    ]'
TCP_ROTATION = (
    '"rotation": [\n      [\n        1.0,\n        0.0,\n        0.0\n      ],\n'
    "      [\n        0.0,\n        -1.0,\n        0.0\n      ],\n"
    "      [\n        0.0,\n        0.0,\n        -1.0\n      ]\n    ]"
)


def put_the_tcp_on_gripperframe(package):
    # Upstream's frame on the fixed pad, 7.9 mm off the roll axis: the TCP moves 7.9 mm everywhere.
    edit(
        package,
        CHAIN_FILE,
        TCP_OFFSET,
        TCP_OFFSET.replace("0.0,\n      0.0,", "-0.0079,\n      -0.000218121,"),
    )


def leave_the_tcp_frame_unturned(package):
    # Identity instead of the half turn about x: the TCP's frame is wrong but its origin is not.
    edit(package, CHAIN_FILE, TCP_ROTATION, TCP_ROTATION.replace("-1.0", "1.0"))


def point_home_straight_up(package):
    edit(package, CHAIN_FILE, "1.5707963267948966", "-1.5707963267948966")


def turn_every_joint_the_wrong_way(package):
    edit(package, KINEMATICS, "frame @ _turn_about_z(angle)", "frame @ _turn_about_z(-angle)")


def import_mujoco_in_the_core(package):
    edit(
        package,
        KINEMATICS,
        "import numpy as np\n",
        "import mujoco  # noqa: F401\nimport numpy as np\n",
    )


def reverse_the_cross_product(package):
    # Tip velocity is axis x lever arm; the other order gives every column the wrong sign.
    edit(
        package,
        KINEMATICS,
        "np.cross(axis, tcp[:3, 3] - position)",
        "np.cross(tcp[:3, 3] - position, axis)",
    )


def flip_the_pitch_normal(package):
    edit(
        package,
        KINEMATICS,
        "normal = np.cross(heading, axes[0])",
        "normal = np.cross(axes[0], heading)",
    )


def drop_the_pitch_row(package):
    edit(
        package,
        KINEMATICS,
        "return np.vstack([np.array(linear).T, [axis @ normal for axis in axes]])",
        "return np.array(linear).T",
    )


def flip_the_pitch_sign(package):
    edit(
        package,
        KINEMATICS,
        "np.arctan2(-z_axis[2], z_axis @ heading)",
        "np.arctan2(z_axis[2], z_axis @ heading)",
    )


def ignore_the_pan_in_the_plane_heading(package):
    edit(
        package,
        KINEMATICS,
        "heading = _plane_heading(frames[0][:3, 2], angles[0])",
        "heading = np.array([1.0, 0.0, 0.0])",
    )


def use_the_elevation_angle(package):
    edit(
        package,
        KINEMATICS,
        "np.arctan2(-z_axis[2], z_axis @ heading)",
        "np.arctan2(-z_axis[2], np.hypot(z_axis[0], z_axis[1]))",
    )


MUTANTS = [
    (
        put_the_tcp_on_gripperframe,
        FK_TESTS,
        "test_fk_puts_the_tcp_where_mujoco_does_over_random_in_limit_poses",
    ),
    (
        leave_the_tcp_frame_unturned,
        FK_TESTS,
        "test_fk_turns_the_tcp_frame_the_way_mujoco_does_over_random_in_limit_poses",
    ),
    (
        point_home_straight_up,
        FK_TESTS,
        "test_the_home_pose_holds_the_tcp_under_the_wrist_pointing_straight_down",
    ),
    (
        turn_every_joint_the_wrong_way,
        FK_TESTS,
        "test_turning_the_pan_joint_turns_the_tcp_by_the_same_angle_about_the_pan_axis",
    ),
    (put_the_tcp_on_gripperframe, FK_TESTS, "test_rolling_the_wrist_leaves_the_tcp_where_it_is"),
    (
        reverse_the_cross_product,
        JACOBIAN_TESTS,
        "test_the_position_rows_match_central_finite_differences_of_fk_over_random_poses",
    ),
    (
        reverse_the_cross_product,
        JACOBIAN_TESTS,
        "test_the_position_rows_match_mujocos_site_jacobian",
    ),
    (
        flip_the_pitch_normal,
        JACOBIAN_TESTS,
        "test_the_pitch_row_matches_central_finite_differences_of_pitch_over_random_poses",
    ),
    (
        flip_the_pitch_normal,
        JACOBIAN_TESTS,
        "test_the_pitch_row_is_the_turn_rate_about_the_plane_normal_from_mujocos_angular_jacobian",
    ),
    (
        flip_the_pitch_normal,
        JACOBIAN_TESTS,
        "test_the_three_parallel_planar_joints_each_add_one_for_one_to_pitch_and_pan_and_roll_add_none",
    ),
    (
        drop_the_pitch_row,
        JACOBIAN_TESTS,
        "test_the_jacobian_has_a_row_for_x_y_z_and_pitch_and_a_column_for_each_arm_joint",
    ),
    (
        put_the_tcp_on_gripperframe,
        JACOBIAN_TESTS,
        "test_rolling_the_wrist_moves_neither_the_tcp_nor_the_pitch",
    ),
    (
        flip_the_pitch_sign,
        PITCH_TESTS,
        "test_home_points_straight_down_so_its_pitch_is_plus_ninety_degrees",
    ),
    (
        flip_the_pitch_sign,
        PITCH_TESTS,
        "test_pitch_is_zero_for_a_tool_pointing_straight_out_and_negative_for_one_pointing_up",
    ),
    (
        use_the_elevation_angle,
        PITCH_TESTS,
        "test_pitch_keeps_growing_when_the_tool_tilts_back_past_straight_down",
    ),
    (
        ignore_the_pan_in_the_plane_heading,
        PITCH_TESTS,
        "test_pitch_is_the_angle_of_mujocos_tcp_z_axis_below_the_arms_plane_heading",
    ),
    (
        ignore_the_pan_in_the_plane_heading,
        PITCH_TESTS,
        "test_pitch_is_the_sum_of_the_three_parallel_planar_joint_angles",
    ),
    (
        import_mujoco_in_the_core,
        CORE_TESTS,
        "test_forward_kinematics_imports_and_runs_with_mujoco_unavailable",
    ),
]


@pytest.mark.parametrize(
    ("fault", "test_module", "failing_test"),
    MUTANTS,
    ids=[f"{f.__name__}-{t[5:40]}" for f, _, t in MUTANTS],
)
def test_the_kinematics_tests_fail_on_a_mutated_package(tmp_path, fault, test_module, failing_test):
    result = run_on_mutated_package(tmp_path, test_module, fault)
    assert_the_test_failed(result, test_module, failing_test)


def narrow_the_elbow_range(folder):
    edit(
        folder,
        MJCF,
        'name="elbow_flex" type="hinge" range="-1.69 1.69"',
        'name="elbow_flex" type="hinge" range="-1.5 1.5"',
    )


def move_the_shoulder_one_millimetre(folder):
    # What an upstream model update looks like to the frozen file.
    edit(folder, MJCF, 'pos="0.0388353 -8.97657e-09 0.0624"', 'pos="0.0398353 -8.97657e-09 0.0624"')


DAMAGES = [
    (narrow_the_elbow_range, "test_the_chain_lists_the_five_arm_joints_in_order_with_their_limits"),
    (
        move_the_shoulder_one_millimetre,
        "test_the_committed_chain_file_is_what_the_generator_writes_from_the_model_today",
    ),
]


@pytest.mark.parametrize(
    ("fault", "failing_test"), DAMAGES, ids=[f"{f.__name__}-{t[5:40]}" for f, t in DAMAGES]
)
def test_the_chain_file_tests_fail_on_a_damaged_model(tmp_path, fault, failing_test):
    result = run_on_damaged_copy(tmp_path, CHAIN_TESTS, fault)
    assert_the_test_failed(result, CHAIN_TESTS, failing_test)

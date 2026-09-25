"""The model tests are run against damaged copies of the model and must notice the damage."""

import pytest
from damage import assert_the_test_failed, edit, run_on_damaged_copy

MODEL_TESTS = "tests/model/test_so101_model.py"


def rename_joint_in_mjcf(folder):
    edit(folder, "so101_new_calib.xml", "wrist_roll", "wrist_spin")


def shrink_shoulder_pan_range(folder):
    edit(
        folder,
        "so101_new_calib.xml",
        'range="-1.9198621771937616 1.9198621771937634"',
        'range="-1.5 1.5"',
    )


def delete_a_mesh(folder):
    (folder / "assets" / "base_so101_v2.stl").unlink()


def rename_joint_in_urdf(folder):
    edit(folder, "so101_new_calib.urdf", 'name="elbow_flex"', 'name="elbow_bend"')


def remove_the_gripper_actuator(folder):
    text = (folder / "so101_new_calib.xml").read_text()
    kept = [
        line
        for line in text.splitlines(keepends=True)
        if 'name="gripper" joint="gripper"' not in line
    ]
    assert len(kept) == len(text.splitlines(keepends=True)) - 1
    (folder / "so101_new_calib.xml").write_text("".join(kept))


def point_the_urdf_at_another_mesh(folder):
    edit(
        folder,
        "so101_new_calib.urdf",
        "assets/upper_arm_so101_v1.stl",
        "assets/upper_arm_so101_v9.stl",
    )


def put_zero_outside_a_joint_range(folder):
    edit(
        folder,
        "so101_new_calib.xml",
        'range="-1.9198621771937616 1.9198621771937634"',
        'range="0.1 1.9"',
    )


def change_the_servo_gain(folder):
    edit(folder, "so101_new_calib.xml", 'kp="998.22"', 'kp="500"')


def make_mujoco_recompute_masses_from_geometry(folder):
    edit(
        folder,
        "so101_new_calib.xml",
        'autolimits="true"',
        'autolimits="true" inertiafromgeom="true"',
    )


def remove_the_gripper_frame_site(folder):
    text = (folder / "so101_new_calib.xml").read_text()
    kept = [line for line in text.splitlines(keepends=True) if 'name="gripperframe"' not in line]
    assert len(kept) == len(text.splitlines(keepends=True)) - 1
    (folder / "so101_new_calib.xml").write_text("".join(kept))


def give_the_base_a_collision_mesh(folder):
    edit(
        folder,
        "so101_new_calib.xml",
        'class="visual" pos="-0.00636471 -9.94414e-05 -0.0024"',
        'class="collision" pos="-0.00636471 -9.94414e-05 -0.0024"',
    )


def flip_the_elbow_axis_in_the_urdf(folder):
    edit(
        folder,
        "so101_new_calib.urdf",
        '<child link="lower_arm_link"/>\n    <axis xyz="0 0 1"/>',
        '<child link="lower_arm_link"/>\n    <axis xyz="0 0 -1"/>',
    )


def move_the_elbow_origin_in_the_urdf_by_a_millimetre(folder):
    edit(
        folder,
        "so101_new_calib.urdf",
        'xyz="-0.11257 -0.028 1.73763e-16"',
        'xyz="-0.11357 -0.028 1.73763e-16"',
    )


def change_a_link_mass_in_the_urdf(folder):
    edit(folder, "so101_new_calib.urdf", '<mass value="0.104"/>', '<mass value="0.204"/>')


FAULTS = [
    (
        rename_joint_in_mjcf,
        "test_the_mjcf_has_six_hinge_joints_and_six_actuators_in_the_expected_order",
    ),
    (
        shrink_shoulder_pan_range,
        "test_each_joint_range_matches_the_expected_value_and_the_urdf_limit[shoulder_pan]",
    ),
    (delete_a_mesh, "test_the_mjcf_and_the_urdf_reference_the_same_thirteen_existing_meshes"),
    (
        rename_joint_in_urdf,
        "test_the_urdf_has_the_same_six_revolute_joints_and_one_fixed_frame_joint",
    ),
    (
        remove_the_gripper_actuator,
        "test_the_mjcf_has_six_hinge_joints_and_six_actuators_in_the_expected_order",
    ),
    (
        point_the_urdf_at_another_mesh,
        "test_the_mjcf_and_the_urdf_reference_the_same_thirteen_existing_meshes",
    ),
    (put_zero_outside_a_joint_range, "test_zero_is_strictly_inside_every_joint_range"),
    (
        change_the_servo_gain,
        "test_actuators_are_position_servos_with_the_recorded_gains_and_limits",
    ),
    (
        make_mujoco_recompute_masses_from_geometry,
        "test_total_mass_is_the_sum_of_the_inertial_masses_written_in_the_file",
    ),
    (remove_the_gripper_frame_site, "test_the_named_frames_exist[gripperframe]"),
    (give_the_base_a_collision_mesh, "test_only_the_base_has_no_colliding_geometry"),
    (flip_the_elbow_axis_in_the_urdf, "test_the_urdf_puts_every_link_where_the_mjcf_does"),
    (
        move_the_elbow_origin_in_the_urdf_by_a_millimetre,
        "test_the_urdf_puts_every_link_where_the_mjcf_does",
    ),
    (change_a_link_mass_in_the_urdf, "test_every_link_has_the_same_mass_in_the_mjcf_and_the_urdf"),
]


@pytest.mark.parametrize(("fault", "failing_test"), FAULTS, ids=[f.__name__ for f, _ in FAULTS])
def test_the_model_tests_fail_on_a_damaged_copy(tmp_path, fault, failing_test):
    result = run_on_damaged_copy(tmp_path, MODEL_TESTS, fault)
    assert_the_test_failed(result, MODEL_TESTS, failing_test)

"""The vendored SO-101 model is the follower arm with the new calibration, as expected."""

import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from so101 import MJCF, SO101, URDF, model_facts, referenced_meshes, urdf_facts

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# Radians. Transcribed from the upstream URDF <limit> elements; see the log for the caveat.
EXPECTED_RANGES = {
    "shoulder_pan": (-math.radians(110), math.radians(110)),
    "shoulder_lift": (-math.radians(100), math.radians(100)),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-math.radians(95), math.radians(95)),
    "wrist_roll": (-2.74385, 2.84121),
    "gripper": (-math.radians(10), math.radians(100)),
}
TOLERANCE = 1e-4


def test_the_mjcf_has_six_hinge_joints_and_six_actuators_in_the_expected_order():
    facts = model_facts(SO101)
    assert facts["joint_names"] == JOINTS
    assert facts["nq"] == 6
    assert facts["actuator_names"] == JOINTS


@pytest.mark.parametrize("joint", JOINTS)
def test_each_joint_range_matches_the_expected_value_and_the_urdf_limit(joint):
    lower, upper = model_facts(SO101)["joint_ranges"][joint]
    assert (lower, upper) == pytest.approx(EXPECTED_RANGES[joint], abs=TOLERANCE)
    assert (lower, upper) == pytest.approx(urdf_facts(SO101)["limits"][joint], abs=TOLERANCE)


@pytest.mark.parametrize("joint", JOINTS)
def test_each_actuator_drives_the_joint_of_the_same_name_within_its_limits(joint):
    facts = model_facts(SO101)
    assert facts["actuator_targets"][joint] == joint
    assert facts["actuator_ctrlranges"][joint] == pytest.approx(
        facts["joint_ranges"][joint], abs=TOLERANCE
    )


def test_the_urdf_has_the_same_six_revolute_joints_and_one_fixed_frame_joint():
    facts = urdf_facts(SO101)
    assert sorted(facts["revolute"]) == sorted(JOINTS)
    assert facts["fixed"] == ["gripper_frame_joint"]


def test_the_mjcf_and_the_urdf_reference_the_same_thirteen_existing_meshes():
    from_mjcf, from_urdf = referenced_meshes(SO101)
    assert len(from_mjcf) == 13
    assert from_mjcf == from_urdf
    missing = [str(path) for path in from_mjcf if not path.is_file()]
    assert missing == []


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(SO101 / MJCF))


def test_zero_is_strictly_inside_every_joint_range(model):
    # New calibration: each joint's virtual zero is the middle of its range (upstream README).
    assert np.all(model.jnt_range[:, 0] < 0) and np.all(model.jnt_range[:, 1] > 0)


def test_actuators_are_position_servos_with_the_recorded_gains_and_limits(model):
    # Values written in the MJCF; the class default force limit (2.94) is overridden per actuator.
    np.testing.assert_allclose(model.actuator_gainprm[:, 0], 998.22)
    np.testing.assert_allclose(model.actuator_biasprm[:, 1], -998.22)
    np.testing.assert_allclose(model.actuator_biasprm[:, 2], -2.731)
    np.testing.assert_allclose(model.actuator_forcerange, [[-3.35, 3.35]] * 6)


def test_total_mass_is_the_sum_of_the_inertial_masses_written_in_the_file(model):
    written = sum(float(i.get("mass")) for i in ET.parse(SO101 / MJCF).getroot().iter("inertial"))
    assert model.body_mass.sum() == pytest.approx(written, abs=1e-9)


@pytest.mark.parametrize("site", ["baseframe", "gripperframe"])
def test_the_named_frames_exist(model, site):
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site) >= 0


def test_only_the_base_has_no_colliding_geometry(model):
    # Upstream README: "Base collision meshes were removed due to problematic collision behavior".
    colliding = (model.geom_contype > 0) & (model.geom_conaffinity > 0)
    with_collider = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[i])
        for i in np.where(colliding)[0]
    }
    links = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody)}
    assert with_collider == links - {"base"}


# MJCF body names; the URDF calls each one "<name>_link". The URDF's base fuses into MuJoCo's world.
LINKS = ["shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1"]
POSES = 200
# The URDF writes angles to 5 decimals (up to 2.65e-6 rad each); over five joints and a 0.4 m arm
# that is a few micrometres. Measured worst: 2.8e-6 m and 1.6e-5.
LINK_POSITION_TOLERANCE = 1e-5
LINK_ROTATION_TOLERANCE = 1e-4


def test_the_urdf_puts_every_link_where_the_mjcf_does():
    # Gazebo simulates the URDF, so an origin or axis that differs from the MJCF would move or
    # mirror the arm there while the name and limit checks above stay green.
    mjcf = mujoco.MjModel.from_xml_path(str(SO101 / MJCF))
    urdf = mujoco.MjModel.from_xml_path(str(SO101 / URDF))
    assert [urdf.joint(i).name for i in range(urdf.njnt)] == JOINTS
    from_mjcf, from_urdf = mujoco.MjData(mjcf), mujoco.MjData(urdf)
    rng = np.random.default_rng(0)
    for _ in range(POSES):
        from_mjcf.qpos[:] = from_urdf.qpos[:] = rng.uniform(*mjcf.jnt_range.T)
        mujoco.mj_kinematics(mjcf, from_mjcf)
        mujoco.mj_kinematics(urdf, from_urdf)
        for link in LINKS:
            a, b = from_mjcf.body(link), from_urdf.body(f"{link}_link")
            assert np.linalg.norm(a.xpos - b.xpos) < LINK_POSITION_TOLERANCE, link
            assert np.abs(a.xmat - b.xmat).max() < LINK_ROTATION_TOLERANCE, link


def test_every_link_has_the_same_mass_in_the_mjcf_and_the_urdf(model):
    # Read as plain XML, not through MuJoCo: a mass that differs would make the arm sag and load its
    # joints differently in Gazebo.
    links = ET.parse(SO101 / URDF).getroot().iter("link")
    written = {
        link.get("name"): float(link.find("inertial/mass").get("value"))
        for link in links
        if link.find("inertial") is not None
    }
    for link in ["base", *LINKS]:
        assert model.body(link).mass[0] == pytest.approx(written[f"{link}_link"], abs=1e-9), link

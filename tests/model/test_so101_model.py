"""The vendored SO-101 model is the follower arm with the new calibration, as expected."""

import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from so101 import MJCF, SO101, model_facts, referenced_meshes, urdf_facts

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

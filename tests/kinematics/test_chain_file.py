"""The frozen chain file: generated from MuJoCo, it is the robot spec the NumPy kinematics reads."""

import importlib.util
import shutil

import pytest
from support.damage import edit
from support.paths import ROOT
from support.so101 import EXPECTED_RANGES, JOINTS, MJCF, SO101

from beverage_disposal.sim.arm import load_arm

GENERATOR = ROOT / "scripts" / "generate_so101_chain.py"
# The URDF writes limits to 5 decimals, so the transcribed table is only that exact.
LIMIT_TOLERANCE = 1e-4


def load_generator():
    spec = importlib.util.spec_from_file_location("generate_so101_chain", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator():
    return load_generator()


@pytest.fixture(scope="module")
def arm():
    return load_arm(SO101 / MJCF)


def test_the_chain_lists_the_five_arm_joints_in_order_with_their_limits(generator, arm):
    # The gripper is not in the chain (it does not move the TCP), but its limits belong in the spec.
    chain = generator.chain(arm)
    assert [joint["name"] for joint in chain["joints"]] == JOINTS[:5]
    for joint in chain["joints"]:
        expected = EXPECTED_RANGES[joint["name"]]
        assert joint["range"] == pytest.approx(expected, abs=LIMIT_TOLERANCE), joint["name"]
    assert chain["gripper_range"] == pytest.approx(EXPECTED_RANGES["gripper"], abs=LIMIT_TOLERANCE)


ELBOW = '<joint axis="0 0 1" name="elbow_flex" type="hinge"'


@pytest.mark.parametrize(
    "damaged_joint",
    [
        '<joint axis="0 1 0" name="elbow_flex" type="hinge"',
        '<joint axis="0 0 1" pos="0.01 0 0" name="elbow_flex" type="hinge"',
        '<joint axis="0 0 1" name="elbow_flex" type="slide"',
    ],
    ids=["tilted-axis", "off-origin", "slide"],
)
def test_the_generator_refuses_a_joint_the_numpy_chain_cannot_represent(
    generator, tmp_path, damaged_joint
):
    # The NumPy chain turns each joint about its body's own z axis, through the body origin. A model
    # built any other way must stop the generator, not produce a file that is quietly wrong.
    copy = tmp_path / "so101"
    shutil.copytree(SO101, copy)
    edit(copy, MJCF, ELBOW, damaged_joint)
    with pytest.raises(ValueError, match="elbow_flex"):
        generator.chain(load_arm(copy / MJCF))


COMMITTED = ROOT / "src" / "beverage_disposal" / "core" / "so101_chain.json"


def test_the_committed_chain_file_is_what_the_generator_writes_from_the_model_today(
    generator, tmp_path
):
    # The file is frozen so the ROS side needs no MuJoCo; this keeps it from going stale when the
    # vendored model or our loader changes. Regenerate with scripts/generate_so101_chain.py.
    fresh = tmp_path / "so101_chain.json"
    generator.main(["--model", str(SO101 / MJCF), "--out", str(fresh)])
    assert COMMITTED.read_text() == fresh.read_text()

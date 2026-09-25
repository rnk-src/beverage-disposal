"""Write the SO-101's kinematic chain, read from MuJoCo, to the NumPy kinematics' JSON file."""

import argparse
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

from beverage_disposal.sim.arm import load_arm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "assets" / "so101" / "so101_new_calib.xml"
DEFAULT_OUT = ROOT / "src" / "beverage_disposal" / "core" / "so101_chain.json"

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
# The owner's choice (D-79): arm straight out, wrist turned so the gripper points straight down.
HOME = [0.0, 0.0, 0.0, math.pi / 2, 0.0]


def chain(model):
    """The chain of `model` (the arm as `load_arm` builds it) as plain lists, ready for JSON."""
    for name in ARM_JOINTS:
        joint = model.joint(name)
        # The NumPy chain turns each joint about its body's z axis through the body origin, which is
        # how the vendored model is built; anything else would give a file that is quietly wrong.
        hinge = joint.type[0] == mujoco.mjtJoint.mjJNT_HINGE
        if (
            not hinge
            or joint.axis.tolist() != [0.0, 0.0, 1.0]
            or joint.pos.tolist() != [0.0, 0.0, 0.0]
        ):
            raise ValueError(f"{name} is not a hinge about its body's z axis through the origin")
    return {
        "units": "metres and radians; each rotation is a 3x3 matrix, one row per list",
        "joints": [
            {
                "name": name,
                # Where the joint's body sits in its parent's frame when the joint is at zero.
                **frame(model.body(model.joint(name).bodyid[0])),
                "range": model.joint(name).range.tolist(),
            }
            for name in ARM_JOINTS
        ],
        "gripper_range": model.joint("gripper").range.tolist(),
        "tcp": frame(model.site("tcp")),
        "home": HOME,
    }


def frame(element):
    """Offset and rotation of a MuJoCo body or site in its parent body's frame."""
    rotation = np.zeros(9)
    # MuJoCo converts its own quaternions, so the NumPy side never has to know their convention.
    mujoco.mju_quat2Mat(rotation, element.quat)
    return {"offset": element.pos.tolist(), "rotation": rotation.reshape(3, 3).tolist()}


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="MJCF file to read")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="JSON file to write")
    args = parser.parse_args(argv)
    args.out.write_text(json.dumps(chain(load_arm(args.model)), indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])

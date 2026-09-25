"""Forward kinematics of the SO-101 in plain NumPy, from the frozen chain file (no MuJoCo)."""

import json
from importlib import resources

import numpy as np


def load_chain():
    # Read through importlib.resources so the file is found wherever the package is installed.
    return json.loads(resources.files(__package__).joinpath("so101_chain.json").read_text())


def _transform(rotation, offset):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = offset
    return matrix


def _turn_about_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    matrix = np.eye(4)
    matrix[:2, :2] = [[c, -s], [s, c]]
    return matrix


_CHAIN = load_chain()
_LINKS = [_transform(joint["rotation"], joint["offset"]) for joint in _CHAIN["joints"]]
_TCP = _transform(_CHAIN["tcp"]["rotation"], _CHAIN["tcp"]["offset"])
# Arm straight out with the gripper pointing straight down (pitch +90 degrees).
HOME = np.array(_CHAIN["home"])


def fk(angles):
    """TCP pose in the arm's base frame, as a 4x4 homogeneous transform.

    `angles` are the five arm joints, shoulder_pan to wrist_roll, in radians. The gripper is left
    out: the TCP is fixed to the gripper body, so opening the jaws does not move it.
    """
    pose = np.eye(4)
    for link, angle in zip(_LINKS, angles, strict=True):
        pose = pose @ link @ _turn_about_z(angle)
    return pose @ _TCP

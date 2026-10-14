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


def _joint_frames(angles):
    """Each joint's frame in the base frame, and the TCP pose.

    A joint's frame is its body's frame just before the joint turns, so the joint's axis is that
    frame's z column and its position is that frame's origin.
    """
    frames = []
    pose = np.eye(4)
    for link, angle in zip(_LINKS, angles, strict=True):
        frame = pose @ link
        frames.append(frame)
        pose = frame @ _turn_about_z(angle)
    return frames, pose @ _TCP


def fk(angles):
    """TCP pose in the arm's base frame, as a 4x4 homogeneous transform.

    `angles` are the five arm joints, shoulder_pan to wrist_roll, in radians. The gripper is left
    out: the TCP is fixed to the gripper body, so opening the jaws does not move it.
    """
    return _joint_frames(angles)[1]


def _plane_heading(pan_axis, pan_angle):
    """Horizontal direction the arm's vertical plane points in: base x turned by the pan angle."""
    x_axis = np.array([1.0, 0.0, 0.0])
    c, s = np.cos(pan_angle), np.sin(pan_angle)
    # Rodrigues' rotation formula.
    return x_axis * c + np.cross(pan_axis, x_axis) * s + pan_axis * (pan_axis @ x_axis) * (1 - c)


def pitch(angles):
    """How far below horizontal the TCP's z axis points, in radians, positive down (D-15).

    Top-down is +pi/2 and straight out is 0. It is the signed angle inside the arm's vertical plane,
    not the elevation angle: the elevation has no derivative where the tool points straight down,
    which is the home pose, and the Jacobian and IK have to work there. The range is (-pi, pi].
    """
    frames, tcp = _joint_frames(angles)
    heading = _plane_heading(frames[0][:3, 2], angles[0])
    z_axis = tcp[:3, 2]
    return np.arctan2(-z_axis[2], z_axis @ heading)


def jacobian(angles):
    """How the TCP's x, y, z (base frame) and pitch change per radian of each arm joint.

    A 4x5 matrix: rows x, y, z, pitch; columns shoulder_pan to wrist_roll. A revolute joint moves
    the tip with velocity axis x (tip - joint position). Pitch grows when the tool z axis turns
    about the plane's normal, which points so that the turn tips the tool downwards, so each pitch
    entry is that joint's axis along the normal. The pan axis and the roll axis are both
    perpendicular to it, so those two entries are zero.
    """
    frames, tcp = _joint_frames(angles)
    axes = [frame[:3, 2] for frame in frames]
    positions = [frame[:3, 3] for frame in frames]
    heading = _plane_heading(axes[0], angles[0])
    normal = np.cross(heading, axes[0])
    linear = [
        np.cross(axis, tcp[:3, 3] - position)
        for axis, position in zip(axes, positions, strict=True)
    ]
    return np.vstack([np.array(linear).T, [axis @ normal for axis in axes]])

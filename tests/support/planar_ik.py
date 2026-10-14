"""Closed-form IK for the arm's vertical plane: a test oracle for the numerical solver.

With the pan at 0 and the roll locked, lift, elbow and wrist_flex turn about parallel axes, so the
arm is a three-link planar chain. Fixing the tool's pitch fixes the wrist point, and the other two
links are then the classic two-link problem, solved with the law of cosines.
"""

import mujoco
import numpy as np

from support.so101 import MJCF, SO101


def _angle(vector):
    return np.arctan2(vector[1], vector[0])


def _plane_anchors():
    """Joint anchors and the tool offset at the zero pose, as (x, z) pairs read from MuJoCo."""
    from beverage_disposal.sim.arm import load_arm

    model = load_arm(SO101 / MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)

    def xz(position):
        return np.array([position[0], position[2]])

    shoulder, elbow, wrist = (
        xz(data.body(name).xpos) for name in ("upper_arm", "lower_arm", "wrist")
    )
    return shoulder, elbow - shoulder, wrist - elbow, xz(data.site("tcp").xpos) - wrist


def planar_solutions(tip_xz, pitch, start):
    """The (lift, elbow, wrist_flex) solutions that put the tip at `tip_xz` with this pitch."""
    shoulder, upper, forearm, tool = _plane_anchors()
    # Angles in the (x, z) plane: a positive joint turn is a clockwise one (x swings down).
    pitch_turn = np.array([[np.cos(pitch), np.sin(pitch)], [-np.sin(pitch), np.cos(pitch)]])
    wrist_point = tip_xz - pitch_turn @ tool
    reach = wrist_point - shoulder
    cos_between = (reach @ reach - upper @ upper - forearm @ forearm) / (
        2 * np.linalg.norm(upper) * np.linalg.norm(forearm)
    )
    solutions = []
    for sign in (1, -1):
        elbow = _angle(forearm) - _angle(upper) - sign * np.arccos(np.clip(cos_between, -1, 1))
        # The forearm points at (angle of forearm - elbow) once the upper arm has turned by lift.
        stacked = upper + np.array(
            [
                np.linalg.norm(forearm) * np.cos(_angle(forearm) - elbow),
                np.linalg.norm(forearm) * np.sin(_angle(forearm) - elbow),
            ]
        )
        lift = _angle(stacked) - _angle(reach)
        solutions.append(np.array([lift, elbow, pitch - lift - elbow]))
    return min(solutions, key=lambda q: np.linalg.norm(q - np.asarray(start)[1:4])), solutions

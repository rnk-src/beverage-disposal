"""Turns a policy action [dx, dy, dz, gripper] into joint targets, in plain NumPy."""

from dataclasses import dataclass

import numpy as np

from beverage_disposal.core.ik import POSITION_TOLERANCE, solve_ik
from beverage_disposal.core.kinematics import fk, load_chain

# Metres the tip target moves for an action of 1.
MAX_DELTA = 0.015
# Base frame, metres. Placeholder until 2.1 places the table: chosen from the arm's reach (position
# reachable at 97 % of the grid points inside it, measured in 1.5).
WORKSPACE = (np.array([0.12, -0.16, 0.0]), np.array([0.30, 0.16, 0.20]))
# How far the target may run ahead of the measured tip. Tuned by measurement in 1.6.
MAX_TIP_LEAD = 0.05
# Radians the gripper target may move per control step: the STS3215 servo turns about 270 deg/s
# unloaded (recalled from its datasheet, unverified), which is 0.24 rad per 50 ms.
MAX_GRIPPER_STEP = 0.2
_GRIPPER_RANGE = load_chain()["gripper_range"]


@dataclass(frozen=True)
class Command:
    joints: np.ndarray
    gripper: float
    target: np.ndarray
    ik_ok: bool


class EEDeltaController:
    """Previous commanded target + scaled delta, kept in the box and near the tip, then IK.

    Relative to the previous *target*, a delta means what it says and gravity sag gets corrected;
    the price is that a blocked arm lets the target run away, which the clamp on the distance from
    the measured tip bounds (D-19).
    """

    def __init__(
        self,
        max_delta=MAX_DELTA,
        workspace=WORKSPACE,
        max_tip_lead=MAX_TIP_LEAD,
        max_gripper_step=MAX_GRIPPER_STEP,
        pitch=np.pi / 2,
    ):
        self._max_delta = max_delta
        self._workspace = workspace
        self._max_tip_lead = max_tip_lead
        self._max_gripper_step = max_gripper_step
        self._pitch = (
            pitch  # tool tilt below horizontal, radians; the roll is whatever reset was given
        )

    def reset(self, joints, gripper):
        """Start from the measured arm: the first target is where the tip is."""
        self._joints = np.array(joints, dtype=float)
        self._target = fk(self._joints)[:3, 3]
        self._gripper = float(gripper)

    def step(self, action, joints, gripper):
        action = np.asarray(action, dtype=float)
        if action.shape != (4,) or not np.all(np.isfinite(action)):
            raise ValueError(f"action must be four finite numbers, got {action}")
        action = np.clip(action, -1.0, 1.0)
        target = np.clip(self._target + self._max_delta * action[:3], *self._workspace)
        lead = target - fk(np.asarray(joints, dtype=float))[:3, 3]
        distance = np.linalg.norm(lead)
        if distance > self._max_tip_lead:
            target = target - lead * (1 - self._max_tip_lead / distance)
        # The tip comes first; if the pitch cannot be had there, that is not a failure.
        result = solve_ik(target, self._pitch, start=self._joints)
        ik_ok = result.position_error < POSITION_TOLERANCE
        if ik_ok:
            self._joints, self._target = result.angles, target
        self._gripper = self._slew_gripper(action[3])
        return Command(self._joints.copy(), self._gripper, self._target.copy(), ik_ok)

    def _slew_gripper(self, action):
        low, high = _GRIPPER_RANGE
        wanted = low + (action + 1) / 2 * (high - low)
        step = np.clip(wanted - self._gripper, -self._max_gripper_step, self._max_gripper_step)
        return float(self._gripper + step)
